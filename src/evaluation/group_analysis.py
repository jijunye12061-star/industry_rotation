#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: group_analysis.py
@time: 2025/12/18
@description:
分组测试模块 - 基类与具体实现
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod
import logging
from datetime import timedelta

from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


class BaseGroupAnalyzer(ABC):
    """
    分组测试基类

    定义分组测试的通用流程，子类实现具体的持仓构建逻辑
    """

    def __init__(self,
                 trading_days_per_year: int = 252,
                 risk_free_rate: float = 0.0):
        """
        Args:
            trading_days_per_year: 每年交易日数
            risk_free_rate: 无风险利率（年化，小数形式）
        """
        self.trading_days_per_year = trading_days_per_year
        self.risk_free_rate = risk_free_rate

    def analyze(self,
                factor: pd.DataFrame,
                config: FactorConfig,
                backtest_start_date: str,
                backtest_end_date: Optional[str] = None) -> Dict:
        """
        执行分组测试

        Args:
            factor: 因子值 DataFrame (调仓日 × 行业代码)
            config: 配置对象
            backtest_start_date: 回测起始日期 'YYYY-MM-DD'
                - 必须是调仓日（在factor.index中）
                - 该日净值=1.0，收盘价确定持仓
            backtest_end_date: 回测结束日期 'YYYY-MM-DD' (可选)
                - 净值计算到该日期的最后一个交易日
                - 如不指定，则计算到数据的最后一天

        Returns:
            {
                'nav': 净值曲线 DataFrame,
                'metrics': 绩效指标 DataFrame,
                'holdings': 持仓记录
            }
        """
        logger.info("开始分组测试...")

        # 1. 验证并确定回测起始调仓日
        backtest_start = pd.to_datetime(backtest_start_date)
        if backtest_start not in factor.index:
            raise ValueError(f"回测起始日期 {backtest_start_date} 不在因子调仓日中")

        logger.info(f"回测起始调仓日: {backtest_start.strftime('%Y-%m-%d')}")

        # 2. 获取行业日收益率
        industry_returns = self._get_industry_returns(config)

        # 3. 确定回测结束日期
        if backtest_end_date:
            backtest_end = pd.to_datetime(backtest_end_date)
            # 找到 <= backtest_end 的最后一个交易日
            valid_dates = industry_returns.index[industry_returns.index <= backtest_end]
            if len(valid_dates) == 0:
                raise ValueError(f"收益率数据中没有 <= {backtest_end_date} 的交易日")
            last_trading_date = valid_dates[-1]
        else:
            last_trading_date = industry_returns.index[-1]

        logger.info(f"回测结束日期: {last_trading_date.strftime('%Y-%m-%d')}")

        # 4. 筛选回测期间的因子和收益率
        factor_backtest = factor.loc[backtest_start:]
        returns_backtest = industry_returns.loc[backtest_start:last_trading_date]

        # 5. 生成持仓（子类实现）
        holdings = self._create_holdings(factor_backtest, returns_backtest.columns)

        # 6. 计算各组日收益率
        group_returns = self._calculate_group_returns(holdings, returns_backtest, backtest_start)

        # 7. 计算净值曲线
        nav = self._calculate_nav(group_returns, backtest_start)

        # 8. 计算绩效指标
        metrics = self._calculate_metrics(group_returns, nav, factor_backtest.index)

        logger.info("分组测试完成")

        return {
            'nav': nav,
            'metrics': metrics,
            'holdings': holdings
        }

    @abstractmethod
    def _create_holdings(self,
                         factor: pd.DataFrame,
                         all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """
        生成持仓记录（抽象方法，子类实现）

        Args:
            factor: 因子值 DataFrame
            all_industries: 所有行业代码

        Returns:
            持仓字典 {组名: 持仓DataFrame}
        """
        pass

    @staticmethod
    def _get_industry_returns(config: FactorConfig) -> pd.DataFrame:
        """获取行业日收益率"""
        extended_end = (pd.to_datetime(config.end_date) + timedelta(days=60)).strftime('%Y-%m-%d')
        loader = get_loader()
        returns_df = loader.get_industry_returns(config.start_date, extended_end)
        logger.info(f"获取行业日收益率，形状: {returns_df.shape}")
        return returns_df

    @staticmethod
    def _calculate_group_returns(holdings: Dict[str, pd.DataFrame],
                                 industry_returns: pd.DataFrame,
                                 first_rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        计算各组日收益率

        逻辑说明（以20151231为例）：
        - 20151231（调仓日）：收盘价确定持仓，当日不产生收益
        - 20160101（下一交易日）：开始产生收益
            收益率 = 20160101的行业收益率 × 持仓权重

        Args:
            holdings: 持仓记录
            industry_returns: 行业日收益率
            first_rebalance_date: 第一个调仓日
        """
        # 从调仓日的下一个交易日开始计算收益
        all_dates = industry_returns.index[industry_returns.index > first_rebalance_date]

        if len(all_dates) == 0:
            raise ValueError(f"调仓日 {first_rebalance_date} 之后没有交易数据")

        group_returns_dict = {}

        for group_name, holding_df in holdings.items():
            # 持仓权重前向填充
            holding_daily = holding_df.reindex(all_dates, method='ffill')

            # 组合收益率 = Σ(权重 × 行业收益率)
            group_ret = (holding_daily * industry_returns.loc[all_dates]).sum(axis=1)
            group_returns_dict[group_name] = group_ret

        group_returns = pd.DataFrame(group_returns_dict)

        logger.info(f"计算组合日收益率，形状: {group_returns.shape}")
        return group_returns

    @staticmethod
    def _calculate_nav(returns: pd.DataFrame,
                       first_rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        计算净值曲线

        逻辑说明（以20151231为例）：
        - 20151231（调仓日）：净值 = 1.0（初始值）
        - 20160101（下一交易日）：净值 = 1.0 × (1 + 收益率_20160101 / 100)
        - 20160104：净值 = 上一日净值 × (1 + 收益率_20160104 / 100)

        Args:
            returns: 日收益率 DataFrame（从调仓日下一交易日开始）
            first_rebalance_date: 第一个调仓日
        """
        # 收益率转为小数
        daily_returns = returns / 100.0

        # 计算累积净值
        nav = (1 + daily_returns).cumprod()

        # 在调仓日插入初始净值1.0
        nav = pd.concat([
            pd.DataFrame(1.0, index=[first_rebalance_date], columns=nav.columns),
            nav
        ]).sort_index()

        logger.info("计算净值曲线完成")
        return nav

    def _calculate_metrics(self,
                           returns: pd.DataFrame,
                           nav: pd.DataFrame,
                           rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """计算绩效指标"""
        metrics = {}

        n_days = len(returns)
        n_years = n_days / self.trading_days_per_year

        for col in returns.columns:
            ret = returns[col]
            nav_series = nav[col]

            # 总收益
            total_return = (nav_series.iloc[-1] - 1) * 100

            # 年化收益
            annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

            # 年化波动率
            annual_vol = ret.std() * np.sqrt(self.trading_days_per_year)

            # 最大回撤
            cum_max = nav_series.cummax()
            drawdown = (nav_series - cum_max) / cum_max * 100
            max_drawdown = drawdown.min()

            # 调仓胜率
            rebalance_returns = self._get_period_returns(ret, rebalance_dates)
            win_rate = (rebalance_returns > 0).sum() / len(rebalance_returns) * 100 if len(
                rebalance_returns) > 0 else 0

            # 夏普比率
            excess_return = ret.mean() - self.risk_free_rate / self.trading_days_per_year
            sharpe = (excess_return * self.trading_days_per_year) / (
                    ret.std() * np.sqrt(self.trading_days_per_year)) if ret.std() > 0 else 0

            metrics[col] = {
                '总收益(%)': total_return,
                '年化收益(%)': annual_return,
                '年化波动(%)': annual_vol,
                '最大回撤(%)': max_drawdown,
                '调仓胜率(%)': win_rate,
                '夏普比率': sharpe
            }

        metrics_df = pd.DataFrame(metrics).T
        logger.info("计算绩效指标完成")
        return metrics_df

    @staticmethod
    def _get_period_returns(daily_returns: pd.Series,
                            rebalance_dates: pd.DatetimeIndex) -> pd.Series:
        """
        计算调仓期间收益率

        注意：调仓日当天不计入收益（当日收盘价才调仓）
        """
        period_returns = []

        for i in range(len(rebalance_dates) - 1):
            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i + 1]

            # 获取期间收益（不包含调仓日当天）
            period_ret = daily_returns.loc[start_date:end_date]

            if len(period_ret) > 1:
                period_ret = period_ret.iloc[1:]  # 排除调仓日
                cum_ret = ((1 + period_ret / 100).prod() - 1) * 100
                period_returns.append(cum_ret)

        return pd.Series(period_returns)


class LongShortAnalyzer(BaseGroupAnalyzer):
    """
    多空模式分析器

    将行业分为：多头组、空头组、基准组
    """

    def __init__(self,
                 long_size: int = 6,
                 short_size: int = 6,
                 trading_days_per_year: int = 252,
                 risk_free_rate: float = 0.0):
        """
        Args:
            long_size: 多头组行业数量
            short_size: 空头组行业数量
        """
        super().__init__(trading_days_per_year, risk_free_rate)
        self.long_size = long_size
        self.short_size = short_size
        logger.info(f"初始化多空分析器: 多头{long_size}个, 空头{short_size}个")

    def _create_holdings(self,
                         factor: pd.DataFrame,
                         all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """生成多空持仓"""
        long_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)
        short_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)
        benchmark_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)

        for date in factor.index:
            factor_values = factor.loc[date].dropna()

            if len(factor_values) < self.long_size + self.short_size:
                logger.warning(f"{date}: 有效行业数不足，跳过")
                continue

            sorted_industries = factor_values.sort_values(ascending=False)

            # 多头：因子值最高的N个
            long_industries = sorted_industries.head(self.long_size).index
            long_holdings.loc[date, long_industries] = 1.0 / self.long_size

            # 空头：因子值最低的N个
            short_industries = sorted_industries.tail(self.short_size).index
            short_holdings.loc[date, short_industries] = 1.0 / self.short_size

            # 基准：所有有效行业等权
            valid_industries = factor_values.index
            benchmark_holdings.loc[date, valid_industries] = 1.0 / len(valid_industries)

        logger.info(f"生成多空持仓，调仓次数: {len(factor)}")

        return {
            'long': long_holdings,
            'short': short_holdings,
            'benchmark': benchmark_holdings
        }

    def _calculate_group_returns(self,
                                 holdings: Dict[str, pd.DataFrame],
                                 industry_returns: pd.DataFrame,
                                 first_rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """计算多空组收益（添加派生指标）"""
        # 调用基类方法
        group_returns = super()._calculate_group_returns(
            holdings, industry_returns, first_rebalance_date
        )

        # 添加派生指标
        group_returns['多头/基准'] = group_returns['long'] - group_returns['benchmark']
        group_returns['多空'] = group_returns['long'] - group_returns['short']

        # 重命名
        group_returns = group_returns.rename(columns={
            'long': '多头',
            'short': '空头',
            'benchmark': '基准'
        })

        return group_returns


class MultiGroupAnalyzer(BaseGroupAnalyzer):
    """
    多分组模式分析器

    将行业按因子值分为N组，每组M个行业
    """

    def __init__(self,
                 n_groups: int = 5,
                 industries_per_group: int = 6,
                 trading_days_per_year: int = 252,
                 risk_free_rate: float = 0.0):
        """
        Args:
            n_groups: 分组数量
            industries_per_group: 每组行业数量
        """
        super().__init__(trading_days_per_year, risk_free_rate)
        self.n_groups = n_groups
        self.industries_per_group = industries_per_group
        logger.info(f"初始化多分组分析器: {n_groups}组, 每组{industries_per_group}个行业")

    def _create_holdings(self,
                         factor: pd.DataFrame,
                         all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """生成多分组持仓"""
        holdings = {}

        # 初始化各组
        for i in range(self.n_groups):
            holdings[f'G{i + 1}'] = pd.DataFrame(0.0, index=factor.index, columns=all_industries)

        for date in factor.index:
            factor_values = factor.loc[date].dropna()

            required_industries = self.n_groups * self.industries_per_group
            if len(factor_values) < required_industries:
                logger.warning(f"{date}: 有效行业数({len(factor_values)})不足{required_industries}，跳过")
                continue

            # 按因子值降序排列
            sorted_industries = factor_values.sort_values(ascending=False)

            # 分配到各组
            for i in range(self.n_groups):
                start_idx = i * self.industries_per_group
                end_idx = start_idx + self.industries_per_group
                group_industries = sorted_industries.iloc[start_idx:end_idx].index

                # 组内等权
                holdings[f'G{i + 1}'].loc[date, group_industries] = 1.0 / self.industries_per_group

        logger.info(f"生成{self.n_groups}组持仓，调仓次数: {len(factor)}")

        return holdings


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    from factors.base import create_factor

    test_config = FactorConfig(
        start_date='2015-12-31',
        end_date='2025-03-31',
        frequency='monthly'
    )

    # 计算因子
    factor_calc = create_factor("upside_beta", window=21, preprocess=False)
    marginal_beta = factor_calc(test_config)

    print("\n=== 测试1: 多空模式 ===")
    analyzer1 = LongShortAnalyzer(long_size=6, short_size=6)
    results1 = analyzer1.analyze(
        marginal_beta,
        test_config,
        backtest_start_date='2015-12-31',
        backtest_end_date='2025-03-31'
    )
    print("\n绩效指标:")
    print(results1['metrics'].round(2))
    print("\n净值曲线（前3期）:")
    print(results1['nav'].head(3))

    # print("\n=== 测试2: 多分组模式 ===")
    # analyzer2 = MultiGroupAnalyzer(n_groups=5, industries_per_group=6)
    # results2 = analyzer2.analyze(
    #     marginal_beta,
    #     test_config,
    #     backtest_start_date='2016-01-31',
    #     backtest_end_date='2024-12-31'
    # )
    # print("\n绩效指标:")
    # print(results2['metrics'].round(2))
    # print("\n净值曲线（前3期）:")
    # print(results2['nav'].head(3))