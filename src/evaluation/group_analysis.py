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
    """分组测试基类 - 统一计算净值曲线"""

    def __init__(self, trading_days_per_year: int = 252, risk_free_rate: float = 0.0):
        self.trading_days_per_year = trading_days_per_year
        self.risk_free_rate = risk_free_rate

    def analyze(self,
                factor: pd.DataFrame,
                config: FactorConfig,
                backtest_start_date: str,
                backtest_end_date: Optional[str] = None) -> Dict:
        """执行分组测试"""
        logger.info("开始分组测试...")

        # 1. 验证回测起始日必须是调仓日
        backtest_start = pd.to_datetime(backtest_start_date)
        if backtest_start not in factor.index:
            raise ValueError(f"回测起始日期 {backtest_start_date} 不在因子调仓日中")

        # 2. 获取行业日收益率（扩展到回测结束后60天，确保数据充足）
        industry_returns = self._get_industry_returns(config)

        # 3. 确定实际回测结束日（最后一个可用交易日）
        if backtest_end_date:
            backtest_end = pd.to_datetime(backtest_end_date)
            valid_dates = industry_returns.index[industry_returns.index <= backtest_end]
            if len(valid_dates) == 0:
                raise ValueError(f"收益率数据中没有 <= {backtest_end_date} 的交易日")
            last_trading_date = valid_dates[-1]
        else:
            last_trading_date = industry_returns.index[-1]

        logger.info(f"回测区间: {backtest_start.strftime('%Y-%m-%d')} -> {last_trading_date.strftime('%Y-%m-%d')}")

        # 4. 筛选回测期间的数据
        factor_backtest = factor.loc[backtest_start:]
        returns_backtest = industry_returns.loc[backtest_start:last_trading_date]

        # 5. 生成持仓记录（子类实现）
        holdings = self._create_holdings(factor_backtest, returns_backtest.columns)

        # 6. 计算净值曲线
        nav = self._calculate_nav(holdings, returns_backtest, backtest_start)

        # 7. 计算绩效指标
        metrics = self._calculate_metrics(nav, factor_backtest.index)

        logger.info("分组测试完成")
        return {'nav': nav, 'metrics': metrics, 'holdings': holdings}

    @abstractmethod
    def _create_holdings(self, factor: pd.DataFrame, all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """
        生成持仓记录（子类实现）

        返回: {组名: 持仓权重DataFrame (调仓日 × 行业代码)}
        """
        pass

    @staticmethod
    def _get_industry_returns(config: FactorConfig) -> pd.DataFrame:
        """获取行业日收益率（扩展到config.end_date后60天）"""
        extended_end = (pd.to_datetime(config.end_date) + timedelta(days=60)).strftime('%Y-%m-%d')
        loader = get_loader()
        returns_df = loader.get_industry_returns(config.start_date, extended_end)
        logger.info(f"获取行业日收益率，形状: {returns_df.shape}")
        return returns_df

    @staticmethod
    def _calculate_nav(holdings: Dict[str, pd.DataFrame],
                       industry_returns: pd.DataFrame,
                       first_rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        计算净值曲线（考虑权重漂移）

        核心逻辑：
        1. 调仓日：老持仓先按当日行业收益率涨跌 → 得到新总净值 → 收盘价按新权重重新分配
        2. 非调仓日：持仓随行业收益率自然漂移

        Args:
            holdings: {组名: 持仓权重DataFrame}
            industry_returns: 行业日收益率 (%, 日期 × 行业)
            first_rebalance_date: 首个调仓日

        Returns:
            净值曲线 DataFrame (日期 × 组名)
        """
        all_dates = industry_returns.index[industry_returns.index >= first_rebalance_date]
        nav_dict = {}

        for group_name, holding_df in holdings.items():
            # 子资产净值矩阵（日期 × 行业），记录每个行业持仓的净值
            sub_navs = pd.DataFrame(0.0, index=all_dates, columns=industry_returns.columns)

            # === 首日：必须是调仓日 ===
            date = all_dates[0]
            if date not in holding_df.index:
                raise ValueError(f"首个交易日 {date.strftime('%Y-%m-%d')} 不是调仓日")

            # 初始建仓：按权重分配初始净值1.0
            current_holdings = holding_df.loc[date]
            sub_navs.loc[date] = current_holdings

            # 从第二日开始循环
            for i in range(1, len(all_dates)):
                date = all_dates[i]
                prev_date = all_dates[i - 1]

                # === 调仓日（非首日）===
                if date in holding_df.index:
                    # Step 1: 老持仓先按当日行业收益率涨跌
                    daily_ret = industry_returns.loc[date] / 100  # 转为小数
                    sub_navs.loc[date] = sub_navs.loc[prev_date] * (1 + daily_ret)

                    # Step 2: 收盘价按新权重重新分配（总净值保持为涨跌后的值）
                    total_nav = sub_navs.loc[date].sum()
                    current_holdings = holding_df.loc[date]
                    sub_navs.loc[date] = total_nav * current_holdings

                # === 非调仓日 ===
                else:
                    daily_ret = industry_returns.loc[date] / 100
                    sub_navs.loc[date] = sub_navs.loc[prev_date] * (1 + daily_ret)

            # 组合净值 = 所有子资产净值之和
            nav_dict[group_name] = sub_navs.sum(axis=1)

        return pd.DataFrame(nav_dict)

    def _calculate_metrics(self, nav: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        从净值曲线计算绩效指标

        Args:
            nav: 净值曲线 (日期 × 组名)
            rebalance_dates: 调仓日序列（用于计算调仓胜率）
        """
        metrics = {}
        n_days = len(nav) - 1
        n_years = n_days / self.trading_days_per_year

        for col in nav.columns:
            nav_series = nav[col]

            # 日收益率（用于波动率/夏普计算）
            daily_ret = nav_series.pct_change().dropna() * 100

            # 总收益
            total_return = (nav_series.iloc[-1] - 1) * 100

            # 年化收益
            annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

            # 年化波动率
            annual_vol = daily_ret.std() * np.sqrt(self.trading_days_per_year)

            # 最大回撤
            cum_max = nav_series.cummax()
            drawdown = (nav_series - cum_max) / cum_max * 100
            max_drawdown = drawdown.min()

            # 调仓胜率（每个调仓期间的收益率）
            period_returns = self._get_period_returns(nav_series, rebalance_dates)
            win_rate = (period_returns > 0).sum() / len(period_returns) * 100 if len(period_returns) > 0 else 0

            # 夏普比率
            excess_return = daily_ret.mean() - self.risk_free_rate / self.trading_days_per_year
            sharpe = (excess_return * self.trading_days_per_year) / (
                    daily_ret.std() * np.sqrt(self.trading_days_per_year)) if daily_ret.std() > 0 else 0

            metrics[col] = {
                '总收益(%)': total_return,
                '年化收益(%)': annual_return,
                '年化波动(%)': annual_vol,
                '最大回撤(%)': max_drawdown,
                '调仓胜率(%)': win_rate,
                '夏普比率': sharpe
            }

        return pd.DataFrame(metrics).T

    @staticmethod
    def _get_period_returns(nav: pd.Series, rebalance_dates: pd.DatetimeIndex) -> pd.Series:
        """
        计算调仓期间收益率

        Args:
            nav: 净值序列（索引为所有交易日）
            rebalance_dates: 调仓日序列

        Returns:
            每个调仓区间的收益率 (%)
        """
        period_returns = []

        for i in range(len(rebalance_dates) - 1):
            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i + 1]

            # 找到 <= end_date 的最后一个交易日（处理调仓日可能不在净值索引中的情况）
            valid_end_dates = nav.index[nav.index <= end_date]
            if len(valid_end_dates) == 0:
                continue
            actual_end_date = valid_end_dates[-1]

            # 期间的净值变化
            start_nav = nav.loc[start_date]
            end_nav = nav.loc[actual_end_date]

            period_ret = (end_nav / start_nav - 1) * 100
            period_returns.append(period_ret)

        return pd.Series(period_returns)


class LongShortAnalyzer(BaseGroupAnalyzer):
    """多空模式分析器"""

    def __init__(self, long_size: int = 6, short_size: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.long_size = long_size
        self.short_size = short_size

    def _create_holdings(self, factor: pd.DataFrame, all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """
        生成多空持仓

        每个调仓日：
        - 多头：因子值最高的 long_size 个行业，等权
        - 空头：因子值最低的 short_size 个行业，等权
        - 基准：所有有效行业，等权
        """
        long_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)
        short_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)
        benchmark_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)

        for date in factor.index:
            factor_values = factor.loc[date].dropna()
            if len(factor_values) < self.long_size + self.short_size:
                logger.warning(f"{date}: 有效行业数不足，跳过")
                continue

            sorted_industries = factor_values.sort_values(ascending=False)

            # 多头
            long_industries = sorted_industries.head(self.long_size).index
            long_holdings.loc[date, long_industries] = 1.0 / self.long_size

            # 空头
            short_industries = sorted_industries.tail(self.short_size).index
            short_holdings.loc[date, short_industries] = 1.0 / self.short_size

            # 基准
            benchmark_holdings.loc[date, factor_values.index] = 1.0 / len(factor_values)

        return {'long': long_holdings, 'short': short_holdings, 'benchmark': benchmark_holdings}

    def _calculate_nav(self, holdings: Dict[str, pd.DataFrame],
                       industry_returns: pd.DataFrame,
                       first_rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        计算净值并添加派生组合

        派生组合逻辑（相对净值）：
        - 多头/基准 = 1 + (多头累计收益 - 基准累计收益)
        - 多空 = 1 + (多头累计收益 - 空头累计收益)
        """
        # 基础三组净值
        nav = super()._calculate_nav(holdings, industry_returns, first_rebalance_date)

        # 派生组合：相对净值（初始=1，后续=1+超额收益）
        nav['多头/基准'] = 1 + (nav['long'] - 1) - (nav['benchmark'] - 1)
        nav['多空'] = 1 + (nav['long'] - 1) - (nav['short'] - 1)

        # 重命名
        nav = nav.rename(columns={'long': '多头', 'short': '空头', 'benchmark': '基准'})

        return nav


class MultiGroupAnalyzer(BaseGroupAnalyzer):
    """多分组模式分析器"""

    def __init__(self,
                 n_groups: int = 5,
                 industries_per_group: int = 6,
                 allow_partial_last_group: bool = True,
                 **kwargs):
        """
        Args:
            n_groups: 分组数量
            industries_per_group: 每组目标行业数
            allow_partial_last_group: 是否允许最后一组不满员
                - True: 前n-1组满员，最后一组包含剩余行业（推荐）
                - False: 严格模式，必须所有组都满员
        """
        super().__init__(**kwargs)
        self.n_groups = n_groups
        self.industries_per_group = industries_per_group
        self.allow_partial = allow_partial_last_group

    def _create_holdings(self, factor: pd.DataFrame, all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """生成多分组持仓（支持动态分组）"""
        holdings = {f'G{i + 1}': pd.DataFrame(0.0, index=factor.index, columns=all_industries)
                    for i in range(self.n_groups)}

        for date in factor.index:
            factor_values = factor.loc[date].dropna()
            n_available = len(factor_values)

            # 计算最小需求
            if self.allow_partial:
                # 宽松模式：前n-1组满员即可
                min_required = (self.n_groups - 1) * self.industries_per_group + 1
            else:
                # 严格模式：所有组必须满员
                min_required = self.n_groups * self.industries_per_group

            if n_available < min_required:
                logger.warning(
                    f"{date}: 可用行业数({n_available})不足最小需求({min_required})，跳过"
                )
                continue

            sorted_industries = factor_values.sort_values(ascending=False)

            # 分配到各组
            for i in range(self.n_groups):
                start = i * self.industries_per_group
                end = min(start + self.industries_per_group, len(sorted_industries))

                # 最后一组可能不满员
                group_industries = sorted_industries.iloc[start:end].index

                if len(group_industries) > 0:
                    # 组内等权（动态调整权重）
                    holdings[f'G{i + 1}'].loc[date, group_industries] = 1.0 / len(group_industries)

        logger.info(f"生成{self.n_groups}组持仓，调仓次数: {len(factor)}")
        return holdings


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from factors.base import create_factor

    test_config = FactorConfig(
        start_date='2012-12-31',
        end_date='2023-12-29',
        frequency='monthly'
    )

    factor_calc = create_factor("marginal_average_momentum", preprocess=False)
    test_factor_values = factor_calc(test_config)

    analyzer = LongShortAnalyzer(long_size=6, short_size=6)
    results = analyzer.analyze(test_factor_values, test_config, '2013-01-31', '2023-12-29')

    print("\n绩效指标:")
    print(results['metrics'].round(2))
    print("\n净值曲线（前3期）:")
    print(results['nav'].head(3))

    # # 2. 测试多分组回测
    # analyzer = MultiGroupAnalyzer(n_groups=4, industries_per_group=7)
    # results = analyzer.analyze(test_factor_values, test_config, '2013-01-31', '2023-12-29')
    #
    # print("\n绩效指标:")
    # print(results['metrics'].round(2))
    # print("\n净值曲线（前3期）:")
    # print(results['nav'].head(3))