#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: group_analysis.py
@time: 2025/12/12 09:58
@description:
分组测试模块

基于因子值对行业进行分组，计算多头、空头、基准组合的表现
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging
from datetime import timedelta

from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


class GroupAnalyzer:
    """
    分组测试分析器

    根据因子值将行业分为多头组、空头组，计算各组表现
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
            trading_days_per_year: 每年交易日数
            risk_free_rate: 无风险利率（年化，小数形式，如0.03表示3%）
        """
        self.long_size = long_size
        self.short_size = short_size
        self.trading_days_per_year = trading_days_per_year
        self.risk_free_rate = risk_free_rate

    def analyze(self,
                factor: pd.DataFrame,
                config: FactorConfig) -> Dict:
        """
        执行分组测试

        Args:
            factor: 因子值 DataFrame (调仓日 × 行业代码)
            config: 配置对象

        Returns:
            {
                'nav': 净值曲线 DataFrame (日期 × ['多头', '空头', '基准', '多头/基准', '多空']),
                'metrics': 绩效指标 DataFrame,
                'holdings': 持仓记录（可选）
            }
        """
        logger.info("开始分组测试...")

        # 1. 获取行业收益率数据
        industry_returns = self._get_industry_returns(config)

        # 2. 生成分组持仓
        holdings = self._create_holdings(factor, industry_returns.columns)

        # 3. 计算各组日收益率
        group_returns = self._calculate_group_returns(holdings, industry_returns)

        # 4. 计算净值曲线
        nav = self._calculate_nav(group_returns)

        # 5. 计算绩效指标
        metrics = self._calculate_metrics(group_returns, nav, factor.index)  # type: ignore

        logger.info("分组测试完成")

        return {
            'nav': nav,
            'metrics': metrics,
            'holdings': holdings
        }

    @staticmethod
    def _get_industry_returns(config: FactorConfig) -> pd.DataFrame:
        """
        获取行业日收益率

        Returns:
            行业日收益率 DataFrame (交易日 × 行业代码)
        """
        # 扩展日期以包含回测后的数据
        extended_end = (pd.to_datetime(config.end_date) + timedelta(days=60)).strftime('%Y-%m-%d')  # type: ignore

        loader = get_loader()
        returns_df = loader.get_industry_returns(config.start_date, extended_end)

        logger.info(f"获取行业日收益率，形状: {returns_df.shape}")
        return returns_df

    def _create_holdings(self,
                         factor: pd.DataFrame,
                         all_industries: pd.Index) -> Dict[str, pd.DataFrame]:
        """
        根据因子值生成分组持仓

        Args:
            factor: 因子值 DataFrame (调仓日 × 行业代码)
            all_industries: 所有行业代码列表

        Returns:
            {
                'long': 多头持仓 DataFrame (调仓日 × 行业代码, 值为权重),
                'short': 空头持仓,
                'benchmark': 基准持仓
            }
        """
        # 初始化持仓矩阵
        long_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)
        short_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)
        benchmark_holdings = pd.DataFrame(0.0, index=factor.index, columns=all_industries)

        for date in factor.index:
            # 获取当期因子值（去除NaN）
            factor_values = factor.loc[date].dropna()  # type: ignore

            if len(factor_values) < self.long_size + self.short_size:
                logger.warning(f"{date}: 有效行业数不足，跳过")
                continue

            # 按因子值排序
            sorted_industries = factor_values.sort_values(ascending=False)

            # 多头组：前N个（等权）
            long_industries = sorted_industries.head(self.long_size).index
            long_holdings.loc[date, long_industries] = 1.0 / self.long_size

            # 空头组：后N个（等权）
            short_industries = sorted_industries.tail(self.short_size).index
            short_holdings.loc[date, short_industries] = 1.0 / self.short_size

            # 基准组：所有有效行业（等权）
            valid_industries = factor_values.index
            benchmark_holdings.loc[date, valid_industries] = 1.0 / len(valid_industries)

        logger.info(f"生成持仓记录，调仓次数: {len(factor)}")

        return {
            'long': long_holdings,
            'short': short_holdings,
            'benchmark': benchmark_holdings
        }

    @staticmethod
    def _calculate_group_returns(holdings: Dict[str, pd.DataFrame],
                                 industry_returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算各组日收益率

        Args:
            holdings: 持仓记录
            industry_returns: 行业日收益率

        Returns:
            组合日收益率 DataFrame (交易日 × ['多头', '空头', '基准', '多头/基准', '多空'])
        """
        # 找到第一个有效调仓日（权重非全0）
        valid_rebalance_dates = holdings['long'].index[
            (holdings['long'].sum(axis=1) > 0)
        ]

        if len(valid_rebalance_dates) == 0:
            raise ValueError("没有有效的调仓日")

        first_valid_date = valid_rebalance_dates[0]
        all_dates = industry_returns.index[industry_returns.index >= first_valid_date]

        long_daily = holdings['long'].reindex(all_dates, method='ffill')
        short_daily = holdings['short'].reindex(all_dates, method='ffill')
        benchmark_daily = holdings['benchmark'].reindex(all_dates, method='ffill')

        # 计算各组收益率（持仓权重 * 行业收益率）
        long_ret = (long_daily * industry_returns).sum(axis=1)
        short_ret = (short_daily * industry_returns).sum(axis=1)
        benchmark_ret = (benchmark_daily * industry_returns).sum(axis=1)

        # 计算相对收益和多空收益
        long_vs_benchmark = long_ret - benchmark_ret
        long_short = long_ret - short_ret

        group_returns = pd.DataFrame({
            '多头': long_ret,
            '空头': short_ret,
            '基准': benchmark_ret,
            '多头/基准': long_vs_benchmark,
            '多空': long_short
        })

        logger.info(f"计算组合日收益率，形状: {group_returns.shape}")
        return group_returns

    @staticmethod
    def _calculate_nav(returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算净值曲线

        Args:
            returns: 日收益率 DataFrame

        Returns:
            净值曲线 (初始值=1)
        """
        # 将百分比收益转换为小数
        daily_returns = returns / 100.0

        # 计算累计净值
        nav = (1 + daily_returns).cumprod()

        logger.info("计算净值曲线完成")
        return nav

    def _calculate_metrics(self,
                           returns: pd.DataFrame,
                           nav: pd.DataFrame,
                           rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        计算绩效指标

        Args:
            returns: 日收益率 DataFrame (%)
            nav: 净值曲线
            rebalance_dates: 调仓日期列表

        Returns:
            绩效指标 DataFrame
        """
        metrics = {}

        # 年化因子（假设252个交易日）
        n_days = len(returns)
        n_years = n_days / self.trading_days_per_year

        for col in returns.columns:
            ret = returns[col]
            nav_series = nav[col]

            # 1. 总收益
            total_return = (nav_series.iloc[-1] - 1) * 100

            # 2. 年化收益
            annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

            # 3. 年化波动率
            annual_vol = ret.std() * np.sqrt(self.trading_days_per_year)

            # 4. 最大回撤
            cum_max = nav_series.cummax()
            drawdown = (nav_series - cum_max) / cum_max * 100
            max_drawdown = drawdown.min()

            # 5. 调仓胜率（调仓期收益为正的比例）
            rebalance_returns = self._get_period_returns(ret, rebalance_dates)
            win_rate = (rebalance_returns > 0).sum() / len(rebalance_returns) * 100 if len(rebalance_returns) > 0 else 0

            # 6. 夏普比率
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

        Args:
            daily_returns: 日收益率序列 (%)
            rebalance_dates: 调仓日期

        Returns:
            期间收益率序列
        """
        period_returns = []

        for i in range(len(rebalance_dates) - 1):
            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i + 1]

            # 获取期间内的日收益率
            period_ret = daily_returns.loc[start_date:end_date]

            # 累计收益 = (1+r1)*(1+r2)*...*(1+rn) - 1
            if len(period_ret) > 0:
                cum_ret = ((1 + period_ret / 100).prod() - 1) * 100
                period_returns.append(cum_ret)

        return pd.Series(period_returns)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 使用边际贝塔因子测试
    from factors.marginal_beta import MarginalBetaFactor

    test_config = FactorConfig(
        start_date='2016-01-01',
        end_date='2024-12-31',
        frequency='monthly'
    )

    # 计算因子
    factor_calc = MarginalBetaFactor(window=21)
    marginal_beta = factor_calc(test_config)

    # 分组测试
    analyzer = GroupAnalyzer(long_size=6, short_size=6)
    results = analyzer.analyze(marginal_beta, test_config)

    print("\n=== 分组测试结果 ===")
    print("\n绩效指标:")
    print(results['metrics'].round(2))

    print("\n净值曲线（最近5期）:")
    print(results['nav'].tail())

    # 绘制净值曲线（如果需要）
    from utils.plot_funcs import plot_net_value_curve

    plot_net_value_curve(results['nav'].to_dict('series'))
