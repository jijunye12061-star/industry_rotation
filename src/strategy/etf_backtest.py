#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: etf_backtest.py
@time: 2025/01/08
@description:
ETF回测引擎 - 基于权重的简化回测
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from strategy.etf_loader import ETFLoader

logger = logging.getLogger(__name__)


class ETFBacktest:
    """ETF回测引擎（权重模式）"""

    def __init__(self,
                 loader: ETFLoader,
                 trading_cost: float = 0.0003,
                 initial_capital: float = 1.0):
        """
        Args:
            loader: ETF数据加载器
            trading_cost: 单边交易费率（默认千分之0.3）
            initial_capital: 初始资金（默认1.0）
        """
        self.loader = loader
        self.trading_cost = trading_cost
        self.initial_capital = initial_capital

    def run(self,
            positions: pd.DataFrame,
            start_date: str,
            end_date: str,
            benchmark_etf: Optional[str] = None,
            use_industry_benchmark: bool = False
            ) -> Dict:
        """
        执行回测

        Args:
            positions: 持仓权重 DataFrame (调仓日 × ETF代码)
            start_date: 回测开始日期
            end_date: 回测结束日期
            benchmark_etf: 基准ETF代码（可选）
            use_industry_benchmark: 是否使用行业等权基准

        Returns:
            {
                'nav': 净值曲线 Series,
                'metrics': 绩效指标 Dict,
                'turnover': 换手率 Series,
                'holdings': 持仓记录 DataFrame（用于调试）
            }
        """
        logger.info(f"开始ETF回测: {start_date} -> {end_date}")

        # 1. 获取所有涉及的ETF净值
        all_etfs = positions.columns.tolist()
        etf_prices = self.loader.get_etf_prices(start_date, end_date, all_etfs)

        if etf_prices.empty:
            raise ValueError("未获取到ETF净值数据")

        # 2. 筛选回测区间
        backtest_dates = etf_prices.index[
            (etf_prices.index >= pd.to_datetime(start_date)) &
            (etf_prices.index <= pd.to_datetime(end_date))
        ]

        if len(backtest_dates) == 0:
            raise ValueError("回测区间无有效交易日")

        # 3. 确保首个交易日是调仓日
        first_date = backtest_dates[0]
        if first_date not in positions.index:
            raise ValueError(f"回测起始日 {first_date.strftime('%Y-%m-%d')} 不是调仓日")

        logger.info(f"回测交易日数: {len(backtest_dates)}")
        logger.info(f"调仓次数: {len(positions)}")

        # 4. 计算净值曲线
        nav_series, holdings_df, turnover_series = self._calculate_nav(
            positions, etf_prices, backtest_dates
        )

        # 5. 计算绩效指标
        metrics = self._calculate_metrics(nav_series, turnover_series, positions.index)
        yearly_returns = self._calculate_yearly_returns(nav_series)

        # 6. 基准对比
        benchmark_nav = None

        if benchmark_etf:
            benchmark_nav = self._calculate_benchmark(
                benchmark_etf, etf_prices, backtest_dates
            )
        elif use_industry_benchmark:
            benchmark_nav = self._calculate_industry_benchmark(start_date, end_date, backtest_dates)
            benchmark_nav = benchmark_nav.reindex(backtest_dates).fillna(method='ffill')

        if benchmark_nav is not None:
            metrics['benchmark'] = self._calculate_metrics(
                benchmark_nav, pd.Series(0, index=backtest_dates), positions.index
            )
            metrics['excess_return'] = metrics['annual_return'] - metrics['benchmark']['annual_return']
        logger.info(f"回测完成 - 年化收益: {metrics['annual_return']:.2%}, 夏普: {metrics['sharpe']:.2f}")

        return {
            'nav': nav_series,
            'metrics': metrics,
            'turnover': turnover_series,
            'holdings': holdings_df,
            'yearly_returns': yearly_returns
        }

    def _calculate_nav(self,
                       positions: pd.DataFrame,
                       etf_prices: pd.DataFrame,
                       backtest_dates: pd.DatetimeIndex) -> tuple:
        """
        计算净值曲线（考虑权重漂移和交易成本）

        Returns:
            (净值序列, 持仓记录, 换手率序列)
        """
        # 初始化记录
        nav_series = pd.Series(self.initial_capital, index=backtest_dates)
        holdings_df = pd.DataFrame(0.0, index=backtest_dates, columns=etf_prices.columns)
        turnover_series = pd.Series(0.0, index=backtest_dates)

        # 首日建仓
        first_date = backtest_dates[0]
        target_weights = positions.loc[first_date]
        holdings_df.loc[first_date] = self.initial_capital * target_weights
        turnover_series.loc[first_date] = target_weights.abs().sum()  # 首日换手 = 全部买入

        # 循环计算
        for i in range(1, len(backtest_dates)):
            date = backtest_dates[i]
            prev_date = backtest_dates[i - 1]

            # 前一日持仓市值（按当日价格）
            prev_holdings = holdings_df.loc[prev_date]
            current_prices = etf_prices.loc[date]
            prev_prices = etf_prices.loc[prev_date]

            # 价格变化率
            price_change = (current_prices / prev_prices - 1).fillna(0)

            # 持仓随价格漂移
            current_holdings = prev_holdings * (1 + price_change)

            # 是否调仓日
            if date in positions.index:
                # 当前总资产
                total_value = current_holdings.sum()

                # 目标权重
                target_weights = positions.loc[date]
                target_holdings = total_value * target_weights

                # 计算换手
                weight_change = (target_holdings - current_holdings).abs()
                turnover = weight_change.sum() / total_value

                # 扣除交易费用（双边，买入和卖出都扣）
                trading_cost = turnover * self.trading_cost * total_value
                total_value -= trading_cost

                # 按新权重重新分配（基于扣费后的总资产）
                current_holdings = total_value * target_weights
                turnover_series.loc[date] = turnover

                logger.debug(f"{date.strftime('%Y-%m-%d')}: 调仓, 换手={turnover:.2%}, 费用={trading_cost:.4f}")

            # 记录
            holdings_df.loc[date] = current_holdings
            nav_series.loc[date] = current_holdings.sum()

        return nav_series, holdings_df, turnover_series

    @staticmethod
    def _calculate_metrics(nav: pd.Series,
                           turnover: pd.Series,
                           rebalance_dates: pd.DatetimeIndex,
                           trading_days_per_year: int = 252) -> Dict:
        """计算绩效指标"""
        n_days = len(nav) - 1
        n_years = n_days / trading_days_per_year

        # 日收益率
        daily_ret = nav.pct_change().dropna() * 100

        # 总收益
        total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100

        # 年化收益
        annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # 年化波动
        annual_vol = daily_ret.std() * np.sqrt(trading_days_per_year)

        # 最大回撤
        cum_max = nav.cummax()
        drawdown = (nav - cum_max) / cum_max * 100
        max_drawdown = drawdown.min()

        # 夏普比率（假设无风险利率=0）
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0

        # 卡玛比率
        calmar = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0

        # 调仓胜率
        period_returns = []
        for i in range(len(rebalance_dates) - 1):
            start = rebalance_dates[i]
            end = rebalance_dates[i + 1]
            valid_end = nav.index[nav.index <= end]
            if len(valid_end) == 0:
                continue
            actual_end = valid_end[-1]
            period_ret = (nav.loc[actual_end] / nav.loc[start] - 1) * 100
            period_returns.append(period_ret)

        win_rate = (pd.Series(period_returns) > 0).sum() / len(period_returns) * 100 if period_returns else 0

        # 平均换手率
        avg_turnover = turnover[turnover > 0].mean() if (turnover > 0).any() else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'calmar': calmar,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover * 100  # 转为百分比
        }

    def _calculate_benchmark(self,
                             benchmark_etf: str,
                             etf_prices: pd.DataFrame,
                             backtest_dates: pd.DatetimeIndex) -> pd.Series:
        """计算基准净值（买入持有）"""
        if benchmark_etf not in etf_prices.columns:
            logger.warning(f"基准ETF {benchmark_etf} 无数据，跳过")
            return pd.Series(self.initial_capital, index=backtest_dates)

        prices = etf_prices.loc[backtest_dates, benchmark_etf]
        nav = self.initial_capital * (prices / prices.iloc[0])
        return nav

    @staticmethod
    def _calculate_yearly_returns(nav: pd.Series) -> pd.DataFrame:
        """计算分年度收益"""
        yearly_data = []

        for year in nav.index.year.unique():
            year_nav = nav[nav.index.year == year]
            if len(year_nav) < 2:
                continue

            year_return = (year_nav.iloc[-1] / year_nav.iloc[0] - 1) * 100
            yearly_data.append({
                'year': year,
                'return': year_return,
                'start_nav': year_nav.iloc[0],
                'end_nav': year_nav.iloc[-1]
            })

        return pd.DataFrame(yearly_data).set_index('year')

    # def _calculate_industry_benchmark(self,
    #                                   start_date: str,
    #                                   end_date: str) -> pd.Series:
    #     """计算行业等权基准"""
    #     from data.loader import get_loader
    #
    #     ind_loader = get_loader()
    #     industry_returns = ind_loader.get_industry_returns(start_date, end_date)
    #
    #     # 等权组合日收益 = 各行业日收益的均值
    #     equal_weight_returns = industry_returns.mean(axis=1) / 100  # 转为小数
    #
    #     # 累计净值
    #     nav = (1 + equal_weight_returns).cumprod() * self.initial_capital
    #     return nav

    def _calculate_industry_benchmark(self,
                                      start_date: str,
                                      end_date: str,
                                      backtest_dates: pd.DatetimeIndex) -> pd.Series:
        """计算中证全指基准（买入持有）"""
        from data.loader import get_loader

        ind_loader = get_loader()
        market_prices = ind_loader.get_market_prices(start_date, end_date, market_code='000985')

        # 对齐到回测日期
        prices = market_prices.reindex(backtest_dates, method='ffill')
        nav = self.initial_capital * (prices / prices.iloc[0])
        return nav


def format_metrics(metrics: Dict) -> pd.DataFrame:
    """格式化指标输出"""
    data = {
        '总收益(%)': f"{metrics['total_return']:.2f}",
        '年化收益(%)': f"{metrics['annual_return']:.2f}",
        '年化波动(%)': f"{metrics['annual_vol']:.2f}",
        '最大回撤(%)': f"{metrics['max_drawdown']:.2f}",
        '夏普比率': f"{metrics['sharpe']:.2f}",
        '卡玛比率': f"{metrics['calmar']:.2f}",
        '调仓胜率(%)': f"{metrics['win_rate']:.2f}",
        '平均换手(%)': f"{metrics['avg_turnover']:.2f}",
    }

    if 'benchmark' in metrics:
        data['基准年化(%)'] = f"{metrics['benchmark']['annual_return']:.2f}"
        data['超额收益(%)'] = f"{metrics['excess_return']:.2f}"

    return pd.DataFrame([data]).T.rename(columns={0: '策略'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试示例
    from strategy.etf_loader import ETFLoader

    loader = ETFLoader()

    # 构造
    positions_dynamic = pd.read_parquet(r'./positions_dynamic.parquet')
    test_positions = positions_dynamic.pivot_table(
        index='trade_date',
        columns='etf_code',
        values='weight',
        fill_value=0.0
    )

    # 回测
    backtest = ETFBacktest(loader, trading_cost=0.0003)
    results = backtest.run(
        test_positions,
        start_date='2016-12-30',
        end_date='2025-12-31',
        use_industry_benchmark=True  # 使用中证全指
    )

    # 输出结果
    print("\n" + "=" * 60)
    print(format_metrics(results['metrics']))
    print("\n净值曲线（前5期）:")
    print(results['nav'].head())