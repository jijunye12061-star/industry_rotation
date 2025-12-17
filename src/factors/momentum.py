#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: momentum.py
@time: 2025/12/12 15:04
@description:
动量类因子

包含：
1. 20日动量：当前价格相对20日前的涨跌幅
2. 平均动量：最新收盘价相对于20日均价的涨跌幅
3. 边际平均动量：当前平均动量 - 20日前平均动量
"""
import pandas as pd
import logging
from datetime import timedelta
from typing import Optional, Literal, Union

from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


@register_factor("momentum")
class MomentumFactor(BaseFactor):
    """
    20日动量因子

    计算逻辑：(当前价格 / 20日前价格 - 1) * 100
    """

    def __init__(self,
                 name: str = "momentum",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 动量计算窗口（交易日数）
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法
            winsorize: 去极值方法
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算20日动量因子"""
        # 扩展开始日期以获取足够的历史数据
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=60)).strftime('%Y-%m-%d')  # type: ignore

        # 获取价格数据
        loader = get_loader()
        prices = loader.get_industry_prices(extended_start, config.end_date)

        # 计算动量：(P_t / P_{t-window} - 1) * 100
        momentum = (prices / prices.shift(self.window) - 1) * 100

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        momentum_rebalance = momentum.reindex(rebalance_dates)

        logger.info(f"20日动量因子计算完成，形状: {momentum_rebalance.shape}")
        return momentum_rebalance


@register_factor("average_momentum")
class AverageMomentumFactor(BaseFactor):
    """
    平均动量因子

    计算逻辑：(最新收盘价 / 20日均价 - 1) * 100
    """

    def __init__(self,
                 name: str = "average_momentum",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 均线窗口（交易日数）
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法
            winsorize: 去极值方法
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算平均动量因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=60)).strftime('%Y-%m-%d')  # type: ignore

        # 获取价格数据
        loader = get_loader()
        prices = loader.get_industry_prices(extended_start, config.end_date)

        # 计算均线
        ma = prices.rolling(window=self.window, min_periods=self.window).mean()

        # 计算平均动量：(当前价格 / 均线 - 1) * 100
        avg_momentum = (prices / ma - 1) * 100

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        avg_momentum_rebalance = avg_momentum.reindex(rebalance_dates)

        logger.info(f"平均动量因子计算完成，形状: {avg_momentum_rebalance.shape}")
        return avg_momentum_rebalance


@register_factor("marginal_average_momentum")
class MarginalAverageMomentumFactor(BaseFactor):
    """
    边际平均动量因子

    计算逻辑：当前平均动量 - 20日前平均动量
    """

    def __init__(self,
                 name: str = "marginal_average_momentum",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 均线窗口及差分窗口（交易日数）
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法
            winsorize: 去极值方法
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算边际平均动量因子"""
        # 扩展开始日期（需要更多历史数据）
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=120)).strftime('%Y-%m-%d')  # type: ignore

        # 获取价格数据
        loader = get_loader()
        prices = loader.get_industry_prices(extended_start, config.end_date)

        # 计算均线
        ma = prices.rolling(window=self.window, min_periods=self.window).mean()

        # 计算平均动量
        avg_momentum = (prices / ma - 1) * 100

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        avg_momentum_rebalance = avg_momentum.reindex(rebalance_dates)

        # 计算边际平均动量：当期 - 上期
        marginal_avg_momentum = avg_momentum_rebalance.diff(periods=1)

        logger.info(f"边际平均动量因子计算完成，形状: {marginal_avg_momentum.shape}")
        return marginal_avg_momentum


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试配置
    test_config = FactorConfig(
        start_date='2024-01-01',
        end_date='2025-03-31',
        frequency='monthly'
    )

    print("\n=== 测试 20日动量因子 ===")
    momentum_factor = MomentumFactor(window=20)
    momentum_values = momentum_factor(test_config)
    print(f"因子形状: {momentum_values.shape}")
    print(momentum_values.tail())

    print("\n=== 测试平均动量因子 ===")
    avg_momentum_factor = AverageMomentumFactor(window=20)
    avg_momentum_values = avg_momentum_factor(test_config)
    print(f"因子形状: {avg_momentum_values.shape}")
    print(avg_momentum_values.tail())

    print("\n=== 测试边际平均动量因子 ===")
    marginal_avg_momentum_factor = MarginalAverageMomentumFactor(window=20)
    marginal_avg_momentum_values = marginal_avg_momentum_factor(test_config)
    print(f"因子形状: {marginal_avg_momentum_values.shape}")
    print(marginal_avg_momentum_values.tail())
