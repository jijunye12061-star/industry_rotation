#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: elasticity.py
@time: 2025/12/12 15:25
@description:
价格成交弹性因子

基于价格变化和成交金额计算行业的成交弹性及其变化
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Optional, Literal, Union

from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


@register_factor("day_elasticity")
class DayElasticityFactor(BaseFactor):
    """
    日度成交弹性因子

    计算每日的价格成交弹性：|(close/open - 1)| / amt
    表示单位成交金额产生的价格变化幅度
    """

    def __init__(self,
                 name: str = "day_elasticity",
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            **params
        )

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算日度成交弹性"""
        loader = get_loader()

        # 获取数据
        open_prices = loader.get_industry_open_prices(config.start_date, config.end_date)
        close_prices = loader.get_industry_prices(config.start_date, config.end_date)
        amounts = loader.get_industry_amounts(config.start_date, config.end_date)

        # 计算日度弹性：|(close/open - 1)| / amt
        price_change = (close_prices / open_prices - 1).abs()
        day_elasticity = price_change / amounts

        # 处理无效值（成交金额为0或接近0导致的极值）
        day_elasticity = day_elasticity.replace([np.inf, -np.inf], np.nan)

        # 提取调仓日
        rebalance_dates = config.get_rebalance_dates()
        result = day_elasticity.reindex(rebalance_dates)

        logger.info(f"日度成交弹性因子计算完成，形状: {result.shape}")
        return result


@register_factor("elasticity")
class ElasticityFactor(BaseFactor):
    """
    成交弹性变化因子

    计算最近window日弹性之和减去前window日弹性之和
    elasticity_t = sum(day_elasticity[T-20:T]) - sum(day_elasticity[T-41:T-21])

    衡量近期成交弹性相对于早期的变化，反映市场流动性和价格敏感度的变化
    """

    def __init__(self,
                 name: str = "elasticity",
                 window: int = 21,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 滚动窗口期（交易日数），默认21天
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
        """计算成交弹性变化因子"""
        # 扩展开始日期（需要2倍窗口的历史数据）
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=120)).strftime('%Y-%m-%d')

        loader = get_loader()

        # 获取数据
        open_prices = loader.get_industry_open_prices(extended_start, config.end_date)
        close_prices = loader.get_industry_prices(extended_start, config.end_date)
        amounts = loader.get_industry_amounts(extended_start, config.end_date)

        # 计算日度弹性
        price_change = (close_prices / open_prices - 1).abs()
        day_elasticity = price_change / amounts
        day_elasticity = day_elasticity.replace([np.inf, -np.inf], np.nan)

        # 计算滚动弹性变化
        elasticity_change = self._calculate_elasticity_change(day_elasticity)

        # 提取调仓日
        rebalance_dates = config.get_rebalance_dates()
        result = elasticity_change.reindex(rebalance_dates)

        logger.info(f"成交弹性变化因子计算完成，形状: {result.shape}")
        return result

    def _calculate_elasticity_change(self, day_elasticity: pd.DataFrame) -> pd.DataFrame:
        """
        计算弹性变化：sum(T-window+1:T) - sum(T-2*window+1:T-window)

        Args:
            day_elasticity: 日度弹性 DataFrame

        Returns:
            弹性变化 DataFrame
        """
        # 计算最近window日的和
        recent_sum = day_elasticity.rolling(
            window=self.window,
            min_periods=int(self.window * 0.8)
        ).sum()

        # 计算前一个window的和（先shift再rolling）
        previous_sum = day_elasticity.shift(self.window).rolling(
            window=self.window,
            min_periods=int(self.window * 0.8)
        ).sum()

        # 变化 = 近期 - 早期
        elasticity_change = recent_sum - previous_sum

        return elasticity_change


@register_factor("elasticity_momentum")
class ElasticityMomentumFactor(BaseFactor):
    """
    成交弹性动量因子

    计算短期和长期成交弹性均值的差，类似MACD指标
    momentum = mean(day_elasticity, short_window) - mean(day_elasticity, long_window)

    正值表示近期弹性上升，可能预示价格对资金更敏感
    """

    def __init__(self,
                 name: str = "elasticity_momentum",
                 short_window: int = 21,
                 long_window: int = 63,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            short_window: 短期窗口（默认21天）
            long_window: 长期窗口（默认63天）
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            short_window=short_window,
            long_window=long_window,
            **params
        )
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算成交弹性动量因子"""
        # 扩展开始日期（需要长窗口的历史数据）
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=180)).strftime('%Y-%m-%d')

        loader = get_loader()

        # 获取数据
        open_prices = loader.get_industry_open_prices(extended_start, config.end_date)
        close_prices = loader.get_industry_prices(extended_start, config.end_date)
        amounts = loader.get_industry_amounts(extended_start, config.end_date)

        # 计算日度弹性
        price_change = (close_prices / open_prices - 1).abs()
        day_elasticity = price_change / amounts
        day_elasticity = day_elasticity.replace([np.inf, -np.inf], np.nan)

        # 计算短期和长期弹性均值
        short_mean = day_elasticity.rolling(
            window=self.short_window,
            min_periods=int(self.short_window * 0.8)
        ).mean()

        long_mean = day_elasticity.rolling(
            window=self.long_window,
            min_periods=int(self.long_window * 0.8)
        ).mean()

        # 动量 = 短期均值 - 长期均值
        momentum = short_mean - long_mean

        # 提取调仓日
        rebalance_dates = config.get_rebalance_dates()
        result = momentum.reindex(rebalance_dates)

        logger.info(f"成交弹性动量因子计算完成，形状: {result.shape}")
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 创建配置
    main_config = FactorConfig(
        start_date='2024-01-01',
        end_date='2025-03-31',
        frequency='monthly'
    )

    # 测试日度弹性因子
    print("\n=== 测试日度成交弹性因子 ===")
    day_factor = DayElasticityFactor()
    day_elasticity_df = day_factor(main_config)
    print(f"形状: {day_elasticity_df.shape}")
    print(day_elasticity_df.tail())

    # 测试弹性变化因子
    print("\n=== 测试成交弹性变化因子 ===")
    elasticity_factor = ElasticityFactor(window=21)
    elasticity_df = elasticity_factor(main_config)
    print(f"形状: {elasticity_df.shape}")
    print(elasticity_df.tail())

    # 测试弹性动量因子
    print("\n=== 测试成交弹性动量因子 ===")
    momentum_factor = ElasticityMomentumFactor(short_window=21, long_window=63)
    momentum_df = momentum_factor(main_config)
    print(f"形状: {momentum_df.shape}")
    print(momentum_df.tail())