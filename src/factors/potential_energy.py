#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: potential_energy.py
@time: 2025/12/12 15:16
@description:
累积势能因子

日势能 = (收盘价 - 最低价 + 0.1) / (最高价 - 收盘价 + 0.1)
累积势能 = -Σ(过去20日的日势能)
"""
import pandas as pd
import logging
from datetime import timedelta
from typing import Optional, Literal, Union

from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


@register_factor("potential_energy")
class PotentialEnergyFactor(BaseFactor):
    """
    累积势能因子

    计算逻辑：
    1. 日势能 = (close - low + 0.1) / (high - close + 0.1)
    2. 累积势能 = -Σ(过去20日的日势能)
    """

    def __init__(self,
                 name: str = "potential_energy",
                 window: int = 20,
                 epsilon: float = 0.1,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 累积窗口（交易日数）
            epsilon: 避免除零的常数项
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
            epsilon=epsilon,
            **params
        )
        self.window = window
        self.epsilon = epsilon

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算累积势能因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=60)).strftime('%Y-%m-%d')

        # 获取OHLC数据
        loader = get_loader()
        close_prices = loader.get_industry_prices(extended_start, config.end_date)
        high_prices = loader.get_industry_high_prices(extended_start, config.end_date)
        low_prices = loader.get_industry_low_prices(extended_start, config.end_date)

        # 计算日势能
        # day_potential_energy = (close - low + epsilon) / (high - close + epsilon)
        numerator = close_prices - low_prices + self.epsilon
        denominator = high_prices - close_prices + self.epsilon
        day_potential = numerator / denominator

        # 计算累积势能（过去20日求和后取负）
        cumulative_potential = -day_potential.rolling(
            window=self.window,
            min_periods=self.window
        ).sum()

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        cumulative_potential_rebalance = cumulative_potential.reindex(rebalance_dates)

        logger.info(f"累积势能因子计算完成，形状: {cumulative_potential_rebalance.shape}")
        return cumulative_potential_rebalance


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试配置
    test_config = FactorConfig(
        start_date='2024-01-01',
        end_date='2025-03-31',
        frequency='monthly'
    )

    print("\n=== 测试累积势能因子 ===")
    potential_factor = PotentialEnergyFactor(window=20, epsilon=0.1)
    potential_values = potential_factor(test_config)
    print(f"因子形状: {potential_values.shape}")
    print(potential_values.tail())