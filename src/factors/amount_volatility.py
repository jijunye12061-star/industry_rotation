#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: amount_volatility.py
@time: 2025/12/15 10:19
@description:
成交波动因子

计算过去N日成交额的标准差
"""
import pandas as pd
import logging
from datetime import timedelta
from typing import Optional, Literal, Union

from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


@register_factor("amount_volatility")
class AmountVolatilityFactor(BaseFactor):
    """
    成交波动因子

    计算过去N日成交额的标准差，衡量成交活跃度的波动性
    """

    def __init__(self,
                 name: str = "amount_volatility",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 滚动窗口期（交易日数），默认20日
            preprocess: 是否自动预处理
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
        """计算成交波动因子"""
        # 扩展开始日期以获取足够的历史数据
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=60)).strftime('%Y-%m-%d')

        # 获取成交额数据
        loader = get_loader()
        amounts = loader.get_industry_amounts(extended_start, config.end_date)

        # 计算滚动标准差
        amount_vol = amounts.rolling(
            window=self.window,
            min_periods=int(self.window * 0.8)
        ).std()

        # 筛选到调仓日
        rebalance_dates = config.get_rebalance_dates()
        result = amount_vol.reindex(rebalance_dates)

        logger.info(f"成交波动因子计算完成，形状: {result.shape}")
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试
    test_config = FactorConfig(
        start_date='2024-01-01',
        end_date='2025-03-31',
        frequency='monthly'
    )

    # 计算因子
    factor = AmountVolatilityFactor(window=20)
    amount_vol = factor(test_config)

    print("\n成交波动因子（最近5期）：")
    print(amount_vol.tail())

    print(f"\n调仓日数量: {len(amount_vol)}")
    print(f"有效值占比: {amount_vol.notna().sum().sum() / amount_vol.size:.2%}")