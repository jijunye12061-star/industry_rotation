#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: amount_heat.py
@time: 2025/12/15 10:48
@description:
成交热度因子

个股近20日成交额均值相对过去120日均值的比值，按行业权重加权
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


@register_factor("amount_heat")
class AmountHeatFactor(BaseFactor):
    """
    成交热度因子

    计算逻辑：
    1. 个股层面：近20日成交额均值 / 过去120日成交额均值
    2. 行业层面：按个股成交额的行业占比加权
    3. 取负值（成交热度下降为正信号）
    """

    def __init__(self,
                 name: str = "amount_heat",
                 short_window: int = 20,
                 long_window: int = 120,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            short_window: 短期窗口（默认20日）
            long_window: 长期窗口（默认120日）
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
        """计算成交热度因子"""
        loader = get_loader()
        rebalance_dates = config.get_rebalance_dates()
        industry_codes = loader.get_all_industry_codes()

        # 初始化结果
        factor_values = pd.DataFrame(
            index=rebalance_dates,
            columns=industry_codes,
            dtype=float
        )

        # 按调仓日循环计算
        for i, date in enumerate(rebalance_dates):
            date_str = date.strftime('%Y-%m-%d')  # type: ignore
            logger.info(f"处理 {date_str} ({i + 1}/{len(rebalance_dates)})")

            # 计算数据起始日期（需要 long_window 个交易日）
            data_start = (date - timedelta(days=int(self.long_window * 1.8))).strftime('%Y-%m-%d')  # type: ignore

            # 逐行业计算
            for industry_code in industry_codes:
                try:
                    factor_value = self._calculate_industry_heat(
                        loader, industry_code, date_str, data_start
                    )
                    factor_values.loc[date, industry_code] = factor_value
                except Exception as e:
                    logger.warning(f"{date_str} {industry_code} 计算失败: {e}")
                    continue

        logger.info(f"成交热度因子计算完成，形状: {factor_values.shape}")
        return factor_values

    def _calculate_industry_heat(self,
                                 loader,
                                 industry_code: str,
                                 date: str,
                                 data_start: str) -> float:
        """计算单个行业在某日的成交热度"""

        # 1. 获取成分股及权重
        constituents = loader.get_index_constituents(industry_code, date)
        if len(constituents) == 0:
            return np.nan

        stock_codes = constituents['证券内码'].tolist()
        weights = constituents.set_index('证券内码')['权重']

        # 2. 获取个股成交额
        amounts = loader.get_stock_amounts(stock_codes, data_start, date)
        if amounts.empty or len(amounts) < self.long_window:
            return np.nan

        # 3. 计算个股成交额变化率
        stock_ratios = pd.Series(index=amounts.columns, dtype=float)

        for stock in amounts.columns:
            stock_amount = amounts[stock].dropna()
            if len(stock_amount) < self.long_window:
                continue

            # 近20日均值 / 过去120日均值
            recent_mean = stock_amount.iloc[-self.short_window:].mean()
            long_mean = stock_amount.iloc[-self.long_window:].mean()

            if long_mean > 0:
                stock_ratios[stock] = recent_mean / long_mean

        # 4. 按权重加权（仅对有数据的股票）
        valid_stocks = stock_ratios.dropna().index
        valid_weights = weights.loc[valid_stocks]
        valid_ratios = stock_ratios.loc[valid_stocks]

        if len(valid_stocks) == 0:
            return np.nan

        # 权重归一化
        valid_weights = valid_weights / valid_weights.sum()

        # 加权平均并取负值
        industry_heat = -(valid_weights * valid_ratios).sum()

        return industry_heat


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试（建议先用短期数据测试）
    test_config = FactorConfig(
        start_date='2024-10-01',
        end_date='2024-12-31',
        frequency='monthly'
    )

    factor = AmountHeatFactor(short_window=20, long_window=120)
    heat_factor = factor(test_config)

    print("\n成交热度因子：")
    print(heat_factor)

    print(f"\n有效值占比: {heat_factor.notna().sum().sum() / heat_factor.size:.2%}")