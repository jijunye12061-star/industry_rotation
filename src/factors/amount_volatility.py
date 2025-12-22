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
import numpy as np
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
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=60)).strftime('%Y-%m-%d')  # type: ignore

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
        result = - amount_vol.reindex(rebalance_dates)

        logger.info(f"成交波动因子计算完成，形状: {result.shape}")
        return result


@register_factor("super_large_volatility")
class SuperLargeVolatilityFactor(BaseFactor):
    """
    超大单成交额波动因子

    计算逻辑：
    1. 个股层面：过去N日超大单成交额（流入+流出）的标准差
    2. 行业层面：按个股权重加权
    3. 取负值（波动降低为正信号）
    """

    def __init__(self,
                 name: str = "super_large_volatility",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 计算波动的窗口期（默认20日）
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
        """计算超大单波动因子"""
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

            # 计算数据起始日期
            data_start = (date - timedelta(days=int(self.window * 1.8))).strftime('%Y-%m-%d')  # type: ignore

            # 逐行业计算
            for industry_code in industry_codes:
                try:
                    factor_value = self._calculate_industry_volatility(
                        loader, industry_code, date_str, data_start
                    )
                    factor_values.loc[date, industry_code] = factor_value
                except Exception as e:
                    logger.warning(f"{date_str} {industry_code} 计算失败: {e}")
                    continue

        logger.info(f"超大单波动因子计算完成，形状: {factor_values.shape}")
        return factor_values

    def _calculate_industry_volatility(self,
                                       loader,
                                       industry_code: str,
                                       date: str,
                                       data_start: str) -> float:
        """计算单个行业在某日的超大单波动"""

        # 1. 获取成分股及权重
        constituents = loader.get_index_constituents(industry_code, date)
        if len(constituents) == 0:
            return np.nan

        stock_codes = constituents['证券代码'].tolist()
        weights = constituents.set_index('证券代码')['权重']

        # 2. 获取个股超大单成交额
        amounts = loader.get_stock_super_large_flow(stock_codes, data_start, date)
        if amounts.empty or len(amounts) < self.window:
            return np.nan

        # 3. 计算个股波动率（标准差）
        stock_vols = pd.Series(index=amounts.columns, dtype=float)

        for stock in amounts.columns:
            stock_amount = amounts[stock].dropna()
            if len(stock_amount) < self.window:
                continue

            # 过去N日标准差
            recent_vol = stock_amount.iloc[-self.window:].std()
            stock_vols[stock] = recent_vol

        # 4. 按权重加权
        valid_stocks = stock_vols.dropna().index
        valid_weights = weights.loc[valid_stocks]
        valid_vols = stock_vols.loc[valid_stocks]

        if len(valid_stocks) == 0:
            return np.nan

        # 权重归一化
        valid_weights = valid_weights / valid_weights.sum()

        # 加权平均并取负值（波动降低为正信号）
        industry_vol = -(valid_weights * valid_vols).sum()

        return industry_vol


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试
    test_config = FactorConfig(
        start_date='2015-01-01',
        end_date='2015-03-31',
        frequency='monthly'
    )

    # 计算因子
    factor = SuperLargeVolatilityFactor(window=20)
    amount_vol = factor(test_config)

    print("\n成交波动因子（最近5期）：")
    print(amount_vol.tail())

    print(f"\n调仓日数量: {len(amount_vol)}")
    print(f"有效值占比: {amount_vol.notna().sum().sum() / amount_vol.size:.2%}")