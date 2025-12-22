#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: factor_config.py
@time: 2025/12/10 17:00
@description:
因子计算通用配置类
"""

import pandas as pd
from dataclasses import dataclass
from typing import Literal
from data.loader import get_loader
from datetime import timedelta

loader = get_loader()


@dataclass
class FactorConfig:
    start_date: str
    end_date: str
    frequency: Literal['monthly', 'biweekly', 'weekly', 'daily'] = 'monthly'
    market_code: str = '1000157271'

    def __post_init__(self):
        """验证日期格式"""
        pd.to_datetime(self.start_date)
        pd.to_datetime(self.end_date)

    def _compute_rebalance_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        计算调仓日的通用逻辑

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            调仓日期序列
        """
        trading_dates = loader.get_trading_dates(start_date, end_date)

        if self.frequency == 'daily':
            return trading_dates

        elif self.frequency == 'weekly':
            df = pd.DataFrame(index=trading_dates)
            df['week'] = df.index.to_series().dt.isocalendar().week
            df['year'] = df.index.year  # type: ignore
            return pd.DatetimeIndex(df.groupby(['year', 'week']).tail(1).index)

        elif self.frequency == 'biweekly':
            df = pd.DataFrame(index=trading_dates)
            df['biweek'] = (df.index.to_series().dt.isocalendar().week - 1) // 2
            df['year'] = df.index.year  # type: ignore
            return pd.DatetimeIndex(df.groupby(['year', 'biweek']).tail(1).index)

        elif self.frequency == 'monthly':
            df = pd.DataFrame(index=trading_dates)
            return pd.DatetimeIndex(
                df.groupby([df.index.year, df.index.month]).apply(lambda x: x.index[-1])  # type: ignore
            )

        else:
            raise ValueError(f"不支持的频率: {self.frequency}")

    def get_rebalance_dates(self) -> pd.DatetimeIndex:
        """获取回测期的调仓日（不含扩展）"""
        return self._compute_rebalance_dates(self.start_date, self.end_date)

    def get_extended_rebalance_dates(self, extra_periods: int = 1) -> pd.DatetimeIndex:
        """
        获取扩展后的调仓日（用于因子计算）

        Args:
            extra_periods: 向前扩展的期数（默认1）

        Returns:
            扩展后的调仓日期序列

        Example:
            正常调仓日: [2024-01-31, 2024-02-29, 2024-03-31]
            扩展1期后:  [2023-12-29, 2024-01-31, 2024-02-29, 2024-03-31]
        """
        # 根据频率估算需要扩展的天数
        days_map = {
            'daily': 2,
            'weekly': 10,
            'biweekly': 20,
            'monthly': 60
        }
        extend_days = days_map.get(self.frequency, 60) * extra_periods

        # 扩展开始日期
        extended_start = pd.to_datetime(self.start_date) - timedelta(days=extend_days)
        extended_start_str = extended_start.strftime('%Y-%m-%d')  # type: ignore

        return self._compute_rebalance_dates(extended_start_str, self.end_date)


if __name__ == "__main__":
    config = FactorConfig(
        start_date='2013-01-31',
        end_date='2023-12-29',
        frequency='monthly'
    )

    dts = config.get_rebalance_dates()
    e_dts = config.get_extended_rebalance_dates()