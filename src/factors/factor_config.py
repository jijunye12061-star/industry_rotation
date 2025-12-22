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

loader = get_loader()

@dataclass
class FactorConfig:
    """
    因子计算通用配置

    Attributes:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        frequency: 调仓频率 ('monthly', 'biweekly', 'weekly', 'daily')
        market_code: 市场指数内码（默认中证全指）
    """
    start_date: str
    end_date: str
    frequency: Literal['monthly', 'biweekly', 'weekly', 'daily'] = 'monthly'
    market_code: str = '000985.CSI'

    def __post_init__(self):
        """验证日期格式"""
        pd.to_datetime(self.start_date)
        pd.to_datetime(self.end_date)

    def get_rebalance_dates(self) -> pd.DatetimeIndex:
        trading_dates = loader.get_trading_dates(self.start_date, self.end_date)

        if self.frequency == 'daily':
            return trading_dates

        elif self.frequency == 'weekly':
            df = pd.DataFrame(index=trading_dates)
            df['week'] = df.index.to_series().dt.isocalendar().week
            df['year'] = df.index.year  # type: ignore
            return pd.DatetimeIndex(df.groupby(['year', 'week']).tail(1).index)

        elif self.frequency == 'biweekly':
            df = pd.DataFrame(index=trading_dates)
            df['biweek'] = (df.index.to_series().dt.isocalendar().week - 1) // 2  # 修改这里
            df['year'] = df.index.year  # type: ignore
            return pd.DatetimeIndex(df.groupby(['year', 'biweek']).tail(1).index)

        elif self.frequency == 'monthly':
            df = pd.DataFrame(index=trading_dates)
            return pd.DatetimeIndex(df.groupby([df.index.year, df.index.month]).apply(lambda x: x.index[-1]))  # type: ignore

        else:
            raise ValueError(f"不支持的频率: {self.frequency}")


if __name__ == "__main__":
    config = FactorConfig(
        start_date='2024-01-01',
        end_date='2024-12-31',
        frequency='weekly',
    )

    dts = config.get_rebalance_dates()