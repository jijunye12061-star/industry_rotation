# src/data/sync/trading_calendar.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: trading_calendar.py
@time: 2025/01/05
@description:
交易日历管理
"""
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

CALENDAR_FILE = Path(__file__).resolve().parents[3] / "data" / "trading_calendar" / "calendar.parquet"


def get_trading_calendar(start_date: str = '2015-01-01',
                         end_date: str = '2030-12-31',
                         force_update: bool = False) -> pd.DataFrame:
    """
    获取交易日历（优先读缓存）

    Args:
        start_date: 查询起始日期
        end_date: 查询结束日期
        force_update: 强制更新缓存

    Returns:
        DataFrame with columns: ['date']
    """
    # 读缓存
    if CALENDAR_FILE.exists() and not force_update:
        cached = pd.read_parquet(CALENDAR_FILE)
        cached_start = cached['date'].min()
        cached_end = cached['date'].max()

        # 检查是否需要扩展
        if start_date >= cached_start and end_date <= cached_end:
            logger.info(f"交易日历缓存命中: {cached_start} ~ {cached_end}")
            return cached[(cached['date'] >= start_date) & (cached['date'] <= end_date)]

        logger.warning(f"查询范围超出缓存，扩展中...")
        start_date = min(start_date, cached_start)
        end_date = max(end_date, cached_end)

    # 获取交易日
    logger.info(f"获取交易日历: {start_date} ~ {end_date}")
    from utils.query_data_from_choice import get_fetcher
    fetcher = get_fetcher()
    dates = fetcher.get_trading_dates(start_date, end_date)

    # 保存
    calendar = pd.DataFrame({'date': dates.strftime('%Y-%m-%d')})
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(CALENDAR_FILE, index=False)
    logger.info(f"交易日历已缓存: {len(calendar)} 天")

    return calendar


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 初始化：获取2015-2030所有交易日
    calendar = get_trading_calendar('2015-01-01', '2030-12-31', force_update=True)
    print(f"\n交易日历初始化完成: {len(calendar)} 天")
    print(f"范围: {calendar['date'].min()} ~ {calendar['date'].max()}")
    print(f"\n最近5个交易日:\n{calendar.tail()}")
