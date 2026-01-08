# src/data/sync/sync_all.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: sync_all.py
@time: 2025/01/05
@description:
数据同步统一入口
"""
import pandas as pd
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def sync_by_periods(sync_func, start_date: str, end_date: str, freq: str = 'M'):
    """
    按时间周期分批同步

    Args:
        sync_func: 同步函数（如 sync_index_daily）
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率 ('M'=月, 'Y'=年)

    Returns:
        总记录数
    """
    periods = pd.period_range(start_date, end_date, freq=freq)
    total = 0

    for period in periods:
        period_start = period.start_time.strftime('%Y-%m-%d')
        period_end = period.end_time.strftime('%Y-%m-%d')

        try:
            count = sync_func(period_start, period_end)
            total += count
            logger.info(f"  {period}: {count} 条")
        except Exception as e:
            logger.error(f"  {period} 失败: {e}")

    return total


def sync_all_init(start_date: str = '2015-01-01',
                  end_date: str = '2030-12-31'):
    """
    初始化：全量同步所有表

    Args:
        start_date: 历史起始日期
        end_date: 结束日期
    """
    logger.info("=" * 60)
    logger.info(f"全量初始化同步: {start_date} -> {end_date}")
    logger.info("=" * 60)

    # 1. 交易日历
    logger.info("\n[1/3] 初始化交易日历...")
    from data.sync.trading_calendar import get_trading_calendar
    get_trading_calendar(start_date, end_date, force_update=True)

    # 2. 指数日线（按月）
    logger.info("\n[2/3] 初始化指数日线...")
    from data.sync.sync_index_daily import sync_index_daily
    total = sync_by_periods(sync_index_daily, start_date, end_date, freq='M')
    logger.info(f"指数日线完成: {total} 条")

    # 3. 大单数据（按月）
    logger.info("\n[3/3] 初始化大单数据...")
    from data.sync.sync_index_large_order import sync_large_order
    total = sync_by_periods(sync_large_order, start_date, end_date, freq='M')
    logger.info(f"大单数据完成: {total} 条")

    logger.info("\n" + "=" * 60)
    logger.info("全量初始化完成")
    logger.info("=" * 60)


def sync_all_incremental(end_date: Optional[str] = None, lookback_days: int = 5):
    """
    增量同步：更新最近N天数据

    Args:
        end_date: 结束日期（默认今天）
        lookback_days: 回看天数
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info(f"增量同步: {start_date} -> {end_date}")
    logger.info("=" * 60)

    # 1. 交易日历
    logger.info("\n[1/3] 更新交易日历...")
    from data.sync.trading_calendar import get_trading_calendar
    get_trading_calendar(start_date, end_date)

    # 2. 指数日线
    logger.info("\n[2/3] 更新指数日线...")
    from data.sync.sync_index_daily import sync_index_daily
    count = sync_index_daily(start_date, end_date)
    logger.info(f"指数日线: {count} 条")

    # 3. 大单数据
    logger.info("\n[3/3] 更新大单数据...")
    from data.sync.sync_index_large_order import sync_large_order
    count = sync_large_order(start_date, end_date)
    logger.info(f"大单数据: {count} 条")

    logger.info("\n" + "=" * 60)
    logger.info("增量同步完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO
    )

    # import sys
    #
    # if len(sys.argv) > 1 and sys.argv[1] == '--init':
    #     # 全量初始化
    #     sync_all_init('2015-01-01', '2030-12-31')
    # else:
    #     # 增量更新（默认）
    #     sync_all_incremental(lookback_days=5)
    sync_all_init('2012-01-01', '2025-12-31')