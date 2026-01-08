#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: sync_index_daily.py
@time: 2025/01/04
@description:
指数日线数据同步 - 从Choice API获取并缓存到Parquet
"""
import pandas as pd
import logging
from typing import Optional, List

from data.decorators import parquet_cache
from utils.query_data_from_choice import get_fetcher
from config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)

# Choice代码映射
INDUSTRY_CODES = list(INDUSTRY_CONFIG.keys())
CHOICE_CODES = [f"{code}.CI" for code in INDUSTRY_CODES] + ['000985.CSI']
CHOICE_TO_STD = {
    **{f"{code}.CI": code for code in INDUSTRY_CODES},
    '000985.CSI': '000985'
}


@parquet_cache(
    table="tb_index_daily",
    keys=['trade_date', 'index_code'],
    date_col='trade_date',
    overlap_days=3,
    partition_by='month',
    auto_split=True
)
def _fetch_index_daily(
        start_date: str,
        end_date: str,
        index_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    从Choice API获取指数日线数据

    装饰器自动处理：
    - 历史月：永久缓存，命中直接返回
    - 当前月：回看3天增量更新

    Args:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        index_codes: 指数代码列表（None=全部行业+市场指数）

    Returns:
        DataFrame with columns:
            trade_date, index_code, open, high, low, close, preclose, amount
    """
    fetcher = get_fetcher()

    # 确定查询代码
    if index_codes:
        choice_codes = [
            f"{code}.CI" if code in INDUSTRY_CODES else code
            for code in index_codes
        ]
    else:
        choice_codes = CHOICE_CODES

    codes_str = ','.join(choice_codes)

    # 调用API
    try:
        raw_data = fetcher.query(
            'csd',
            codes=codes_str,
            indicators='CLOSE,HIGH,LOW,OPEN,AMOUNT,PRECLOSE',
            start_date=start_date,
            end_date=end_date,
            options='period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1'
        )
    except Exception as e:
        logger.error(f"Choice API查询失败: {e}")
        return pd.DataFrame()

    if raw_data.empty:
        logger.warning(f"API返回空数据: {start_date} -> {end_date}")
        return pd.DataFrame()

    # 数据转换
    df = raw_data.reset_index()
    df['trade_date'] = pd.to_datetime(df['DATES']).dt.strftime('%Y-%m-%d')
    df['index_code'] = df['CODES'].map(CHOICE_TO_STD)

    result = df[df['index_code'].notna()][
        ['trade_date', 'index_code', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'PRECLOSE', 'AMOUNT']
    ].rename(columns={
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'PRECLOSE': 'preclose',
        'AMOUNT': 'amount'
    })

    # 过滤异常值
    result = result.dropna(subset=['open', 'close'], how='any')

    logger.info(f"获取 {len(result)} 条记录: {start_date} -> {end_date}")
    return result


def sync_index_daily(start_date: str, end_date: str) -> int:
    """
    公开接口：同步指数日线数据

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        同步记录数
    """
    logger.info(f"同步指数日线: {start_date} -> {end_date}")
    df = _fetch_index_daily(start_date, end_date)
    logger.info(f"同步完成: {len(df)} 条")
    return len(df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试
    count = sync_index_daily('2024-11-01', '2024-12-31')
    print(f"\n同步完成: {count} 条记录")