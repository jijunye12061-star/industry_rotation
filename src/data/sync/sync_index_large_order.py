# src/data/sync/sync_index_large_order.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: sync_index_large_order.py
@time: 2025/01/05
@description:
指数大单数据同步
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
CHOICE_CODES = [f"{code}.CI" for code in INDUSTRY_CODES]
CHOICE_TO_STD = {f"{code}.CI": code for code in INDUSTRY_CODES}


@parquet_cache(
    table="tb_index_large_order",
    keys=['trade_date', 'index_code'],
    date_col='trade_date',
    overlap_days=3,
    partition_by='month',
    auto_split=True
)
def _fetch_large_order(start_date: str, end_date: str,
                       index_codes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    获取指数大单数据（内部函数，装饰器自动缓存）

    Returns:
        DataFrame with columns: trade_date, index_code, super_large_inflow, super_large_outflow
    """
    fetcher = get_fetcher()

    # 确定查询代码
    if index_codes:
        choice_codes = [f"{code}.CI" for code in index_codes if code in INDUSTRY_CODES]
    else:
        choice_codes = CHOICE_CODES

    codes_str = ','.join(choice_codes)

    # 调用API
    try:
        raw_data = fetcher.query(
            'csd',
            codes=codes_str,
            indicators='FLOWINXL,FLOWOUTXL',
            start_date=start_date,
            end_date=end_date,
            options='period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1'
        )
    except Exception as e:
        logger.error(f"API查询失败: {e}")
        return pd.DataFrame()

    if raw_data.empty:
        return pd.DataFrame()

    # 数据转换
    df = raw_data.reset_index()
    df['trade_date'] = pd.to_datetime(df['DATES']).dt.strftime('%Y-%m-%d')
    df['index_code'] = df['CODES'].map(CHOICE_TO_STD)

    result = df[df['index_code'].notna()][
        ['trade_date', 'index_code', 'FLOWINXL', 'FLOWOUTXL']
    ].rename(columns={
        'FLOWINXL': 'super_large_inflow',
        'FLOWOUTXL': 'super_large_outflow'
    })

    # 过滤空值（至少一个流入/流出非空）
    result = result.dropna(subset=['super_large_inflow', 'super_large_outflow'], how='all')

    return result


def sync_large_order(start_date: str, end_date: str) -> int:
    """
    同步大单数据（公开接口）

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        同步记录数
    """
    logger.info(f"同步大单数据: {start_date} -> {end_date}")
    df = _fetch_large_order(start_date, end_date)
    logger.info(f"同步完成: {len(df)} 条记录")
    return len(df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试/批量同步
    sync_large_order('2024-01-01', '2024-12-31')