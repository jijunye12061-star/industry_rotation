#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: decorators.py
@time: 2025/01/05
@description:
Parquet缓存装饰器 - 智能管理历史数据与增量更新
"""
from functools import wraps
from pathlib import Path
import pandas as pd
import json
from inspect import signature
from typing import List
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _get_partition_status(date_str: str) -> str:
    """
    判断分区状态

    Args:
        date_str: 日期字符串 'YYYY-MM-DD'

    Returns:
        'sealed': 完整历史月（永久缓存）
        'active': 当前月（需增量更新）
    """
    query_month = pd.to_datetime(date_str).to_period('M')
    current_month = pd.Timestamp.now().to_period('M')
    return 'active' if query_month == current_month else 'sealed'


def parquet_cache(
        table: str,
        keys: List[str],
        date_col: str,  # ← P0修复：显式指定日期列
        overlap_days: int = 3,
        partition_by: str = 'month',
        auto_split: bool = True
):
    """
    Parquet缓存装饰器 - 按月分区，智能增量更新

    核心逻辑：
    - 历史月（sealed）：命中缓存直接返回，未命中查询后永久存储
    - 当前月（active）：回看N天增量更新，去重后覆盖写入

    Args:
        table: 表名（如 'tb_index_daily'）
        keys: 主键列表（用于去重），如 ['trade_date', 'index_code']
        date_col: 日期列名（用于增量更新时确定回看起点）
        overlap_days: 增量回看天数（覆盖可能不完整的数据）
        partition_by: 分区方式 ('month', 'year', None)
        auto_split: 是否自动按月拆分查询

    示例:
        @parquet_cache(
            table='tb_index_daily',
            keys=['trade_date', 'index_code'],
            date_col='trade_date',
            overlap_days=3
        )
        def fetch_data(start_date, end_date):
            return query_from_api(start_date, end_date)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ba = signature(func).bind(*args, **kwargs)
            ba.apply_defaults()

            start_date = ba.arguments.get('start_date', '')
            end_date = ba.arguments.get('end_date', '')

            # 跨月查询自动拆分
            if auto_split and partition_by == 'month' and start_date and end_date:
                return _auto_split_by_month(
                    func, ba, table, keys, date_col, overlap_days, start_date, end_date
                )

            # 单月查询
            return _single_query(
                func, ba, table, keys, date_col, overlap_days, partition_by, start_date, end_date
            )

        wrapper.__wrapped__ = func
        return wrapper

    return decorator


def _auto_split_by_month(
        func, ba, table: str, keys: List[str], date_col: str,
        overlap_days: int, start_date: str, end_date: str
):
    """按月拆分查询并合并（处理跨月边界去重）"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    results = []

    for month_start in pd.date_range(start, end, freq='MS'):
        month_end = month_start + pd.offsets.MonthEnd(0)
        query_start = max(month_start, start).strftime('%Y-%m-%d')
        query_end = min(month_end, end).strftime('%Y-%m-%d')

        ba.arguments['start_date'] = query_start
        ba.arguments['end_date'] = query_end

        # P1改进：统一使用状态判断
        status = _get_partition_status(query_start)
        skip_cache = (status == 'active')

        df = _single_query(
            func, ba, table, keys, date_col, overlap_days, 'month',
            query_start, query_end, skip_cache=skip_cache
        )
        results.append(df)

    if not results:
        return pd.DataFrame()

    # 合并并去重（处理跨月边界重复数据）
    combined = pd.concat(results, ignore_index=True)
    return combined.drop_duplicates(subset=keys, keep='last').sort_values(keys).reset_index(drop=True)


def _single_query(
        func, ba, table: str, keys: List[str], date_col: str,
        overlap_days: int, partition_by: str, start_date: str, end_date: str,
        skip_cache: bool = False
):
    """
    单次查询核心逻辑

    流程：
    1. 判断分区状态（sealed/active）
    2. sealed月：缓存命中直接返回，未命中查询后存储
    3. active月：读取缓存 → 回看N天增量查询 → 合并去重 → 覆盖写入
    """
    # 构建文件路径
    if partition_by == 'month' and start_date:
        year, month = start_date[:4], start_date[5:7]
        subdir = f"{year}/{month}"
    elif partition_by == 'year' and start_date:
        subdir = start_date[:4]
    else:
        subdir = ""

    filename = f"{start_date}_{end_date}.parquet"
    filepath = DATA_DIR / table / subdir / filename
    metadata_path = DATA_DIR / table / "metadata.json"

    # P1改进：统一状态判断
    status = _get_partition_status(start_date)
    is_active = (status == 'active')

    # 跳过缓存直接查询（当前月的强制刷新）
    if skip_cache:
        df = func(*ba.args, **ba.kwargs)
        _save_with_dedup(df, filepath, keys)
        _update_metadata(metadata_path, subdir, end_date, len(df), status='active')
        return df

    # P1改进：文件缺失时的优雅处理
    if not filepath.exists():
        # 历史月缺失：查询并永久存储
        df = func(*ba.args, **ba.kwargs)
        if df.empty:
            return df
        _save_with_dedup(df, filepath, keys)
        _update_metadata(metadata_path, subdir, end_date, len(df), status=status)
        return df

    # 读取缓存
    cached_df = pd.read_parquet(filepath)

    # 历史月：直接返回缓存
    if not is_active:
        return cached_df

    # 当前月：增量更新逻辑
    if overlap_days > 0 and not cached_df.empty:
        # P0修复：使用显式指定的日期列
        last_date = pd.to_datetime(cached_df[date_col].max())
        new_start = (last_date - pd.Timedelta(days=overlap_days)).strftime('%Y-%m-%d')

        # 更新查询参数
        ba.arguments['start_date'] = max(new_start, start_date)

        # 查询增量数据
        new_df = func(*ba.args, **ba.kwargs)

        if new_df.empty:
            return cached_df

        # 合并去重（仅对overlap区间去重，优化性能）
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        result = combined.drop_duplicates(subset=keys, keep='last').sort_values(keys).reset_index(drop=True)

        # 覆盖写入
        _save_with_dedup(result, filepath, keys)
        _update_metadata(metadata_path, subdir, end_date, len(result), status='active')
        return result

    # 缓存有效，直接返回
    return cached_df


def _save_with_dedup(df: pd.DataFrame, filepath: Path, keys: List[str]):
    """保存前去重（确保数据唯一性）"""
    if not df.empty and keys:
        df = df.drop_duplicates(subset=keys, keep='last').sort_values(keys).reset_index(drop=True)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, compression='zstd', index=False)


def _update_metadata(
        metadata_path: Path, partition: str, last_date: str,
        rows: int, status: str
):
    """更新元数据文件（记录分区状态和更新时间）"""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取现有元数据
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # 更新分区记录
    metadata[partition] = {
        'status': status,
        'last_date': last_date,
        'rows': rows,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 保存
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)  # type: ignore