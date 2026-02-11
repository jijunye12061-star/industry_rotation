"""
Parquet 缓存装饰器 - 用于 sync 脚本的智能缓存管理

核心逻辑：
- 按月分区存储: data/{table}/{year}/{month}/start_end.parquet
- 历史月(sealed): 缓存命中直接跳过，未命中则查询后永久存储
- 当前月(active): 回看 N 天增量查询，合并去重后覆盖写入
- 元数据: metadata.json 记录分区状态和更新时间
"""
from functools import wraps
from pathlib import Path
from inspect import signature
from typing import List
from datetime import datetime
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _is_current_month(date_str: str) -> bool:
    """判断日期是否属于当前月"""
    return pd.to_datetime(date_str).to_period('M') == pd.Timestamp.now().to_period('M')


def _save_parquet(df: pd.DataFrame, filepath: Path, keys: List[str]):
    """去重后保存 Parquet"""
    if not df.empty and keys:
        df = df.drop_duplicates(subset=keys, keep='last').sort_values(keys).reset_index(drop=True)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, compression='zstd', index=False)


def _update_metadata(metadata_path: Path, partition: str, last_date: str, rows: int, status: str):
    """更新 metadata.json"""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    metadata[partition] = {
        'status': status,
        'last_date': last_date,
        'rows': rows,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parquet_cache(
        table: str,
        keys: List[str],
        date_col: str,
        overlap_days: int = 3,
        partition_by: str = 'month',
        auto_split: bool = True
):
    """
    Parquet 缓存装饰器

    Args:
        table: 表名（data/{table}/ 目录）
        keys: 主键列（去重用），如 ['trade_date', 'index_code']
        date_col: 日期列名
        overlap_days: 当前月增量回看天数
        partition_by: 分区方式 ('month')
        auto_split: 跨月查询是否自动拆分
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ba = signature(func).bind(*args, **kwargs)
            ba.apply_defaults()
            start_date = ba.arguments.get('start_date', '')
            end_date = ba.arguments.get('end_date', '')

            if auto_split and partition_by == 'month' and start_date and end_date:
                return _split_by_month(func, ba, table, keys, date_col, overlap_days, start_date, end_date)

            return _query_single_month(func, ba, table, keys, date_col, overlap_days, start_date, end_date)

        wrapper.__wrapped__ = func
        return wrapper

    return decorator


def _split_by_month(
        func, ba, table: str, keys: List[str], date_col: str,
        overlap_days: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """按月拆分查询，合并去重"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    results = []

    for month_start in pd.date_range(start, end, freq='MS'):
        month_end = month_start + pd.offsets.MonthEnd(0)
        q_start = max(month_start, start).strftime('%Y-%m-%d')
        q_end = min(month_end, end).strftime('%Y-%m-%d')

        ba.arguments['start_date'] = q_start
        ba.arguments['end_date'] = q_end

        df = _query_single_month(
            func, ba, table, keys, date_col, overlap_days, q_start, q_end,
            force_refresh=_is_current_month(q_start)
        )
        results.append(df)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined.drop_duplicates(subset=keys, keep='last').sort_values(keys).reset_index(drop=True)


def _query_single_month(
        func, ba, table: str, keys: List[str], date_col: str,
        overlap_days: int, start_date: str, end_date: str,
        force_refresh: bool = False
) -> pd.DataFrame:
    """
    单月查询核心逻辑

    sealed月: 缓存命中 → 返回; 未命中 → 查询 → 永久存储
    active月: 读缓存 → 回看N天增量 → 合并去重 → 覆盖写入
    """
    year, month = start_date[:4], start_date[5:7]
    subdir = f"{year}/{month}"
    filepath = DATA_DIR / table / subdir / f"{start_date}_{end_date}.parquet"
    metadata_path = DATA_DIR / table / "metadata.json"
    is_active = _is_current_month(start_date)
    status = 'active' if is_active else 'sealed'

    # ---- 当前月强制刷新 ----
    if force_refresh:
        df = func(*ba.args, **ba.kwargs)
        if not df.empty:
            _save_parquet(df, filepath, keys)
            _update_metadata(metadata_path, subdir, end_date, len(df), status='active')
        return df

    # ---- 文件不存在: 查询并存储 ----
    if not filepath.exists():
        df = func(*ba.args, **ba.kwargs)
        if df.empty:
            return df
        _save_parquet(df, filepath, keys)
        _update_metadata(metadata_path, subdir, end_date, len(df), status=status)
        return df

    # ---- 读取缓存 ----
    cached_df = pd.read_parquet(filepath)

    # 历史月直接返回
    if not is_active:
        return cached_df

    # ---- 当前月: 增量更新 ----
    if overlap_days > 0 and not cached_df.empty:
        last_date = pd.to_datetime(cached_df[date_col].max())
        new_start = (last_date - pd.Timedelta(days=overlap_days)).strftime('%Y-%m-%d')
        ba.arguments['start_date'] = max(new_start, start_date)

        new_df = func(*ba.args, **ba.kwargs)
        if new_df.empty:
            return cached_df

        combined = pd.concat([cached_df, new_df], ignore_index=True)
        result = combined.drop_duplicates(subset=keys, keep='last').sort_values(keys).reset_index(drop=True)
        _save_parquet(result, filepath, keys)
        _update_metadata(metadata_path, subdir, end_date, len(result), status='active')
        return result

    return cached_df