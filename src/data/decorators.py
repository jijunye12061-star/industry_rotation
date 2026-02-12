"""
Parquet 缓存装饰器 - 用于 sync 脚本的智能缓存管理

核心逻辑：
- 按月分区存储: data/{table}/{year}/{month}/start_end.parquet
- 历史月(sealed): 缓存命中直接跳过，未命中则查询后永久存储
- 当前月(active): 回看 N 天增量查询，合并去重后覆盖写入
- 元数据: metadata.json 记录分区状态和更新时间
"""
# src/data/decorators.py
from functools import wraps
from pathlib import Path
from inspect import signature
from typing import List
from datetime import datetime
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _is_sealed(year: str, month: str) -> bool:
    """当前月之前的分区永久缓存"""
    return pd.Period(f"{year}-{month}", freq='M') < pd.Timestamp.now().to_period('M')


def _filepath(table: str, year: str, month: str) -> Path:
    return DATA_DIR / table / year / f"{year}{month}.parquet"


def parquet_cache(table: str, keys: List[str], date_col: str, overlap_days: int = 3):
    """
    Parquet 缓存装饰器 - 按月分区，智能增量更新

    文件路径: data/{table}/{YYYY}/{YYYYMM}.parquet
    - sealed 月（当前月之前）: 命中缓存直接返回，未命中查询后永久存储
    - active 月（当前月）: 回看 overlap_days 增量更新，合并去重后覆盖写入
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ba = signature(func).bind(*args, **kwargs)
            ba.apply_defaults()
            start_date = ba.arguments['start_date']
            end_date   = ba.arguments['end_date']

            results = []
            for period in pd.period_range(start_date, end_date, freq='M'):
                year  = period.strftime('%Y')
                month = period.strftime('%m')
                # API 查询边界（首末月可能是半月）
                api_start = max(pd.to_datetime(start_date), period.start_time).strftime('%Y-%m-%d')
                api_end   = min(pd.to_datetime(end_date),   period.end_time).strftime('%Y-%m-%d')
                results.append(_process_month(
                    func, ba, table, keys, date_col, overlap_days,
                    year, month, api_start, api_end
                ))

            if not results:
                return pd.DataFrame()
            return (pd.concat(results, ignore_index=True)
                    .drop_duplicates(subset=keys, keep='last')
                    .sort_values(keys)
                    .reset_index(drop=True))

        wrapper.__wrapped__ = func
        return wrapper
    return decorator


def _process_month(func, ba, table, keys, date_col, overlap_days,
                   year, month, api_start, api_end) -> pd.DataFrame:
    filepath = _filepath(table, year, month)
    sealed   = _is_sealed(year, month)

    # sealed 且缓存存在 → 直接返回
    if sealed and filepath.exists():
        logger.info(f"[{table}] {year}-{month} 缓存命中，跳过")
        return pd.read_parquet(filepath)

    logger.info(f"[{table}] {year}-{month} 查询 API: {api_start} → {api_end}")
    # 缓存不存在 → 全量查询并存储
    if not filepath.exists():
        ba.arguments['start_date'] = api_start
        ba.arguments['end_date']   = api_end
        df = func(*ba.args, **ba.kwargs)
        if not df.empty:
            _save(df, filepath, keys)
            _update_metadata(table, year, month, df[date_col], 'sealed' if sealed else 'active')
        return df

    # active 月 → 增量更新
    cached    = pd.read_parquet(filepath)
    last_date = pd.to_datetime(cached[date_col].max())
    new_start = (last_date - pd.Timedelta(days=overlap_days)).strftime('%Y-%m-%d')

    ba.arguments['start_date'] = max(new_start, api_start)
    ba.arguments['end_date']   = api_end
    new_df = func(*ba.args, **ba.kwargs)

    if new_df.empty:
        return cached

    result = (pd.concat([cached, new_df], ignore_index=True)
              .drop_duplicates(subset=keys, keep='last')
              .sort_values(keys)
              .reset_index(drop=True))
    _save(result, filepath, keys)
    _update_metadata(table, year, month, result[date_col], 'active')
    return result


def _save(df: pd.DataFrame, filepath: Path, keys: List[str]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    (df.drop_duplicates(subset=keys, keep='last')
       .sort_values(keys)
       .reset_index(drop=True)
       .to_parquet(filepath, compression='zstd', index=False))


def _update_metadata(table: str, year: str, month: str, dates: pd.Series, status: str):
    meta_path = DATA_DIR / table / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # 首次写入时记录表级信息
    if '_meta' not in metadata:
        metadata['_meta'] = {'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    metadata[f"{year}/{month}"] = {
        'status':     status,
        'first_date': pd.to_datetime(dates.min()).strftime('%Y-%m-%d'),
        'last_date':  pd.to_datetime(dates.max()).strftime('%Y-%m-%d'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))