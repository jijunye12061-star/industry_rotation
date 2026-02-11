"""
数据同步统一入口 - 注册式调度

使用方式:
    python -m data.sync.sync_all --init              # 全量初始化 (2012-2025)
    python -m data.sync.sync_all                     # 增量更新 (近5天)
    python -m data.sync.sync_all --start 2024-01-01  # 指定起始日期

扩展新表:
    在 SYNC_STEPS 列表中新增一行即可
"""
import pandas as pd
import logging
import sys
from datetime import datetime
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

# ==================== 同步注册表 ====================
# 格式: (步骤名, 同步函数路径, 是否按月分批)
# 新增数据表只需在这里加一行

SYNC_STEPS: List[Tuple[str, str, bool]] = [
    ("交易日历",   "data.sync.trading_calendar:get_trading_calendar", False),
    ("指数日线",   "data.sync.sync_index_daily:sync_index_daily",    True),
    ("大单数据",   "data.sync.sync_index_large_order:sync_large_order", True),
    # ("ETF日线",  "data.sync.sync_etf_daily:sync_etf_daily",        True),
    # 新表在这里注册...
]


def _import_func(path: str) -> Callable:
    """动态导入函数: 'module.path:func_name' → callable"""
    module_path, func_name = path.rsplit(':', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def sync_by_periods(sync_func: Callable, start_date: str, end_date: str, freq: str = 'M') -> int:
    """按月分批同步"""
    total = 0
    for period in pd.period_range(start_date, end_date, freq=freq):
        p_start = period.start_time.strftime('%Y-%m-%d')
        p_end = period.end_time.strftime('%Y-%m-%d')
        try:
            count = sync_func(p_start, p_end)
            total += count
        except Exception as e:
            logger.error(f"  {period} 失败: {e}")
    return total


def sync_all(start_date: str, end_date: str, batch: bool = True):
    """
    执行所有同步步骤

    Args:
        start_date: 起始日期
        end_date: 结束日期
        batch: 是否按月分批（全量初始化时使用）
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"数据同步: {start_date} → {end_date}")
    logger.info(f"{'=' * 60}")

    for i, (name, func_path, can_batch) in enumerate(SYNC_STEPS, 1):
        step_label = f"[{i}/{len(SYNC_STEPS)}] {name}"
        logger.info(f"\n{step_label}...")

        func = _import_func(func_path)

        if name == "交易日历":
            # 交易日历特殊处理（不返回 count）
            func(start_date, end_date)
            continue

        if batch and can_batch:
            total = sync_by_periods(func, start_date, end_date)
        else:
            total = func(start_date, end_date)

        logger.info(f"  {name}: {total} 条")

    logger.info(f"\n{'=' * 60}")
    logger.info("同步完成")


def sync_incremental(lookback_days: int = 5):
    """增量同步最近 N 天"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    sync_all(start_date, end_date, batch=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if '--init' in sys.argv:
        start = '2012-01-01'
        end = '2025-12-31'
        # 支持 --start 参数
        if '--start' in sys.argv:
            idx = sys.argv.index('--start')
            start = sys.argv[idx + 1]
        sync_all(start, end, batch=True)
    else:
        if '--start' in sys.argv:
            idx = sys.argv.index('--start')
            start = sys.argv[idx + 1]
            end = datetime.now().strftime('%Y-%m-%d')
            sync_all(start, end, batch=False)
        else:
            sync_incremental()