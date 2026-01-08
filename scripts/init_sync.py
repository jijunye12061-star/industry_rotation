#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : init_sync.py
@Time    : 2025/12/18
@Author  : jijunye
@Desc    : 首次全量数据同步脚本（仅部署时运行一次）

Usage:
    python scripts/init_sync.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from data.data_sync import get_sync
import logging

logging.basicConfig(
    level=logging.INFO
)


def main():
    """首次全量同步"""
    sync = get_sync()

    print("=" * 60)
    print("开始首次全量数据同步（2015-01-01 至今）")
    print("=" * 60)

    # 1. 同步交易日历
    print("\n[1/3] 同步交易日历...")
    sync.sync_trading_calendar('2015-01-01', '2025-12-25')

    # 2. 同步指数日线数据
    print("\n[2/3] 同步指数日线数据...")
    sync.sync_table_by_month('tb_index_daily', '2015-01-01', '2025-12-25')

    # 3. 同步指数大单数据
    print("\n[3/3] 同步指数大单数据...")
    sync.sync_table_by_month('tb_index_large_order', '2015-01-01', '2025-12-25')

    # 显示同步状态
    print("\n" + "=" * 60)
    print("同步完成！当前状态：")
    print("=" * 60)
    status = sync.get_sync_status()
    print(status.to_string(index=False))

    print("\n提示：后续使用 sync_table_incremental() 进行增量更新")


if __name__ == '__main__':
    main()