#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日数据同步脚本
Usage: python scripts/daily_sync.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from data.data_sync import sync_all_to_latest
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    results = sync_all_to_latest()

    total = sum(results.values())
    print(f"\n同步完成，共 {total} 条新记录")

    for table, count in results.items():
        if count > 0:
            print(f"  • {table}: +{count}")