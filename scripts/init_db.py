#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : init_db.py
@Time    : 2025/12/18
@Author  : jijunye
@Desc    : 数据库初始化脚本（仅首次部署时运行）

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from data.db_manager import DBManager

if __name__ == '__main__':
    print("开始初始化数据库...")

    db = DBManager()
    db.create_tables()

    print(f"✓ 数据库初始化完成: {db.db_path}")