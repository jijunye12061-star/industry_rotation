# src/data/loader.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: loader.py
@time: 2025/01/04
@description:
数据加载器统一入口（工厂模式）
"""
from typing import Optional
from data.base_loader import BaseDataLoader
from config import DATA_SOURCE, MARKET_CODE

_loader_instance: Optional[BaseDataLoader] = None


def get_loader() -> BaseDataLoader:
    """
    获取数据加载器（根据配置自动选择）

    Returns:
        OracleLoader | CachedLoader
    """
    global _loader_instance

    if _loader_instance is None:
        if DATA_SOURCE == 'remote':
            from data.oracle_loader import OracleLoader
            _loader_instance = OracleLoader()
        elif DATA_SOURCE == 'local':
            from data.cached_loader import CachedLoader
            _loader_instance = CachedLoader()
        else:
            raise ValueError(f"Invalid DATA_SOURCE: {DATA_SOURCE}")

    return _loader_instance


def reset_loader():
    """重置加载器（用于切换数据源）"""
    global _loader_instance
    _loader_instance = None


# 便捷函数（保持向后兼容）
def get_trading_dates(start_date: str, end_date: str):
    return get_loader().get_trading_dates(start_date, end_date)


def get_industry_prices(start_date: str, end_date: str):
    return get_loader().get_industry_prices(start_date, end_date)


def get_industry_returns(start_date: str, end_date: str):
    return get_loader().get_industry_returns(start_date, end_date)


def get_market_returns(start_date: str, end_date: str, market_code: str = MARKET_CODE):
    return get_loader().get_market_returns(start_date, end_date, market_code)
