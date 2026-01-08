#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: base_loader.py
@time: 2025/12/22
@description:
数据加载器抽象基类
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
from config import MARKET_CODE


class BaseDataLoader(ABC):
    """数据加载器接口定义"""

    # ==================== 交易日历 ====================

    @abstractmethod
    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """获取交易日期序列"""
        pass

    # ==================== 价格数据 ====================

    @abstractmethod
    def get_industry_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业收盘价 (日期 × 行业代码)"""
        pass

    @abstractmethod
    def get_market_prices(self, start_date: str, end_date: str,
                          market_code: str = MARKET_CODE) -> pd.Series:
        """获取市场指数收盘价"""
        pass

    def get_industry_amounts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业成交额 (日期 × 行业代码)"""
        pass

    def get_industry_open_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业开盘价 (日期 × 行业代码)"""
        pass

    def get_industry_low_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业最低价 (日期 × 行业代码)"""
        pass

    def get_industry_high_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业最高价 (日期 × 行业代码)"""
        pass

    def get_industry_overnight_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业隔夜收益率 (日期 × 行业代码), 单位: %"""
        pass

    # ==================== 收益率数据 ====================

    @abstractmethod
    def get_industry_returns(self, start_date: str, end_date: str,
                             method: str = 'pct') -> pd.DataFrame:
        """获取行业收益率 (日期 × 行业代码)"""
        pass

    @abstractmethod
    def get_market_returns(self, start_date: str, end_date: str,
                           market_code: str = MARKET_CODE,
                           method: str = 'pct') -> pd.Series:
        """获取市场收益率"""
        pass

    def get_market_ohlc(self, start_date: str, end_date: str,
                        market_code: str = MARKET_CODE) -> pd.DataFrame:
        """获取市场指数OHLC数据（含前收盘），索引为交易日"""

    def get_forward_returns(self,
                            rebalance_dates: pd.DatetimeIndex,
                            start_date: str,
                            end_date: str,
                            forward_periods: int = 1) -> pd.DataFrame:
        """
        获取调仓日的未来收益率（用于IC分析）

        Args:
            rebalance_dates: 调仓日期序列
            start_date: 数据开始日期
            end_date: 数据结束日期
            forward_periods: 前瞻期数（1=下一期）

        Returns:
            未来收益率 DataFrame (调仓日 × 行业代码), 单位: %
        """
        # 扩展结束日期以获取未来数据
        from datetime import timedelta
        extended_end = (pd.to_datetime(end_date) + timedelta(days=60)).strftime('%Y-%m-%d')

        # 获取价格数据
        prices = self.get_industry_prices(start_date, extended_end)

        # 计算区间 收益率
        forward_returns = pd.DataFrame(
            index=rebalance_dates[:-forward_periods],
            columns=prices.columns,
            dtype=float
        )

        for i in range(len(rebalance_dates) - forward_periods):  # type: ignore
            start_dt = rebalance_dates[i]
            end_dt = rebalance_dates[i + forward_periods]

            if start_dt in prices.index and end_dt in prices.index:
                start_prices = prices.loc[start_dt]
                end_prices = prices.loc[end_dt]
                period_return = (end_prices / start_prices - 1) * 100
                forward_returns.loc[start_dt] = period_return

        return forward_returns

    # ==================== 成分股数据 ====================

    @abstractmethod
    def get_index_constituents(self, industry_code: str, date: str) -> pd.DataFrame:
        """获取行业成分股及权重"""
        pass

    # ==================== 个股数据 ====================

    @abstractmethod
    def get_stock_amounts(self, stock_codes: List[str],
                          start_date: str, end_date: str) -> pd.DataFrame:
        """获取个股成交额 (日期 × 股票代码)"""
        pass

    @abstractmethod
    def get_stock_super_large_flow(self, stock_codes: List[str],
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """获取个股超大单成交额 (日期 × 股票代码)"""
        pass

    # ==================== 其他接口 ====================

    @abstractmethod
    def get_all_industry_codes(self) -> List[str]:
        """获取所有行业代码"""
        pass

    @staticmethod
    def get_industry_info(code: str = None) -> Dict:
        """获取行业信息（可选：子类覆盖）"""
        from config import INDUSTRY_CONFIG
        if code is None:
            return INDUSTRY_CONFIG
        return INDUSTRY_CONFIG.get(code, {})

    def get_industry_large_order_amount(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业超大单成交额 (日期 × 行业代码)"""
        pass