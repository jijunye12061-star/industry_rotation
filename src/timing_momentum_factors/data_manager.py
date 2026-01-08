#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: data_manager.py
@time: 2025/12/26
@description:
时点动量因子数据管理器

负责数据获取、预计算技术指标（ATR、新高统计等）
"""
import pandas as pd
import logging
from datetime import timedelta
from typing import Optional

from data.loader import get_loader
from config import MARKET_CODE

logger = logging.getLogger(__name__)


class TimingDataManager:
    """
    时点动量因子数据管理器

    职责：
    1. 复用 DataLoader 获取基础数据
    2. 预计算技术指标（ATR、新高统计）
    3. 提供高效的数据访问接口
    """

    def __init__(self, start_date: str, end_date: str, market_code: str = MARKET_CODE):
        """
        初始化数据管理器（数据一次性加载，避免重复查询）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            market_code: 市场指数内码（默认中证全指）
        """
        self.market_code = market_code
        loader = get_loader()

        # 扩展起始日期以满足回看窗口需求
        extended_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')

        logger.info(f"初始化数据管理器: {extended_start} 至 {end_date}")

        # ==================== 市场数据 ====================
        # 返回 DataFrame with columns: ['close_price', 'high_price', 'low_price', 'preclose_price']
        self.ohlc = loader.get_market_ohlc(extended_start, end_date, market_code)
        self.prices = self.ohlc['close_price']
        self.returns = (self.ohlc['close_price'] / self.ohlc['preclose_price'] - 1)

        # ==================== 行业数据 ====================
        self.industry_prices = loader.get_industry_prices(extended_start, end_date)
        # 行业收益率延迟加载（按需计算，避免内存占用）
        self._industry_returns_cache: Optional[pd.DataFrame] = None
        self._industry_returns_range = (extended_start, end_date)

        # ==================== 交易日历 ====================
        self.trading_dates = loader.get_trading_dates(extended_start, end_date)

        # ==================== 预计算指标 ====================
        logger.info("预计算技术指标...")
        self.atr60 = self._calculate_atr(window=60)
        self.new_high_counts = self._calculate_new_high_counts(window=252)
        logger.info("数据管理器初始化完成")

    # ==================== 技术指标计算 ====================

    def _calculate_atr(self, window: int = 60) -> pd.Series:
        """
        计算平均真实波动幅度（ATR）

        Args:
            window: 滚动窗口期

        Returns:
            ATR序列（以前收盘价归一化）
        """
        h = self.ohlc['high_price']
        l = self.ohlc['low_price']
        c_prev = self.ohlc['preclose_price']

        # 真实波动幅度 = max(H-L, |H-C_prev|, |L-C_prev|) / C_prev
        tr = pd.DataFrame({
            'hl': (h - l) / c_prev,
            'hc': (h - c_prev).abs() / c_prev,
            'lc': (c_prev - l).abs() / c_prev
        }).max(axis=1)

        atr = tr.rolling(window=window, min_periods=int(window * 0.8)).mean()
        logger.info(f"ATR计算完成，窗口={window}")
        return atr

    def _calculate_new_high_counts(self, window: int = 252) -> pd.Series:
        """
        计算每日创N日新高的行业数量

        Args:
            window: 新高回看窗口（252 ≈ 52周）

        Returns:
            每日新高行业数量序列
        """
        prices = self.industry_prices
        counts = pd.Series(index=prices.index, dtype=int)

        # 向量化计算：滚动最大值
        rolling_max = prices.rolling(window=window, min_periods=window).max()

        # 当日价格 >= 滚动最大值，视为创新高
        for i in range(window, len(prices)):
            is_new_high = (prices.iloc[i] >= rolling_max.iloc[i - 1])
            counts.iloc[i] = is_new_high.sum()  # type: ignore

        logger.info(f"新高统计完成，窗口={window}")
        return counts

    # ==================== 市场数据接口 ====================

    def get_market_price(self, date) -> float:
        """获取市场指数收盘价"""
        return self.prices.loc[pd.to_datetime(date)]

    def get_market_return(self, date) -> float:
        """获取市场指数日收益率"""
        return self.returns.loc[pd.to_datetime(date)]

    def get_market_prices_window(self, date, n_days: int) -> pd.Series:
        """
        获取市场指数历史价格窗口

        Args:
            date: 截止日期
            n_days: 回看天数（含当日）

        Returns:
            价格序列（长度 <= n_days）
        """
        date = pd.to_datetime(date)
        idx = self.prices.index.get_loc(date)
        start_idx = max(0, idx - n_days + 1)
        return self.prices.iloc[start_idx:idx + 1]

    def get_market_returns_window(self, date, n_days: int) -> pd.Series:
        """获取市场指数历史收益率窗口"""
        date = pd.to_datetime(date)
        idx = self.returns.index.get_loc(date)
        start_idx = max(0, idx - n_days + 1)
        return self.returns.iloc[start_idx:idx + 1]

    # ==================== 行业数据接口 ====================

    def get_industry_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取行业价格数据

        Returns:
            DataFrame (dates × industries)
        """
        return self.industry_prices.loc[start_date:end_date]

    def get_industry_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取行业收益率（延迟计算+缓存）

        Returns:
            DataFrame (dates × industries)，单位: %
        """
        # 检查缓存
        if self._industry_returns_cache is None:
            loader = get_loader()
            self._industry_returns_cache = loader.get_industry_returns(
                *self._industry_returns_range
            )

        return self._industry_returns_cache.loc[start_date:end_date]

    # ==================== 预计算指标接口 ====================

    def get_atr(self, date) -> float:
        """获取某日ATR值"""
        return self.atr60.loc[pd.to_datetime(date)]

    def get_new_high_count(self, date) -> int:
        """获取某日创新高的行业数量"""
        return int(self.new_high_counts.loc[pd.to_datetime(date)])

    # ==================== 辅助方法 ====================

    def is_trading_day(self, date) -> bool:
        """检查是否交易日"""
        return pd.to_datetime(date) in self.trading_dates

    def get_previous_trading_day(self, date, n: int = 1) -> pd.Timestamp:
        """获取前N个交易日"""
        date = pd.to_datetime(date)
        idx = self.trading_dates.get_loc(date)
        return self.trading_dates[max(0, idx - n)]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试
    data_mgr = TimingDataManager('2023-01-01', '2024-12-31')

    test_date = '2024-03-15'
    print(f"\n=== 测试日期: {test_date} ===")
    print(f"市场收盘价: {data_mgr.get_market_price(test_date):.2f}")
    print(f"市场收益率: {data_mgr.get_market_return(test_date):.2%}")
    print(f"ATR: {data_mgr.get_atr(test_date):.2%}")
    print(f"创新高行业数: {data_mgr.get_new_high_count(test_date)}")

    print("\n=== 测试窗口数据 ===")
    prices_window = data_mgr.get_market_prices_window(test_date, 5)
    print(f"近5日价格:\n{prices_window}")