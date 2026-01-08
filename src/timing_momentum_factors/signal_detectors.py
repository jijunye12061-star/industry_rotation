#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: signal_detectors.py
@time: 2025/12/26
@description:
时点信号检测器
"""
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Optional

from timing_momentum_factors.data_manager import TimingDataManager

logger = logging.getLogger(__name__)


class BaseSignalDetector(ABC):
    """信号检测器基类"""

    def __init__(self, data_manager: TimingDataManager):
        self.data = data_manager

    @abstractmethod
    def check_date(self, date) -> bool:
        """检查单个日期是否触发信号"""
        pass

    def detect(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """批量检测区间内的触发日期"""
        # 修正：使用loc切片或转换为Timestamp比较
        check_dates = self.data.trading_dates[
            (self.data.trading_dates >= pd.to_datetime(start_date)) &
            (self.data.trading_dates <= pd.to_datetime(end_date))
            ]

        signals = [date for date in check_dates if self.check_date(date)]

        logger.info(
            f"{self.__class__.__name__}: "
            f"检测 {len(check_dates)} 日，触发 {len(signals)} 次"
        )

        return pd.DatetimeIndex(signals)


class ReboundSignalDetector(BaseSignalDetector):
    """
    大跌反弹信号检测器

    触发条件：
    1. T日涨幅 > U（动态阈值）
    2. T-1向前存在无反弹区间M（连续 <= U）
    3. 区间存在高低点（间隔>2天）
    4. 高低点下跌幅度 >= D（动态阈值）
    """

    def __init__(self,
                 data_manager: TimingDataManager,
                 base_drawdown: float = 0.05,
                 base_rebound: float = 0.005,
                 lookback_days: int = 180):
        super().__init__(data_manager)
        self.base_D = base_drawdown
        self.base_U = base_rebound
        self.lookback_days = lookback_days

    def _get_dynamic_thresholds(self, date) -> tuple[float, float]:
        """根据ATR动态调整阈值"""
        atr = self.data.get_atr(date)

        if atr < 0.01:
            factor = (atr / 0.01) ** 0.5
        elif atr > 0.02:
            factor = (atr / 0.02) ** 0.5
        else:
            factor = 1.0

        return self.base_D * factor, self.base_U * factor

    def check_date(self, date) -> bool:
        """检查某日是否触发大跌反弹信号"""
        date = pd.to_datetime(date)

        d, u = self._get_dynamic_thresholds(date)

        # 条件1: T日涨幅 > u
        if self.data.get_market_return(date) <= u:
            return False

        # 获取回看窗口数据
        prices = self.data.get_market_prices_window(date, self.lookback_days)
        returns = self.data.get_market_returns_window(date, self.lookback_days)

        if len(prices) < 10:
            return False

        # 条件2: 从T-1向前找无反弹区间M
        end_idx = len(returns) - 2
        start_idx = end_idx

        while start_idx > 0 and returns.iloc[start_idx] <= u:
            start_idx -= 1

        # 删除下面的逻辑 允许区间起点收益率 > U
        # if returns.iloc[start_idx] > u:
        #     start_idx += 1

        if end_idx - start_idx < 3:
            return False

        # 条件3: 找高低点（间隔>2天）
        interval_prices = prices.iloc[start_idx:end_idx + 1]
        high_idx = interval_prices.argmax()

        if high_idx + 3 >= len(interval_prices):
            return False

        low_part = interval_prices.iloc[high_idx + 3:]
        low_idx = high_idx + 3 + low_part.argmin()

        # 条件4: 下跌幅度 >= d
        drawdown = 1 - interval_prices.iloc[low_idx] / interval_prices.iloc[high_idx]

        return drawdown >= d

    def get_trigger_metadata(self, date) -> dict:
        """获取触发日的元数据"""
        date = pd.to_datetime(date)
        d, u = self._get_dynamic_thresholds(date)

        # 找到最后一个下跌日
        returns = self.data.get_market_returns_window(date, 30)
        end_idx = len(returns) - 2

        while end_idx > 0 and returns.iloc[end_idx] <= u:
            end_idx -= 1

        last_down_date = returns.index[end_idx + 1] if end_idx + 1 < len(returns) else date

        return {
            'trigger_date': date,
            'last_down_date': last_down_date,
            'atr': self.data.get_atr(date),
            'thresholds': (d, u)
        }


class TopSwitchSignalDetector(BaseSignalDetector):
    """
    顶部切换信号检测器

    触发条件（同时满足）：
    1. T日创52周新高的行业数量较T-1日减少3个或以上
    2. T日中证全指跌幅超过ATR60
    """

    def __init__(self, data_manager: TimingDataManager, min_decrease: int = 3):
        """
        Args:
            min_decrease: 新高行业减少数量阈值（默认3个）
        """
        super().__init__(data_manager)
        self.min_decrease = min_decrease

    def check_date(self, date) -> bool:
        """检查某日是否触发顶部切换信号"""
        date = pd.to_datetime(date)

        # 获取前一个交易日
        try:
            prev_date = self.data.get_previous_trading_day(date, n=1)
        except (IndexError, KeyError):
            return False

        # 条件1: T日新高数量较T-1日减少 >= min_decrease
        count_t = self.data.get_new_high_count(date)
        count_t1 = self.data.get_new_high_count(prev_date)

        if count_t1 - count_t < self.min_decrease:
            return False

        # 条件2: T日跌幅超过ATR60
        market_return = self.data.get_market_return(date)
        atr = self.data.get_atr(date)

        # 跌幅为负，ATR为正，需要判断 |return| > atr
        if market_return >= 0 or abs(market_return) <= atr:
            return False

        return True

    def get_trigger_metadata(self, date) -> dict:
        """获取触发日元数据"""
        date = pd.to_datetime(date)
        prev_date = self.data.get_previous_trading_day(date, n=1)

        count_t = self.data.get_new_high_count(date)
        count_t1 = self.data.get_new_high_count(prev_date)

        return {
            'trigger_date': date,
            'new_high_count_t': count_t,
            'new_high_count_t1': count_t1,
            'decrease': count_t1 - count_t,
            'market_return': self.data.get_market_return(date),
            'atr': self.data.get_atr(date)
        }


class TriangleBreakoutDetector(BaseSignalDetector):
    """
    三角突破形态检测器

    触发条件（同时满足）：
    1. 突破确认：T日涨跌幅 > B（突破幅度阈值）
    2. 前期收缩：T-5至T-1日，每日涨跌幅绝对值 <= C（收缩幅度阈值）
    3. 通道收窄：T-1日的5日通道宽度 < T-2日的5日通道宽度

    通道宽度 = (滚动5日最高价 - 滚动5日最低价) / 收盘价
    """

    def __init__(self,
                 data_manager: TimingDataManager,
                 breakout_threshold: float = 0.01,  # B：突破幅度阈值（1%）
                 contraction_threshold: float = 0.01):  # C：收缩幅度阈值（1%）
        """
        Args:
            breakout_threshold: 突破幅度阈值（如0.01表示1%）
            contraction_threshold: 收缩幅度阈值
        """
        super().__init__(data_manager)
        self.B = breakout_threshold
        self.C = contraction_threshold

    def check_date(self, date) -> bool:
        """检查某日是否触发三角突破信号"""
        date = pd.to_datetime(date)

        # 条件1：T日涨跌幅 > B
        market_return = self.data.get_market_return(date)
        if market_return <= self.B:
            return False

        # 条件2：T-5至T-1日，每日涨跌幅绝对值 <= C
        returns_window = self.data.get_market_returns_window(date, 6)  # 包含T日，共6天
        if len(returns_window) < 6:
            return False

        recent_5_returns = returns_window.iloc[-6:-1]  # T-5至T-1
        if (recent_5_returns.abs() > self.C).any():
            return False

        # 条件3：T-1日通道宽度 < T-2日通道宽度
        channel_t1 = self._get_channel_width(date, lookback_days=1)  # T-1日
        channel_t2 = self._get_channel_width(date, lookback_days=2)  # T-2日

        if channel_t1 is None or channel_t2 is None:
            return False

        if channel_t1 >= channel_t2:  # 未收窄
            return False

        return True

    def _get_channel_width(self, date, lookback_days: int = 0) -> Optional[float]:
        """
        获取指定日期的5日通道宽度

        通道宽度 = 5日内最高价 - 5日内最低价（绝对价差）

        Args:
            date: 参考日期
            lookback_days: 往前推几天（0=T日，1=T-1日）

        Returns:
            通道宽度（绝对值）
        """
        try:
            target_date = self.data.get_previous_trading_day(date, n=lookback_days)
        except (IndexError, KeyError):
            return None

        # 获取5日窗口的OHLC数据
        idx = self.data.ohlc.index.get_loc(target_date)
        start_idx = max(0, idx - 4)

        if idx - start_idx < 4:  # 不足5天
            return None

        window_ohlc = self.data.ohlc.iloc[start_idx:idx + 1]

        # 5日内的最高价和最低价
        high_5 = window_ohlc['high_price'].max()
        low_5 = window_ohlc['low_price'].min()

        return high_5 - low_5

    def get_trigger_metadata(self, date) -> dict:
        """获取触发日元数据"""
        date = pd.to_datetime(date)

        returns_window = self.data.get_market_returns_window(date, 6)
        recent_5_returns = returns_window.iloc[-6:-1]

        channel_t1 = self._get_channel_width(date, 1)
        channel_t2 = self._get_channel_width(date, 2)

        return {
            'trigger_date': date,
            'market_return': self.data.get_market_return(date),
            'recent_5_returns': recent_5_returns.tolist(),
            'recent_5_max_abs': recent_5_returns.abs().max(),
            'channel_width_t1': channel_t1,
            'channel_width_t2': channel_t2,
            'contraction_confirmed': (recent_5_returns.abs() <= self.C).all(),
            'narrowing_confirmed': channel_t1 < channel_t2 if channel_t1 and channel_t2 else False
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main_start_date = '2013-03-19'
    main_end_date = '2013-04-28'

    data_mgr = TimingDataManager(main_start_date, main_end_date)

    # 测试大跌反弹信号
    print("\n=== 大跌反弹信号 ===")
    rebound = ReboundSignalDetector(data_mgr)
    rebound_signals = rebound.detect(main_start_date, main_end_date)
    print(f"触发次数: {len(rebound_signals)}")
    print(f"触发日期: {rebound_signals[:3]}")

    # # 测试顶部切换信号
    # print("\n=== 顶部切换信号 ===")
    # top_switch = TopSwitchSignalDetector(data_mgr, min_decrease=3)
    # switch_signals = top_switch.detect(main_start_date, main_end_date)
    # print(f"触发次数: {len(switch_signals)}")
    # print(f"触发日期: {switch_signals[:3]}")
    #
    # # 测试三角突破信号
    # print("\n=== 三角突破信号 ===")
    # breakout = TriangleBreakoutDetector(data_mgr, breakout_threshold=0.01, contraction_threshold=0.01)
    # signals = breakout.detect(main_start_date, main_end_date)
    # print(f"触发次数: {len(signals)}")
