#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: factor_calculators.py
@time: 2025/12/26
@description:
时点动量因子计算器
"""
import pandas as pd
import logging
from abc import ABC, abstractmethod
from datetime import timedelta

from timing_momentum_factors.data_manager import TimingDataManager
from timing_momentum_factors.signal_detectors import (
    ReboundSignalDetector,
    TopSwitchSignalDetector,
    TriangleBreakoutDetector
)
from config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)


class BaseFactorCalculator(ABC):
    """因子计算器基类"""

    def __init__(self, data_manager: TimingDataManager):
        self.data = data_manager

    @abstractmethod
    def calculate(self, trigger_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        计算因子值

        Returns:
            pd.DataFrame with columns: ['date', 'industry_code', 'industry_name', 'score']
        """
        pass

    @staticmethod
    def _convert_to_rank(values: pd.Series) -> pd.Series:
        """
        将原始值转为横截面排名

        Args:
            values: 原始因子值（Series，index=行业代码）

        Returns:
            排名（1=最弱，值越大排名越大）
        """
        return values.rank(ascending=True, method='min', na_option='keep')

    @staticmethod
    def _to_long_format(date: pd.Timestamp,
                        scores: pd.Series,
                        include_name: bool = True) -> list:
        """
        将横截面数据转为长表格式

        Args:
            date: 触发日期
            scores: 行业评分（Series，index=行业代码）
            include_name: 是否包含行业名称

        Returns:
            [{date, industry_code, industry_name, score}, ...]
        """
        results = []
        for industry, score in scores.items():
            if pd.notna(score):
                row = {
                    'date': date,
                    'industry_code': industry,
                    'score': int(score)
                }
                if include_name:
                    row['industry_name'] = INDUSTRY_CONFIG[industry]['name']
                results.append(row)
        return results


class ReboundMomentumCalculator(BaseFactorCalculator):
    """
    反弹动量因子（对应 ReboundSignalDetector）

    公式：反弹动量 = 反弹当日涨跌幅 - 反弹前20日日均涨跌幅
    输出：横截面排名（1=最弱，值越大排名越大）
    """

    def __init__(self, data_manager: TimingDataManager, lookback: int = 20):
        super().__init__(data_manager)
        self.lookback = lookback

    def calculate(self, trigger_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """计算反弹动量因子"""
        results = []

        for date in trigger_dates:
            date_str = date.strftime('%Y-%m-%d')

            try:
                # 1. 获取T日行业收益率
                trigger_returns = self.data.get_industry_returns(
                    date_str, date_str
                ).iloc[0]

                # 2. 获取T-1至T-20的行业平均收益率
                hist_avg = self._get_historical_avg_returns(date, self.lookback)

                # 3. 计算反弹动量（原始值）
                momentum = trigger_returns - hist_avg

                # 4. 横截面排名（ascending=True，1=最弱）
                ranks = self._convert_to_rank(momentum)

                # 5. 转为长表格式
                results.extend(self._to_long_format(date, ranks))

                logger.debug(f"{date_str}: 反弹动量，行业数={len(ranks.dropna())}")

            except Exception as e:
                logger.warning(f"{date_str}: 计算失败 - {e}")
                continue

        df = pd.DataFrame(results)
        logger.info(f"反弹动量因子完成，触发{len(trigger_dates)}次，生成{len(df)}条记录")
        return df

    def _get_historical_avg_returns(self,
                                    trigger_date: pd.Timestamp,
                                    n_days: int) -> pd.Series:
        """计算T-1至T-n的行业平均收益率"""
        t_minus_1 = self.data.get_previous_trading_day(trigger_date, n=1)
        start_date = (t_minus_1 - timedelta(days=n_days * 2)).strftime('%Y-%m-%d')
        end_date = t_minus_1.strftime('%Y-%m-%d')

        hist_data = self.data.get_industry_returns(start_date, end_date)

        if len(hist_data) > n_days:
            hist_data = hist_data.tail(n_days)

        return hist_data.mean()


class TopSwitchMomentumCalculator(BaseFactorCalculator):
    """
    顶部切换动量因子（对应 TopSwitchSignalDetector）

    公式：切换动量 = 切换当日涨跌幅
    输出：横截面排名（1=最弱，值越大排名越大）
    """

    def calculate(self, trigger_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """计算切换动量因子"""
        results = []

        for date in trigger_dates:
            date_str = date.strftime('%Y-%m-%d')

            try:
                # 获取触发日当天行业收益率
                trigger_returns = self.data.get_industry_returns(
                    date_str, date_str
                ).iloc[0]

                # 横截面排名（ascending=True，1=最弱）
                ranks = self._convert_to_rank(trigger_returns)

                # 转为长表格式
                results.extend(self._to_long_format(date, ranks))

                logger.debug(f"{date_str}: 切换动量，行业数={len(ranks.dropna())}")

            except Exception as e:
                logger.warning(f"{date_str}: 计算失败 - {e}")
                continue

        df = pd.DataFrame(results)
        logger.info(f"切换动量因子完成，触发{len(trigger_dates)}次，生成{len(df)}条记录")
        return df


class BreakoutMomentumCalculator(BaseFactorCalculator):
    """
    三角突破动量因子（对应 TriangleBreakoutDetector）

    公式：突破动量 = 突破当日涨跌幅
    输出：横截面排名（1=最弱，值越大排名越大）
    """

    def calculate(self, trigger_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """计算突破动量因子"""
        results = []

        for date in trigger_dates:
            date_str = date.strftime('%Y-%m-%d')

            try:
                # 获取突破日当天行业收益率
                trigger_returns = self.data.get_industry_returns(
                    date_str, date_str
                ).iloc[0]

                # 横截面排名（ascending=True，1=最弱）
                ranks = self._convert_to_rank(trigger_returns)

                # 转为长表格式
                results.extend(self._to_long_format(date, ranks))

                logger.debug(f"{date_str}: 突破动量，行业数={len(ranks.dropna())}")

            except Exception as e:
                logger.warning(f"{date_str}: 计算失败 - {e}")
                continue

        df = pd.DataFrame(results)
        logger.info(f"突破动量因子完成，触发{len(trigger_dates)}次，生成{len(df)}条记录")
        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 初始化
    data_mgr = TimingDataManager('2023-01-01', '2024-12-31')

    # 测试反弹动量
    print("\n=== 反弹动量因子 ===")
    rebound_detector = ReboundSignalDetector(data_mgr)
    rebound_signals = rebound_detector.detect('2023-01-01', '2024-12-31')

    if len(rebound_signals) > 0:
        rebound_calc = ReboundMomentumCalculator(data_mgr, lookback=20)
        rebound_factor = rebound_calc.calculate(rebound_signals)
        print(rebound_factor.head(10))
        print(f"\n统计: {rebound_factor['score'].describe()}")

    # 测试切换动量
    print("\n=== 切换动量因子 ===")
    switch_detector = TopSwitchSignalDetector(data_mgr)
    switch_signals = switch_detector.detect('2023-01-01', '2024-12-31')

    if len(switch_signals) > 0:
        switch_calc = TopSwitchMomentumCalculator(data_mgr)
        switch_factor = switch_calc.calculate(switch_signals)
        print(switch_factor.head(10))
        print(f"\n统计: {switch_factor['score'].describe()}")

    # 测试突破动量
    print("\n=== 突破动量因子 ===")
    breakout_detector = TriangleBreakoutDetector(data_mgr)
    breakout_signals = breakout_detector.detect('2023-01-01', '2024-12-31')

    if len(breakout_signals) > 0:
        breakout_calc = BreakoutMomentumCalculator(data_mgr)
        breakout_factor = breakout_calc.calculate(breakout_signals)
        print(breakout_factor.head(10))
        print(f"\n统计: {breakout_factor['score'].describe()}")