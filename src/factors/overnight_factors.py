#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: overnight_factors.py
@time: 2025/12/12 14:32
@description:
隔夜收益率相关因子

基于开盘价与收盘价计算隔夜收益率指标
"""
import pandas as pd
import logging
from datetime import timedelta
from typing import Optional, Literal, Union

from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


@register_factor("overnight_return")
class OvernightReturnFactor(BaseFactor):
    """
    隔夜收益率因子

    隔夜收益率 = (今日开盘价 / 昨日收盘价 - 1) * 100
    因子值 = 过去N个交易日的隔夜收益率均值
    """

    def __init__(self,
                 name: str = "overnight_return",
                 window: int = 250,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = None,
                 **params):
        """
        Args:
            window: 回溯窗口期（交易日数，默认250天）
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法
            winsorize: 去极值方法
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算隔夜收益率因子"""
        # 扩展开始日期以获取足够历史数据
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=self.window * 2)).strftime('%Y-%m-%d')  # type: ignore

        # 获取隔夜收益率数据
        loader = get_loader()
        overnight_returns = loader.get_industry_overnight_returns(extended_start, config.end_date)

        # 计算滚动均值
        factor_values = overnight_returns.rolling(window=self.window, min_periods=int(self.window * 0.8)).mean()

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        factor_values = factor_values.reindex(rebalance_dates)

        logger.info(f"隔夜收益率因子计算完成，形状: {factor_values.shape}")
        return factor_values


@register_factor("standardized_overnight_return")
class StandardizedOvernightReturnFactor(BaseFactor):
    """
    标准化隔夜收益率因子

    每日先对隔夜收益率做横截面Zscore标准化，然后取过去N天均值
    """

    def __init__(self,
                 name: str = "standardized_overnight_return",
                 window: int = 250,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 回溯窗口期（交易日数，默认250天）
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法
            winsorize: 去极值方法
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算标准化隔夜收益率因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=self.window * 2)).strftime('%Y-%m-%d')

        # 获取隔夜收益率数据
        loader = get_loader()
        overnight_returns = loader.get_industry_overnight_returns(extended_start, config.end_date)

        # 横截面标准化
        standardized_returns = self._standardize_zscore(overnight_returns)

        # 计算滚动均值
        factor_values = standardized_returns.rolling(window=self.window, min_periods=int(self.window * 0.8)).mean()

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        factor_values = factor_values.reindex(rebalance_dates)

        logger.info(f"标准化隔夜收益率因子计算完成，形状: {factor_values.shape}")
        return factor_values


@register_factor("improved_overnight_return")
class ImprovedOvernightReturnFactor(BaseFactor):
    """
    改进隔夜收益率因子

    剔除标准差最小的20%样本后，取标准化隔夜收益率均值
    """

    def __init__(self,
                 name: str = "improved_overnight_return",
                 window: int = 250,
                 trim_ratio: float = 0.2,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 回溯窗口期（交易日数，默认250天）
            trim_ratio: 剔除低波动样本的比例（默认0.2，即20%）
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法
            winsorize: 去极值方法
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            trim_ratio=trim_ratio,
            **params
        )
        self.window = window
        self.trim_ratio = trim_ratio

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算改进隔夜收益率因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=self.window * 2)).strftime('%Y-%m-%d')

        # 获取隔夜收益率数据
        loader = get_loader()
        overnight_returns = loader.get_industry_overnight_returns(extended_start, config.end_date)

        # 横截面标准化
        standardized_returns = self._standardize_zscore(overnight_returns)

        # 计算改进因子（剔除低波动样本）
        factor_values = self._calculate_improved_factor(overnight_returns, standardized_returns)

        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        factor_values = factor_values.reindex(rebalance_dates)

        logger.info(f"改进隔夜收益率因子计算完成，形状: {factor_values.shape}")
        return factor_values

    def _calculate_improved_factor(self,
                                   overnight_returns: pd.DataFrame,
                                   standardized_returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算改进因子：剔除低波动日期后取均值

        Args:
            overnight_returns: 原始隔夜收益率
            standardized_returns: 标准化隔夜收益率

        Returns:
            改进因子值
        """
        result = pd.DataFrame(index=overnight_returns.index, columns=overnight_returns.columns, dtype=float)

        for i in range(self.window, len(overnight_returns)):
            # 获取窗口期数据
            window_data = overnight_returns.iloc[i - self.window:i]
            window_std_data = standardized_returns.iloc[i - self.window:i]

            # 计算每一天的横截面标准差
            daily_cross_std = window_data.std(axis=1)

            # 剔除标准差最小的20%的日期
            n_trim = int(len(daily_cross_std) * self.trim_ratio)
            threshold = daily_cross_std.sort_values().iloc[n_trim] if n_trim > 0 else daily_cross_std.min()

            # 保留标准差大于阈值的日期
            valid_dates = daily_cross_std[daily_cross_std > threshold].index

            # 对保留日期的标准化收益率取均值
            if len(valid_dates) > 0:
                factor_value = window_std_data.loc[valid_dates].mean(axis=0)
                result.iloc[i] = factor_value

        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 创建配置
    test_config = FactorConfig(
        start_date='2024-01-01',
        end_date='2025-03-31',
        frequency='monthly'
    )

    # 测试三个因子
    print("\n=== 测试隔夜收益率因子 ===")
    factor1 = OvernightReturnFactor(window=250)
    result1 = factor1(test_config)
    print(f"形状: {result1.shape}")
    print(result1.tail())

    print("\n=== 测试标准化隔夜收益率因子 ===")
    factor2 = StandardizedOvernightReturnFactor(window=250)
    result2 = factor2(test_config)
    print(f"形状: {result2.shape}")
    print(result2.tail())

    print("\n=== 测试改进隔夜收益率因子 ===")
    factor3 = ImprovedOvernightReturnFactor(window=250, trim_ratio=0.2)
    result3 = factor3(test_config)
    print(f"形状: {result3.shape}")
    print(result3.tail())