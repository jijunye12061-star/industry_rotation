#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: ic_analysis.py
@time: 2025/12/10
@description:
IC分析模块

计算因子的IC指标：IC均值、ICIR、IC胜率、T值、P值等
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Literal
import logging

from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


class ICAnalyzer:
    """
    IC分析器

    评估因子预测能力的统计指标
    """

    def __init__(self, method: Literal['pearson', 'spearman'] = 'spearman'):
        """
        Args:
            method: 相关系数计算方法
                - 'pearson': 皮尔逊相关（线性）
                - 'spearman': 斯皮尔曼秩相关（单调，推荐）
        """
        self.method = method

    def analyze(self,
                factor: pd.DataFrame,
                config: FactorConfig,
                forward_periods: int = 1) -> Dict[str, float]:
        """计算因子IC指标（自动获取收益率数据）"""
        # 使用 loader 获取未来收益率
        loader = get_loader()
        future_returns = loader.get_forward_returns(
            rebalance_dates=factor.index,  # type: ignore
            start_date=config.start_date,
            end_date=config.end_date,
            forward_periods=forward_periods
        )

        ic_time_series = self._calculate_ic_series(factor, future_returns)

        # 后续计算逻辑保持不变
        results = {
            'IC_mean': ic_time_series.mean(),
            'IC_std': ic_time_series.std(),
            'ICIR': ic_time_series.mean() / ic_time_series.std() if ic_time_series.std() > 0 else np.nan,
            'IC_win_rate': (ic_time_series > 0).sum() / len(ic_time_series),
            't_value': np.nan,
            'p_value': np.nan,
        }

        if len(ic_time_series) > 1:
            t_stat, p_val = stats.ttest_1samp(ic_time_series.dropna(), 0)
            results['t_value'] = t_stat
            results['p_value'] = p_val

        logger.info(f"IC分析完成 - IC均值: {results['IC_mean']:.4f}, ICIR: {results['ICIR']:.4f}")
        return results

    def _calculate_ic_series(self,
                             factor: pd.DataFrame,
                             returns: pd.DataFrame) -> pd.Series:
        """
        计算每期IC值（横截面相关系数）

        Returns:
            IC时间序列
        """
        common_dates = factor.index.intersection(returns.index)
        factor = factor.loc[common_dates]
        returns = returns.loc[common_dates]

        ic_list = []
        for date in common_dates:
            factor_cross = factor.loc[date]
            return_cross = returns.loc[date]

            # 去除NaN
            valid_mask = factor_cross.notna() & return_cross.notna()  # type: ignore
            if valid_mask.sum() < 3:  # 至少3个有效样本
                ic_list.append(np.nan)
                continue

            f = factor_cross[valid_mask].values  # type: ignore
            r = return_cross[valid_mask].values  # type: ignore

            # 计算相关系数
            if self.method == 'pearson':
                ic, _ = stats.pearsonr(f, r)
            else:  # spearman
                ic, _ = stats.spearmanr(f, r)

            ic_list.append(ic)

        return pd.Series(ic_list, index=common_dates, name='IC')

    def get_ic_series(self,
                      factor: pd.DataFrame,
                      config: FactorConfig,
                      forward_periods: int = 1) -> pd.Series:
        """获取IC时间序列（用于绘图）"""
        loader = get_loader()
        future_returns = loader.get_forward_returns(
            rebalance_dates=factor.index,  # type: ignore
            start_date=config.start_date,
            end_date=config.end_date,
            forward_periods=forward_periods
        )
        return self._calculate_ic_series(factor, future_returns)

    def analyze_with_details(self,
                             factor: pd.DataFrame,
                             config: FactorConfig,
                             forward_periods: int = 1) -> Dict:
        """带详细信息的分析"""
        loader = get_loader()
        future_returns = loader.get_forward_returns(
            rebalance_dates=factor.index,  # type: ignore
            start_date=config.start_date,
            end_date=config.end_date,
            forward_periods=forward_periods
        )
        ic_time_series = self._calculate_ic_series(factor, future_returns)
        metrics = self.analyze(factor, config, forward_periods)

        return {
            'metrics': metrics,
            'ic_series': ic_time_series,
            'cumulative_ic': ic_time_series.cumsum()
        }


def calculate_ic_metrics(factor: pd.DataFrame,
                         config: FactorConfig,
                         forward_periods: int = 1,
                         method: Literal['pearson', 'spearman'] = 'spearman') -> Dict[str, float]:
    """
    便捷函数：计算IC指标
    """
    analyzer = ICAnalyzer(method=method)
    return analyzer.analyze(factor, config, forward_periods)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 使用真实因子测试
    from factors.base import create_factor

    main_config = FactorConfig(
        start_date='2013-01-31',
        end_date='2023-12-29',
        frequency='monthly'
    )

    # 计算因子
    factor_calculator = create_factor("momentum", window=21, preprocess=False)
    marginal_beta = factor_calculator(main_config)

    # IC分析
    ic_analyzer = ICAnalyzer(method='spearman')  # 可选'pearson', 'spearman'
    ic_results = ic_analyzer.analyze(marginal_beta, main_config, forward_periods=1)

    print("\n=== 边际贝塔因子 IC分析结果 ===")
    for key, value in ic_results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")

    # 获取IC序列
    ic_values = ic_analyzer.get_ic_series(marginal_beta, main_config)
    print(f"\nIC序列:\n{ic_values}")

    from utils.plot_funcs import plot_ic_analysis
    plot_ic_analysis(ic_values)