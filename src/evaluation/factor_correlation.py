#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: factor_correlation.py
@time: 2025/12/29
@description:
因子相关性分析
"""
import pandas as pd
import numpy as np
from typing import Dict, Literal
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

from factors.base import create_factor
from factors.factor_config import FactorConfig

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                             title: str = "因子相关性热力图",
                             figsize: tuple = (10, 8),
                             save_path: str = None):
    """
    绘制相关性热力图

    Args:
        corr_matrix: 相关性矩阵
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=figsize)

    # 绘制热力图
    sns.heatmap(
        corr_matrix,
        annot=True,  # 显示数值
        fmt='.3f',  # 保留3位小数
        cmap='RdYlGn_r',  # 红-黄-绿配色（反转）
        center=0,  # 中心值为0
        vmin=-1,
        vmax=1,
        square=True,  # 正方形单元格
        linewidths=0.5,
        cbar_kws={'label': '相关系数'}
    )

    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"热力图已保存至: {save_path}")

    plt.show()


class FactorCorrelationAnalyzer:
    """因子相关性分析器"""

    def __init__(self, method: Literal['pearson', 'spearman'] = 'spearman'):
        """
        Args:
            method: 相关系数方法
                - 'pearson': 皮尔逊相关（线性）
                - 'spearman': 斯皮尔曼秩相关（单调，推荐）
        """
        self.method = method

    def analyze(self,
                factors: Dict[str, pd.DataFrame],
                corr_type: Literal['cross_section', 'time_series'] = 'time_series') -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factors: 因子字典 {因子名: 因子值DataFrame}
            corr_type: 相关性类型
                - 'time_series': 时序相关（默认）- 因子值随时间变化的相关性
                - 'cross_section': 截面相关 - 每个截面分别计算相关性再平均

        Returns:
            相关性矩阵 DataFrame
        """
        if corr_type == 'time_series':
            return self._time_series_correlation(factors)
        else:
            return self._cross_section_correlation(factors)

    def _time_series_correlation(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算时序相关性

        将每个因子展平为时间序列，计算两两相关性
        """
        factor_names = list(factors.keys())
        n = len(factor_names)
        corr_matrix = pd.DataFrame(np.eye(n), index=factor_names, columns=factor_names)

        for i, name1 in enumerate(factor_names):
            for j, name2 in enumerate(factor_names):
                if i >= j:
                    continue

                # 获取共同日期和行业
                common_dates = factors[name1].index.intersection(factors[name2].index)
                common_industries = factors[name1].columns.intersection(factors[name2].columns)

                # 对齐数据
                df1 = factors[name1].loc[common_dates, common_industries]
                df2 = factors[name2].loc[common_dates, common_industries]

                # 展平为一维序列
                values1 = df1.values.flatten()
                values2 = df2.values.flatten()

                # 去除NaN
                mask = ~(np.isnan(values1) | np.isnan(values2))
                values1 = values1[mask]
                values2 = values2[mask]

                if len(values1) < 10:
                    logger.warning(f"{name1} vs {name2}: 有效样本不足")
                    corr_matrix.loc[name1, name2] = np.nan
                    corr_matrix.loc[name2, name1] = np.nan
                    continue

                # 计算相关系数
                if self.method == 'pearson':
                    corr = np.corrcoef(values1, values2)[0, 1]
                else:  # spearman
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(values1, values2)

                corr_matrix.loc[name1, name2] = corr
                corr_matrix.loc[name2, name1] = corr

        logger.info(f"时序相关性计算完成，方法: {self.method}")
        return corr_matrix

    def _cross_section_correlation(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算截面相关性（每期横截面相关性的平均）
        """
        factor_names = list(factors.keys())
        n = len(factor_names)

        # 获取共同日期
        common_dates = None
        for df in factors.values():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)

        # 对齐所有因子到共同日期和行业
        aligned_factors = {}
        common_industries = None
        for name, df in factors.items():
            aligned_factors[name] = df.loc[common_dates]
            if common_industries is None:
                common_industries = df.columns
            else:
                common_industries = common_industries.intersection(df.columns)

        # 进一步对齐到共同行业
        for name in aligned_factors:
            aligned_factors[name] = aligned_factors[name][common_industries]

        # 逐期计算相关性
        period_corrs = []
        for date in common_dates:
            # 提取该期所有因子的横截面值
            cross_section = pd.DataFrame({
                name: df.loc[date]
                for name, df in aligned_factors.items()
            })

            # 去除有NaN的行业
            cross_section = cross_section.dropna()

            if len(cross_section) < 3:
                continue

            # 计算相关性矩阵
            if self.method == 'pearson':
                period_corr = cross_section.corr(method='pearson')
            else:
                period_corr = cross_section.corr(method='spearman')

            period_corrs.append(period_corr)

        # 平均相关性
        avg_corr = pd.concat(period_corrs).groupby(level=0).mean()

        logger.info(f"截面相关性计算完成（{len(period_corrs)}期），方法: {self.method}")
        return avg_corr


# ==================== 便捷函数 ====================

def calculate_factor_correlation(
        factors: Dict[str, pd.DataFrame],
        corr_type: Literal['cross_section', 'time_series'] = 'time_series',
        method: Literal['pearson', 'spearman'] = 'spearman',
        plot: bool = True,
        save_path: str = None
) -> pd.DataFrame:
    """
    快速计算因子相关性

    Args:
        factors: 因子字典
        corr_type: 'time_series' 或 'cross_section'
        method: 'pearson' 或 'spearman'
        plot: 是否绘制热力图
        save_path: 保存路径

    Returns:
        相关性矩阵
    """
    analyzer = FactorCorrelationAnalyzer(method=method)
    corr_matrix = analyzer.analyze(factors, corr_type)

    if plot:
        plot_correlation_heatmap(
            corr_matrix,
            title=f"因子{corr_type}相关性 ({method})",
            save_path=save_path
        )

    return corr_matrix


# ==================== 测试代码 ====================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 配置
    config = FactorConfig(
        start_date='2012-12-31',
        end_date='2023-12-29',
        frequency='monthly'
    )

    improved_overnight_return = create_factor("improved_overnight_return", preprocess=False)
    mom_factor = create_factor("marginal_average_momentum", window=20, preprocess=False)
    potential_energy = create_factor("potential_energy", preprocess=False)
    large_order_volatility = create_factor("large_order_volatility", preprocess=False)
    elasticity = create_factor("elasticity", preprocess=False)

    # 计算多个因子
    print("\n=== 计算因子 ===")
    factors_dict = {
        '隔夜收益率': improved_overnight_return(config),
        "动量加速度": mom_factor(config),
        "累积势能": potential_energy(config),
        "成交波动": large_order_volatility(config),
        "成交弹性": elasticity(config)
    }

    # 时序相关性
    print("\n=== 时序相关性 ===")
    corr_ts = calculate_factor_correlation(
        factors_dict,
        corr_type='time_series',
        method='spearman',
        plot=True
    )
    print(corr_ts)

    # 截面相关性
    print("\n=== 截面相关性 ===")
    corr_cs = calculate_factor_correlation(
        factors_dict,
        corr_type='cross_section',
        method='spearman',
        plot=True
    )
    print(corr_cs)