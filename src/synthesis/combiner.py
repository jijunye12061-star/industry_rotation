#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: combiner.py
@time: 2025/12/15
@description:
因子合成模块 - 基于截面排序得分
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from factors.factor_config import FactorConfig
from factors.base import create_factor

logger = logging.getLogger(__name__)


class FactorCombiner:
    """因子合成器 - 等权或自定义权重"""

    def __init__(self, use_rank_score: bool = True):
        """
        Args:
            use_rank_score: 是否转换为横截面排序得分 [0,1]
        """
        self.use_rank_score = use_rank_score

    def combine(
            self,
            factors: Dict[str, pd.DataFrame],
            custom_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        合成多个因子

        Args:
            factors: {因子名: 因子值DataFrame}
            custom_weights: 自定义权重（None则等权）

        Returns:
            合成因子 DataFrame
        """
        if not factors:
            raise ValueError("至少需要一个因子")

        # 转换为排序得分
        if self.use_rank_score:
            factors = {name: self._to_rank_score(df) for name, df in factors.items()}
            logger.info(f"合成 {len(factors)} 个因子（已转排序得分）")
        else:
            logger.info(f"合成 {len(factors)} 个因子（原始值）")

        # 加权合成
        return self._weighted_average(factors, custom_weights)

    @staticmethod
    def _to_rank_score(factor: pd.DataFrame) -> pd.DataFrame:
        """横截面排序得分 [0,1]"""
        return factor.rank(axis=1, pct=True, method='average')

    @staticmethod
    def _weighted_average(
            factors: Dict[str, pd.DataFrame],
            weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        加权平均合成（自动归一化权重）

        Args:
            factors: 因子字典
            weights: 权重字典（None则等权，自动归一化）
        """
        # 权重归一化（统一处理等权和自定义）
        if weights is None:
            weights = {name: 1.0 for name in factors}

        total = sum(weights.values())
        weights = {name: weights.get(name, 0) / total for name in factors}

        # 收集非零权重因子
        factor_list = [(df, weights[name]) for name, df in factors.items() if weights[name] > 0]

        if not factor_list:
            raise ValueError("所有因子权重为0")

        # 获取统一索引
        all_dates = pd.Index(sorted(set().union(*[df.index for df, _ in factor_list])))
        all_industries = pd.Index(sorted(set().union(*[df.columns for df, _ in factor_list])))

        # 向量化加权
        combined = pd.DataFrame(0.0, index=all_dates, columns=all_industries)
        total_weights = pd.DataFrame(0.0, index=all_dates, columns=all_industries)

        for df, weight in factor_list:
            df_aligned = df.reindex(index=all_dates, columns=all_industries)
            valid_mask = df_aligned.notna()

            combined = combined.add(df_aligned.mul(weight), fill_value=0)
            total_weights = total_weights.add(valid_mask.astype(float).mul(weight), fill_value=0)

        return combined.div(total_weights.replace(0, np.nan))


# ==================== 便捷函数 ====================

def combine_from_configs(
        config: FactorConfig,
        factor_configs: Dict[str, tuple[str, dict]],
        custom_weights: Optional[Dict[str, float]] = None,
        use_rank_score: bool = True
) -> pd.DataFrame:
    """
    从因子配置直接生成合成因子

    Args:
        config: 因子配置
        factor_configs: {因子名: (因子类型, 参数)}
        custom_weights: 自定义权重（None则等权）
        use_rank_score: 是否转排序得分

    Returns:
        合成因子 DataFrame

    Example:
        factor_configs = {
            '动量': ('momentum_20', {'window': 20}),
            '波动': ('volatility', {})
        }
        combined = combine_from_configs(config, factor_configs)
    """
    logger.info(f"批量计算 {len(factor_configs)} 个因子")

    factors_dict = {}
    for name, (factor_type, params) in factor_configs.items():
        try:
            factor = create_factor(factor_type, preprocess=False, **params)
            factors_dict[name] = factor(config)
            logger.info(f"✓ {name}")
        except Exception as e:
            logger.warning(f"✗ {name}: {e}")

    if not factors_dict:
        raise ValueError("所有因子计算失败")

    logger.info(f"成功 {len(factors_dict)}/{len(factor_configs)} 个")

    combiner = FactorCombiner(use_rank_score=use_rank_score)
    return combiner.combine(factors_dict, custom_weights)


# ==================== 测试代码 ====================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main_config = FactorConfig(
        start_date='2012-12-31',
        end_date='2023-12-31',
        frequency='monthly'
    )

    main_factor_configs = {
        '隔夜收益率': ('improved_overnight_return', {}),
        '动量加速度': ('marginal_average_momentum', {'window': 20}),
        '累积势能': ('potential_energy', {}),
        '成交波动': ('large_order_volatility', {}),
        '成交弹性': ('elasticity', {}),
    }

    # 等权合成
    print("\n=== 等权合成 ===")
    combined_df = combine_from_configs(main_config, main_factor_configs)
    print(f"形状: {combined_df.shape}")
    print(f"范围: [{combined_df.min().min():.4f}, {combined_df.max().max():.4f}]")
    print(f"缺失: {combined_df.isna().sum().sum() / combined_df.size:.2%}")
    print(combined_df.tail())

    # # 自定义权重
    # print("\n=== 自定义权重 ===")
    # custom = combine_from_configs(
    #     test_config,
    #     factor_configs,
    #     custom_weights={'隔夜收益率': 0.4, '动量加速度': 0.6}
    # )
    # print(custom.tail())

    # 验证
    from evaluation.ic_analysis import ICAnalyzer
    from evaluation.report_generator import generate_factor_report

    ic_analyzer = ICAnalyzer(method='spearman')
    ic_results = ic_analyzer.analyze(combined_df, main_config)

    print("\n=== IC指标 ===")
    for key, value in ic_results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")

    generate_factor_report(combined_df, "合成因子", main_config)