#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETF策略生成器 - 因子分数转持仓权重"""
import pandas as pd
import logging
from typing import Literal, Optional

from strategy.etf_selector import ETFSelector
from factors.factor_config import FactorConfig

logger = logging.getLogger(__name__)


class ETFStrategy:
    """ETF持仓生成器"""

    def __init__(self, selector: ETFSelector):
        """
        Args:
            selector: ETF筛选器（复用其select_etfs方法）
        """
        self.selector = selector

    def generate_positions(
            self,
            factor: pd.DataFrame,
            config: FactorConfig,
            mode: Literal['fixed', 'dynamic'],
            n_positions: Optional[int] = None,
            top_pct: Optional[float] = None
    ) -> pd.DataFrame:
        """
        根据因子分数生成ETF持仓

        Args:
            factor: 因子值 (日期×CI行业码)
            config: 因子配置（用于获取调仓日）
            mode: 'fixed'=固定数量 | 'dynamic'=动态百分比
            n_positions: 固定模式下选几个ETF
            top_pct: 动态模式下选前X%（如0.2表示前20%）

        Returns:
            长表 DataFrame: [trade_date, etf_code, weight]
        """
        # 参数校验
        if mode == 'fixed' and n_positions is None:
            raise ValueError("固定模式需指定 n_positions")
        if mode == 'dynamic' and top_pct is None:
            raise ValueError("动态模式需指定 top_pct")

        rebalance_dates = config.get_rebalance_dates()
        positions_list = []

        for date in rebalance_dates:
            if date not in factor.index:
                logger.warning(f"{date.strftime('%Y-%m-%d')} 不在因子数据中，跳过")
                continue

            date_str = date.strftime('%Y-%m-%d')

            # 1. 获取该日行业→ETF映射
            mapping = self.selector.select_etfs(date_str)
            if not mapping:
                logger.warning(f"{date_str} 无可用ETF，跳过")
                continue

            # 2. 筛选有映射的行业因子
            factor_values = factor.loc[date]
            valid_industries = [ci for ci in mapping.keys() if ci in factor_values.index]

            if not valid_industries:
                logger.warning(f"{date_str} 无有效行业因子，跳过")
                continue

            valid_scores = factor_values[valid_industries].dropna()

            # 3. 按因子值排序（降序=分数高的排前面）
            ranked = valid_scores.sort_values(ascending=False)

            # 4. 选择前N个
            if mode == 'fixed':
                selected = ranked.head(n_positions)
            else:  # dynamic
                n_select = max(1, int(len(ranked) * top_pct))
                selected = ranked.head(n_select)

            # 5. 等权分配
            weight = 1.0 / len(selected)
            for ci_code in selected.index:
                etf_code = mapping[ci_code]
                positions_list.append({
                    'trade_date': date,
                    'etf_code': etf_code,
                    'weight': weight
                })

            logger.info(
                f"{date_str}: {len(mapping)}个ETF可选 → 选中{len(selected)}个"
            )

        result = pd.DataFrame(positions_list)
        logger.info(f"持仓生成完成，总记录数: {len(result)}")
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    from strategy.etf_loader import ETFLoader
    from synthesis.combiner import combine_from_configs

    factor_configs = {
        '隔夜收益率': ('improved_overnight_return', {}),
        '动量加速度': ('marginal_average_momentum', {}),
        '累积势能': ('potential_energy', {}),
        '成交波动': ('large_order_volatility', {}),
        '成交弹性': ('elasticity', {}),
    }

    # 测试
    test_config = FactorConfig(
        start_date='2016-12-30',
        end_date='2025-11-30',
        frequency='monthly'
    )

    # 1. 计算因子
    print("\n=== 等权合成 ===")
    factor_values = combine_from_configs(test_config, factor_configs)

    # 2. 生成持仓
    loader = ETFLoader()
    selector = ETFSelector(loader, min_scale=5e8)
    strategy = ETFStrategy(selector)

    # # 固定模式
    # positions_fixed = strategy.generate_positions(
    #     factor_values, test_config, mode='fixed', n_positions=4
    # )
    # print("\n固定4个ETF（前3期）:")
    # print(positions_fixed.head(15))

    # 动态模式
    positions_dynamic = strategy.generate_positions(
        factor_values, test_config, mode='dynamic', top_pct=0.2
    )
    print("\n动态前20%（前3期）:")
    print(positions_dynamic.head(15))
    # positions_dynamic.to_parquet(r'./positions_dynamic.parquet')