#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: main.py
@time: 2025/12/17 10:51
@description:
"""
from factors.base import create_factor
from evaluation.report_generator import generate_factor_report
from factors.factor_config import FactorConfig


if __name__ == "__main__":
    # 配置
    config = FactorConfig(
        start_date='2016-01-01',
        end_date='2025-10-31',
        frequency='monthly'
    )

    # 因子列表（直接写参数）
    factors_to_test = [
        {"factor_name": "marginal_average_momentum", "display_name": "边际平均动量", "window": 21},
        {"factor_name": "average_momentum", "display_name": "平均动量因子", "window": 21},
        {"factor_name": "amount_volatility", "display_name": "成交额波动率", "window": 21},
        {"factor_name": "day_elasticity", "display_name": "日度成交弹性"},
        {"factor_name": "elasticity", "display_name": "成交弹性变化", "window": 21},
        {"factor_name": "elasticity_momentum", "display_name": "成交弹性动量", "short_window": 21, "long_window": 63},
        {"factor_name": "overnight_return", "display_name": "隔夜收益因子", "window": 250},
        {"factor_name": "standardized_overnight_return", "display_name": "标准隔夜收益", "window": 250},
        {"factor_name": "improved_overnight_return", "display_name": "改进隔夜收益", "window": 250, "trim_ratio": 0.2},
        {"factor_name": "potential_energy", "display_name": "累积势能因子", "window": 20},
        {"factor_name": "amount_heat", "display_name": "成交额热度", "short_window": 20, "long_window": 120},
        {"factor_name": "marginal_beta", "display_name": "边际贝塔因子", "window": 20},
        {"factor_name": "upside_beta", "display_name": "边际上行贝塔", "window": 20},
        {"factor_name": "downside_beta", "display_name": "边际下行贝塔", "window": 20},
        # 继续添加...
    ]

    # 循环生成报告
    for factor_info in factors_to_test:
        factor_name = factor_info.pop("factor_name")
        display_name = factor_info.pop("display_name")

        print(f"\n{'=' * 50}")
        print(f"正在测试: {display_name}")
        print(f"{'=' * 50}")

        try:
            # 创建因子
            factor_creator = create_factor(factor_name, preprocess=False, **factor_info)
            factor_value = factor_creator(config)

            # 生成报告
            report_path = generate_factor_report(factor_value, display_name, config)
            print(f"✓ 报告已生成: {report_path}")

        except Exception as e:
            print(f"✗ 因子 {display_name} 测试失败: {e}")
            continue

    print("\n所有因子测试完成！")

    pass

