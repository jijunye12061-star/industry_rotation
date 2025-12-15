#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: report_generator.py
@time: 2025/12/15 15:42
@description:
"""
# evaluation/report_generator.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from factors.factor_config import FactorConfig
from evaluation.ic_analysis import ICAnalyzer
from evaluation.group_analysis import GroupAnalyzer

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示


class FactorReportGenerator:
    """因子分析报告生成器"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # 获取项目根目录（evaluation 上两级）
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # evaluation/ -> src/ -> root/
            output_dir = project_root / "reports"
        else:
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self,
                        factor: pd.DataFrame,
                        factor_name: str,
                        config: FactorConfig) -> str:
        """
        生成完整因子报告

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{factor_name}_{timestamp}"
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)

        # 1. IC分析
        ic_analyzer = ICAnalyzer(method='spearman')
        ic_results = ic_analyzer.analyze_with_details(factor, config)

        # 2. 分组测试
        group_analyzer = GroupAnalyzer(long_size=6, short_size=6)
        group_results = group_analyzer.analyze(factor, config)

        # 3. 保存数据（JSON）
        metrics = {
            'ic_metrics': ic_results['metrics'],
            'group_metrics': group_results['metrics'].to_dict()
        }
        with open(report_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # 4. 保存时间序列（CSV）
        ic_results['ic_series'].to_csv(report_dir / 'ic_series.csv')
        group_results['nav'].to_csv(report_dir / 'nav_curve.csv')

        # 5. 生成图表
        self._plot_ic_analysis(ic_results, report_dir)
        self._plot_group_performance(group_results, report_dir, config)
        self._plot_factor_distribution(factor, report_dir)

        # 6. 生成HTML报告
        html_path = self._generate_html(
            factor_name, metrics, report_dir, config
        )

        print(f"报告已生成: {html_path}")
        return str(html_path)

    @staticmethod
    def _plot_ic_analysis(ic_results: dict, save_dir: Path):
        """IC分析图表（交互式）"""
        ic_series = ic_results['ic_series'].dropna()
        cumulative_ic = ic_series.cumsum()

        # 创建双轴图
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # IC柱状图
        fig.add_trace(
            go.Bar(
                x=ic_series.index,
                y=ic_series.values,
                name='IC',
                marker=dict(color='#E8B4B8'),
                hovertemplate='%{x|%Y-%m-%d}<br>IC: %{y:.2%}<extra></extra>'
            ),
            secondary_y=False
        )

        # 累计IC曲线
        fig.add_trace(
            go.Scatter(
                x=cumulative_ic.index,
                y=cumulative_ic.values,
                name='累计IC',
                line=dict(color='#8B1A1A', width=2.5),
                hovertemplate='%{x|%Y-%m-%d}<br>累计IC: %{y:.2%}<extra></extra>'
            ),
            secondary_y=True
        )

        # 布局
        fig.update_layout(
            title='IC净值分析',
            hovermode='x unified',
            height=600,
            showlegend=True
        )
        fig.update_yaxes(title_text="IC", secondary_y=False)
        fig.update_yaxes(title_text="累计IC(右轴)", secondary_y=True)

        # 保存为HTML
        fig.write_html(str(save_dir / 'ic_analysis.html'))

    @staticmethod
    def _plot_group_performance(group_results: dict, save_dir: Path, config: FactorConfig):
        """分组测试净值曲线（交互式）"""
        nav = group_results['nav']
        rebalance_dates = config.get_rebalance_dates()

        # 筛选调仓日数据
        nav_rebalance = nav.loc[nav.index.isin(rebalance_dates)]

        # 创建图表
        fig = go.Figure()

        # 添加四条曲线
        lines = ['多头', '基准', '空头', '多头/基准']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for line, color in zip(lines, colors):
            fig.add_trace(go.Scatter(
                x=nav_rebalance.index,
                y=nav_rebalance[line],
                name=line,
                mode='lines',  # 去掉 +markers
                line=dict(color=color, width=3),  # 增加宽度到3
                hovertemplate='%{x|%Y-%m-%d}<br>' + line + ': %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title='分组测试的净值表现',
            xaxis_title='交易日期',
            yaxis_title='净值',
            hovermode='x unified',
            height=600,
            showlegend=True
        )

        fig.write_html(str(save_dir / 'group_performance.html'))

    @staticmethod
    def _plot_factor_distribution(factor: pd.DataFrame, save_dir: Path):
        """因子分布图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 修改这里：使用 ax.hist() 而不是数组的 .hist()
        factor_values = factor.values.flatten()
        factor_values = factor_values[~pd.isna(factor_values)]

        ax.hist(factor_values, bins=50, edgecolor='black', alpha=0.7)

        ax.set_title('Factor Distribution (All Periods)', fontsize=14)
        ax.set_xlabel('Factor Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'factor_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _generate_html(factor_name: str, metrics: dict,
                       report_dir: Path, config: FactorConfig) -> Path:
        """生成HTML报告"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{factor_name} - 因子分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; margin: 20px 0; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; 
                      background: #f9f9f9; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{factor_name} 因子分析报告</h1>
    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>回测区间: {config.start_date} 至 {config.end_date}</p>
    <p>调仓频率: {config.frequency}</p>

    <h2>一、IC分析</h2>
    <div>
        <div class="metric-box">
            <strong>IC均值:</strong> {metrics['ic_metrics']['IC_mean']:.2%}
        </div>
        <div class="metric-box">
            <strong>ICIR:</strong> {metrics['ic_metrics']['ICIR']:.2f}
        </div>
        <div class="metric-box">
            <strong>IC胜率:</strong> {metrics['ic_metrics']['IC_win_rate']:.2%}
        </div>
        <div class="metric-box">
            <strong>T值:</strong> {metrics['ic_metrics']['t_value']:.2f}
        </div>
        <div class="metric-box">
            <strong>P值:</strong> {metrics['ic_metrics']['p_value']:.4f}
        </div>
    </div>
    <iframe src="ic_analysis.html" width="100%" height="650" frameborder="0"></iframe>

    <h2>二、分组测试</h2>
    {pd.DataFrame(metrics['group_metrics']).round(2).to_html()}
    <iframe src="group_performance.html" width="100%" height="650" frameborder="0"></iframe>

    <h2>三、因子分布</h2>
    <img src="factor_distribution.png" alt="因子分布">

</body>
</html>
        """

        html_path = report_dir / 'report.html'
        html_path.write_text(html_template, encoding='utf-8')
        return html_path


# 便捷函数
def generate_factor_report(factor: pd.DataFrame,
                           factor_name: str,
                           config: FactorConfig,
                           output_dir: str = None) -> str:
    """快速生成因子报告"""
    generator = FactorReportGenerator(output_dir)
    return generator.generate_report(factor, factor_name, config)


if __name__ == '__main__':
    # 使用真实因子测试
    from factors.marginal_beta import UpsideBetaFactor

    main_config = FactorConfig(
        start_date='2016-01-01',
        end_date='2025-03-31',
        frequency='monthly'
    )

    factor_calculator = UpsideBetaFactor(window=21, winsorize=None)
    upside_beta = factor_calculator(main_config)

    # 一键生成报告
    report_path = generate_factor_report(upside_beta, "边际上行贝塔", main_config)
