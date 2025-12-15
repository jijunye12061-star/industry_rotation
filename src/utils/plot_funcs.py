"""
保存会用到的画图工具
coding:utf-8
@Time:2025/04/08 15:55
@Author: 季俊晔
以下为可用函数：
-plot_net_value_curve：传入净值字典，画净值曲线图
-plot_boxplot：画分类箱型图
-create_scatter_plot：画散点图
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import pandas as pd

# 先设置 seaborn 样式
sns.set_style("whitegrid")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）字体来显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_net_value_curve(series_dict, title="净值曲线图", xlabel="日期", ylabel="净值", **kwargs):
    """
    绘制净值曲线图的函数，支持传入多个净值series的字典，并设置美化效果。

    :param series_dict: dict, 包含多个series对象的字典，键为曲线名称，值为pandas.Series对象，索引为Timestamp
    :param title: str, 图表的标题
    :param xlabel: str, x轴的标签
    :param ylabel: str, y轴的标签
    :param kwargs: 其他可选参数用于美化图表，例如color, linewidth, linestyle等
    """
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 循环遍历字典中的每个series，并绘制
    for label, series in series_dict.items():
        ax.plot(pd.to_datetime(series.index), series.values, label=label)

    # 设置图表标题和标签
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # 图例设置
    ax.legend(loc='best', fontsize=10)

    # 设置x轴为日期格式
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动设置日期刻度
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期显示格式

    # 自动旋转日期标签以避免重叠
    fig.autofmt_xdate()

    # 设置y轴网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 美化外观
    plt.tight_layout()

    save_path = kwargs.get('save_path', None)
    if save_path:
        plt.savefig(save_path, dpi=200)

    # 显示图像
    plt.show()


def plot_ic_analysis(ic_series: pd.Series, title: str = "IC分析", save_path: str = None):
    """
    绘制IC分析图：IC柱状图 + 累计IC曲线

    Args:
        ic_series: IC时间序列 (index=日期, values=IC值)
        title: 图表标题
        save_path: 保存路径（可选）
    """
    # 计算累计IC
    # 去除NaN值
    ic_clean = ic_series.dropna()
    cumulative_ic = ic_clean.cumsum()

    # 创建图形和主轴
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # 绘制IC柱状图（浅粉色）
    ax1.bar(ic_clean.index, ic_clean.values,
            color='#E8B4B8', alpha=0.8, width=15, label='RankIC')
    ax1.set_ylabel('IC', fontsize=11, color='#666666')
    ax1.set_ylim(-1, 1)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.tick_params(axis='y', labelcolor='#666666')

    # 创建次坐标轴绘制累计IC（深红色）
    ax2 = ax1.twinx()
    ax2.plot(cumulative_ic.index, cumulative_ic.values,
             color='#8B1A1A', linewidth=2.5, label='累计IC(右轴)')
    ax2.set_ylabel('累计IC', fontsize=11, color='#666666')
    ax2.tick_params(axis='y', labelcolor='#666666')

    # 设置x轴日期格式（紧凑显示）
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, fontsize=8)

    # 合并图例（底部居中）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=2, frameon=False, fontsize=10)

    plt.title(title, fontsize=13, pad=15, color='#333333')
    plt.tight_layout()

    # 添加保存逻辑
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 创建示例数据
    data = {
        'category': ['A'] * 50 + ['B'] * 50,
        'value': list(np.random.normal(100, 15, 50)) + list(np.random.normal(80, 20, 50))
    }
    df = pd.DataFrame(data)


