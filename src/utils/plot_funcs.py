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
from typing import Optional, Dict, Any
import os

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


def plot_ic_analysis(ic_series: pd.Series, title: str = "IC分析"):
    """
    绘制IC分析图：IC柱状图 + 累计IC曲线

    Args:
        ic_series: IC时间序列 (index=日期, values=IC值)
        title: 图表标题
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
    plt.show()


def plot_boxplot(df, category_col, value_col, title=None, figsize=(10, 6),
                 xlabel=None, ylabel=None, save_path=None,
                 color_palette=None, show_points=True, hue=None):
    """
    创建分类箱型图

    参数：
    df: DataFrame, 包含数据的DataFrame
    category_col: str, 类别列的名称
    value_col: str, 值列的名称
    title: str, 可选，图表标题
    figsize: tuple, 可选，图表大小
    xlabel: str, 可选，x轴标签
    ylabel: str, 可选，y轴标签
    save_path: str, 可选，保存图片的路径
    color_palette: list, 可选，自定义颜色列表
    show_points: bool, 可选，是否显示离群点
    """
    # 创建图表
    plt.figure(figsize=figsize)

    # 绘制箱型图
    ax = sns.boxplot(x=category_col,
                     y=value_col,
                     data=df,
                     palette=color_palette,
                     showfliers=show_points,
                     hue=hue)  # showfliers控制是否显示离群点

    # 添加散点图来显示实际数据点分布
    sns.swarmplot(x=category_col,
                  y=value_col,
                  data=df,
                  color='0.25',  # 深灰色
                  alpha=0.5,  # 设置透明度
                  size=4)  # 点的大小

    # 设置标题和标签
    if title:
        plt.title(title, pad=20, fontsize=12)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # 优化布局
    plt.tight_layout()

    # 添加一些基本统计信息
    stats = df.groupby(category_col)[value_col].describe()
    print("\n基本统计信息：")
    print(stats)

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def create_scatter_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        **kwargs: Dict[str, Any]
) -> Optional[plt.Figure]:
    """
    创建高度可定制的散点图函数

    必需参数:
        df: pandas DataFrame，包含要绘制的数据
        x_col: str，x轴列名
        y_col: str，y轴列名

    可选参数（通过kwargs传入）:
        figsize: tuple, 图表大小，默认 (10, 6)
        title: str, 图表标题
        xlabel: str, x轴标签，默认使用x_col
        ylabel: str, y轴标签，默认使用y_col
        xlim: tuple, x轴范围，如 (0, 100)
        ylim: tuple, y轴范围
        save_path: str, 图表保存路径，如 'plots/scatter.png'
        dpi: int, 保存图片的DPI，默认300
        color: str, 点的颜色，如 'blue', '#FF5733'
        size: float, 点的大小，默认30
        alpha: float, 点的透明度，默认0.6
        grid: bool, 是否显示网格，默认True
        style: str, 图表样式，如'default', 'whitegrid'等
        add_reg_line: bool, 是否添加回归线，默认False
        reg_line_color: str, 回归线颜色，默认'red'
        font_size: dict, 字体大小设置，如{'title': 14, 'label': 12, 'tick': 10}

    返回:
        matplotlib.figure.Figure 对象，如果save_path不为None则返回None
    """
    # 默认参数设置
    default_kwargs = {
        'figsize': (10, 6),
        'title': None,
        'xlabel': x_col,
        'ylabel': y_col,
        'xlim': None,
        'ylim': None,
        'save_path': None,
        'dpi': 300,
        'color': '#1f77b4',
        'size': 30,
        'alpha': 0.6,
        'grid': True,
        'style': 'whitegrid',
        'add_reg_line': False,
        'reg_line_color': 'red',
        'font_size': {'title': 14, 'label': 12, 'tick': 10}
    }

    # 更新默认参数
    plot_params = {**default_kwargs, **kwargs}

    # 设置样式
    # sns.set_style(plot_params['style'])

    # 创建图表
    fig, ax = plt.subplots(figsize=plot_params['figsize'])

    # 绘制散点图
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        color=plot_params['color'],
        s=plot_params['size'],
        alpha=plot_params['alpha'],
        ax=ax
    )

    # 添加回归线
    if plot_params['add_reg_line']:
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            scatter=False,
            color=plot_params['reg_line_color'],
            ax=ax
        )

    # 设置标题和标签
    if plot_params['title']:
        ax.set_title(plot_params['title'], fontsize=plot_params['font_size']['title'])
    ax.set_xlabel(plot_params['xlabel'], fontsize=plot_params['font_size']['label'])
    ax.set_ylabel(plot_params['ylabel'], fontsize=plot_params['font_size']['label'])

    # 设置刻度字体大小
    ax.tick_params(labelsize=plot_params['font_size']['tick'])

    # 设置坐标轴范围
    if plot_params['xlim']:
        ax.set_xlim(plot_params['xlim'])
    if plot_params['ylim']:
        ax.set_ylim(plot_params['ylim'])

    # 设置网格
    ax.grid(plot_params['grid'], linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if plot_params['save_path']:
        # 确保保存路径的目录存在
        save_dir = os.path.dirname(plot_params['save_path'])
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(plot_params['save_path'], dpi=plot_params['dpi'], bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_bar(df, label_col, value_col, **kwargs):
    """
    Create a bar chart from DataFrame columns.

    Parameters:
        df: DataFrame with data
        label_col: Column for x-axis labels
        value_col: Column for y-axis values
        **kwargs: Optional (title, xlabel, ylabel, figsize, color,
                 save_path, rotation, grid, fontsize, show_values)

    Returns:
        matplotlib figure object
    """
    # Extract parameters with defaults
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', label_col)
    ylabel = kwargs.get('ylabel', value_col)
    figsize = kwargs.get('figsize', (10, 6))
    color = kwargs.get('color', 'royalblue')
    save_path = kwargs.get('save_path', None)
    rotation = kwargs.get('rotation', 0)
    grid = kwargs.get('grid', False)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[label_col], df[value_col], color=color)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotation)

    # Add grid if requested
    if grid:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels if requested
    if kwargs.get('show_values', False):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=kwargs.get('dpi', 300))

    plt.show()


if __name__ == "__main__":
    # 创建示例数据
    data = {
        'category': ['A'] * 50 + ['B'] * 50,
        'value': list(np.random.normal(100, 15, 50)) + list(np.random.normal(80, 20, 50))
    }
    df = pd.DataFrame(data)

    # 基本用法
    plot_boxplot(df, 'category', 'value', title='分类箱型图示例')

    # 自定义颜色和大小的用法
    plot_boxplot(df,
                 'category',
                 'value',
                 title='自定义箱型图',
                 figsize=(8, 5),
                 xlabel='类别',
                 ylabel='数值',
                 color_palette=['lightblue', 'lightgreen'],
                 save_path='boxplot.png')

    plt.show()


# 额外的分析功能
def analyze_boxplot_data(df, category_col, value_col):
    """
    分析箱型图数据，提供详细的统计信息
    """
    # 计算每个类别的详细统计信息
    stats = df.groupby(category_col)[value_col].agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
    ]).round(2)

    stats.columns = ['数量', '平均值', '中位数', '标准差', '最小值', '最大值', 'Q1', 'Q3']

    # 计算类别间的统计检验
    from scipy import stats as spstats
    categories = df[category_col].unique()
    if len(categories) == 2:
        # 执行t检验
        cat1_data = df[df[category_col] == categories[0]][value_col]
        cat2_data = df[df[category_col] == categories[1]][value_col]
        t_stat, p_value = spstats.ttest_ind(cat1_data, cat2_data)

        print(f"\n两组数据t检验结果:")
        print(f"t统计量: {t_stat:.4f}")
        print(f"p值: {p_value:.4f}")

        # 执行Mann-Whitney U检验
        u_stat, p_value = spstats.mannwhitneyu(cat1_data, cat2_data)
        print(f"\nMann-Whitney U检验结果:")
        print(f"U统计量: {u_stat:.4f}")
        print(f"p值: {p_value:.4f}")

    return stats
