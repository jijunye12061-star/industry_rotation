# 行业轮动量化策略框架

基于因子的行业轮动策略研发框架，支持因子快速开发、回测验证和合成。

## 核心功能

- **数据静态化**：将数据获取与数据应用分离，将数据保存至本地
- **因子开发**：模块化因子类，支持快速添加新因子
- **单因子检验**：IC分析、分组回测、自动报告生成
- **因子合成**：等权、IC加权等多种合成方法
- **策略回测**：完整的回测引擎与绩效分析

## 项目结构

```
src/
├── config.py              # 全局配置（行业代码、数据库连接）
├── data/
│   └── base_loader.py     # 统一数据接口（交易日历、价格、收益率）
│   └── loader.py          # 具体数据接口，含数据库和本地数据实现
│   └── db_manager.py      # 本地数据库实现
├── factors/
│   ├── base.py            # 因子基类 + 注册机制
│   ├── factor_config.py   # 因子配置类
│   └── *.py               # 具体因子实现
├── evaluation/
│   ├── ic_analysis.py         # IC/ICIR分析
│   ├── group_analysis.py      # 分组回测
│   └── report_generator.py    # HTML报告生成
├── synthesis/
│   └── combiner.py        # 因子合成
└── utils/
│   ├── query_data_from_choice.py    # 数据库查询
│   └── plot_funcs.py                # 可视化工具
```

## 快速开始
### 0. 配置
主要是要配置一些参数和api接口的账密。

- `config.py`中是指数代码的详情以及数据库加载方式
- `.env`中需配置choice数据的账密
```
CHOICE_USERNAME=此处填写你的账号（手机号）
CHOICE_PASSWORD=此处填写你的密码
```

### 1. 计算单因子

```python
from factors.marginal_beta import MarginalBetaFactor
from factors.factor_config import FactorConfig

# 配置回测参数
config = FactorConfig(
    start_date='2016-01-01',
    end_date='2024-12-31',
    frequency='monthly'  # 'daily', 'weekly', 'biweekly', 'monthly'
)

# 计算因子
factor = MarginalBetaFactor(
    window=21,                 # 边际因子使用的交易日数据长度
    preprocess=True,           # 自动预处理，做单因子测试可以不开启，但因子合成时必须标准化
    standardize_method='zscore',  # 'zscore', 'rank', 'minmax'
    winsorize='mad'            # 'mad' 或 (0.025, 0.975)
)
factor_values = factor(config)
```

### 2. 生成分析报告

```python
from evaluation.report_generator import generate_factor_report
from factors.base import create_factor

# 配置回测参数
config = ...

factor_calculator = create_factor("upside_beta", window=21, winsorize=None)
factor_values = factor_calculator(config)

report_path = generate_factor_report(
    factor=factor_values,
    factor_name="边际贝塔",
    config=config
)
# 输出: reports/边际贝塔_20241216_153045/report.html
```

报告包含：
- IC分析（IC均值、ICIR、胜率、显著性检验）
- 分组回测（多头/空头/基准净值曲线）
- 因子分布统计
- 交互式图表（支持鼠标悬停查看详情）

### 3. 单独运行IC分析

```python
from evaluation.ic_analysis import ICAnalyzer
from factors.base import create_factor

# 配置回测参数
config = ...
factor_calculator = create_factor("upside_beta", window=21, winsorize=None)
factor_values = factor_calculator(config)

analyzer = ICAnalyzer(method='spearman')
results = analyzer.analyze(factor_values, config, forward_periods=1)

print(f"IC均值: {results['IC_mean']:.4f}")
print(f"ICIR: {results['ICIR']:.4f}")
print(f"IC胜率: {results['IC_win_rate']:.2%}")
```

### 4. 分组回测

支持仅计算多头组和空头组的净值`LongShortAnalyzer`类与多分组回测`MultiGroupAnalyzer`两种方式

```python
from evaluation.group_analysis import LongShortAnalyzer
from factors.base import create_factor

# 配置回测参数
config = ...
factor_calculator = create_factor("upside_beta", window=21, winsorize=None)
factor_values = factor_calculator(config)

analyzer = LongShortAnalyzer(long_size=6, short_size=6)  # 表示多头组选择几个行业，空头组选择几个行业
results = analyzer.analyze(factor_values, config)

print("绩效指标:")
print(results['metrics'])  # 年化收益、夏普、最大回撤等

# 净值曲线
nav_curve = results['nav']  # 包含多头/空头/基准/多空组合
```

## 数据格式约定

**统一DataFrame格式**：
```
DataFrame:
    index   = DatetimeIndex  # 调仓日期
    columns = 行业代码        # 'CI005001', 'CI005002', ...
    values  = 因子值/收益率
```

**因子方向**：值越大，预期收益越高

## 添加新因子

### 步骤1：创建因子类

```python
# factors/my_factor.py
from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
import pandas as pd

@register_factor("my_factor")
class MyFactor(BaseFactor):
    def __init__(self, name="my_factor", window=20, **kwargs):
        super().__init__(name, **kwargs)
        self.window = window
    
    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算因子值（只使用T及之前数据）"""
        from data.loader import get_loader
        loader = get_loader()
        
        # 获取数据
        prices = loader.get_industry_prices(
            config.start_date, 
            config.end_date
        )
        
        # 计算因子
        factor = prices.pct_change(self.window)
        
        # 筛选调仓日
        rebalance_dates = config.get_rebalance_dates()
        return factor.reindex(rebalance_dates)
```

### 步骤2：使用新因子

```python
from factors.my_factor import MyFactor

factor = MyFactor(window=20, preprocess=True)
factor_values = factor(config)
```

## 因子预处理

自动预处理包括（按顺序）：
1. **去极值**：MAD方法（小样本推荐）或分位数截断
2. **标准化**：横截面Z-score/Rank/MinMax

```python
factor = MyFactor(
    preprocess=True,              # 启用预处理
    winsorize='mad',              # MAD去极值（或 (0.025, 0.975)）
    standardize_method='zscore'   # Z-score标准化
)
```

也可手动处理：
```python
from factors.base import BaseFactor

factor_processed = BaseFactor.preprocess_factor(
    factor=raw_factor,
    winsorize='mad',
    standardize_method='zscore'
)
```

## 数据接口

所有数据获取通过 `DataLoader` 统一管理：

```python
from data.loader import get_loader

loader = get_loader()

# 交易日历
dates = loader.get_trading_dates('2024-01-01', '2024-12-31')

# 价格数据
prices = loader.get_industry_prices('2024-01-01', '2024-12-31')
market_prices = loader.get_market_prices('2024-01-01', '2024-12-31')

# 收益率数据
returns = loader.get_industry_returns('2024-01-01', '2024-12-31')
market_returns = loader.get_market_returns('2024-01-01', '2024-12-31')

# 未来收益（用于IC分析）
forward_returns = loader.get_forward_returns(
    rebalance_dates=dates,
    start_date='2024-01-01',
    end_date='2024-12-31',
    forward_periods=1
)
```

## 技术特性

- **解耦设计**：因子计算、评估、回测模块独立，可单独使用
- **工厂模式**：装饰器注册因子，支持动态创建
- **统一接口**：所有因子输出相同格式，便于合成和比较
- **可扩展性**：添加新因子无需修改框架代码
- **数据抽象**：隔离数据源实现，支持数据库/API切换

## 注意事项

1. **无未来信息**：因子计算仅使用T及之前数据
2. **缺失值处理**：框架保留NaN，便于下游灵活处理
3. **横截面操作**：标准化/去极值在横截面维度进行
4. **调仓执行**：T日因子值决定T+1开盘仓位

## 依赖环境

```
pandas >= 1.5.0
numpy >= 1.23.0
scipy >= 1.9.0
plotly >= 5.0.0
matplotlib >= 3.5.0
```

## License

MIT
