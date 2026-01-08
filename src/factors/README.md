# 因子说明文档

## 动量类因子

代码：momentum.py

| 因子名称   | 注册名                         | 核心逻辑              | 关键参数        | 预期方向 |
|--------|-----------------------------|-------------------|-------------|------|
| 动量因子   | `momentum`                  | N日涨跌幅             | `window=20` | 正向   |
| 平均动量   | `average_momentum`          | 最新收盘价相对于20日均价的涨跌幅 | `window=20` | 正向   |
| 边际平均动量 | `marginal_average_momentum` | 当前平均动量 - 20日前平均动量 | `window=20` | 正向   |

**说明**：动量效应捕捉趋势延续，边际动量关注动量加速度。

---

## 波动率类因子

代码：amount_volatility.py

| 因子名称      | 注册名                      | 核心逻辑    | 关键参数                     | 预期方向      |
|-----------|--------------------------|---------|--------------------------|-----------|
| 成交额波动率    | `amount_volatility`      | 成交额标准差  | `window=20`              | 负向（低波动更稳） |
| 超大单成交额波动率 | `large_order_volatility` | 大单成交额波动 | `window=20`, `threshold` | 负向        |

---

## 量价弹性因子

代码：elasticity.py

| 因子名称     | 注册名                   | 核心逻辑                              | 关键参数                             | 预期方向 |
|----------|-----------------------|-----------------------------------|----------------------------------|------|
| 日度成交弹性因子 | `day_elasticity`      | 表示单位成交金额产生的价格变化幅度                 | `None`                           | 正向   |
| 成交弹性变化因子 | `elasticity`          | 衡量近期成交弹性相对于早期的变化，反映市场流动性和价格敏感度的变化 | `window=21`                      | 正向   |
| 成交弹性动量因子 | `elasticity_momentum` | 计算短期和长期成交弹性均值的差                   | `short_window=21,long_window=63` | 正向   |

---

## 隔夜收益因子

代码：overnight_factors.py
定义：过去250个交易日的隔夜收益率均值，剔除隔夜收益率标准差较小的（前20%）的样本，并在每个截面进行标准化

| 因子名称    | 注册名                             | 核心逻辑                       | 关键参数                          | 预期方向 |
|---------|---------------------------------|----------------------------|-------------------------------|------|
| 基础隔夜收益  | `overnight_return`              | (开盘价/昨收盘-1)                | `window=250`                  | 正向   |
| 标准化隔夜收益 | `standardized_overnight_return` | 标准化后的隔夜收益率                 | `window=250`                  | 正向   |
| 改进隔夜收益  | `improved_overnight_return`     | 剔除标准差最小的20%样本后，取标准化隔夜收益率均值 | `window=20`, `trim_ratio=0.2` | 正向   |

---

## 累积势能因子

代码： potential_energy.py

| 因子名称   | 注册名                | 核心逻辑                | 关键参数        | 预期方向 |
|--------|--------------------|---------------------|-------------|------|
| 累积势能因子 | `potential_energy` | (开盘价/昨收盘-1),并累积20天的 | `window=20` | 正向   |

---

## 边际贝塔系数

代码：marginal_beta.py

| 因子名称 | 注册名             | 核心逻辑     | 关键参数        | 预期方向       |
|------|-----------------|----------|-------------|------------|
| 边际贝塔 | `marginal_beta` | 月度贝塔变化率  | `window=20` | 负向（贝塔下降更优） |
| 上行贝塔 | `upside_beta`   | 市场上涨时的贝塔 | `window=20` | 负向         |
| 下行贝塔 | `downside_beta` | 市场下跌时的贝塔 | `window=20` | 正向（低下行风险）  |

**计算公式**：

```
滚动计算贝塔系数（OLS回归，零截距）

使用 LinearRegression(fit_intercept=False) 强制截距为零，
假设行业收益无独立漂移项，仅捕捉相对市场的系统性风险暴露变化。

Beta = Cov(R_industry, R_market) / Var(R_market)
边际Beta = Beta(t) - Beta(t-1)
```

---

## 成交额热度因子

代码: amount_heat.py

| 因子名称  | 注册名           | 核心逻辑                             | 关键参数                                 | 预期方向 |
|-------|---------------|----------------------------------|--------------------------------------|------|
| 成交额热度 | `amount_heat` | 近20日成交额均值 / 过去120日成交额均值得到个股热度并聚合 | `short_window=20`, `long_window=120` | 正向   |

---

## 使用示例

```python
from factors.marginal_beta import MarginalBetaFactor
from factors.factor_config import FactorConfig

config = FactorConfig(
    start_date='2016-01-01',
    end_date='2024-12-31',
    frequency='monthly'
)

factor = MarginalBetaFactor(
    window=21,
    preprocess=True,
    standardize_method='zscore',
    winsorize='mad'
)

factor_values = factor(config)
```

## 添加新因子

1. 在 `factors/` 下创建新文件
2. 继承 `BaseFactor` 并实现 `compute()` 方法
3. 使用 `@register_factor("name")` 装饰器
4. 更新本文档

详见 [主README]