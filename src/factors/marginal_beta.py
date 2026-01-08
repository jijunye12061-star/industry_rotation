#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: marginal_beta.py
@time: 2025/12/10 14:43
@description:
边际贝塔系数因子

计算行业相对市场的月度贝塔系数变化率
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Optional, Literal, Union

from factors.base import BaseFactor, register_factor
from factors.factor_config import FactorConfig
from data.loader import get_loader

logger = logging.getLogger(__name__)


@register_factor("marginal_beta")
class MarginalBetaFactor(BaseFactor):
    """
    边际贝塔系数因子

    月度贝塔 = Cov(行业收益, 市场收益) / Var(市场收益)
    边际贝塔 = 当期贝塔 - 上期贝塔
    """

    def __init__(self,
                 name: str = "marginal_beta",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 回归窗口期(交易日数)
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法 ('zscore', 'rank', 'minmax')
            winsorize: 去极值的分位数 (lower, upper) (0.025, 0.975)
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算边际贝塔因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=90)).strftime('%Y-%m-%d')  # type: ignore

        # 使用 loader 获取数据
        loader = get_loader()
        market_ret = loader.get_market_returns(extended_start, config.end_date, config.market_code)
        industry_ret = loader.get_industry_returns(extended_start, config.end_date)

        # 计算滚动贝塔（保持不变）
        beta_df = self._calculate_rolling_beta(industry_ret, market_ret)

        # 筛选调仓日并差分（保持不变）
        extended_dates = config.get_extended_rebalance_dates()
        beta_extended = beta_df.reindex(extended_dates)
        marginal_beta = beta_extended.diff(periods=1)

        logger.info(f"边际贝塔因子计算完成，形状: {marginal_beta.shape}")
        return marginal_beta

    def _calculate_rolling_beta(self, industry_ret: pd.DataFrame,
                                market_ret: pd.Series) -> pd.DataFrame:
        """滚动计算贝塔系数（OLS回归，强制截距=0）"""
        from sklearn.linear_model import LinearRegression

        # 对齐日期
        common_dates = industry_ret.index.intersection(market_ret.index)
        industry_ret = industry_ret.loc[common_dates]
        market_ret = market_ret.loc[common_dates]

        beta_df = pd.DataFrame(index=common_dates, columns=industry_ret.columns, dtype=float)

        for industry in industry_ret.columns:
            y = industry_ret[industry].values
            x = market_ret.values

            for i in range(self.window, len(common_dates)):
                y_win = y[i - self.window:i]
                x_win = x[i - self.window:i]

                # 过滤 NaN
                mask = ~(np.isnan(y_win) | np.isnan(x_win))
                if mask.sum() < self.window * 0.8:
                    continue

                y_clean, x_clean = y_win[mask], x_win[mask]

                # OLS回归（fit_intercept=False强制截距为0）
                if len(x_clean) > 0 and x_clean.std() > 0:
                    model = LinearRegression(fit_intercept=False)
                    model.fit(x_clean.reshape(-1, 1), y_clean)
                    beta = model.coef_[0]
                    beta_df.iloc[i, beta_df.columns.get_loc(industry)] = beta

        return beta_df


@register_factor("upside_beta")
class UpsideBetaFactor(BaseFactor):
    """
    边际上行贝塔因子

    上行贝塔 = Cov(行业收益, 市场收益 | 市场收益 > 0) / Var(市场收益 | 市场收益 > 0)
    边际上行贝塔 = 当期上行贝塔 - 上期上行贝塔
    """

    def __init__(self,
                 name: str = "upside_beta",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 回归窗口期(交易日数)
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法 ('zscore', 'rank', 'minmax')
            winsorize: 去极值方法 ('mad' 或 (lower, upper) 如 (0.025, 0.975))
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算边际上行贝塔因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=90)).strftime('%Y-%m-%d')  # type: ignore

        # 获取数据
        loader = get_loader()
        market_ret = loader.get_market_returns(extended_start, config.end_date, config.market_code)
        industry_ret = loader.get_industry_returns(extended_start, config.end_date)

        # 计算滚动上行贝塔
        upside_beta_df = self._calculate_rolling_upside_beta(industry_ret, market_ret)

        # 筛选调仓日并差分
        extended_dates = config.get_extended_rebalance_dates()
        beta_extended = upside_beta_df.reindex(extended_dates)
        marginal_upside_beta = beta_extended.diff(periods=1)

        logger.info(f"边际上行贝塔因子计算完成，形状: {marginal_upside_beta.shape}")
        return marginal_upside_beta

    def _calculate_rolling_upside_beta(self,
                                       industry_ret: pd.DataFrame,
                                       market_ret: pd.Series) -> pd.DataFrame:
        """滚动计算上行贝塔系数（OLS回归，强制截距=0，仅市场上行日）"""
        # 对齐日期
        common_dates = industry_ret.index.intersection(market_ret.index)
        industry_ret = industry_ret.loc[common_dates]
        market_ret = market_ret.loc[common_dates]

        beta_df = pd.DataFrame(index=common_dates, columns=industry_ret.columns, dtype=float)

        for industry in industry_ret.columns:
            y = industry_ret[industry].values
            x = market_ret.values

            for i in range(self.window, len(common_dates)):
                y_win = y[i - self.window:i]
                x_win = x[i - self.window:i]

                # 过滤 NaN
                mask = ~(np.isnan(y_win) | np.isnan(x_win))
                if mask.sum() < self.window * 0.5:
                    continue

                y_clean, x_clean = y_win[mask], x_win[mask]

                # 筛选市场收益为正的日期
                upside_mask = x_clean > 0
                if upside_mask.sum() < 3:
                    continue

                y_upside = y_clean[upside_mask]
                x_upside = x_clean[upside_mask]

                # OLS回归（无截距）：beta = (X'y) / (X'X)
                if len(x_upside) > 0 and x_upside.std() > 0:
                    beta = np.dot(x_upside, y_upside) / np.dot(x_upside, x_upside)
                    beta_df.iloc[i, beta_df.columns.get_loc(industry)] = beta

        return beta_df


@register_factor("downside_beta")
class DownsideBetaFactor(BaseFactor):
    """
    边际下行贝塔因子

    下行贝塔 = Cov(行业收益, 市场收益 | 市场收益 < 0) / Var(市场收益 | 市场收益 < 0)
    边际下行贝塔 = 当期下行贝塔 - 上期下行贝塔
    """

    def __init__(self,
                 name: str = "downside_beta",
                 window: int = 20,
                 preprocess: bool = True,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = 'mad',
                 **params):
        """
        Args:
            window: 回归窗口期(交易日数)
            preprocess: 是否自动预处理因子值
            standardize_method: 标准化方法 ('zscore', 'rank', 'minmax')
            winsorize: 去极值方法 ('mad' 或 (lower, upper) 如 (0.025, 0.975))
        """
        super().__init__(
            name,
            preprocess=preprocess,
            standardize_method=standardize_method,
            winsorize=winsorize,
            window=window,
            **params
        )
        self.window = window

    def compute(self, config: FactorConfig) -> pd.DataFrame:
        """计算边际下行贝塔因子"""
        # 扩展开始日期
        extended_start = (pd.to_datetime(config.start_date) - timedelta(days=90)).strftime('%Y-%m-%d')  # type:ignore

        # 获取数据
        loader = get_loader()
        market_ret = loader.get_market_returns(extended_start, config.end_date, config.market_code)
        industry_ret = loader.get_industry_returns(extended_start, config.end_date)

        # 计算滚动下行贝塔
        downside_beta_df = self._calculate_rolling_downside_beta(industry_ret, market_ret)

        # 筛选调仓日并差分
        extended_dates = config.get_extended_rebalance_dates()
        beta_extended = downside_beta_df.reindex(extended_dates)
        marginal_downside_beta = beta_extended.diff(periods=1)

        logger.info(f"边际下行贝塔因子计算完成，形状: {marginal_downside_beta.shape}")
        return marginal_downside_beta

    def _calculate_rolling_downside_beta(self,
                                         industry_ret: pd.DataFrame,
                                         market_ret: pd.Series) -> pd.DataFrame:
        """滚动计算下行贝塔系数（OLS回归，强制截距=0，仅市场下行日）"""
        # 对齐日期
        common_dates = industry_ret.index.intersection(market_ret.index)
        industry_ret = industry_ret.loc[common_dates]
        market_ret = market_ret.loc[common_dates]

        beta_df = pd.DataFrame(index=common_dates, columns=industry_ret.columns, dtype=float)

        for industry in industry_ret.columns:
            y = industry_ret[industry].values
            x = market_ret.values

            for i in range(self.window, len(common_dates)):
                y_win = y[i - self.window:i]
                x_win = x[i - self.window:i]

                # 过滤 NaN
                mask = ~(np.isnan(y_win) | np.isnan(x_win))
                if mask.sum() < self.window * 0.5:
                    continue

                y_clean, x_clean = y_win[mask], x_win[mask]

                # 筛选市场收益为负的日期
                downside_mask = x_clean < 0
                if downside_mask.sum() < 3:
                    continue

                y_downside = y_clean[downside_mask]
                x_downside = x_clean[downside_mask]

                # OLS回归（无截距）：beta = (X'y) / (X'X)
                if len(x_downside) > 0 and x_downside.std() > 0:
                    beta = np.dot(x_downside, y_downside) / np.dot(x_downside, x_downside)
                    beta_df.iloc[i, beta_df.columns.get_loc(industry)] = beta

        return beta_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 创建配置
    main_config = FactorConfig(
        start_date='2023-12-29',
        end_date='2025-03-31',
        frequency='monthly'
    )

    # # 计算因子
    # factor = MarginalBetaFactor(window=21)
    # marginal_beta_df = factor(main_config)
    #
    # print("\n边际贝塔因子（最近5期）：")
    # print(marginal_beta_df.tail())

    # 计算因子
    factor = UpsideBetaFactor(window=21)
    upside_beta_df = factor(main_config)

    print("\n边际贝塔因子（最近5期）：")
    print(upside_beta_df.tail())