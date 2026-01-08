#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: base.py
@time: 2025/12/10 11:19
@description:
因子基类与注册机制

提供统一的因子接口和工厂模式注册机制，方便扩展新因子。
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Callable, Literal, Union
from factors.factor_config import FactorConfig
from factors.cache_manager import get_cache
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """
    因子基类

    所有因子需继承此类并实现 compute() 方法。

    Attributes:
        name: 因子名称
        params: 因子参数字典
        preprocess: 是否自动预处理因子值
        standardize_method: 标准化方法 ('zscore', 'rank', 'minmax', None)
        winsorize: 是否去极值 (bool 或 tuple[float, float])
    """

    def __init__(self,
                 name: str,
                 preprocess: bool = False,
                 standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = None,
                 winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = None,  # ← 修改
                 **params):
        """
        初始化因子

        Args:
            name: 因子名称
            preprocess: 是否自动预处理（默认False，保持兼容性）
            standardize_method: 标准化方法
                - 'zscore': 横截面Z-score标准化
                - 'rank': 横截面排序标准化
                - 'minmax': 横截面Min-Max标准化
                - None: 不标准化
            winsorize: 去极值的分位数 (lower, upper)，如 (0.025, 0.975)
            **params: 其他因子参数
        """
        self.name = name
        self.params = params
        self.preprocess = preprocess
        self.standardize_method = standardize_method
        self.winsorize = winsorize
        logger.info(f"初始化因子: {name}, 参数: {params}, 预处理: {preprocess}")

    @abstractmethod
    def compute(self, config: 'FactorConfig') -> pd.DataFrame:
        """
        计算因子值（抽象方法，子类必须实现）

        Args:
            config: 因子配置对象（包含日期范围、频率等）

        Returns:
            因子值 DataFrame
                - index: DatetimeIndex（调仓日）
                - columns: 行业代码
                - values: 因子值

        Notes:
            - 确保只使用 T 及之前的数据（无未来信息）
            - 因子方向：值越大，预期收益越高
            - 处理好 NaN 值
        """
        pass

    def __call__(self, config: 'FactorConfig', use_cache: bool = True) -> pd.DataFrame:
        """使因子对象可调用（支持缓存）"""
        cache = get_cache()

        # 尝试从缓存读取
        if use_cache:
            cached_data = cache.get(self.name, config, self.params)
            if cached_data is not None:
                logger.info(f"使用缓存的因子: {self.name}")

                # 如果需要预处理且缓存的是原始值
                if self.preprocess:
                    cached_data = self.preprocess_factor(
                        cached_data,
                        standardize_method=self.standardize_method,
                        winsorize=self.winsorize
                    )
                return cached_data

        # 计算因子
        logger.info(f"开始计算因子: {self.name}")
        factor_values = self.compute(config)
        logger.info(f"因子 {self.name} 计算完成，形状: {factor_values.shape}")

        # 写入缓存（缓存原始值，不缓存预处理后的）
        if use_cache:
            cache.set(self.name, config, self.params, factor_values)

        # 预处理
        if self.preprocess:
            factor_values = self.preprocess_factor(
                factor_values,
                standardize_method=self.standardize_method,
                winsorize=self.winsorize
            )
            logger.info(f"因子 {self.name} 预处理完成")

        return factor_values

    # ==================== 预处理方法 ====================

    @staticmethod
    def preprocess_factor(
            factor: pd.DataFrame,
            standardize_method: Optional[Literal['zscore', 'rank', 'minmax']] = 'zscore',
            winsorize: Optional[Union[tuple[float, float], Literal['mad']]] = None,  # ← 默认None
            fill_na: bool = False
    ) -> pd.DataFrame:
        """
        因子预处理（横截面处理）

        Args:
            factor: 因子值 DataFrame (dates × industries)
            standardize_method: 标准化方法
            winsorize: 去极值的分位数 (lower, upper)
            fill_na: 是否用截面中位数填充NaN

        Returns:
            预处理后的因子值

        Processing Order:
            1. 去极值 (Winsorize)
            2. 填充缺失值（可选）
            3. 标准化
        """
        result = factor.copy()

        # 1. 去极值（横截面）
        if winsorize is not None:
            result = BaseFactor._winsorize_cross_section(result, winsorize)

        # 2. 填充缺失值（横截面中位数）
        if fill_na:
            result = result.T.fillna(result.median(axis=1)).T

        # 3. 标准化（横截面）
        if standardize_method == 'zscore':
            result = BaseFactor._standardize_zscore(result)
        elif standardize_method == 'rank':
            result = BaseFactor._standardize_rank(result)
        elif standardize_method == 'minmax':
            result = BaseFactor._standardize_minmax(result)
        elif standardize_method is None:
            pass
        else:
            raise ValueError(f"不支持的标准化方法: {standardize_method}")

        return result

    @staticmethod
    def _winsorize_cross_section(
            factor: pd.DataFrame,
            method: Union[tuple[float, float], Literal['mad']] = 'mad'
    ) -> pd.DataFrame:
        """
        横截面去极值

        Args:
            factor: 因子值 DataFrame
            method:
                - 'mad': MAD方法（推荐用于小样本）
                - (lower, upper): 分位数方法，如(0.01, 0.99)
        """
        if method == 'mad':
            # MAD方法：更适合小样本
            def winsorize_mad(row: pd.Series, n_sigma: float = 3.0) -> pd.Series:
                if row.isna().all():
                    return row
                median = row.median()
                mad = (row - median).abs().median()
                if mad == 0:
                    return row
                lower = median - n_sigma * mad * 1.4826  # 1.4826是正态分布下的调整系数
                upper = median + n_sigma * mad * 1.4826
                return row.clip(lower=lower, upper=upper)

            return factor.apply(winsorize_mad, axis=1)

        else:
            # 分位数方法
            lower_q, upper_q = method

            def winsorize_quantile(row: pd.Series) -> pd.Series:
                if row.isna().all():
                    return row
                lower = row.quantile(lower_q)
                upper = row.quantile(upper_q)
                return row.clip(lower=lower, upper=upper)

            return factor.apply(winsorize_quantile, axis=1)

    @staticmethod
    def _standardize_zscore(factor: pd.DataFrame) -> pd.DataFrame:
        """
        横截面Z-score标准化

        Formula: (X - mean) / std

        Args:
            factor: 因子值 DataFrame

        Returns:
            标准化后的因子值
        """

        def zscore_row(row: pd.Series) -> pd.Series:
            """对单行（单个截面）标准化"""
            if row.isna().all() or row.std() == 0:
                return row
            return (row - row.mean()) / row.std()

        return factor.apply(zscore_row, axis=1)

    @staticmethod
    def _standardize_rank(factor: pd.DataFrame) -> pd.DataFrame:
        """
        横截面排序标准化

        将因子值转换为排名百分位 [0, 1]

        Args:
            factor: 因子值 DataFrame

        Returns:
            标准化后的因子值
        """

        def rank_row(row: pd.Series) -> pd.Series:
            """对单行（单个截面）排序标准化"""
            if row.isna().all():
                return row
            # rank(method='average') 处理并列值
            # pct=True 转换为百分位 [0, 1]
            return row.rank(pct=True)

        return factor.apply(rank_row, axis=1)

    @staticmethod
    def _standardize_minmax(factor: pd.DataFrame) -> pd.DataFrame:
        """
        横截面Min-Max标准化

        Formula: (X - min) / (max - min)

        Args:
            factor: 因子值 DataFrame

        Returns:
            标准化后的因子值 [0, 1]
        """

        def minmax_row(row: pd.Series) -> pd.Series:
            """对单行（单个截面）Min-Max标准化"""
            if row.isna().all():
                return row
            min_val = row.min()
            max_val = row.max()
            if max_val == min_val:
                return row
            return (row - min_val) / (max_val - min_val)

        return factor.apply(minmax_row, axis=1)

    # ==================== 其他工具方法 ====================

    @staticmethod
    def neutralize(
            factor: pd.DataFrame,
            industry_weights: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        行业中性化（去除行业平均效应）

        Args:
            factor: 因子值 DataFrame
            industry_weights: 行业权重 DataFrame（可选，默认等权）

        Returns:
            中性化后的因子值
        """
        if industry_weights is None:
            # 等权去均值
            return factor.sub(factor.mean(axis=1), axis=0)
        else:
            # 加权去均值
            weighted_mean = (factor * industry_weights).sum(axis=1) / industry_weights.sum(axis=1)
            return factor.sub(weighted_mean, axis=0)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"params={self.params}, preprocess={self.preprocess})")


# ==================== FactorRegistry 保持不变 ====================

class FactorRegistry:
    """
    因子注册器（单例模式）

    使用工厂模式管理所有因子类，支持动态注册和创建。
    """

    _instance: Optional['FactorRegistry'] = None
    _registry: Dict[str, Type[BaseFactor]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        因子注册装饰器

        Args:
            name: 因子注册名称

        Returns:
            装饰器函数

        Example:
            @register_factor("momentum_20")
            class MomentumFactor(BaseFactor):
                def compute(self, data):
                    return data.pct_change(20)
        """

        def decorator(factor_class: Type[BaseFactor]) -> Type[BaseFactor]:
            if not issubclass(factor_class, BaseFactor):
                raise TypeError(f"{factor_class.__name__} 必须继承 BaseFactor")

            if name in cls._registry:
                logger.warning(f"因子 '{name}' 已存在，将被覆盖")

            cls._registry[name] = factor_class
            logger.info(f"注册因子: {name} -> {factor_class.__name__}")
            return factor_class

        return decorator

    @classmethod
    def create(cls, name: str, **params) -> BaseFactor:
        """
        根据注册名称创建因子实例

        Args:
            name: 因子注册名称
            **params: 因子参数

        Returns:
            因子实例

        Raises:
            KeyError: 因子未注册

        Example:
            factor = FactorRegistry.create("momentum_20", window=20)
        """
        if name not in cls._registry:
            raise KeyError(
                f"因子 '{name}' 未注册。"
                f"可用因子: {list(cls._registry.keys())}"
            )

        factor_class = cls._registry[name]
        return factor_class(name=name, **params)

    @classmethod
    def list_factors(cls) -> list:
        """
        列出所有已注册的因子

        Returns:
            因子名称列表
        """
        return list(cls._registry.keys())

    @classmethod
    def get_factor_class(cls, name: str) -> Type[BaseFactor]:
        """
        获取因子类（不实例化）

        Args:
            name: 因子注册名称

        Returns:
            因子类
        """
        if name not in cls._registry:
            raise KeyError(f"因子 '{name}' 未注册")
        return cls._registry[name]


# 便捷函数
def register_factor(name: str) -> Callable:
    """
    因子注册装饰器（便捷接口）

    Args:
        name: 因子注册名称

    Returns:
        装饰器函数
    """
    return FactorRegistry.register(name)


def create_factor(name: str, **params) -> BaseFactor:
    """
    创建因子实例（便捷接口）

    Args:
        name: 因子注册名称
        **params: 因子参数

    Returns:
        因子实例
    """
    return FactorRegistry.create(name, **params)


def list_factors() -> list:
    """
    列出所有已注册因子（便捷接口）

    Returns:
        因子名称列表
    """
    return FactorRegistry.list_factors()