#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: loader.py
@time: 2025/12/12 10:08
数据加载模块 - 统一数据获取接口

所有数据获取逻辑集中在此，便于后续数据源迁移
"""
import pandas as pd
import logging
from typing import Optional, List
from datetime import timedelta

from utils.query_data_funcs import fetcher
from config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    统一管理所有数据获取接口，隔离数据源实现细节
    """

    def __init__(self):
        """初始化数据加载器"""
        self.industry_codes = list(INDUSTRY_CONFIG.keys())
        self.inner_codes = [info['inner_code'] for info in INDUSTRY_CONFIG.values()]
        self.inner_to_code = {v['inner_code']: k for k, v in INDUSTRY_CONFIG.items()}

    # ==================== 交易日历 ====================

    @staticmethod
    def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        获取交易日期序列

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            交易日期索引
        """
        dates = fetcher.get_trading_dt(start_date, end_date)
        logger.info(f"获取交易日期: {start_date} 至 {end_date}, 共 {len(dates)} 天")
        return dates

    # ==================== 价格数据 ====================

    def get_industry_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取所有行业收盘价

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            价格 DataFrame (日期 × 行业代码)
        """
        sql = """
              SELECT TDATE               AS 交易日期,
                     SECURITYVARIETYCODE AS 内码,
                     NEW                 AS 收盘价
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE 
              """
        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)

        result = data.pivot(index='交易日期', columns='行业代码', values='收盘价')
        logger.info(f"获取行业价格数据: {result.shape}")
        return result

    @staticmethod
    def get_market_prices(start_date: str, end_date: str,
                          market_code: str = '1000157271') -> pd.Series:
        """
        获取市场指数收盘价

        Args:
            start_date: 开始日期
            end_date: 结束日期
            market_code: 市场指数内码（默认中证全指）

        Returns:
            价格序列 (日期)
        """
        sql = """
              SELECT TDATE AS 交易日期,
                     NEW   AS 收盘价
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE = :market_code
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE \
              """
        data = fetcher.query_data_tytfund(
            sql,
            market_code=market_code,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        result = data.set_index('交易日期')['收盘价']
        logger.info(f"获取市场价格数据: {len(result)} 条")
        return result

    # ==================== 收益率数据 ====================

    def get_industry_returns(self, start_date: str, end_date: str,
                             method: str = 'pct') -> pd.DataFrame:
        """
        获取所有行业收益率

        Args:
            start_date: 开始日期
            end_date: 结束日期
            method: 计算方法
                - 'pct': 使用价格计算 (NEW/LCLOSE - 1) * 100
                - 'log': 对数收益率

        Returns:
            收益率 DataFrame (日期 × 行业代码), 单位: %
        """
        if method == 'pct':
            sql = """
                  SELECT TDATE                    AS 交易日期,
                         SECURITYVARIETYCODE      AS 内码,
                         (NEW / LCLOSE - 1) * 100 AS 收益率
                  FROM TYTFUND.TRAD_ID_DAILY
                  WHERE SECURITYVARIETYCODE IN (:code_list)
                    AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                      AND TO_DATE(:end_date, 'YYYY-MM-DD')
                  ORDER BY TDATE \
                  """
        else:  # log returns
            sql = """
                  SELECT TDATE                  AS 交易日期,
                         SECURITYVARIETYCODE    AS 内码,
                         LN(NEW / LCLOSE) * 100 AS 收益率
                  FROM TYTFUND.TRAD_ID_DAILY
                  WHERE SECURITYVARIETYCODE IN (:code_list)
                    AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                      AND TO_DATE(:end_date, 'YYYY-MM-DD')
                  ORDER BY TDATE \
                  """

        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)

        result = data.pivot(index='交易日期', columns='行业代码', values='收益率')
        logger.info(f"获取行业收益率数据: {result.shape}")
        return result

    @staticmethod
    def get_market_returns(start_date: str, end_date: str,
                           market_code: str = '1000157271',
                           method: str = 'pct') -> pd.Series:
        """
        获取市场指数收益率

        Args:
            start_date: 开始日期
            end_date: 结束日期
            market_code: 市场指数内码
            method: 计算方法 ('pct' 或 'log')

        Returns:
            收益率序列 (日期), 单位: %
        """
        if method == 'pct':
            sql = """
                  SELECT TDATE                    AS 交易日期,
                         (NEW / LCLOSE - 1) * 100 AS 收益率
                  FROM TYTFUND.TRAD_ID_DAILY
                  WHERE SECURITYVARIETYCODE = :market_code
                    AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                      AND TO_DATE(:end_date, 'YYYY-MM-DD')
                  ORDER BY TDATE \
                  """
        else:  # log returns
            sql = """
                  SELECT TDATE                  AS 交易日期,
                         LN(NEW / LCLOSE) * 100 AS 收益率
                  FROM TYTFUND.TRAD_ID_DAILY
                  WHERE SECURITYVARIETYCODE = :market_code
                    AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                      AND TO_DATE(:end_date, 'YYYY-MM-DD')
                  ORDER BY TDATE \
                  """

        data = fetcher.query_data_tytfund(
            sql,
            market_code=market_code,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        result = data.set_index('交易日期')['收益率']
        logger.info(f"获取市场收益率数据: {len(result)} 条")
        return result

    def get_industry_overnight_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取所有行业隔夜收益率

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            隔夜收益率 DataFrame (日期 × 行业代码), 单位: %
        """
        sql = """
              SELECT TDATE                     AS 交易日期,
                     SECURITYVARIETYCODE       AS 内码,
                     (OPEN / LCLOSE - 1) * 100 AS 隔夜收益率
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE
              """
        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)

        result = data.pivot(index='交易日期', columns='行业代码', values='隔夜收益率')
        logger.info(f"获取行业隔夜收益率数据: {result.shape}")
        return result

    def get_industry_high_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业最高价"""
        sql = """
              SELECT TDATE               AS 交易日期,
                     SECURITYVARIETYCODE AS 内码,
                     HIGH                AS 最高价
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE \
              """
        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)
        result = data.pivot(index='交易日期', columns='行业代码', values='最高价')
        logger.info(f"获取行业最高价数据: {result.shape}")
        return result

    def get_industry_low_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业最低价"""
        sql = """
              SELECT TDATE               AS 交易日期,
                     SECURITYVARIETYCODE AS 内码,
                     LOW                 AS 最低价
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE \
              """
        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)
        result = data.pivot(index='交易日期', columns='行业代码', values='最低价')
        logger.info(f"获取行业最低价数据: {result.shape}")
        return result

    def get_industry_open_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取所有行业开盘价

        Returns:
            开盘价 DataFrame (日期 × 行业代码)
        """
        sql = """
              SELECT TDATE               AS 交易日期,
                     SECURITYVARIETYCODE AS 内码,
                     OPEN                AS 开盘价
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE \
              """
        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)

        result = data.pivot(index='交易日期', columns='行业代码', values='开盘价')
        logger.info(f"获取行业开盘价数据: {result.shape}")
        return result

    def get_industry_amounts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取所有行业成交金额

        Returns:
            成交金额 DataFrame (日期 × 行业代码)
        """
        sql = """
              SELECT TDATE               AS 交易日期,
                     SECURITYVARIETYCODE AS 内码,
                     TVAL                AS 成交金额
              FROM TYTFUND.TRAD_ID_DAILY
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE \
              """
        data = fetcher.query_data_generic(
            sql,
            code_list=self.inner_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])
        data['行业代码'] = data['内码'].map(self.inner_to_code)

        result = data.pivot(index='交易日期', columns='行业代码', values='成交金额')
        logger.info(f"获取行业成交金额数据: {result.shape}")
        return result

    def get_index_constituents(self,
                               industry_code: str,
                               date: str) -> pd.DataFrame:
        """
        获取某日指数成分股及权重

        Args:
            industry_code: 行业代码（如 '801010'）
            date: 查询日期

        Returns:
            DataFrame with columns: ['证券内码', '权重']
        """
        industry_inner_code = INDUSTRY_CONFIG[industry_code]['inner_code']

        sql = """
              SELECT EMSECURITYVARIETYCODE AS 证券内码,
                     SECINDEXR             AS 权重
              FROM TYTFUND.IDEX_YS_WEIGHT
              WHERE SECURITYVARIETYCODE = :industry_code
                AND TRADEDATE = TO_DATE(:trade_date, 'YYYY-MM-DD')
              """

        data = fetcher.query_data_tytfund(
            sql,
            industry_code=industry_inner_code,
            trade_date=date
        )
        logger.info(f"获取 {industry_code} 在 {date} 的成分股: {len(data)} 只")
        return data

    def get_stock_amounts(self,
                          stock_codes: list,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """
        获取个股成交额

        Args:
            stock_codes: 股票内码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame (日期 × 股票内码)
        """
        if not stock_codes:
            return pd.DataFrame()

        sql = """
              SELECT TDATE               AS 交易日期,
                     SECURITYVARIETYCODE AS 证券内码,
                     TVAL                AS 成交金额
              FROM TYTFUND.TRAD_SK_DAILY_JC
              WHERE SECURITYVARIETYCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              ORDER BY TDATE
              """

        data = fetcher.query_data_generic(
            sql,
            code_list=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])

        result = data.pivot(index='交易日期', columns='证券内码', values='成交金额')
        logger.info(f"获取 {len(stock_codes)} 只股票成交额: {result.shape}")
        return result



    # ==================== 区间收益率 ====================

    def get_forward_returns(self,
                            rebalance_dates: pd.DatetimeIndex,
                            start_date: str,
                            end_date: str,
                            forward_periods: int = 1) -> pd.DataFrame:
        """
        获取调仓日的未来收益率（用于IC分析）

        Args:
            rebalance_dates: 调仓日期序列
            start_date: 数据开始日期
            end_date: 数据结束日期
            forward_periods: 前瞻期数（1=下一期）

        Returns:
            未来收益率 DataFrame (调仓日 × 行业代码), 单位: %
        """
        # 扩展结束日期以获取未来数据
        extended_end = (pd.to_datetime(end_date) + timedelta(days=60)).strftime('%Y-%m-%d')

        # 获取价格数据
        prices = self.get_industry_prices(start_date, extended_end)

        # 计算区间收益率
        forward_returns = pd.DataFrame(
            index=rebalance_dates[:-forward_periods],
            columns=prices.columns,
            dtype=float
        )

        for i in range(len(rebalance_dates) - forward_periods):
            start_dt = rebalance_dates[i]
            end_dt = rebalance_dates[i + forward_periods]

            if start_dt in prices.index and end_dt in prices.index:
                start_prices = prices.loc[start_dt]
                end_prices = prices.loc[end_dt]
                period_return = (end_prices / start_prices - 1) * 100
                forward_returns.loc[start_dt] = period_return

        logger.info(f"计算未来{forward_periods}期收益率: {forward_returns.shape}")
        return forward_returns

    # ==================== 其他数据接口 ====================

    @staticmethod
    def get_industry_info(code: Optional[str] = None) -> dict:
        """
        获取行业信息

        Args:
            code: 行业代码（None则返回所有）

        Returns:
            行业信息字典
        """
        if code is None:
            return INDUSTRY_CONFIG
        return INDUSTRY_CONFIG.get(code, {})

    def get_all_industry_codes(self) -> List[str]:
        """获取所有行业代码列表"""
        return self.industry_codes.copy()


# ==================== 全局单例 ====================

_loader_instance: Optional[DataLoader] = None


def get_loader() -> DataLoader:
    """
    获取数据加载器单例

    Returns:
        DataLoader实例
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DataLoader()
    return _loader_instance


# ==================== 便捷函数 ====================

def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """获取交易日期（便捷函数）"""
    return get_loader().get_trading_dates(start_date, end_date)


def get_industry_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """获取行业价格（便捷函数）"""
    return get_loader().get_industry_prices(start_date, end_date)


def get_industry_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """获取行业收益率（便捷函数）"""
    return get_loader().get_industry_returns(start_date, end_date)


def get_market_returns(start_date: str, end_date: str,
                       market_code: str = '1000157271') -> pd.Series:
    """获取市场收益率（便捷函数）"""
    return get_loader().get_market_returns(start_date, end_date, market_code)


# ==================== 测试代码 ====================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    loader = get_loader()

    # 测试各个接口
    print("\n=== 测试交易日期 ===")
    trading_dates = loader.get_trading_dates('2024-01-01', '2024-01-31')
    print(f"交易日数量: {len(trading_dates)}")
    print(trading_dates[:5])

    print("\n=== 测试行业价格 ===")
    prices = loader.get_industry_prices('2024-01-01', '2024-01-05')
    print(prices.head())

    print("\n=== 测试行业收益率 ===")
    returns = loader.get_industry_returns('2024-01-01', '2024-01-05')
    print(returns.head())

    print("\n=== 测试市场收益率 ===")
    market_ret = loader.get_market_returns('2024-01-01', '2024-01-05')
    print(market_ret.head())

    print("\n=== 测试行业信息 ===")
    print(f"行业数量: {len(loader.get_all_industry_codes())}")
    print(f"801010信息: {loader.get_industry_info('801010')}")
