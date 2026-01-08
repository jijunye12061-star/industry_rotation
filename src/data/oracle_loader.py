#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: oracle_loader.py
@time: 2025/01/04
@description:
远程Oracle数据库加载器
"""
import pandas as pd
import logging
from typing import List
from datetime import timedelta

from data.base_loader import BaseDataLoader
from utils.query_data_funcs import fetcher
from config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)
# noinspection PyUnresolvedReferences,SqlNoDataSourceInspection,SqlResolve


class OracleLoader(BaseDataLoader):
    """
    远程Oracle数据库加载器

    直接查询TYTFUND数据库，无本地缓存
    """
    def __init__(self):
        """初始化数据加载器"""
        self.fetcher = fetcher
        self.industry_codes = list(INDUSTRY_CONFIG.keys())
        self.inner_codes = [info['inner_code'] for info in INDUSTRY_CONFIG.values()]
        self.inner_to_code = {v['inner_code']: k for k, v in INDUSTRY_CONFIG.items()}

    # ==================== 交易日历 ====================

    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        获取交易日期序列

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            交易日期索引
        """
        dates = self.fetcher.get_trading_dt(start_date, end_date)
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
        data = self.fetcher.query_data_generic(
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

    def get_market_prices(self, start_date: str, end_date: str,
                          market_code: str = MARKET_CODE) -> pd.Series:
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
        data = self.fetcher.query_data_tytfund(
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

        data = self.fetcher.query_data_generic(
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

    def get_market_returns(self, start_date: str, end_date: str,
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

        data = self.fetcher.query_data_tytfund(
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
        data = self.fetcher.query_data_generic(
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
        data = self.fetcher.query_data_generic(
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
        data = self.fetcher.query_data_generic(
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
        data = self.fetcher.query_data_generic(
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
        data = self.fetcher.query_data_generic(
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
            DataFrame with columns: ['证券代码', '权重']
        """
        industry_inner_code = INDUSTRY_CONFIG[industry_code]['inner_code']

        sql = """
              SELECT SECURITYCODE AS 证券代码,
                     SECINDEXR    AS 权重
              FROM TYTFUND.IDEX_YS_WEIGHT
              WHERE SECURITYCODE = :industry_code
                AND TRADEDATE = TO_DATE(:trade_date, 'YYYY-MM-DD')
              """

        data = self.fetcher.query_data_tytfund(
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
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame (日期 × 股票代码)
        """
        if not stock_codes:
            return pd.DataFrame()

        sql = """
              SELECT TDATE               AS 交易日期,
                     SECUCODE            AS 证券代码,
                     TVAL                AS 成交金额
              FROM TYTFUND.TRAD_SK_DAILY_JC
              WHERE SECUCODE IN (:code_list)
                AND TDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              """

        data = self.fetcher.query_data_generic(
            sql,
            code_list=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
        data['交易日期'] = pd.to_datetime(data['交易日期'])

        result = data.pivot(index='交易日期', columns='证券代码', values='成交金额')
        logger.info(f"获取 {len(stock_codes)} 只股票成交额: {result.shape}")
        return result

    def get_stock_super_large_flow(self,
                                   stock_codes: List[str],
                                   start_date: str,
                                   end_date: str) -> pd.DataFrame:
        """
        获取个股超大单成交额（流入+流出）

        Args:
            stock_codes: 证券代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            超大单成交额 DataFrame (日期 × 股票代码)
        """
        sql = """
              SELECT TRADEDATE              AS 交易日期,
                     TRADECODE              AS 证券代码,
                     (FLOWINXL + FLOWOUTXL) AS 超大单成交额
              FROM TYTFUND.TD_ASHAREFUNDFLOW
              WHERE TRADECODE IN (:code_list)
                AND TRADEDATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                  AND TO_DATE(:end_date, 'YYYY-MM-DD')
              """

        data = self.fetcher.query_data_generic(
            sql,
            code_list=stock_codes,
            start_date=start_date,
            end_date=end_date
        )

        data['交易日期'] = pd.to_datetime(data['交易日期'])

        result = data.pivot(index='交易日期', columns='证券代码', values='超大单成交额')
        logger.info(f"获取超大单成交额数据: {result.shape}")
        return result

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

