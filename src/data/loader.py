#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : loader.py
@Time    : 2025/12/15 13:34
@Author  : jijunye
@Desc    : 
"""
import pandas as pd
import logging
from typing import Optional, List
from datetime import timedelta

from utils.query_data_from_choice import get_fetcher
from config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)


class APILoader:
    """使用Choice API的数据加载器"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.industry_codes = list(INDUSTRY_CONFIG.keys())
        # 动态生成Choice代码（加.CI后缀）
        self.choice_codes = {code: f"{code}.CI" for code in self.industry_codes}
        self.choice_to_code = {v: k for k, v in self.choice_codes.items()}

    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        获取交易日期序列

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            交易日期索引
        """
        dates = self.fetcher.get_trading_dates(start_date, end_date)
        logger.info(f"获取交易日期: {start_date} 至 {end_date}, 共 {len(dates)} 天")
        return dates

    def _query_industry_data(self, start_date: str, end_date: str,
                             indicators: str, options: str = "") -> pd.DataFrame:
        """查询行业数据的通用方法"""
        codes = ','.join(self.choice_codes.values())
        base_options = "period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1"
        if options:
            base_options += f",{options}"

        data = self.fetcher.query(
            'csd',
            codes=codes,
            indicators=indicators,
            start_date=start_date,
            end_date=end_date,
            options=base_options
        )
        return data

    def _pivot_data(self, data: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        转换Choice API返回格式为标准格式

        Input (Choice):
            CODES (index) | DATES | CLOSE | OPEN | ...

        Output (标准):
            DATES (index) | 801010 | 801020 | ...
        """
        df = data.reset_index()  # CODES变为列
        df['DATES'] = pd.to_datetime(df['DATES'])
        df['行业代码'] = df['CODES'].map(self.choice_to_code)

        result = df.pivot(index='DATES', columns='行业代码', values=value_col)
        result.index.name = None
        logger.info(f"获取{value_col}数据: {result.shape}")
        return result

    def get_industry_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业收盘价"""
        data = self._query_industry_data(start_date, end_date, "CLOSE")
        return self._pivot_data(data, 'CLOSE')

    def get_industry_high_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业最高价"""
        data = self._query_industry_data(start_date, end_date, "HIGH")
        return self._pivot_data(data, 'HIGH')

    def get_industry_low_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业最低价"""
        data = self._query_industry_data(start_date, end_date, "LOW")
        return self._pivot_data(data, 'LOW')

    def get_industry_open_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业开盘价"""
        data = self._query_industry_data(start_date, end_date, "OPEN")
        return self._pivot_data(data, 'OPEN')

    def get_industry_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业收益率（%）"""
        data = self._query_industry_data(start_date, end_date, "PCTCHANGE")
        return self._pivot_data(data, 'PCTCHANGE')

    def get_industry_amount(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业成交额"""
        data = self._query_industry_data(start_date, end_date, "AMOUNT")
        return self._pivot_data(data, 'AMOUNT')

    def get_industry_preclose_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业前收盘价"""
        data = self._query_industry_data(start_date, end_date, "PRECLOSE")
        return self._pivot_data(data, 'PRECLOSE')

    def get_industry_overnight_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取行业隔夜收益率

        计算公式: (OPEN / PRECLOSE - 1) * 100
        """
        # 一次查询获取OPEN和PRECLOSE
        data = self._query_industry_data(start_date, end_date, "OPEN,PRECLOSE")

        # 转换格式
        df = data.reset_index()
        df['DATES'] = pd.to_datetime(df['DATES'])
        df['行业代码'] = df['CODES'].map(self.choice_to_code)

        # 计算隔夜收益率
        df['隔夜收益率'] = (df['OPEN'] / df['PRECLOSE'] - 1) * 100

        result = df.pivot(index='DATES', columns='行业代码', values='隔夜收益率')
        result.index.name = None
        logger.info(f"获取行业隔夜收益率数据: {result.shape}")
        return result

    def get_index_constituents(self, industry_code: str, date: str) -> pd.DataFrame:
        """
        获取某日指数成分股及权重

        Args:
            industry_code: 行业代码（如 'CI005001'）
            date: 查询日期 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: ['证券代码', '权重']
        """
        choice_code = f"{industry_code}.CI"

        data = self.fetcher.query(
            'ctr',
            codes="INDEXCONSTITUENT",
            indicators="SECUCODE,WEIGHT",
            options=f"IndexCode={choice_code},EndDate={date},isPandas=1"
        )

        # 重命名列以保持接口一致
        result = data[['SECUCODE', 'WEIGHT']].rename(columns={
            'SECUCODE': '证券代码',
            'WEIGHT': '权重'
        })

        logger.info(f"获取 {industry_code} 在 {date} 的成分股: {len(result)} 只")
        return result

    def get_stock_amounts(self,
                          stock_codes: list,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """
        获取个股成交额

        Args:
            stock_codes: 股票代码列表（如 ['600506.SH', '000819.SZ']）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame (日期 × 股票代码)
        """
        if not stock_codes:
            return pd.DataFrame()

        codes = ','.join(stock_codes)

        data = self.fetcher.query(
            'csd',
            codes=codes,
            indicators='AMOUNT',
            start_date=start_date,
            end_date=end_date,
            options='period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1'
        )

        # 转换为标准格式
        df = data.reset_index()
        df['DATES'] = pd.to_datetime(df['DATES'])

        result = df.pivot(index='DATES', columns='CODES', values='AMOUNT')
        result.index.name = None
        logger.info(f"获取 {len(stock_codes)} 只股票成交额: {result.shape}")
        return result

    # ==================== 市场收益率 ====================

    def get_market_prices(self, start_date: str, end_date: str,
                          market_code: str = '000985.CSI') -> pd.Series:
        """获取市场指数收盘价（中证全指的Choice代码）"""
        data = self.fetcher.query(
            'csd',
            codes=market_code,
            indicators='CLOSE',
            start_date=start_date,
            end_date=end_date,
            options='period=1,adjustflag=1,curtype=1,order=1,isPandas=1'
        )
        df = data.reset_index()
        df['DATES'] = pd.to_datetime(df['DATES'])
        result = df.set_index('DATES')['CLOSE']
        logger.info(f"获取市场价格数据: {len(result)} 条")
        return result

    def get_market_returns(self, start_date: str, end_date: str,
                          market_code: str = '000985.CSI') -> pd.Series:
        """获取市场指数收盘价（中证全指的Choice代码）"""
        data = self.fetcher.query(
            'csd',
            codes=market_code,
            indicators='PCTCHANGE',
            start_date=start_date,
            end_date=end_date,
            options='period=1,adjustflag=1,curtype=1,order=1,isPandas=1'
        )
        df = data.reset_index()
        df['DATES'] = pd.to_datetime(df['DATES'])
        result = df.set_index('DATES')['PCTCHANGE']
        logger.info(f"获取市场收益率数据: {len(result)} 条")
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

        # 计算区间的收益率
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

    def get_all_industry_codes(self) -> List[str]:
        """获取所有行业代码列表"""
        return self.industry_codes.copy()


# ==================== 全局单例 ====================

_loader_instance: Optional[APILoader] = None


def get_loader() -> APILoader:
    """
    获取数据加载器单例

    Returns:
        DataLoader实例
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = APILoader()
    return _loader_instance


if __name__ == "__main__":
    pass
    logging.basicConfig(level=logging.INFO)

    loader = get_loader()

    trading_dates = loader.get_industry_prices('2024-10-01', '2024-12-31')