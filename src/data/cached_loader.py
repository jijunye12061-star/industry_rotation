#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jijunye
@file: cached_loader.py
@time: 2025/01/05
@description:
本地缓存数据加载器 - 只读Parquet文件，不调用API
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional
from src.config import MARKET_CODE, INDUSTRY_CONFIG
from data.base_loader import BaseDataLoader

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class CachedLoader(BaseDataLoader):
    """本地缓存数据加载器（只读模式）"""

    def __init__(self):
        """初始化缓存加载器"""
        logger.info("初始化 CachedLoader（只读Parquet模式）")

        # 行业代码映射
        self.industry_codes = list(INDUSTRY_CONFIG.keys())
        self.inner_codes = [info['inner_code'] for info in INDUSTRY_CONFIG.values()]
        self.inner_to_code = {v['inner_code']: k for k, v in INDUSTRY_CONFIG.items()}

    # ==================== 核心读取方法 ====================

    @staticmethod
    def _read_parquet_by_month(
            table_name: str,
            start_date: str,
            end_date: str,
            index_codes: Optional[List[str]] = None,
            date_col: str = "trade_date"
    ) -> pd.DataFrame:
        """
        通用按月读取Parquet方法

        Args:
            table_name: 表名（如 'tb_index_daily'）
            start_date: 开始日期
            end_date: 结束日期
            index_codes: 可选的代码过滤列表

        Returns:
            合并后的DataFrame（长表格式）

        Raises:
            FileNotFoundError: 关键数据文件缺失时抛出
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        results = []
        missing_files = []

        # 调整到所在月的月初
        start_month = start.to_period('M').to_timestamp()
        end_month = end.to_period('M').to_timestamp()

        for month_start in pd.date_range(start_month, end_month, freq='MS'):
            year, month = month_start.strftime('%Y'), month_start.strftime('%m')
            month_end = month_start + pd.offsets.MonthEnd(0)

            query_start = month_start.strftime('%Y-%m-%d')
            query_end = month_end.strftime('%Y-%m-%d')

            filepath = DATA_DIR / table_name / year / month / f"{query_start}_{query_end}.parquet"

            # P1改进：优雅处理文件缺失
            if not filepath.exists():
                missing_files.append(str(filepath))
                logger.warning(f"数据文件缺失: {filepath}")
                continue

            try:
                df = pd.read_parquet(filepath)
                results.append(df)
            except Exception as e:
                logger.error(f"读取文件失败 {filepath}: {e}")
                continue

        # 如果所有文件都缺失，抛出友好的错误提示
        if not results:
            error_msg = (
                f"未找到任何数据文件（{start_date} -> {end_date}）\n"
                f"缺失文件: {missing_files[:3]}...\n"
                f"请先运行: python -m data.sync.sync_all --init"
            )
            raise FileNotFoundError(error_msg)

        # 合并数据
        combined = pd.concat(results, ignore_index=True)

        if index_codes:
            combined = combined[combined['index_code'].isin(index_codes)]

        # 确保日期是 datetime 类型
        combined[date_col] = pd.to_datetime(combined[date_col])

        # 过滤日期
        combined = combined[
            (combined[date_col] >= start) &
            (combined[date_col] <= end)
            ]

        return combined

    # ==================== 交易日历 ====================

    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """获取交易日期序列"""
        from data.sync.trading_calendar import get_trading_calendar
        df = get_trading_calendar(start_date, end_date)
        return pd.DatetimeIndex(df['date'])

    # ==================== 价格数据 ====================

    def get_industry_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业收盘价 (日期 × 行业代码)"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        pivot = df.pivot(index='trade_date', columns='index_code', values='close')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    def get_market_prices(
            self,
            start_date: str,
            end_date: str,
            market_code: str = MARKET_CODE
    ) -> pd.Series:
        """获取市场指数收盘价"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=[market_code]
        )
        result = df.set_index('trade_date')['close']
        result.index = pd.to_datetime(result.index)
        return result

    def get_industry_amounts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取行业成交额 (日期 × 行业代码)"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        pivot = df.pivot(index='trade_date', columns='index_code', values='amount')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    def get_industry_open_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业开盘价 (日期 × 行业代码)"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        pivot = df.pivot(index='trade_date', columns='index_code', values='open')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    def get_industry_low_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业最低价 (日期 × 行业代码)"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        pivot = df.pivot(index='trade_date', columns='index_code', values='low')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    def get_industry_high_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业最高价 (日期 × 行业代码)"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        pivot = df.pivot(index='trade_date', columns='index_code', values='high')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    def get_industry_overnight_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业隔夜收益率 (日期 × 行业代码), 单位: %"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        df['overnight_return'] = (df['open'] / df['preclose'] - 1) * 100

        pivot = df.pivot(index='trade_date', columns='index_code', values='overnight_return')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    # ==================== 收益率数据 ====================

    def get_industry_returns(
            self,
            start_date: str,
            end_date: str,
            method: str = 'pct'
    ) -> pd.DataFrame:
        """获取行业收益率 (日期 × 行业代码), 单位: %"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=self.industry_codes
        )

        if method == 'pct':
            df['return'] = (df['close'] / df['preclose'] - 1) * 100
        else:
            df['return'] = np.log1p(df['close'] / df['preclose'])

        pivot = df.pivot(index='trade_date', columns='index_code', values='return')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    def get_market_returns(
            self,
            start_date: str,
            end_date: str,
            market_code: str = MARKET_CODE,
            method: str = 'pct'
    ) -> pd.Series:
        """获取市场收益率, 单位: %"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=[market_code]
        )

        if method == 'pct':
            df['return'] = (df['close'] / df['preclose'] - 1) * 100
        else:
            df['return'] = (df['close'] / df['preclose'] - 1)

        result = df.set_index('trade_date')['return']
        result.index = pd.to_datetime(result.index)
        return result

    def get_market_ohlc(
            self,
            start_date: str,
            end_date: str,
            market_code: str = MARKET_CODE
    ) -> pd.DataFrame:
        """获取市场指数OHLC数据（含前收盘），索引为交易日"""
        df = self._read_parquet_by_month(
            'tb_index_daily', start_date, end_date,
            index_codes=[market_code]
        )

        result = df[['trade_date', 'open', 'high', 'low', 'close', 'preclose']].rename(columns={
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price',
            'preclose': 'preclose_price'
        })

        result['trade_date'] = pd.to_datetime(result['trade_date'])
        result = result.set_index('trade_date')
        return result

    # ==================== 成分股数据 ====================

    def get_index_constituents(self, industry_code: str, date: str) -> pd.DataFrame:
        """获取行业成分股及权重"""
        raise NotImplementedError("请先实现 sync_index_constituents")

    # ==================== 个股数据 ====================

    def get_stock_amounts(
            self,
            stock_codes: List[str],
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """获取个股成交额 (日期 × 股票代码)"""
        raise NotImplementedError("请先实现 sync_stock_daily")

    def get_stock_super_large_flow(
            self,
            stock_codes: List[str],
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """获取个股超大单成交额 (日期 × 股票代码)"""
        raise NotImplementedError("请先实现相关同步脚本")

    def get_industry_large_order_amount(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有行业超大单净流入 (日期 × 行业代码)"""
        df = self._read_parquet_by_month(
            'tb_index_large_order', start_date, end_date,
            index_codes=self.industry_codes
        )

        # 计算净流入
        df['net_inflow'] = df['super_large_inflow'] - df['super_large_outflow']

        pivot = df.pivot(index='trade_date', columns='index_code', values='net_inflow')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot

    # ==================== 其他接口 ====================

    def get_all_industry_codes(self) -> List[str]:
        """获取所有行业代码"""
        return self.industry_codes

    # ==================== 临时接口 ====================
    @staticmethod
    def read_parquet_by_month(table_name: str, start_date: str, end_date: str,
                              index_codes: Optional[List[str]] = None,
                              date_col: str = None) -> pd.DataFrame:
        """公开静态方法供外部调用"""
        return CachedLoader._read_parquet_by_month(table_name, start_date, end_date, index_codes, date_col)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    loader = CachedLoader()

    # 测试
    try:
        prices = loader.get_industry_high_prices('2024-11-01', '2024-11-20')
        print(f"\n行业价格: {prices.shape}")
        print(prices.head())
    except FileNotFoundError as e:
        print(f"\n错误: {e}")