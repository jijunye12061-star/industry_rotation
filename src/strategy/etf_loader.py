#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETF数据加载器 - 纯数据加载与预计算"""
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

from data.cached_loader import CachedLoader

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class ETFLoader:
    """ETF数据加载器（缓存 + 预计算）"""

    def __init__(self):
        self._etf_info = None
        self._nav_monthly = None
        self._portfolio = None
        self._stock_industry = None

    def get_etf_info(self) -> pd.DataFrame:
        """
        获取ETF基本信息

        Returns:
            DataFrame: [c_fd_code, c_estabdate, c_terminate_date, ...]
        """
        if self._etf_info is None:
            self._etf_info = pd.read_parquet(
                DATA_DIR / "etf_basic_info" / "etf_basic_info.parquet"
            )
            self._etf_info['c_estabdate'] = pd.to_datetime(self._etf_info['c_estabdate'])
            self._etf_info['c_terminate_date'] = pd.to_datetime(
                self._etf_info['c_terminate_date'], errors='coerce'
            )
        return self._etf_info

    def get_etf_scales(self, date: str) -> pd.Series:
        """
        获取所有ETF截至date的最新规模

        Returns:
            Series: index=c_fd_code, values=c_total_nav (元)
        """
        if self._nav_monthly is None:
            self._nav_monthly = pd.read_parquet(
                DATA_DIR / "etf_nav_monthly" / "etf_nav_monthly.parquet"
            )
            self._nav_monthly['c_trade_date'] = pd.to_datetime(
                self._nav_monthly['c_trade_date']
            )

        date = pd.to_datetime(date)
        valid = self._nav_monthly[self._nav_monthly['c_trade_date'] <= date]

        # 每个ETF取最新记录
        latest_idx = valid.groupby('c_fd_code')['c_trade_date'].idxmax()
        return valid.loc[latest_idx].set_index('c_fd_code')['c_total_nav']

    def get_all_holdings(self, date: str) -> pd.DataFrame:
        """
        批量获取所有ETF截至date的最新持仓（含行业）

        Returns:
            DataFrame: [c_fd_code, c_stk_code, c_nav_ratio, c_industry_code]
        """
        if self._portfolio is None:
            self._portfolio = pd.read_parquet(
                DATA_DIR / "etf_portfolio" / "etf_portfolio.parquet"
            )
            self._portfolio['c_notice_date'] = pd.to_datetime(
                self._portfolio['c_notice_date']
            )

        if self._stock_industry is None:
            self._stock_industry = pd.read_parquet(
                DATA_DIR / "stock_industry" / "stock_industry.parquet"
            )

        date = pd.to_datetime(date)
        valid = self._portfolio[self._portfolio['c_notice_date'] <= date]

        # 每个ETF取最新公告
        latest = valid.groupby('c_fd_code')['c_notice_date'].max().reset_index()
        holdings = valid.merge(latest, on=['c_fd_code', 'c_notice_date'])

        # 关联股票行业（仅精确匹配c_report_date）
        holdings = holdings.merge(
            self._stock_industry[['c_stk_code', 'c_industry_code', 'c_report_date']],
            on=['c_stk_code', 'c_report_date'],
            how='left'
        )

        return holdings[['c_fd_code', 'c_stk_code', 'c_nav_ratio', 'c_industry_code']]

    @staticmethod
    def calculate_industry_weights(holdings: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个ETF的行业权重统计

        Args:
            holdings: get_all_holdings()的输出

        Returns:
            DataFrame: [c_fd_code, primary_industry, primary_weight,
                       top_tertiary, tertiary_in_primary_ratio, total_valid_weight]
        """
        # 过滤无行业股票
        valid = holdings[holdings['c_industry_code'].notna()].copy()
        valid['industry_6'] = valid['c_industry_code'].str[:6]
        valid['industry_12'] = valid['c_industry_code']

        results = []
        for etf_code, group in valid.groupby('c_fd_code'):
            # 有效权重归一化
            total_weight = group['c_nav_ratio'].sum()
            if total_weight == 0:
                continue

            group = group.copy()
            group['weight_norm'] = group['c_nav_ratio'] / total_weight

            # 一级行业汇总
            primary_weights = group.groupby('industry_6')['weight_norm'].sum()
            if len(primary_weights) == 0:
                continue

            top_primary = primary_weights.idxmax()
            top_primary_weight = primary_weights.max()

            # 该一级下的三级行业
            primary_holdings = group[group['industry_6'] == top_primary]
            tertiary_weights = primary_holdings.groupby('industry_12')['weight_norm'].sum()

            top_tertiary_weight = tertiary_weights.max()
            tertiary_ratio = top_tertiary_weight / top_primary_weight

            results.append({
                'c_fd_code': etf_code,
                'primary_industry': top_primary,
                'primary_weight': top_primary_weight,
                'top_tertiary': tertiary_weights.idxmax(),
                'tertiary_in_primary_ratio': tertiary_ratio,
                'total_valid_weight': total_weight  # 原始权重和（用于判断数据完整性）
            })

        return pd.DataFrame(results)

    @staticmethod
    def get_etf_prices(start_date: str, end_date: str,
                       etf_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取ETF净值 (日期 × ETF代码)

        Returns:
            DataFrame: index=日期, columns=c_fd_code, values=c_adj_nav
        """
        df = CachedLoader.read_parquet_by_month(
            'tb_etf_daily', start_date, end_date, date_col="c_trade_date"
        )

        if etf_codes:
            df = df[df['c_fd_code'].isin(etf_codes)]

        from data.loader import get_loader

        ind_loader = get_loader()
        trading_dates = ind_loader.get_trading_dates(start_date, end_date)
        df = df[df['c_trade_date'].isin(trading_dates)]

        pivot = df.pivot(index='c_trade_date', columns='c_fd_code', values='c_adj_nav')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot


if __name__ == "__main__":
    loader = ETFLoader()
    etf_info = loader.get_etf_prices('2016-12-30', '2017-01-07', ['159928'])

