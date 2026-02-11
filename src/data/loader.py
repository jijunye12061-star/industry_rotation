"""
数据加载器 - 只读 Parquet，提供统一数据接口

所有方法输出格式: DataFrame(DatetimeIndex × 行业代码) 或 Series(DatetimeIndex)
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from config import MARKET_CODE, INDUSTRY_CONFIG

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# 模块级单例
_loader_instance: Optional['DataLoader'] = None


def get_loader() -> 'DataLoader':
    """获取 DataLoader 单例"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DataLoader()
    return _loader_instance


def reset_loader():
    """重置单例（测试或切换数据目录时使用）"""
    global _loader_instance
    _loader_instance = None


class DataLoader:
    """
    本地 Parquet 数据加载器（只读）

    设计原则：
    - 只读本地文件，不触网
    - 会话级缓存，避免重复 IO
    - 统一输出 DataFrame(DatetimeIndex × 行业代码)
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.industry_codes = list(INDUSTRY_CONFIG.keys())
        self._cache: Dict[Tuple, pd.DataFrame] = {}
        logger.info(f"DataLoader 初始化 (数据目录: {self.data_dir})")

    # ==================== 通用读取 ====================

    def read_parquet(
            self,
            table_name: str,
            start_date: str,
            end_date: str,
            index_codes: Optional[List[str]] = None,
            date_col: str = "trade_date"
    ) -> pd.DataFrame:
        """
        按月读取 Parquet 并拼接（长表格式）

        公开方法，供 DataLoader 内部和外部模块（如 ETFLoader）使用。

        Args:
            table_name: 表名，对应 data/{table_name}/ 目录
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            index_codes: 可选的 index_code 过滤列表
            date_col: 日期列名

        Returns:
            长表 DataFrame，包含原始列（如 trade_date, index_code, close, ...）
        """
        # 会话级缓存
        cache_key = (table_name, start_date, end_date, tuple(index_codes or []))
        if cache_key in self._cache:
            return self._cache[cache_key]

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        chunks = []
        missing = []

        # 遍历每个月的分区文件
        for month_start in pd.date_range(
            start.to_period('M').to_timestamp(),
            end.to_period('M').to_timestamp(),
            freq='MS'
        ):
            year = month_start.strftime('%Y')
            month = month_start.strftime('%m')
            month_end = (month_start + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            month_start_str = month_start.strftime('%Y-%m-%d')

            filepath = self.data_dir / table_name / year / month / f"{month_start_str}_{month_end}.parquet"

            if not filepath.exists():
                missing.append(str(filepath))
                continue

            chunks.append(pd.read_parquet(filepath))

        if not chunks:
            raise FileNotFoundError(
                f"未找到数据文件 ({table_name}: {start_date} → {end_date})\n"
                f"缺失: {missing[:3]}...\n请先运行: python -m data.sync.sync_all --init"
            )

        df = pd.concat(chunks, ignore_index=True)

        if missing:
            logger.warning(f"{table_name}: {len(missing)} 个月份文件缺失")

        # 过滤
        if index_codes:
            df = df[df['index_code'].isin(index_codes)]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df[(df[date_col] >= start) & (df[date_col] <= end)]

        self._cache[cache_key] = df
        return df

    def clear_cache(self):
        """清空会话缓存"""
        self._cache.clear()

    # ==================== 交易日历 ====================

    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """获取交易日序列"""
        calendar_file = self.data_dir / "trading_calendar" / "calendar.parquet"

        if not calendar_file.exists():
            raise FileNotFoundError(
                "交易日历不存在，请先运行: python -m data.sync.trading_calendar"
            )

        df = pd.read_parquet(calendar_file)
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        return pd.DatetimeIndex(df.loc[mask, 'date'])

    # ==================== 行业指数数据 ====================

    def _get_index_field(
            self, start_date: str, end_date: str,
            field: str, codes: Optional[List[str]] = None,
            table: str = 'tb_index_daily'
    ) -> pd.DataFrame:
        """通用：读取指数表某字段，输出宽表 (日期 × 行业代码)"""
        codes = codes or self.industry_codes
        df = self.read_parquet(table, start_date, end_date, index_codes=codes)
        pivot = df.pivot(index='trade_date', columns='index_code', values=field)
        pivot.index = pd.to_datetime(pivot.index)
        return pivot.sort_index()

    def get_industry_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业收盘价 (日期 × 行业代码)"""
        return self._get_index_field(start_date, end_date, 'close')

    def get_industry_open_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业开盘价"""
        return self._get_index_field(start_date, end_date, 'open')

    def get_industry_high_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业最高价"""
        return self._get_index_field(start_date, end_date, 'high')

    def get_industry_low_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业最低价"""
        return self._get_index_field(start_date, end_date, 'low')

    def get_industry_amounts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业成交额"""
        return self._get_index_field(start_date, end_date, 'amount')

    def get_industry_preclose(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业前收盘价"""
        return self._get_index_field(start_date, end_date, 'preclose')

    # ==================== 市场指数 ====================

    def get_market_prices(
            self, start_date: str, end_date: str,
            market_code: str = MARKET_CODE
    ) -> pd.Series:
        """市场指数收盘价"""
        df = self.read_parquet(
            'tb_index_daily', start_date, end_date, index_codes=[market_code]
        )
        result = df.set_index('trade_date')['close']
        result.index = pd.to_datetime(result.index)
        result.name = market_code
        return result.sort_index()

    def get_market_returns(
            self, start_date: str, end_date: str,
            market_code: str = MARKET_CODE
    ) -> pd.Series:
        """市场指数日收益率"""
        df = self.read_parquet(
            'tb_index_daily', start_date, end_date, index_codes=[market_code]
        )
        df = df.set_index('trade_date')
        df.index = pd.to_datetime(df.index)
        result = (df['close'] / df['preclose'] - 1).sort_index()
        result.name = market_code
        return result

    # ==================== 收益率 ====================

    def get_industry_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业日收益率 (日期 × 行业代码)"""
        prices = self.get_industry_prices(start_date, end_date)
        return prices.pct_change().dropna(how='all')

    def get_industry_overnight_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业隔夜收益率: (open - preclose) / preclose"""
        open_prices = self.get_industry_open_prices(start_date, end_date)
        preclose = self.get_industry_preclose(start_date, end_date)
        # 对齐索引
        common_idx = open_prices.index.intersection(preclose.index)
        return (open_prices.loc[common_idx] - preclose.loc[common_idx]) / preclose.loc[common_idx]

    def get_forward_returns(
            self,
            rebalance_dates: pd.DatetimeIndex,
            start_date: str,
            end_date: str,
            forward_periods: int = 1
    ) -> pd.DataFrame:
        """
        计算未来 N 期收益率（对齐调仓日）

        Args:
            rebalance_dates: 调仓日序列
            start_date: 价格数据起始日
            end_date: 价格数据结束日
            forward_periods: 向前看的期数

        Returns:
            DataFrame(调仓日 × 行业代码)，值为该调仓日到下一调仓日的收益率
        """
        prices = self.get_industry_prices(start_date, end_date)

        # 取调仓日价格
        valid_dates = rebalance_dates[rebalance_dates.isin(prices.index)]
        rebalance_prices = prices.loc[valid_dates]

        # shift(-N) 计算未来收益
        forward_returns = rebalance_prices.pct_change(periods=forward_periods).shift(-forward_periods)
        return forward_returns.dropna(how='all')

    # ==================== 大单数据 ====================

    def get_industry_large_order_amount(self, start_date: str, end_date: str) -> pd.DataFrame:
        """行业超大单净流入 (日期 × 行业代码)"""
        df = self.read_parquet(
            'tb_index_large_order', start_date, end_date,
            index_codes=self.industry_codes
        )
        df['net_inflow'] = df['super_large_inflow'] - df['super_large_outflow']

        pivot = df.pivot(index='trade_date', columns='index_code', values='net_inflow')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot.sort_index()

    # ==================== 辅助方法 ====================

    def get_all_industry_codes(self) -> List[str]:
        """获取所有行业代码列表"""
        return self.industry_codes.copy()


# ==================== 便捷函数（向后兼容） ====================

def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return get_loader().get_trading_dates(start_date, end_date)

def get_industry_prices(start_date: str, end_date: str) -> pd.DataFrame:
    return get_loader().get_industry_prices(start_date, end_date)

def get_industry_returns(start_date: str, end_date: str) -> pd.DataFrame:
    return get_loader().get_industry_returns(start_date, end_date)

def get_market_returns(start_date: str, end_date: str, market_code: str = MARKET_CODE) -> pd.Series:
    return get_loader().get_market_returns(start_date, end_date, market_code)


if __name__ == '__main__':
    loader = get_loader()
    test_data = loader.get_market_returns('2024-12-31', '2025-12-31', '000985')
