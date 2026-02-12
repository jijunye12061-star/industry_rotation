"""
交易日历管理 - 从 Choice API 获取并缓存

存储路径: data/trading_calendar/calendar.parquet
缓存策略: 范围足够则直接返回，不足则扩展后覆盖
"""
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

CALENDAR_FILE = Path(__file__).resolve().parents[3] / "data" / "trading_calendar" / "calendar.parquet"


def get_trading_calendar(
        start_date: str = '2015-01-01',
        end_date: str = '2030-12-31',
        force_update: bool = False
) -> pd.DataFrame:
    """
    获取交易日历（优先读缓存）

    Returns:
        DataFrame with column: ['date'] (str 'YYYY-MM-DD')
    """
    if CALENDAR_FILE.exists() and not force_update:
        cached = pd.read_parquet(CALENDAR_FILE)
        if start_date >= cached['date'].min() and end_date <= cached['date'].max():
            return cached[(cached['date'] >= start_date) & (cached['date'] <= end_date)]

        # 扩展范围
        start_date = min(start_date, cached['date'].min())
        end_date = max(end_date, cached['date'].max())

    logger.info(f"获取交易日历: {start_date} ~ {end_date}")
    from utils.query_data_from_choice import get_fetcher
    fetcher = get_fetcher()
    dates = fetcher.get_trading_dates(start_date, end_date)

    calendar = pd.DataFrame({'date': dates.strftime('%Y-%m-%d')})
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(CALENDAR_FILE, index=False)
    logger.info(f"交易日历已缓存: {len(calendar)} 天")
    return calendar


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cal = get_trading_calendar('2012-01-01', '2030-12-31', force_update=True)
    print(f"交易日历: {len(cal)} 天 ({cal['date'].min()} ~ {cal['date'].max()})")