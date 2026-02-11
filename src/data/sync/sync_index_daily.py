"""
指数日线数据同步 - Choice API → Parquet

表: tb_index_daily
字段: trade_date, index_code, open, high, low, close, preclose, amount
分区: data/tb_index_daily/{year}/{month}/start_end.parquet
"""
import pandas as pd
import logging
from typing import Optional, List

from data.decorators import parquet_cache
from utils.query_data_from_choice import get_fetcher
from config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)

# Choice 代码映射（集中定义）
INDUSTRY_CODES = list(INDUSTRY_CONFIG.keys())
CHOICE_CODES = [f"{code}.CI" for code in INDUSTRY_CODES] + ['000985.CSI']
CHOICE_TO_STD = {
    **{f"{code}.CI": code for code in INDUSTRY_CODES},
    '000985.CSI': '000985'
}


@parquet_cache(
    table="tb_index_daily",
    keys=['trade_date', 'index_code'],
    date_col='trade_date',
    overlap_days=3,
    partition_by='month',
    auto_split=True
)
def _fetch_index_daily(
        start_date: str,
        end_date: str,
        index_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """从 Choice API 获取指数日线（装饰器自动管理缓存）"""
    fetcher = get_fetcher()

    if index_codes:
        choice_codes = [
            f"{code}.CI" if code in INDUSTRY_CODES else code
            for code in index_codes
        ]
    else:
        choice_codes = CHOICE_CODES

    raw_data = fetcher.query(
        'csd',
        codes=','.join(choice_codes),
        indicators='CLOSE,HIGH,LOW,OPEN,AMOUNT,PRECLOSE',
        start_date=start_date,
        end_date=end_date,
        options='period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1'
    )

    if raw_data.empty:
        logger.warning(f"API 返回空数据: {start_date} → {end_date}")
        return pd.DataFrame()

    df = raw_data.reset_index()
    df['trade_date'] = pd.to_datetime(df['DATES']).dt.strftime('%Y-%m-%d')
    df['index_code'] = df['CODES'].map(CHOICE_TO_STD)

    result = df[df['index_code'].notna()][
        ['trade_date', 'index_code', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'PRECLOSE', 'AMOUNT']
    ].rename(columns={
        'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
        'CLOSE': 'close', 'PRECLOSE': 'preclose', 'AMOUNT': 'amount'
    })

    result = result.dropna(subset=['open', 'close'], how='any')
    logger.info(f"获取 {len(result)} 条: {start_date} → {end_date}")
    return result


def sync_index_daily(start_date: str, end_date: str) -> int:
    """同步指数日线数据（公开接口）"""
    logger.info(f"同步指数日线: {start_date} → {end_date}")
    df = _fetch_index_daily(start_date, end_date)
    count = len(df)
    logger.info(f"完成: {count} 条")
    return count


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sync_index_daily('2024-11-01', '2024-12-31')