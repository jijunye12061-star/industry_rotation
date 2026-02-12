from EmQuantAPI import *
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


class RateLimiter:
    """API调用频率限制器"""

    def __init__(self, max_requests: int = 650, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.calls = []

    def wait_if_needed(self):
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # 清理过期记录
        self.calls = [t for t in self.calls if t > cutoff]

        if len(self.calls) >= self.max_requests:
            sleep_time = (self.calls[0] - cutoff).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.calls.clear()

        self.calls.append(now)


class ChoiceDataFetcher:
    """Choice数据查询客户端（支持上下文管理器）"""

    def __init__(self, max_retries: int = 3):
        self.username = os.getenv('CHOICE_USERNAME')  # 更明确的变量名
        self.password = os.getenv('CHOICE_PASSWORD')
        self.limiter = RateLimiter(max_requests=650, window_seconds=60)
        self.max_retries = max_retries
        self._logged_in = False
        self._connect()

    def _connect(self):
        """建立连接"""
        result = c.start(
            f"username={self.username},"
            f"password={self.password},"
            f"ForceLogin=1"
        )
        if result.ErrorCode != 0:
            raise ConnectionError(f"Choice login failed: {result.ErrorMsg}")
        self._logged_in = True
        logger.info("Choice API connected")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def disconnect(self):
        """断开连接"""
        if self._logged_in:
            c.stop()
            self._logged_in = False
            logger.info("Choice API disconnected")

    def query(self, query_type: str, **params) -> pd.DataFrame:
        """
        查询数据（自动重试+频率限制）

        Args:
            query_type: 'csd', 'css', 'edb', 'ctr', 'sector'
            **params: 查询参数
                - csd: codes, indicators, start_date, end_date, options
                - css: codes, indicators, options
                - edb: codes, options
                - ctr: codes, indicators, options
                - sector: codes, tradedate

        Returns:
            查询结果DataFrame
        """
        for attempt in range(self.max_retries):
            try:
                self.limiter.wait_if_needed()
                result = self._execute_query(query_type, **params)

                if hasattr(result, "ErrorCode"):
                    if query_type == 'sector' and result.ErrorCode == 0:
                        return result.Data
                    else:
                        raise RuntimeError(f"Query error: {result.ErrorMsg}")

                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"Query failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {self.max_retries} attempts")
                    raise

    @staticmethod
    def _execute_query(query_type: str, **params) -> Any:
        """执行具体查询"""
        query_map = {
            'csd': lambda: c.csd(
                params['codes'],
                params['indicators'],
                params['start_date'],
                params['end_date'],
                params.get('options', '')
            ),
            'css': lambda: c.css(
                params['codes'],
                params['indicators'],
                params.get('options', '')
            ),
            'edb': lambda: c.edb(
                params['codes'],
                params.get('options', '')
            ),
            'ctr': lambda: c.ctr(
                params['codes'],
                params['indicators'],
                params.get('options', '')
            ),
            'sector': lambda: c.sector(
                params['codes'],
                params['tradedate']
            )
        }

        if query_type not in query_map:
            raise ValueError(f"Unsupported query type: {query_type}")

        return query_map[query_type]()

    @staticmethod
    def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        获取指定日期范围内的交易日期
        Args:
            start_date: 开始日期，格式为 'YYYY-MM-DD'
            end_date: 结束日期，格式为 'YYYY-MM-DD'
        Returns:
            pd.DatetimeIndex: 交易日期索引
        """
        trading_dates = c.tradedates(start_date, end_date, "period=1,order=1,market=CNSESH")
        return pd.DatetimeIndex(trading_dates.Data)


# 全局单例
_fetcher_instance: Optional[ChoiceDataFetcher] = None


def get_fetcher() -> ChoiceDataFetcher:
    """获取Choice数据查询器单例"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = ChoiceDataFetcher()
    return _fetcher_instance


if __name__ == '__main__':
    fetcher = get_fetcher()
    data = fetcher.query('sector', codes="519002001002", tradedate="2016-01-05")
    # data = c.csd('CI005002.CI', 'FLOWINXL,FLOWOUTXL',
    #              '2025-01-01', '2025-02-01', 'period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1')
    # # 方式1：使用上下文管理器（推荐）
    # with ChoiceDataFetcher() as client:
    #     # data = client.query(
    #     #     'csd',
    #     #     codes='CI005001.CI,CI005002.CI',
    #     #     start_date='2015-01-01',
    #     #     end_date='2015-02-01',
    #     #     indicators='CLOSE,HIGH,LOW,OPEN,AMOUNT,PRECLOSE',
    #     #     options='period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,isPandas=1'
    #     # )
    #     dates = client.get_trading_dates('2015-01-01', '2015-01-10')
    #     data1 = client.query(
    #         'ctr',
    #         codes="INDEXCONSTITUENT",
    #         indicators="SECUCODE,WEIGHT",
    #         options=f"IndexCode=CI005001.CI,EndDate=2015-01-06,isPandas=1"
    #     )

    # # 方式2：使用全局单例
    # fetcher = get_fetcher()
    # data = fetcher.query('css', codes='...', indicators='...', options='...')