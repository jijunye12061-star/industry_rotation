"""
数据查询工具集 - 支持Oracle和Doris数据库
包含基金净值、指数数据、持仓数据等常用查询函数

@Time: 2024/10/25 13:43
@Author: 季俊晔
"""
import os
from contextlib import contextmanager
from typing import List, Optional, Union, Iterator

import pandas as pd
import oracledb
import logging
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from sqlalchemy import create_engine, text, Table, MetaData, Column, String
from sqlalchemy.engine import URL, Engine
from sqlalchemy.pool import QueuePool

# 加载环境变量
load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """数据库配置单例类"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Oracle配置
            cls._instance.host = os.getenv('DB_HOST')
            cls._instance.port = os.getenv('DB_PORT')
            cls._instance.service_name = os.getenv('DB_SERVICE_NAME')
            cls._instance.username = os.getenv('DB_USERNAME')
            cls._instance.password = os.getenv('DB_PASSWORD')

            # Doris配置
            cls._instance.c_host = os.getenv('DORIS_HOST')
            cls._instance.c_port = os.getenv('DORIS_PORT')
            cls._instance.c_username = os.getenv('DORIS_USERNAME')
            cls._instance.c_password = os.getenv('DORIS_PASSWORD')
            cls._instance.c_database = os.getenv('DORIS_DATABASE')
        return cls._instance

    @property
    def dsn_tns(self) -> str:
        """Oracle数据库连接字符串"""
        return f"{self.host}:{self.port}/{self.service_name}"


class TytDataUtils:
    """Oracle数据库查询工具类"""
    _initialized = False

    def __init__(self):
        if not TytDataUtils._initialized:
            try:
                oracledb.init_oracle_client()
                TytDataUtils._initialized = True
            except Exception as e:
                logger.warning(f"Oracle客户端初始化失败: {e}")
                logger.warning("将仅使用本地SQLite数据库")
        self.db_config = DatabaseConfig()
        # 创建连接池
        self.pool = oracledb.create_pool(
            user=self.db_config.username,
            password=self.db_config.password,
            dsn=self.db_config.dsn_tns,
            min=2,
            max=10,
            increment=1
        )

    @contextmanager
    def get_connection(self) -> Iterator[oracledb.Connection]:
        """获取Oracle连接的上下文管理器"""
        connection = self.pool.acquire()
        try:
            yield connection
        finally:
            self.pool.release(connection)

    def query_data_tytfund(self, query: str, **kwargs) -> pd.DataFrame:
        """
        执行TYTFUND数据库查询
        :param query: SQL查询语句
        :param kwargs: 查询参数，支持report_dts参数替换为日期列表
        :return: 查询结果DataFrame
        """
        with self.get_connection() as connection:
            with connection.cursor() as cursor:
                # 处理日期列表参数
                if 'report_dts' in kwargs:
                    dates = kwargs.pop('report_dts')
                    date_string = ",".join(f"TO_DATE('{date}', 'YYYY-MM-DD')" for date in dates)
                    query = query.replace(":report_dts", f"({date_string})")

                cursor.execute(query, **kwargs)
                result = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
                return pd.DataFrame(result, columns=columns)

    def query_data_generic(self, query: str, code_list: Optional[List[str]] = None,
                           batch_size: int = 450, **kwargs) -> pd.DataFrame:
        """
        通用批量查询方法 - 支持代码列表分批处理
        :param query: SQL查询语句，代码列表用IN (:code_list)表示
        :param code_list: 代码列表
        :param batch_size: 每批处理数量
        :param kwargs: 其他绑定变量
        :return: 查询结果DataFrame
        """
        with self.get_connection() as connection:
            with connection.cursor() as cursor:
                # 无代码列表时直接查询
                if code_list is None:
                    cursor.execute(query, **kwargs)
                    result = cursor.fetchall()
                    columns = [col[0] for col in cursor.description]
                    return pd.DataFrame(result, columns=columns)

                # 分批处理代码列表
                all_results = []
                for i in range(0, len(code_list), batch_size):
                    batch = code_list[i:i + batch_size]
                    # 动态构建绑定变量
                    placeholders = ','.join(f':{j}' for j in range(len(batch)))
                    modified_query = query.replace("IN (:code_list)", f"IN ({placeholders})")
                    bind_vars = {str(j): code for j, code in enumerate(batch)}
                    bind_vars.update(kwargs)

                    cursor.execute(modified_query, bind_vars)
                    all_results.extend(cursor.fetchall())

                columns = [col[0] for col in cursor.description]
                return pd.DataFrame(all_results, columns=columns)

    def get_trading_dt(self, begin_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        获取交易日序列
        :param begin_date: 开始日期 (YYYY-MM-DD)
        :param end_date: 结束日期 (YYYY-MM-DD)
        :return: 交易日DatetimeIndex
        """
        query = """
            SELECT 
                TRADE_DT AS 交易日期
            FROM TYTFUND.QT_TRADE_CALENDAR
            WHERE IS_D = '1'
            AND TRADE_DT >= TO_DATE(:begin_date, 'YYYY-MM-DD')
            AND TRADE_DT <= TO_DATE(:end_date, 'YYYY-MM-DD')
        """
        trading_dt = self.query_data_tytfund(query, begin_date=begin_date, end_date=end_date)
        return pd.DatetimeIndex(trading_dt['交易日期'])

    def query_fund_nav(self, fund_codes: List[str], begin_date: str, end_date: str) -> pd.DataFrame:
        """
        获取基金净值数据
        :param fund_codes: 基金代码列表
        :param begin_date: 开始日期
        :param end_date: 结束日期
        :return: 基金净值DataFrame
        """
        query = """
            SELECT SECURITYCODE AS 基金代码,
                   ENDDATE      AS 交易日期,
                   AANVPER      AS 复权净值
            FROM TYTFUND.FUND_DR_FUNDNV
            WHERE SECURITYCODE IN (:code_list)
              AND ENDDATE >= TO_DATE(:begin_date, 'YYYY-MM-DD')
              AND ENDDATE <= TO_DATE(:end_date, 'YYYY-MM-DD')
        """
        return self.query_data_generic(query, code_list=fund_codes, batch_size=500,
                                       begin_date=begin_date, end_date=end_date)

    def query_index_nav(self, index_codes: Union[str, List[str]],
                       begin_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数净值数据
        :param index_codes: 指数代码或代码列表
        :param begin_date: 开始日期
        :param end_date: 结束日期
        :return: 指数数据DataFrame
        """
        if isinstance(index_codes, str):
            index_codes = [index_codes]

        query = """
            SELECT a.INDEXCODE AS 指数代码,
                   a.TDATE     AS 交易日期,
                   a.NEW       AS 收盘价,
                   a.LCLOSE    AS 前收盘价
            FROM TYTFUND.TRAD_ID_DAILY a
            INNER JOIN TYTFUND.INDEX_BA_INFO b ON a.SECURITYVARIETYCODE = b.SECURITYVARIETYCODE
            WHERE b.INDEXCODE = :index_code
              AND a.TDATE >= TO_DATE(:begin_date, 'YYYY-MM-DD')
              AND a.TDATE <= TO_DATE(:end_date, 'YYYY-MM-DD')
        """

        results = [
            self.query_data_tytfund(query, index_code=code, begin_date=begin_date, end_date=end_date)
            for code in index_codes
        ]
        return pd.concat(results, ignore_index=True)

    def query_fund_iv_cb(self, fund_codes: List[str], report_dt: str) -> pd.DataFrame:
        """
        获取基金转债持仓数据
        :param fund_codes: 基金代码列表
        :param report_dt: 报告日期
        :return: 转债持仓DataFrame
        """
        query = """
            SELECT FUNDCODE  AS 基金代码,
                   ENDDATE   AS 报告日期,
                   BONDCODE  AS 债券代码,
                   INNERCODE AS 债券内码,
                   PCTNV     AS 持仓占比,
                   STYLE
            FROM TYTFUND.FUND_IV_BONDINVESTD
            WHERE FUNDCODE IN (:code_list)
              AND ENDDATE = TO_DATE(:report_dt, 'YYYY-MM-DD')
              AND BONDTYPE = '2'
        """
        iv_cb_data = self.query_data_generic(query, code_list=fund_codes, report_dt=report_dt)

        # 去重：按STYLE排序，保留第一条
        return (iv_cb_data.sort_values('STYLE', ascending=True)
                .drop_duplicates(subset=['基金代码', '报告日期', '债券内码'], keep='first')
                .drop('STYLE', axis=1))

    def query_fund_iv_stock(self, fund_codes: List[str], report_dt: str) -> pd.DataFrame:
        """
        获取基金股票持仓数据
        :param fund_codes: 基金代码列表
        :param report_dt: 报告日期
        :return: 股票持仓DataFrame
        """
        query = """
            SELECT FUNDCODE              AS 基金代码,
                   ENDDATE               AS 报告日期,
                   STOCKCODE             AS 股票代码,
                   EMSECURITYVARIETYCODE AS 股票内码,
                   PCTNV                 AS 持仓占比
            FROM TYTFUND.FUND_IV_STOCKINVESTO
            WHERE FUNDCODE IN (:code_list)
              AND ENDDATE = TO_DATE(:report_dt, 'YYYY-MM-DD')
              AND STYLE IN ('01', '02', '03', '04')
        """
        return self.query_data_generic(query, code_list=fund_codes, report_dt=report_dt)

    def close(self):
        """关闭连接池"""
        self.pool.close()


class DorisDBUtils:
    """Doris数据库查询工具类"""

    def __init__(self):
        self.db_config = DatabaseConfig()
        self.engine = self._create_engine()

    def _create_engine(self) -> Engine:
        """创建SQLAlchemy引擎"""
        return create_engine(
            URL.create(
                "mysql+pymysql",
                username=self.db_config.c_username,
                password=self.db_config.c_password,
                host=self.db_config.c_host,
                port=self.db_config.c_port,
                database=self.db_config.c_database
            ),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )

    def execute(self, sql: str) -> None:
        """
        执行DDL/DML语句
        :param sql: SQL语句
        """
        with self.engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()

    def execute_sql_file(self, file_path: str) -> None:
        """
        执行SQL文件
        :param file_path: SQL文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        self.execute(sql_content)
        print(f"成功执行SQL文件: {file_path}")

    def query(self, query: str, chunksize: Optional[int] = 50000, **kwargs) -> pd.DataFrame:
        """
        执行Doris查询
        :param query: SQL查询语句
        :param chunksize: 分块大小
        :param kwargs: 查询参数
        :return: 查询结果DataFrame
        """
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn, params=kwargs, chunksize=chunksize)
            return pd.concat(result) if chunksize else result

    def batch_insert(self, table_name: str, df: pd.DataFrame, batch_size: int = 8000) -> None:
        """
        批量插入数据到Doris
        :param table_name: 表名
        :param df: 待插入的DataFrame
        :param batch_size: 批次大小
        """
        if df.empty:
            return

        total_rows = len(df)
        metadata = MetaData()
        columns = [Column(col, String) for col in df.columns]
        table = Table(table_name, metadata, *columns)

        with self.engine.connect() as conn:
            # 小数据量直接插入
            if total_rows <= batch_size:
                values = df.to_dict('records')
                conn.execute(table.insert(), values)  # type: ignore
                conn.commit()
                return

            # 大数据量分批插入
            with tqdm(total=total_rows, desc="Inserting data") as pbar:
                for i in range(0, total_rows, batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    values = batch_df.to_dict('records')
                    conn.execute(table.insert(), values)  # type: ignore
                    conn.commit()
                    pbar.update(len(batch_df))

    def close(self):
        """关闭引擎"""
        self.engine.dispose()


# 全局实例
fetcher = TytDataUtils()
doris_fetcher = DorisDBUtils()


if __name__ == '__main__':
    # 测试样例
    try:
        trading_dt = fetcher.get_trading_dt('2023-01-01', '2023-02-01')
        print(f"交易日数量: {len(trading_dt)}")

        target_date = pd.to_datetime("2023-01-25")
        pos = int(trading_dt.searchsorted(target_date, side='left'))
        nearest_dt = trading_dt[pos]
        print(f"最近交易日: {nearest_dt}")

    except Exception as e:
        print(f"执行出错: {e}")
    finally:
        fetcher.close()
        doris_fetcher.close()