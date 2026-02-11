"""
数据模块 - 统一数据加载与同步

架构：Sync-Load 分离
  - sync/: 独立运行的数据同步脚本（API → Parquet）
  - loader.py: 只读 Parquet，提供业务数据接口
  - decorators.py: 缓存装饰器（sync层使用）

使用方式：
    from data import get_loader
    loader = get_loader()
    prices = loader.get_industry_prices('2024-01-01', '2024-12-31')

扩展新数据表：
    1. 在 sync/ 下新建 sync_xxx.py，用 @parquet_cache 装饰
    2. 在 DataLoader 中新增对应的 get_xxx() 方法
    3. 在 sync_all.py 中注册同步步骤
"""
from data.loader import get_loader, DataLoader

__all__ = ['get_loader', 'DataLoader']