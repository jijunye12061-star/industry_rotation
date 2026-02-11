"""
数据同步脚本 - 从 Choice API 获取数据并写入 Parquet

独立运行，不被业务代码直接调用。

使用方式：
    python -m data.sync.sync_all --init       # 全量初始化
    python -m data.sync.sync_all              # 增量更新

扩展新表：
    1. 新建 sync_xxx.py
    2. 定义 @parquet_cache 装饰的内部函数 + 公开 sync_xxx() 函数
    3. 在 sync_all.py 的 SYNC_STEPS 中注册
"""