#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: cache_manager.py
@description: 因子缓存管理器
"""
import pandas as pd
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FactorCache:
    """因子缓存管理器"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            # 默认在项目根目录下
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "factor_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _get_cache_key(config, params: dict) -> str:
        """生成缓存键（基于配置和参数）"""
        key_dict = {
            'start_date': config.start_date,
            'end_date': config.end_date,
            'frequency': config.frequency,
            'params': params
        }
        # 生成hash
        key_str = json.dumps(key_dict, sort_keys=True)
        hash_key = hashlib.md5(key_str.encode()).hexdigest()[:8]

        # 文件名：日期范围_频率_hash
        filename = f"{config.start_date.replace('-', '')}_{config.end_date.replace('-', '')}_{config.frequency}_{hash_key}"
        return filename

    def get(self, factor_name: str, config, params: dict) -> Optional[pd.DataFrame]:
        """读取缓存"""
        cache_key = self._get_cache_key(config, params)
        cache_path = self.cache_dir / factor_name / f"{cache_key}.parquet"

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"从缓存加载因子: {factor_name} ({cache_key})")
            return df
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
            return None

    def set(self, factor_name: str, config, params: dict, data: pd.DataFrame):
        """写入缓存"""
        cache_key = self._get_cache_key(config, params)
        factor_dir = self.cache_dir / factor_name
        factor_dir.mkdir(exist_ok=True)

        cache_path = factor_dir / f"{cache_key}.parquet"

        # 保存数据
        data.to_parquet(cache_path)

        # 保存元数据
        metadata = {
            'factor_name': factor_name,
            'config': {
                'start_date': config.start_date,
                'end_date': config.end_date,
                'frequency': config.frequency,
                'market_code': config.market_code
            },
            'params': params,
            'created_at': datetime.now().isoformat(),
            'shape': data.shape
        }

        meta_path = factor_dir / f"{cache_key}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)  # type: ignore

        logger.info(f"因子已缓存: {factor_name} ({cache_key})")

    def clear(self, factor_name: Optional[str] = None):
        """清除缓存"""
        if factor_name:
            factor_dir = self.cache_dir / factor_name
            if factor_dir.exists():
                import shutil
                shutil.rmtree(factor_dir)
                logger.info(f"已清除因子缓存: {factor_name}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            logger.info("已清除所有因子缓存")

    def list_cached_factors(self) -> list:
        """列出已缓存的因子"""
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]


# 全局缓存实例
_cache_instance = None


def get_cache() -> FactorCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = FactorCache()
    return _cache_instance