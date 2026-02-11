#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETF筛选器 - 分步筛选与行业映射"""
import pandas as pd
from typing import Dict, List
import logging

from strategy.etf_loader import ETFLoader
from src.config import INDUSTRY_CONFIG

logger = logging.getLogger(__name__)

# 12位代码前6位 -> 8位中信一级代码
INDUSTRY_CODE_MAPPING = {
    '025001': 'CI005001',  # 石油石化
    '025002': 'CI005002',  # 煤炭
    '025003': 'CI005003',  # 有色金属
    '025004': 'CI005004',  # 电力及公用事业
    '025005': 'CI005005',  # 钢铁
    '025006': 'CI005006',  # 基础化工
    '025007': 'CI005007',  # 建筑
    '025008': 'CI005008',  # 建材
    '025009': 'CI005009',  # 轻工制造
    '025010': 'CI005010',  # 机械
    '025011': 'CI005011',  # 电力设备及新能源
    '025012': 'CI005012',  # 国防军工
    '025013': 'CI005013',  # 汽车
    '025014': 'CI005014',  # 商贸零售
    '025015': 'CI005015',  # 消费者服务
    '025016': 'CI005016',  # 家电
    '025017': 'CI005017',  # 纺织服装
    '025018': 'CI005018',  # 医药
    '025019': 'CI005019',  # 食品饮料
    '025020': 'CI005020',  # 农林牧渔
    '025021': 'CI005021',  # 银行
    '025022': 'CI005022',  # 非银行金融
    '025023': 'CI005023',  # 房地产
    '025024': 'CI005030',  # 综合金融
    '025025': 'CI005024',  # 交通运输
    '025026': 'CI005025',  # 电子
    '025027': 'CI005026',  # 通信
    '025028': 'CI005027',  # 计算机
    '025029': 'CI005028',  # 传媒
    '025030': 'CI005029',  # 综合
}

exclude_keywords = ['科创板50']  # 临时做法

class ETFSelector:
    """ETF筛选器"""

    def __init__(self,
                 loader: ETFLoader,
                 min_scale: float = 5e8,
                 min_primary_weight: float = 0.5,
                 max_tertiary_ratio: float = 0.8):
        """
        Args:
            min_scale: 最小规模（元）
            min_primary_weight: 第一大一级行业最小权重
            max_tertiary_ratio: 第一大三级占一级的最大比例
        """
        self.loader = loader
        self.min_scale = min_scale
        self.min_primary_weight = min_primary_weight
        self.max_tertiary_ratio = max_tertiary_ratio

    def select_etfs(self, date: str) -> Dict[str, str]:
        """
        筛选合格ETF并映射到行业

        Returns:
            {CI005001: 'etf_code', ...}
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"筛选 {date} 的ETF")

        # 1. 时间筛选
        candidates = self._filter_by_time(date)
        logger.info(f"  [1/4] 时间筛选: {len(candidates)} 个")

        # 2. 规模筛选
        candidates = self._filter_by_scale(candidates, date)
        logger.info(f"  [2/4] 规模筛选: {len(candidates)} 个")

        if len(candidates) == 0:
            logger.warning("  无合格ETF")
            return {}

        # 3. 集中度筛选
        qualified = self._filter_by_concentration(candidates, date)
        logger.info(f"  [3/4] 集中度筛选: {len(qualified)} 个")

        if qualified.empty:
            logger.warning("  无符合集中度要求的ETF")
            return {}

        # 4. 每个行业选规模最大的
        mapping = self._map_to_industries(qualified, date)
        logger.info(f"  [4/4] 行业映射: {len(mapping)} 个")
        logger.info(f"{'=' * 60}\n")

        return mapping

    def _filter_by_time(self, date: str) -> List[str]:
        """筛选已成立且未清盘的ETF"""
        date = pd.to_datetime(date)
        info = self.loader.get_etf_info()

        valid = info[
            (info['c_estabdate'] <= date) &
            (info['c_terminate_date'].isna() | (info['c_terminate_date'] > date))
            ]

        # 排除包含特定关键词的ETF
        for keyword in exclude_keywords:
            valid = valid[~valid['c_short_name'].str.contains(keyword, na=False)]

        return valid['c_fd_code'].tolist()

    def _filter_by_scale(self, candidates: List[str], date: str) -> List[str]:
        """筛选规模达标的ETF"""
        scales = self.loader.get_etf_scales(date)
        qualified = scales[scales >= self.min_scale]
        return [code for code in candidates if code in qualified.index]

    def _filter_by_concentration(self, candidates: List[str], date: str) -> pd.DataFrame:
        """筛选集中度达标的ETF（返回带权重信息的DataFrame）"""
        holdings = self.loader.get_all_holdings(date)
        holdings = holdings[holdings['c_fd_code'].isin(candidates)]

        if holdings.empty:
            return pd.DataFrame()

        weights = self.loader.calculate_industry_weights(holdings)

        qualified = weights[
            (weights['primary_weight'] >= self.min_primary_weight) &
            (weights['tertiary_in_primary_ratio'] <= self.max_tertiary_ratio)
            ]

        return qualified

    def _map_to_industries(self, qualified: pd.DataFrame, date: str) -> Dict[str, str]:
        """每个一级行业选规模最大的ETF，映射到8位CI码"""
        scales = self.loader.get_etf_scales(date)
        qualified = qualified.merge(
            scales.rename('scale'),
            left_on='c_fd_code',
            right_index=True
        )

        # 按一级行业分组，取规模最大的
        idx = qualified.groupby('primary_industry')['scale'].idxmax()
        selected = qualified.loc[idx]

        mapping = {}
        for _, row in selected.iterrows():
            industry_6 = row['primary_industry']
            if industry_6 in INDUSTRY_CODE_MAPPING:
                ci_code = INDUSTRY_CODE_MAPPING[industry_6]
                mapping[ci_code] = row['c_fd_code']
                logger.info(
                    f"  {ci_code} -> {row['c_fd_code']} "
                    f"(一级={row['primary_weight']:.1%}, 规模={row['scale'] / 1e8:.1f}亿)"
                )

        return mapping

    def display_selection_results(self, mapping: Dict[str, str], date: str):
        """展示ETF选择结果与未匹配行业"""
        # 获取ETF简称
        etf_info = self.loader.get_etf_info()
        etf_names = etf_info.set_index('c_fd_code')['c_short_name'].to_dict()  # 假设简称字段是c_fd_abbr

        print(f"\n{'=' * 70}")
        print(f"ETF选择结果 ({date})")
        print(f"{'=' * 70}")
        print(f"{'行业代码':<12} {'行业名称':<15} {'ETF代码':<12} {'ETF简称':<20}")
        print(f"{'-' * 70}")

        for ci_code, etf_code in sorted(mapping.items()):
            industry_name = INDUSTRY_CONFIG.get(ci_code, {}).get('name', '未知')
            etf_name = etf_names.get(etf_code, '未知')
            print(f"{ci_code:<12} {industry_name:<15} {etf_code:<12} {etf_name:<20}")

        # 未匹配行业
        all_industries = set(INDUSTRY_CONFIG.keys())
        matched_industries = set(mapping.keys())
        unmatched = all_industries - matched_industries

        if unmatched:
            print(f"\n{'-' * 70}")
            print(f"未匹配行业 ({len(unmatched)}个):")
            print(f"{'-' * 70}")
            for ci_code in sorted(unmatched):
                industry_name = INDUSTRY_CONFIG[ci_code]['name']
                print(f"{ci_code:<12} {industry_name:<15}")

        print(f"{'=' * 70}\n")

    def batch_select_and_analyze(self, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        批量选择ETF并生成分析表

        Returns:
            (详情表, 统计表)
        """
        etf_info = self.loader.get_etf_info()
        etf_names = etf_info.set_index('c_fd_code')['c_short_name'].to_dict()

        details = []
        stats = []

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"处理 {date_str}")

            mapping = self.select_etfs(date_str)

            # 详情记录
            for ci_code, etf_code in mapping.items():
                details.append({
                    '行业代码': ci_code,
                    '行业名称': INDUSTRY_CONFIG.get(ci_code, {}).get('name', '未知'),
                    'ETF代码': etf_code,
                    'ETF简称': etf_names.get(etf_code, '未知'),
                    '截面日期': date
                })

            # 统计记录
            stats.append({
                '截面日期': date,
                '匹配行业数': len(mapping)
            })

        details_df = pd.DataFrame(details)
        stats_df = pd.DataFrame(stats)

        return details_df, stats_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    loader = ETFLoader()
    selector = ETFSelector(loader, min_scale=1e8)
    main_mapping = selector.select_etfs("2017-01-26")

    # from factors.factor_config import FactorConfig
    #
    # config = FactorConfig(
    #     start_date='2017-01-01',
    #     end_date='2025-12-31',
    #     frequency='monthly'
    # )
    #
    # details_df, stats_df = selector.batch_select_and_analyze(config.get_rebalance_dates())
    # a_df = details_df.drop_duplicates(subset=["ETF代码"])


