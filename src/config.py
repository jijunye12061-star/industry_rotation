import logging
from logging.handlers import RotatingFileHandler


# 设置变量区域
# config.py

# 中信一级行业配置（29个行业）
INDUSTRY_CONFIG = {
    'CI005001': {'name': '石油石化', 'inner_code': '1000414414'},
    'CI005002': {'name': '煤炭', 'inner_code': '1000414415'},
    'CI005003': {'name': '有色金属', 'inner_code': '1000414416'},
    'CI005004': {'name': '电力及公用事业', 'inner_code': '1000414417'},
    'CI005005': {'name': '钢铁', 'inner_code': '1000414418'},
    'CI005006': {'name': '基础化工', 'inner_code': '1000414419'},
    'CI005007': {'name': '建筑', 'inner_code': '1000414420'},
    'CI005008': {'name': '建材', 'inner_code': '1000414421'},
    'CI005009': {'name': '轻工制造', 'inner_code': '1000414422'},
    'CI005010': {'name': '机械', 'inner_code': '1000414423'},
    'CI005011': {'name': '电力设备及新能源', 'inner_code': '1000414424'},
    'CI005012': {'name': '国防军工', 'inner_code': '1000414425'},
    'CI005013': {'name': '汽车', 'inner_code': '1000414426'},
    'CI005014': {'name': '商贸零售', 'inner_code': '1000414427'},
    'CI005015': {'name': '消费者服务', 'inner_code': '1000414428'},
    'CI005016': {'name': '家电', 'inner_code': '1000414429'},
    'CI005017': {'name': '纺织服装', 'inner_code': '1000414430'},
    'CI005018': {'name': '医药', 'inner_code': '1000414431'},
    'CI005019': {'name': '食品饮料', 'inner_code': '1000414432'},
    'CI005020': {'name': '农林牧渔', 'inner_code': '1000414433'},
    'CI005021': {'name': '银行', 'inner_code': '1000414434'},
    'CI005022': {'name': '非银行金融', 'inner_code': '1000414435'},
    'CI005023': {'name': '房地产', 'inner_code': '1000414436'},
    'CI005024': {'name': '交通运输', 'inner_code': '1000414437'},
    'CI005025': {'name': '电子', 'inner_code': '1000414438'},
    'CI005026': {'name': '通信', 'inner_code': '1000414439'},
    'CI005027': {'name': '计算机', 'inner_code': '1000414440'},
    'CI005028': {'name': '传媒', 'inner_code': '1000414441'},
    'CI005029': {'name': '综合', 'inner_code': '1000414442'},
    # 'CI005030': {'name': '综合金融', 'inner_code': '1002118426'},
}

# 便捷访问函数
def get_industry_codes():
    """获取所有行业代码列表"""
    return list(INDUSTRY_CONFIG.keys())

def get_inner_codes():
    """获取所有内码列表"""
    return [v['inner_code'] for v in INDUSTRY_CONFIG.values()]

def get_industry_name(code):
    """根据代码获取行业名称"""
    return INDUSTRY_CONFIG.get(code, {}).get('name', '')

def get_inner_code(code):
    """根据代码获取内码"""
    return INDUSTRY_CONFIG.get(code, {}).get('inner_code', '')

# 变量设置完成


def setup_logger(name: str = None,
                 level: int = logging.INFO,
                 log_file: str = 'app.log',
                 if_console: bool = True,
                 if_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    logger.propagate = False  # 添加这行
    # 清除已存在的处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if if_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if if_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
