import logging
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_file: str = 'app.log',
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    创建并配置 Logger。
    同名 Logger 重复调用时会清空旧 Handler，避免重复输出。
    """
    logger = logging.getLogger(name or __name__)
    logger.propagate = False
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if file:
        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger