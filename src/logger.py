"""
日志配置模块
提供结构化的日志配置和获取方法
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.config import config


def setup_logger(name: str = 'rag_system',
                level: str = None,
                log_file: Path = None,
                console: bool = True) -> logging.Logger:
    """
    设置并返回一个配置好的日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        console: 是否输出到控制台

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    level = level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    formatter = logging.Formatter(config.LOG_FORMAT)

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file is None:
        log_file = config.LOG_FILE

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        Logger实例
    """
    return logging.getLogger(name)


# 预配置的日志记录器
logger = setup_logger('rag_system')
