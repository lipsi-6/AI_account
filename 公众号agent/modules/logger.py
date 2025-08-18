"""
统一日志服务模块

严格遵循无副作用导入原则：
- 模块导入时绝不创建任何日志文件或目录
- 所有日志配置必须通过 setup_logging() 函数显式完成
- 提供结构化日志支持，便于后续分析和监控
"""

import logging
import logging.handlers
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

from .config_manager import ConfigManager


class StructuredFormatter(logging.Formatter):
    """
    结构化日志格式化器
    
    当 python-json-logger 不可用时的备用方案
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为结构化格式"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 尽可能合并 LoggerAdapter 注入的上下文（避免丢失 extra）
        try:
            for key, value in record.__dict__.items():
                if key in (
                    'name','msg','args','levelname','levelno','pathname','filename','module','exc_info',
                    'exc_text','stack_info','lineno','funcName','created','msecs','relativeCreated','thread',
                    'threadName','processName','process','message','asctime'
                ):
                    continue
                # 仅序列化基础类型，复杂类型转字符串
                try:
                    json.dumps(value)
                    log_data[key] = value
                except Exception:
                    log_data[key] = str(value)
        except Exception:
            pass

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class DeepScholarLoggerAdapter(logging.LoggerAdapter):
    """
    Deep Scholar AI 专用日志适配器
    
    为所有日志消息添加上下文信息
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """处理日志消息，添加上下文"""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # 添加系统上下文
        kwargs['extra'].update({
            'component': 'deep_scholar_ai',
            'version': '1.0.0'
        })
        
        # 合并适配器的额外信息
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs


async def setup_logging(config: ConfigManager) -> None:
    """
    异步设置日志系统
    
    Args:
        config: 配置管理器实例
        
    Raises:
        ValueError: 配置无效
        OSError: 无法创建日志文件或目录
    """
    if not config.is_initialized():
        raise ValueError("ConfigManager must be initialized before setting up logging")
    
    if not config.logging:
        raise ValueError("Logging configuration is missing")
    
    try:
        # 确保日志目录存在
        log_path = Path(config.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取根日志器
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # 清除默认处理器
        
        # 设置日志级别
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # 创建文件处理器（支持轮转）
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.logging.file_path,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # 配置格式化器
        if config.logging.enable_structured_logging:
            if JSON_LOGGER_AVAILABLE:
                # 使用 python-json-logger
                json_formatter = jsonlogger.JsonFormatter(
                    fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(json_formatter)
            else:
                # 使用自定义结构化格式化器
                structured_formatter = StructuredFormatter()
                file_handler.setFormatter(structured_formatter)
            
            # 控制台使用简单格式
            console_formatter = logging.Formatter(
                fmt=config.logging.format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        else:
            # 使用传统格式
            formatter = logging.Formatter(
                fmt=config.logging.format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
        
        # 添加处理器到根日志器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 配置第三方库的日志级别
        await _configure_third_party_loggers()
        
        # 记录日志系统启动
        logger = get_logger('logger_setup')
        logger.info("Logging system initialized", extra={
            'log_file': config.logging.file_path,
            'log_level': config.logging.level,
            'structured_logging': config.logging.enable_structured_logging,
            'json_logger_available': JSON_LOGGER_AVAILABLE
        })
        
    except Exception as e:
        # 如果日志设置失败，至少确保有基本的控制台输出
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logger = logging.getLogger('logger_setup')
        logger.error(f"Failed to setup logging: {e}")
        raise


async def _configure_third_party_loggers() -> None:
    """配置第三方库的日志级别"""
    third_party_configs = {
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'aiohttp': logging.INFO,
        'chromadb': logging.WARNING,
        'openai': logging.WARNING,
        'anthropic': logging.WARNING,
        'google': logging.WARNING,
        'arxiv': logging.WARNING,
        'PyMuPDF': logging.WARNING,
        'transformers': logging.WARNING,
        'torch': logging.WARNING,
        'tensorflow': logging.WARNING
    }
    
    for logger_name, level in third_party_configs.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def get_logger(name: str, extra_context: Optional[Dict[str, Any]] = None) -> DeepScholarLoggerAdapter:
    """
    获取Deep Scholar AI专用日志器
    
    Args:
        name: 日志器名称
        extra_context: 额外的上下文信息
        
    Returns:
        配置好的日志适配器
    """
    logger = logging.getLogger(name)
    return DeepScholarLoggerAdapter(logger, extra_context)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    记录函数调用
    
    Args:
        func_name: 函数名称
        **kwargs: 函数参数
    """
    logger = get_logger('function_calls')
    logger.debug(f"Function called: {func_name}", extra={
        'function': func_name,
        'arguments': {k: str(v) for k, v in kwargs.items()}
    })


def log_performance(operation: str, duration: float, **metrics) -> None:
    """
    记录性能指标（同步方法，避免未await导致的协程泄漏）。
    
    Args:
        operation: 操作名称
        duration: 持续时间（秒）
        **metrics: 其他性能指标
    """
    logger = get_logger('performance')
    logger.info(
        f"Performance: {operation}",
        extra={
            'operation': operation,
            'duration_seconds': duration,
            'metrics': metrics,
        },
    )


def log_api_call(
    provider: str,
    model: str,
    tokens_used: Optional[int] = None,
    cost: Optional[float] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> None:
    """
    记录API调用
    
    Args:
        provider: API提供商
        model: 模型名称
        tokens_used: 使用的token数量
        cost: 调用成本
        success: 是否成功
        error_message: 错误消息
    """
    logger = get_logger('api_calls')
    log_data = {
        'provider': provider,
        'model': model,
        'success': success
    }
    
    if tokens_used is not None:
        log_data['tokens_used'] = tokens_used
    if cost is not None:
        log_data['cost_usd'] = cost
    if error_message:
        log_data['error'] = error_message
    
    if success:
        logger.info(f"API call successful: {provider}/{model}", extra=log_data)
    else:
        logger.error(f"API call failed: {provider}/{model}", extra=log_data)


def log_paper_processing(
    paper_id: str,
    stage: str,
    success: bool = True,
    error_message: Optional[str] = None,
    **metrics
) -> None:
    """
    记录论文处理过程
    
    Args:
        paper_id: 论文ID
        stage: 处理阶段
        success: 是否成功
        error_message: 错误消息
        **metrics: 处理指标
    """
    logger = get_logger('paper_processing')
    log_data = {
        'paper_id': paper_id,
        'stage': stage,
        'success': success,
        **metrics
    }
    
    if error_message:
        log_data['error'] = error_message
    
    if success:
        logger.info(f"Paper processing: {stage} completed for {paper_id}", extra=log_data)
    else:
        logger.error(f"Paper processing: {stage} failed for {paper_id}", extra=log_data)


def log_memory_operation(
    operation: str,
    memory_type: str,
    success: bool = True,
    error_message: Optional[str] = None,
    **details
) -> None:
    """
    记录记忆系统操作
    
    Args:
        operation: 操作类型（存储、检索、更新等）
        memory_type: 记忆类型（向量、图谱等）
        success: 是否成功
        error_message: 错误消息
        **details: 操作详情
    """
    logger = get_logger('memory_system')
    log_data = {
        'operation': operation,
        'memory_type': memory_type,
        'success': success,
        **details
    }
    
    if error_message:
        log_data['error'] = error_message
    
    if success:
        logger.info(f"Memory operation: {operation} on {memory_type}", extra=log_data)
    else:
        logger.error(f"Memory operation failed: {operation} on {memory_type}", extra=log_data)


async def log_synthesis_operation(
    operation: str,
    article_id: str,
    success: bool = True,
    error_message: Optional[str] = None,
    **details
) -> None:
    """
    记录文章合成操作
    
    Args:
        operation: 操作类型（生成、保存、格式化等）
        article_id: 文章ID
        success: 是否成功
        error_message: 错误消息
        **details: 操作详情
    """
    logger = get_logger('synthesis_engine')
    log_data = {
        'operation': operation,
        'article_id': article_id,
        'success': success,
        **details
    }
    
    if error_message:
        log_data['error'] = error_message
    
    if success:
        logger.info(f"Synthesis operation: {operation} for article {article_id}", extra=log_data)
    else:
        logger.error(f"Synthesis operation failed: {operation} for article {article_id}", extra=log_data)


class LoggingContextManager:
    """
    日志上下文管理器
    
    用于在特定操作期间添加上下文信息
    """
    
    def __init__(self, logger_name: str, context: Dict[str, Any]):
        self.logger_name = logger_name
        self.context = context
        self.logger: Optional[DeepScholarLoggerAdapter] = None
    
    def __enter__(self) -> DeepScholarLoggerAdapter:
        self.logger = get_logger(self.logger_name, self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.logger:
            self.logger.error(f"Exception in context: {exc_type.__name__}: {exc_val}")
        return False
