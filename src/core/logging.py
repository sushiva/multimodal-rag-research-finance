"""
Structured Logging Configuration

What: Centralized logging setup with structured (JSON) output
Why: Machine-readable logs for aggregation, searching, and analysis
How: Uses structlog for structured logging with context propagation

Features:
- JSON output for production (easy to parse)
- Colored console output for development
- Request ID tracking across services
- Automatic timestamp and log level
- Context variables (user_id, request_id, etc.)

Usage:
    from src.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("User logged in", user_id="123", ip_address="192.168.1.1")
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import FilteringBoundLogger

from src.core.config import get_settings


def configure_logging() -> None:
    """
    Configure structured logging for the application.

    Sets up structlog with appropriate processors based on environment:
    - Development: Colored console output
    - Production: JSON output to stdout

    Called once at application startup.
    """
    settings = get_settings()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_development:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            )
        ]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log_level = getattr(logging, settings.log_level)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of calling module)

    Returns:
        Structured logger with bound context

    Example:
        logger = get_logger(__name__)
        logger.info("Processing request", request_id="abc123")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to all subsequent log statements.

    Useful for request ID, user ID, session ID that should appear in all logs.

    Args:
        **kwargs: Context key-value pairs

    Example:
        bind_context(request_id="abc123", user_id="user_456")
        logger.info("User action")
        Output: {"request_id": "abc123", "user_id": "user_456", "event": "User action"}
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Remove context variables from subsequent log statements.

    Args:
        *keys: Context keys to remove

    Example:
        unbind_context("request_id", "user_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """
    Clear all context variables.

    Useful at the end of request processing to avoid context leakage.
    """
    structlog.contextvars.clear_contextvars()


class LoggerAdapter:
    """
    Adapter for standard Python logging to structured logging.

    Allows gradual migration from logging.getLogger to structured logging.

    Usage:
        logger = LoggerAdapter(__name__)
        logger.info("Message", extra={"user_id": "123"})
    """

    def __init__(self, name: str) -> None:
        """
        Initialize logger adapter.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self._name = name

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with structured data."""
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message with structured data."""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with structured data."""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message with structured data."""
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message with structured data."""
        self.logger.critical(msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg, **kwargs)


def log_function_call(logger: FilteringBoundLogger) -> Any:
    """
    Decorator to log function entry/exit.

    Args:
        logger: Structured logger instance

    Returns:
        Decorator function

    Example:
        @log_function_call(get_logger(__name__))
        def process_data(data: dict) -> dict:
            return {"result": "success"}
    """
    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.debug(
                "Function called",
                function=func_name,
                args=args,
                kwargs=kwargs,
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    "Function completed",
                    function=func_name,
                )
                return result
            except Exception as e:
                logger.error(
                    "Function failed",
                    function=func_name,
                    error=str(e),
                    exc_info=True,
                )
                raise

        return wrapper
    return decorator
