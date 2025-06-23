"""Logging configuration for AGR MCP Server.

This module provides centralized logging configuration with support for
request ID tracking, log rotation, and multiple output formats.
"""

import logging
import os
import sys
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Optional, Generator
import uuid

# Defensive import to handle potential configuration errors
try:
    from ..config import Config
except Exception as e:
    # If Config import fails, use defaults to prevent import-time crashes
    import warnings
    warnings.warn(f"Failed to import Config, using defaults: {e}")

    class Config:
        """Fallback configuration with safe defaults."""
        LOG_LEVEL = 'INFO'
        LOG_DIR = './logs'
        LOG_FILE_MAX_SIZE = 100 * 1024 * 1024  # 100MB
        LOG_FILE_BACKUP_COUNT = 5


class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records.

    This filter adds a unique request ID to each log record, allowing
    correlation of log entries across a single request or operation.
    """

    def __init__(self, request_id: Optional[str] = None):
        """Initialize the filter with a request ID.

        Args:
            request_id: Optional request ID. If not provided, generates a new UUID.
        """
        super().__init__()
        self.request_id = request_id or str(uuid.uuid4())

    def filter(self, record):
        """Add request_id to the log record.

        Args:
            record: LogRecord to filter

        Returns:
            True (always allows the record through)
        """
        record.request_id = self.request_id
        return True


def setup_logging(logger_name: str = 'agr_mcp') -> logging.Logger:
    """Configure logging for the AGR MCP server.

    Sets up logging with console and file handlers, formatters, and
    request ID tracking. Ensures log directory exists and configures
    log rotation.

    Args:
        logger_name: Name for the logger (default: 'agr_mcp')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Set log level from config
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent propagation to root logger
    logger.propagate = False

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(filename)s:%(lineno)d - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Add request ID filter to console handler
    console_handler.addFilter(RequestIdFilter())

    # Create log directory if it doesn't exist
    log_dir = Config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # Create file handler with rotation
    log_file = os.path.join(log_dir, 'agr_mcp.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=Config.LOG_FILE_MAX_SIZE,
        backupCount=Config.LOG_FILE_BACKUP_COUNT
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Add request ID filter to file handler
    file_handler.addFilter(RequestIdFilter())

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Creates a logger with the name prefixed with 'agr_mcp.' to maintain
    a consistent naming hierarchy. Ensures the parent logger is properly
    configured.

    Args:
        name: Module or component name

    Returns:
        Logger instance for the specified module

    Example:
        logger = get_logger('api.search')
        logger.info('Processing search request')
    """
    # Ensure the parent logger is set up
    setup_logging()

    # Return child logger with consistent naming
    return logging.getLogger(f'agr_mcp.{name}')


@contextmanager
def with_request_id(request_id: Optional[str] = None) -> Generator[str, None, None]:
    """Context manager for associating logs with a specific request ID.

    This context manager temporarily sets a request ID for all logs
    within its scope, allowing correlation of log entries for a single
    request or operation.

    Args:
        request_id: Optional request ID. If not provided, generates a new UUID.

    Yields:
        The request ID being used

    Example:
        with with_request_id('req-123') as rid:
            logger.info('Processing request')  # Will include [req-123] in log
            # ... do work ...
            logger.info('Request complete')    # Will include [req-123] in log
    """
    # Generate request ID if not provided
    rid = request_id or str(uuid.uuid4())

    # Get all handlers from all loggers
    all_handlers = []
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if logger_name.startswith('agr_mcp'):
            all_handlers.extend(logger.handlers)

    # Also include root agr_mcp logger handlers
    root_logger = logging.getLogger('agr_mcp')
    all_handlers.extend(root_logger.handlers)

    # Remove duplicates
    all_handlers = list(set(all_handlers))

    # Store old filters and add new ones
    old_filters = {}
    new_filter = RequestIdFilter(rid)

    for handler in all_handlers:
        # Store existing filters
        old_filters[handler] = handler.filters.copy() if hasattr(handler, 'filters') else []

        # Remove any existing RequestIdFilter
        for f in handler.filters[:]:
            if isinstance(f, RequestIdFilter):
                handler.removeFilter(f)

        # Add new filter
        handler.addFilter(new_filter)

    try:
        yield rid
    finally:
        # Restore original filters
        for handler, filters in old_filters.items():
            # Remove our filter
            handler.removeFilter(new_filter)

            # Restore original filters
            for f in filters:
                if isinstance(f, RequestIdFilter):
                    handler.addFilter(f)
