"""Logging configuration for AGR MCP server.

This module provides utilities for setting up structured logging
with appropriate formatting and handlers.
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_rich: Whether to use rich formatting for console output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if use_rich and sys.stderr.isatty():
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        console_handler.setLevel(numeric_level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(file_formatter)
        console_handler.setLevel(numeric_level)

    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Configure specific loggers

    # Reduce noise from HTTP client libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Set AGR MCP loggers to appropriate level
    logging.getLogger("agr_mcp").setLevel(numeric_level)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured at {level} level")
    if log_file:
        logger.debug(f"Logging to file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
