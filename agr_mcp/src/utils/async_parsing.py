"""
Asynchronous parsing utilities for AGR MCP with thread pool execution.

This module provides async wrappers for CPU-intensive parsing operations,
moving them to a thread pool to avoid blocking the main event loop.
"""

import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class AsyncParsingPool:
    """Thread pool for CPU-intensive parsing operations.

    This singleton class manages a thread pool executor for parsing operations
    that are CPU-intensive and would otherwise block the async event loop.
    """

    _instance: Optional['AsyncParsingPool'] = None
    _lock = threading.Lock()

    def __new__(cls, max_workers: Optional[int] = None) -> 'AsyncParsingPool':
        """Ensure only one instance exists (thread-safe singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize the parsing pool.

        Args:
            max_workers: Maximum number of worker threads. If None,
                        defaults to min(32, (cpu_count() or 1) + 4)
        """
        # Prevent reinitialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers = max_workers
        self._initialized = False

        logger.info("AsyncParsingPool singleton instance created")

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Ensure the thread pool executor is initialized."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="async_parsing"
            )
            self._initialized = True

            logger.info(
                "Thread pool executor initialized for parsing operations",
                extra={
                    "max_workers": self._executor._max_workers,
                    "thread_name_prefix": "async_parsing"
                }
            )

        return self._executor

    async def run_in_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run a function in the thread pool.

        Args:
            func: Function to execute in thread pool
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution

        Raises:
            Any exception raised by the function
        """
        executor = self._ensure_executor()
        loop = asyncio.get_event_loop()

        # Use functools.partial for better error handling
        partial_func = functools.partial(func, *args, **kwargs)

        try:
            result = await loop.run_in_executor(executor, partial_func)
            return result
        except Exception as e:
            logger.error(
                f"Error in thread pool execution: {e}",
                extra={"function": func.__name__, "error": str(e)}
            )
            raise

    def close(self) -> None:
        """Close the thread pool executor and clean up resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._initialized = False
            logger.info("AsyncParsingPool thread pool executor shut down")

    @classmethod
    def get_instance(cls, max_workers: Optional[int] = None) -> 'AsyncParsingPool':
        """Get the singleton instance."""
        return cls(max_workers)


# Global parsing pool instance
_parsing_pool: Optional[AsyncParsingPool] = None


async def get_parsing_pool(max_workers: Optional[int] = None) -> AsyncParsingPool:
    """Get the global parsing pool instance.

    Args:
        max_workers: Maximum number of worker threads

    Returns:
        AsyncParsingPool singleton instance
    """
    global _parsing_pool
    if _parsing_pool is None:
        _parsing_pool = AsyncParsingPool(max_workers)
    return _parsing_pool


def _parse_html_sync(html_content: str, parser: str = 'html.parser') -> BeautifulSoup:
    """Synchronous HTML parsing function for thread execution.

    Args:
        html_content: HTML content to parse
        parser: Parser to use ('html.parser', 'lxml', 'html5lib')

    Returns:
        BeautifulSoup object
    """
    return BeautifulSoup(html_content, parser)


def _parse_json_sync(json_content: str) -> Union[Dict[str, Any], List[Any]]:
    """Synchronous JSON parsing function for thread execution.

    Args:
        json_content: JSON content to parse

    Returns:
        Parsed JSON data

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    return json.loads(json_content)


def _parse_xml_sync(xml_content: str) -> ET.Element:
    """Synchronous XML parsing function for thread execution.

    Args:
        xml_content: XML content to parse

    Returns:
        XML Element tree root

    Raises:
        ET.ParseError: If XML parsing fails
    """
    return ET.fromstring(xml_content)


async def parse_html_async(
    html_content: str,
    parser: str = 'html.parser',
    max_workers: Optional[int] = None
) -> BeautifulSoup:
    """Parse HTML content asynchronously using thread pool.

    Args:
        html_content: HTML content to parse
        parser: Parser to use ('html.parser', 'lxml', 'html5lib')
        max_workers: Maximum number of worker threads

    Returns:
        BeautifulSoup object

    Raises:
        Exception: Any parsing errors from BeautifulSoup
    """
    pool = await get_parsing_pool(max_workers)

    logger.debug(
        f"Parsing HTML content asynchronously",
        extra={"content_size": len(html_content), "parser": parser}
    )

    return await pool.run_in_thread(_parse_html_sync, html_content, parser)


async def parse_json_async(
    json_content: str,
    max_workers: Optional[int] = None
) -> Union[Dict[str, Any], List[Any]]:
    """Parse JSON content asynchronously using thread pool.

    Args:
        json_content: JSON content to parse
        max_workers: Maximum number of worker threads

    Returns:
        Parsed JSON data

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    pool = await get_parsing_pool(max_workers)

    logger.debug(
        f"Parsing JSON content asynchronously",
        extra={"content_size": len(json_content)}
    )

    return await pool.run_in_thread(_parse_json_sync, json_content)


async def parse_xml_async(
    xml_content: str,
    max_workers: Optional[int] = None
) -> ET.Element:
    """Parse XML content asynchronously using thread pool.

    Args:
        xml_content: XML content to parse
        max_workers: Maximum number of worker threads

    Returns:
        XML Element tree root

    Raises:
        ET.ParseError: If XML parsing fails
    """
    pool = await get_parsing_pool(max_workers)

    logger.debug(
        f"Parsing XML content asynchronously",
        extra={"content_size": len(xml_content)}
    )

    return await pool.run_in_thread(_parse_xml_sync, xml_content)


async def run_cpu_intensive_task(
    func: Callable[..., T],
    *args,
    max_workers: Optional[int] = None,
    **kwargs
) -> T:
    """Run any CPU-intensive task asynchronously using thread pool.

    This is a general-purpose function for moving CPU-intensive operations
    to a thread pool to avoid blocking the async event loop.

    Args:
        func: Function to execute in thread pool
        *args: Positional arguments for the function
        max_workers: Maximum number of worker threads
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function execution

    Raises:
        Any exception raised by the function
    """
    pool = await get_parsing_pool(max_workers)

    logger.debug(
        f"Running CPU-intensive task asynchronously",
        extra={"function": func.__name__}
    )

    return await pool.run_in_thread(func, *args, **kwargs)


async def close_parsing_pool() -> None:
    """Close the global parsing pool and clean up resources.

    This should be called during application shutdown to ensure
    proper cleanup of thread pool resources.
    """
    global _parsing_pool
    if _parsing_pool is not None:
        _parsing_pool.close()
        _parsing_pool = None
        logger.info("Global parsing pool closed and cleaned up")
