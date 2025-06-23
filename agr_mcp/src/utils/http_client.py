"""
HTTP client module for AGR MCP with retry logic and rate limiting.

This module provides an async HTTP client with the following features:
- Automatic retry logic with exponential backoff
- Rate limiting to respect API limits
- Request/response logging
- Error handling with custom exceptions
- Connection pooling for efficiency

Example:
    Basic usage::

        from agr_mcp.src.utils.http_client import http_client

        # Simple GET request
        response = await http_client.get("https://api.example.com/data")

        # POST with data
        response = await http_client.post(
            "https://api.example.com/users",
            json={"name": "John", "email": "john@example.com"}
        )

        # Custom timeout and retries
        response = await http_client.get(
            "https://api.example.com/slow-endpoint",
            timeout=30.0,
            max_retries=5
        )
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log,
)

# Import errors and logging according to supervisor guidance
from ..errors import (
    ConnectionError,
    TimeoutError,
    RateLimitError,
    ToolExecutionError,
)
from .logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float, burst: int):
        """
        Initialize rate limiter.

        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            while self.tokens < tokens:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now

                if self.tokens < tokens:
                    sleep_time = (tokens - self.tokens) / self.rate
                    await asyncio.sleep(sleep_time)

            self.tokens -= tokens


class AGRHttpClient:
    """Async HTTP client with retry logic and rate limiting."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit: Optional[float] = None,
        rate_burst: Optional[int] = None,
    ):
        """
        Initialize HTTP client.

        Args:
            base_url: Base URL for relative requests
            timeout: Default timeout in seconds
            max_retries: Maximum number of retries
            rate_limit: Requests per second (None for no limit)
            rate_burst: Maximum burst size
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize rate limiter if specified
        self.rate_limiter = None
        if rate_limit is not None:
            burst = rate_burst or int(rate_limit * 10)  # Default burst
            self.rate_limiter = RateLimiter(rate_limit, burst)

        # Create httpx client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
            ),
            follow_redirects=True,
        )

        logger.info(
            "Initialized AGRHttpClient",
            extra={
                "base_url": base_url,
                "timeout": timeout,
                "max_retries": max_retries,
                "rate_limit": rate_limit,
            }
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

    def _build_url(self, url: str) -> str:
        """Build full URL from relative or absolute URL."""
        if self.base_url and not url.startswith(("http://", "https://")):
            return urljoin(self.base_url, url)
        return url

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response object

        Raises:
            ConnectionError: On connection issues
            TimeoutError: On timeout
            RateLimitError: On rate limit exceeded
            ToolExecutionError: On other HTTP errors
        """
        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        try:
            response = await self.client.request(method, url, **kwargs)

            # Check for rate limit headers
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds",
                    retry_after=int(retry_after),
                )

            # Raise for other HTTP errors
            response.raise_for_status()

            return response

        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {url}")
            raise TimeoutError(f"Request to {url} timed out") from e

        except httpx.ConnectError as e:
            logger.error(f"Connection error: {url}")
            raise ConnectionError(f"Failed to connect to {url}") from e

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code}: {url}",
                extra={"status_code": e.response.status_code},
            )
            raise ToolExecutionError(
                f"HTTP {e.response.status_code} error from {url}",
                details={"status_code": e.response.status_code},
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error during request: {e}")
            raise ToolExecutionError(
                f"Unexpected error during request to {url}",
                details={"error": str(e)},
            ) from e

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], bytes, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request (relative or absolute)
            json: JSON data to send
            data: Form data or raw bytes to send
            params: Query parameters
            headers: Request headers
            timeout: Override default timeout
            max_retries: Override default max retries
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response object
        """
        full_url = self._build_url(url)

        # Override timeout if specified
        if timeout is not None:
            kwargs["timeout"] = timeout

        # Build request kwargs
        request_kwargs = {
            "json": json,
            "data": data,
            "params": params,
            "headers": headers,
            **kwargs,
        }

        # Remove None values
        request_kwargs = {k: v for k, v in request_kwargs.items() if v is not None}

        # Log request
        logger.debug(
            f"Making {method} request to {full_url}",
            extra={
                "method": method,
                "url": full_url,
                "has_json": json is not None,
                "has_data": data is not None,
                "params": params,
            }
        )

        # Make request with retries
        if max_retries is not None:
            # Custom retry decorator
            custom_retry = retry(
                retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                after=after_log(logger, logging.INFO),
            )
            response = await custom_retry(self._make_request)(method, full_url, **request_kwargs)
        else:
            response = await self._make_request(method, full_url, **request_kwargs)

        # Log response
        logger.debug(
            f"Received {response.status_code} response from {full_url}",
            extra={
                "status_code": response.status_code,
                "elapsed": response.elapsed.total_seconds(),
            }
        )

        return response

    # Convenience methods
    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make PATCH request."""
        return await self.request("PATCH", url, **kwargs)


# Create a default client instance
http_client = AGRHttpClient()
