"""Request deduplication cache for AGR MCP.

This module provides request deduplication using configurable caching backends
to prevent redundant API calls. Supports Redis and in-memory caching.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import asdict, dataclass

from ..module_config import ENABLE_CACHING, CACHE_TTL
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached request/response entry."""
    data: Any
    timestamp: float
    ttl: int

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > (self.timestamp + self.ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create cache entry from dictionary."""
        return cls(**data)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set cache entry by key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete cache entry by key."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close cache backend connection."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache backend using asyncio-safe dictionary."""

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries to store
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry and entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                return None
            return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set cache entry by key."""
        async with self._lock:
            # Remove oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                # Remove first entry (oldest)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = entry

    async def delete(self, key: str) -> None:
        """Delete cache entry by key."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def close(self) -> None:
        """Close cache backend (no-op for in-memory)."""
        await self.clear()


class RedisCache(CacheBackend):
    """Redis cache backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379",
                 key_prefix: str = "agr_mcp:"):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._redis = None
        self._initialized = False

    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established."""
        if self._initialized:
            return self._redis is not None

        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )

            # Test connection
            await self._redis.ping()
            self._initialized = True
            logger.info(f"Connected to Redis cache at {self._redis_url}")
            return True

        except ImportError:
            logger.warning("Redis library not available, falling back to in-memory cache")
            self._redis = None
            self._initialized = True
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._redis = None
            self._initialized = True
            return False

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        if not await self._ensure_connected():
            return None

        try:
            data = await self._redis.get(self._make_key(key))
            if data:
                cache_data = json.loads(data)
                entry = CacheEntry.from_dict(cache_data)
                if entry.is_expired():
                    await self.delete(key)
                    return None
                return entry
        except Exception as e:
            logger.warning(f"Redis get error: {e}")

        return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set cache entry by key."""
        if not await self._ensure_connected():
            return

        try:
            data = json.dumps(entry.to_dict())
            await self._redis.setex(
                self._make_key(key),
                entry.ttl,
                data
            )
        except Exception as e:
            logger.warning(f"Redis set error: {e}")

    async def delete(self, key: str) -> None:
        """Delete cache entry by key."""
        if not await self._ensure_connected():
            return

        try:
            await self._redis.delete(self._make_key(key))
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")

    async def clear(self) -> None:
        """Clear all cache entries with our prefix."""
        if not await self._ensure_connected():
            return

        try:
            keys = await self._redis.keys(f"{self._key_prefix}*")
            if keys:
                await self._redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")


class RequestCache:
    """Request deduplication cache with configurable backends."""

    def __init__(self,
                 backend: Optional[CacheBackend] = None,
                 default_ttl: int = 300,  # 5 minutes
                 enabled: bool = None):
        """Initialize request cache.

        Args:
            backend: Cache backend to use (auto-detected if None)
            default_ttl: Default TTL for cache entries in seconds
            enabled: Whether caching is enabled (uses module config if None)
        """
        self._backend = backend
        self._default_ttl = default_ttl
        self._enabled = enabled if enabled is not None else ENABLE_CACHING
        self._lock = asyncio.Lock()
        self._initialized = False

        if not self._enabled:
            logger.info("Request caching is disabled")

    async def _ensure_initialized(self) -> None:
        """Ensure cache backend is initialized."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if not self._enabled:
                self._backend = None
                self._initialized = True
                return

            if self._backend is None:
                # Try Redis first, fall back to in-memory
                redis_cache = RedisCache()
                if await redis_cache._ensure_connected():
                    self._backend = redis_cache
                    logger.info("Using Redis cache backend for request deduplication")
                else:
                    self._backend = InMemoryCache()
                    logger.info("Using in-memory cache backend for request deduplication")

            self._initialized = True

    def _generate_cache_key(self,
                           method: str,
                           url: str,
                           params: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, str]] = None) -> str:
        """Generate cache key for request."""
        # Create a deterministic hash of the request components
        key_data = {
            'method': method.upper(),
            'url': url,
            'params': sorted(params.items()) if params else None,
            'headers': sorted(headers.items()) if headers else None
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def get_cached_response(self,
                                 method: str,
                                 url: str,
                                 params: Optional[Dict[str, Any]] = None,
                                 headers: Optional[Dict[str, str]] = None) -> Optional[Any]:
        """Get cached response if available.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            headers: Request headers

        Returns:
            Cached response data if available, None otherwise
        """
        if not self._enabled:
            return None

        await self._ensure_initialized()

        if not self._backend:
            return None

        try:
            cache_key = self._generate_cache_key(method, url, params, headers)
            entry = await self._backend.get(cache_key)

            if entry:
                logger.debug(f"Cache hit for {method} {url}")
                return entry.data
            else:
                logger.debug(f"Cache miss for {method} {url}")
                return None

        except Exception as e:
            logger.warning(f"Error getting cached response: {e}")
            return None

    async def cache_response(self,
                            method: str,
                            url: str,
                            response_data: Any,
                            params: Optional[Dict[str, Any]] = None,
                            headers: Optional[Dict[str, str]] = None,
                            ttl: Optional[int] = None) -> None:
        """Cache response data.

        Args:
            method: HTTP method
            url: Request URL
            response_data: Response data to cache
            params: Query parameters
            headers: Request headers
            ttl: Time to live in seconds (uses default if None)
        """
        if not self._enabled:
            return

        await self._ensure_initialized()

        if not self._backend:
            return

        try:
            cache_key = self._generate_cache_key(method, url, params, headers)
            entry = CacheEntry(
                data=response_data,
                timestamp=time.time(),
                ttl=ttl or self._default_ttl
            )

            await self._backend.set(cache_key, entry)
            logger.debug(f"Cached response for {method} {url} (TTL: {entry.ttl}s)")

        except Exception as e:
            logger.warning(f"Error caching response: {e}")

    async def clear_cache(self) -> None:
        """Clear all cached entries."""
        if not self._enabled:
            return

        await self._ensure_initialized()

        if self._backend:
            try:
                await self._backend.clear()
                logger.info("Cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")

    async def close(self) -> None:
        """Close cache backend."""
        if self._backend:
            try:
                await self._backend.close()
            except Exception as e:
                logger.warning(f"Error closing cache: {e}")


# Global request cache instance
_global_cache: Optional[RequestCache] = None
_cache_lock = asyncio.Lock()


async def get_request_cache() -> RequestCache:
    """Get the global request cache instance."""
    global _global_cache

    if _global_cache is None:
        async with _cache_lock:
            if _global_cache is None:
                # Use module-level configuration
                _global_cache = RequestCache(
                    default_ttl=CACHE_TTL,
                    enabled=ENABLE_CACHING
                )

    return _global_cache


async def close_request_cache() -> None:
    """Close the global request cache."""
    global _global_cache

    if _global_cache:
        await _global_cache.close()
        _global_cache = None
