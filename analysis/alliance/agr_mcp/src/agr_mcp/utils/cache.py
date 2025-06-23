"""Caching utilities for AGR MCP server.

This module provides a simple file-based cache implementation for
storing API responses and reducing redundant requests.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import aiofiles

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple file-based cache manager.

    This cache manager stores JSON-serializable data in files with
    expiration times. It's designed for caching API responses.

    Attributes:
        cache_dir: Directory for storing cache files
        ttl: Time-to-live for cache entries in seconds
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (default: ~/.cache/agr-mcp)
            ttl: Time-to-live in seconds (default: 3600)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/agr-mcp")

        self.cache_dir = Path(cache_dir)
        self.ttl = ttl

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")

    def make_key(self, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from URL and parameters.

        Args:
            url: Request URL
            params: Query parameters

        Returns:
            Hexadecimal cache key
        """
        # Create a unique key from URL and sorted params
        key_parts = [url]

        if params:
            sorted_params = sorted(params.items())
            key_parts.append(str(sorted_params))

        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        return key_hash

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        # Use first 2 chars of hash for subdirectory to avoid too many files in one dir
        subdir = key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)

        return cache_subdir / f"{key}.json"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            async with aiofiles.open(cache_path, "r") as f:
                cache_data = json.loads(await f.read())

            # Check expiration
            if time.time() > cache_data["expires_at"]:
                logger.debug(f"Cache expired for key {key[:8]}...")
                await self.delete(key)
                return None

            logger.debug(f"Cache hit for key {key[:8]}...")
            return cache_data["value"]

        except Exception as e:
            logger.warning(f"Error reading cache {key[:8]}...: {e}")
            return None

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
        """
        cache_path = self._get_cache_path(key)

        cache_data = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + self.ttl,
        }

        try:
            async with aiofiles.open(cache_path, "w") as f:
                await f.write(json.dumps(cache_data))

            logger.debug(f"Cached key {key[:8]}...")

        except Exception as e:
            logger.warning(f"Error writing cache {key[:8]}...: {e}")

    async def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Deleted cache key {key[:8]}...")
        except Exception as e:
            logger.warning(f"Error deleting cache {key[:8]}...: {e}")

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            # Remove all cache files
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        cache_file.unlink()

            logger.info("Cache cleared")

        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    async def cleanup(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        removed = 0
        current_time = time.time()

        try:
            for subdir in self.cache_dir.iterdir():
                if not subdir.is_dir():
                    continue

                for cache_file in subdir.glob("*.json"):
                    try:
                        async with aiofiles.open(cache_file, "r") as f:
                            cache_data = json.loads(await f.read())

                        if current_time > cache_data.get("expires_at", 0):
                            cache_file.unlink()
                            removed += 1

                    except Exception:
                        # Remove corrupted cache files
                        cache_file.unlink()
                        removed += 1

            logger.info(f"Cleaned up {removed} expired cache entries")
            return removed

        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
            return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_files = 0
        total_size = 0
        expired_count = 0
        current_time = time.time()

        try:
            for subdir in self.cache_dir.iterdir():
                if not subdir.is_dir():
                    continue

                for cache_file in subdir.glob("*.json"):
                    total_files += 1
                    total_size += cache_file.stat().st_size

                    # Check if expired
                    try:
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)

                        if current_time > cache_data.get("expires_at", 0):
                            expired_count += 1
                    except:
                        expired_count += 1

            return {
                "cache_dir": str(self.cache_dir),
                "ttl_seconds": self.ttl,
                "total_entries": total_files,
                "expired_entries": expired_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }

        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "cache_dir": str(self.cache_dir),
                "ttl_seconds": self.ttl,
            }
