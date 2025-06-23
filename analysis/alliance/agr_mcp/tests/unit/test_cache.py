"""Unit tests for cache functionality.

This module tests the cache manager implementation.
"""

import json
import time
from pathlib import Path

import pytest

from agr_mcp.utils.cache import CacheManager


@pytest.mark.asyncio
class TestCacheManager:
    """Test cache manager functionality."""

    async def test_cache_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        cache = CacheManager(cache_dir=str(temp_cache_dir), ttl=60)

        assert cache.cache_dir == temp_cache_dir
        assert cache.ttl == 60
        assert cache.cache_dir.exists()

    async def test_make_key(self, cache_manager):
        """Test cache key generation."""
        # Test with URL only
        key1 = cache_manager.make_key("https://api.example.com/gene/123")
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA256 hex digest length

        # Test with URL and params
        key2 = cache_manager.make_key(
            "https://api.example.com/gene/123",
            {"include": "all", "format": "json"},
        )
        assert key1 != key2

        # Test same params in different order produce same key
        key3 = cache_manager.make_key(
            "https://api.example.com/gene/123",
            {"format": "json", "include": "all"},
        )
        assert key2 == key3

    async def test_set_and_get(self, cache_manager):
        """Test setting and getting cache values."""
        test_data = {"id": "test123", "value": "test_value"}
        key = "test_key"

        # Set value
        await cache_manager.set(key, test_data)

        # Get value
        retrieved = await cache_manager.get(key)
        assert retrieved == test_data

    async def test_cache_expiration(self, temp_cache_dir):
        """Test cache expiration."""
        # Create cache with 1 second TTL
        cache = CacheManager(cache_dir=str(temp_cache_dir), ttl=1)

        key = "expiring_key"
        value = {"test": "data"}

        # Set value
        await cache.set(key, value)

        # Should be retrievable immediately
        assert await cache.get(key) == value

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert await cache.get(key) is None

    async def test_delete(self, cache_manager):
        """Test cache deletion."""
        key = "delete_test"
        value = {"delete": "me"}

        # Set and verify
        await cache_manager.set(key, value)
        assert await cache_manager.get(key) == value

        # Delete and verify
        await cache_manager.delete(key)
        assert await cache_manager.get(key) is None

    async def test_clear(self, cache_manager):
        """Test clearing all cache."""
        # Set multiple values
        for i in range(5):
            await cache_manager.set(f"key_{i}", {"value": i})

        # Verify they exist
        assert await cache_manager.get("key_0") is not None
        assert await cache_manager.get("key_4") is not None

        # Clear cache
        await cache_manager.clear()

        # Verify all cleared
        for i in range(5):
            assert await cache_manager.get(f"key_{i}") is None

    async def test_cleanup(self, temp_cache_dir):
        """Test cleanup of expired entries."""
        cache = CacheManager(cache_dir=str(temp_cache_dir), ttl=1)

        # Set some values
        await cache.set("keep_1", {"keep": 1})
        await cache.set("keep_2", {"keep": 2})

        # Wait for expiration
        time.sleep(1.5)

        # Set new values that shouldn't expire
        await cache.set("keep_3", {"keep": 3})

        # Run cleanup
        removed = await cache.cleanup()

        # Should have removed 2 expired entries
        assert removed == 2

        # New entry should still exist
        assert await cache.get("keep_3") is not None

        # Old entries should be gone
        assert await cache.get("keep_1") is None
        assert await cache.get("keep_2") is None

    def test_get_stats(self, cache_manager):
        """Test cache statistics."""
        stats = cache_manager.get_stats()

        assert "cache_dir" in stats
        assert "ttl_seconds" in stats
        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats

        assert stats["ttl_seconds"] == cache_manager.ttl

    async def test_corrupted_cache_handling(self, cache_manager):
        """Test handling of corrupted cache files."""
        key = "corrupted_key"
        cache_path = cache_manager._get_cache_path(key)

        # Create corrupted cache file
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("invalid json {")

        # Should return None for corrupted data
        assert await cache_manager.get(key) is None

    async def test_cache_subdirectory_creation(self, cache_manager):
        """Test that cache creates subdirectories based on key prefix."""
        # Generate keys that will have different prefixes
        key1 = "a" * 64  # Will go in 'aa' subdir
        key2 = "b" * 64  # Will go in 'bb' subdir

        await cache_manager.set(key1, {"test": 1})
        await cache_manager.set(key2, {"test": 2})

        # Check subdirectories were created
        assert (cache_manager.cache_dir / "aa").exists()
        assert (cache_manager.cache_dir / "bb").exists()
