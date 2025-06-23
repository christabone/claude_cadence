#!/usr/bin/env python3
"""Test script for HTTP client functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.http_client import AGRHttpClient, http_client
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_basic_requests():
    """Test basic HTTP client functionality."""
    logger.info("Testing HTTP client basic requests")

    # Test with httpbin.org
    async with AGRHttpClient(base_url="https://httpbin.org") as client:
        # Test GET request
        logger.info("Testing GET request")
        response = await client.get("/get", params={"test": "value"})
        logger.info(f"GET Response: {response.status_code}")
        data = response.json()
        logger.info(f"GET Data: {data.get('args')}")

        # Test POST request
        logger.info("Testing POST request")
        response = await client.post(
            "/post",
            json={"message": "Hello, AGR!"},
            headers={"X-Custom-Header": "test"},
        )
        logger.info(f"POST Response: {response.status_code}")
        data = response.json()
        logger.info(f"POST Data: {data.get('json')}")

        # Test with custom timeout
        logger.info("Testing custom timeout")
        response = await client.get("/delay/1", timeout=5.0)
        logger.info(f"Custom timeout response: {response.status_code}")


async def test_rate_limiting():
    """Test rate limiting functionality."""
    logger.info("Testing rate limiting")

    # Create client with rate limit of 2 requests per second
    client = AGRHttpClient(
        base_url="https://httpbin.org",
        rate_limit=2.0,
        rate_burst=4,
    )

    async with client:
        start_time = asyncio.get_event_loop().time()

        # Make 5 requests rapidly
        for i in range(5):
            logger.info(f"Making request {i + 1}")
            response = await client.get("/get")
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"Request {i + 1} completed at {elapsed:.2f}s")

        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Total time for 5 requests: {total_time:.2f}s")
        logger.info("(Should be ~2 seconds with 2 req/s limit and burst of 4)")


async def test_retry_logic():
    """Test retry logic with failures."""
    logger.info("Testing retry logic")

    client = AGRHttpClient(max_retries=3)

    async with client:
        # Test with endpoint that returns 500 error
        logger.info("Testing retry on 500 error")
        try:
            response = await client.get("https://httpbin.org/status/500")
            logger.error("Should have raised an error!")
        except Exception as e:
            logger.info(f"Got expected error: {type(e).__name__}: {e}")

        # Test with invalid URL
        logger.info("Testing retry on connection error")
        try:
            response = await client.get("https://invalid-domain-that-does-not-exist.com")
            logger.error("Should have raised an error!")
        except Exception as e:
            logger.info(f"Got expected error: {type(e).__name__}: {e}")


async def test_global_client():
    """Test the global client instance."""
    logger.info("Testing global client instance")

    # Use the pre-created global client
    response = await http_client.get("https://httpbin.org/get")
    logger.info(f"Global client response: {response.status_code}")

    # Close when done
    await http_client.close()


async def main():
    """Run all tests."""
    logger.info("Starting HTTP client tests")

    try:
        await test_basic_requests()
        logger.info("-" * 50)

        await test_rate_limiting()
        logger.info("-" * 50)

        await test_retry_logic()
        logger.info("-" * 50)

        await test_global_client()

        logger.info("All tests completed!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
