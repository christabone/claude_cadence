"""
S3 service for AGR data repository access.

This module provides functionality for interacting with AWS S3 buckets
containing Alliance data files and resources.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, AsyncIterator, Dict, Any

import aiofiles
import httpx
from httpx import AsyncClient

from ..core.config import S3Config

logger = logging.getLogger(__name__)


@dataclass
class S3Object:
    """Represents an S3 object."""
    key: str
    size: int
    last_modified: datetime
    etag: str
    storage_class: str = "STANDARD"
    metadata: Dict[str, str] = None

    @property
    def name(self) -> str:
        """Get the object name (last part of key)."""
        return self.key.split("/")[-1]

    @property
    def prefix(self) -> str:
        """Get the object prefix (path without filename)."""
        parts = self.key.split("/")[:-1]
        return "/".join(parts) + "/" if parts else ""


class S3Service:
    """
    Service for AWS S3 operations.

    Provides functionality for listing, downloading, and uploading
    files to Alliance S3 buckets.
    """

    def __init__(self, config: S3Config):
        """
        Initialize S3 service.

        Args:
            config: S3 configuration
        """
        self.config = config
        self._client: Optional[AsyncClient] = None
        self._base_url = f"https://{config.bucket}.s3.{config.region}.amazonaws.com"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize HTTP client for S3 operations."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10)
            )
            logger.info(f"Initialized S3 client for bucket: {self.config.bucket}")

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_url(self, key: str) -> str:
        """Build S3 object URL."""
        # Remove leading slash if present
        key = key.lstrip("/")
        return f"{self._base_url}/{key}"

    async def list_objects(
        self,
        prefix: str = "",
        delimiter: str = "/",
        max_results: int = 1000
    ) -> List[S3Object]:
        """
        List objects in S3 bucket.

        Args:
            prefix: Prefix to filter objects
            delimiter: Delimiter for grouping objects
            max_results: Maximum number of results

        Returns:
            List of S3Object instances
        """
        # For MCP implementation, we'll use the AWS CLI or boto3
        # This is a simplified version for the template
        objects = []

        # TODO: Implement actual S3 listing using AWS SDK
        # This would typically use boto3 or make signed requests

        logger.info(f"Listed {len(objects)} objects with prefix: {prefix}")
        return objects

    async def download_file(
        self,
        key: str,
        destination: Path,
        chunk_size: int = 8192
    ) -> Path:
        """
        Download a file from S3.

        Args:
            key: S3 object key
            destination: Local file path to save to
            chunk_size: Download chunk size

        Returns:
            Path to downloaded file
        """
        if self._client is None:
            await self.connect()

        url = self._get_url(key)

        try:
            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Stream download
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                async with aiofiles.open(destination, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (chunk_size * 100) == 0:
                                logger.debug(f"Download progress: {progress:.1f}%")

            logger.info(f"Downloaded {key} to {destination}")
            return destination

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to download {key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Download error for {key}: {e}")
            raise

    async def upload_file(
        self,
        source: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload a file to S3.

        Args:
            source: Local file path to upload
            key: S3 object key
            metadata: Optional object metadata

        Returns:
            S3 object URL
        """
        if self._client is None:
            await self.connect()

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        url = self._get_url(key)

        try:
            # Read file content
            async with aiofiles.open(source, "rb") as f:
                content = await f.read()

            # Prepare headers
            headers = {}
            if metadata:
                for k, v in metadata.items():
                    headers[f"x-amz-meta-{k}"] = v

            # Upload file
            response = await self._client.put(
                url,
                content=content,
                headers=headers
            )
            response.raise_for_status()

            logger.info(f"Uploaded {source} to {key}")
            return url

        except Exception as e:
            logger.error(f"Upload error for {source}: {e}")
            raise

    async def get_object_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata for an S3 object.

        Args:
            key: S3 object key

        Returns:
            Dictionary of object metadata
        """
        if self._client is None:
            await self.connect()

        url = self._get_url(key)

        try:
            response = await self._client.head(url)
            response.raise_for_status()

            metadata = {
                "content_length": int(response.headers.get("content-length", 0)),
                "content_type": response.headers.get("content-type"),
                "last_modified": response.headers.get("last-modified"),
                "etag": response.headers.get("etag", "").strip('"'),
            }

            # Extract custom metadata
            custom_metadata = {}
            for header, value in response.headers.items():
                if header.startswith("x-amz-meta-"):
                    key = header[11:]  # Remove x-amz-meta- prefix
                    custom_metadata[key] = value

            if custom_metadata:
                metadata["custom_metadata"] = custom_metadata

            return metadata

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(f"Object not found: {key}")
            raise

    def get_public_url(self, key: str) -> str:
        """
        Get public URL for an S3 object.

        Args:
            key: S3 object key

        Returns:
            Public URL for the object
        """
        return self._get_url(key)

    async def object_exists(self, key: str) -> bool:
        """
        Check if an S3 object exists.

        Args:
            key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            await self.get_object_metadata(key)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking object existence: {e}")
            raise
