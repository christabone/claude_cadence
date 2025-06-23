"""File management utilities for AGR MCP Server.

This module provides utilities for managing downloaded files, including
saving, organizing, and cleaning up files.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiofiles
import httpx

from ..errors import (
    AGRMCPError,
    ConfigurationError,
    ValidationError,
    ConnectionError as AGRConnectionError,
    ResourceNotFoundError,
    DataIntegrityError
)

logger = logging.getLogger(__name__)


class FileManager:
    """Manages downloaded files and metadata."""

    # Default maximum file size (1GB)
    DEFAULT_MAX_FILE_SIZE = 1024 * 1024 * 1024

    # Default minimum free space (100MB)
    DEFAULT_MIN_FREE_SPACE = 100 * 1024 * 1024

    # Default chunk size for downloads (8KB)
    DEFAULT_CHUNK_SIZE = 8192

    def __init__(
        self,
        base_dir: Union[str, Path],
        max_file_size: Optional[int] = None,
        min_free_space: Optional[int] = None,
        chunk_size: Optional[int] = None
    ):
        """Initialize file manager.

        Args:
            base_dir: Base directory for file storage
            max_file_size: Maximum allowed file size in bytes
            min_free_space: Minimum required free disk space in bytes
            chunk_size: Chunk size for downloading files
        """
        self.base_dir = Path(base_dir)
        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE
        self.min_free_space = min_free_space or self.DEFAULT_MIN_FREE_SPACE
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

        # Create and validate directories
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir = self.base_dir / ".metadata"
            self.metadata_dir.mkdir(exist_ok=True)
            self.temp_dir = self.base_dir / ".temp"
            self.temp_dir.mkdir(exist_ok=True)

            # Validate write permissions
            test_file = self.base_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError as e:
            raise ConfigurationError(
                f"No write permission for directory: {base_dir}",
                config_key="base_dir"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize file manager: {str(e)}",
                config_key="base_dir"
            )

    def _get_metadata_path(self, file_path: Union[str, Path]) -> Path:
        """Get metadata file path for a given file.

        Args:
            file_path: Path to the data file

        Returns:
            Path to the metadata file
        """
        file_path = Path(file_path)
        metadata_filename = f"{file_path.name}.json"
        return self.metadata_dir / metadata_filename

    def _get_available_disk_space(self) -> int:
        """Get available disk space in bytes.

        Returns:
            Available disk space in bytes
        """
        stat = shutil.disk_usage(self.base_dir)
        return stat.free

    def _validate_disk_space(self, required_bytes: int) -> None:
        """Validate that enough disk space is available.

        Args:
            required_bytes: Number of bytes needed

        Raises:
            ValidationError: If insufficient disk space
        """
        available = self._get_available_disk_space()
        # Ensure we have the required space plus minimum buffer
        needed = required_bytes + self.min_free_space

        if available < needed:
            raise ValidationError(
                f"Insufficient disk space. Need {needed} bytes, have {available} bytes",
                field="disk_space",
                value=available,
                constraints={"minimum": needed}
            )

    async def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        validate_size: bool = True
    ) -> Path:
        """Download a file from a URL with progress tracking.

        Args:
            url: URL to download from
            filename: Optional filename (defaults to URL filename)
            progress_callback: Optional callback for progress updates (bytes_downloaded, total_bytes)
            timeout: Optional timeout in seconds
            headers: Optional HTTP headers
            validate_size: Whether to validate file size before download

        Returns:
            Path to the downloaded file

        Raises:
            ValidationError: If file size exceeds limit or invalid parameters
            AGRConnectionError: If download fails
            DataIntegrityError: If downloaded file is corrupted
        """
        if not url:
            raise ValidationError(
                "URL is required for file download",
                field="url",
                value=url
            )

        # Generate filename if not provided
        if not filename:
            filename = Path(url).name or f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create temporary file for download
        temp_file = self.temp_dir / f"{filename}.download"
        final_file = self.base_dir / filename

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # First, get headers to check file size
                if validate_size:
                    try:
                        response = await client.head(url, headers=headers)
                        response.raise_for_status()

                        if 'content-length' in response.headers:
                            file_size = int(response.headers['content-length'])

                            # Check file size limit
                            if file_size > self.max_file_size:
                                raise ValidationError(
                                    f"File size {file_size} bytes exceeds maximum allowed {self.max_file_size} bytes",
                                    field="file_size",
                                    value=file_size,
                                    constraints={"maximum": self.max_file_size}
                                )

                            # Check disk space
                            self._validate_disk_space(file_size)
                    except httpx.RequestError:
                        # If HEAD request fails, continue with download
                        logger.warning(f"Could not get file size for {url}, proceeding with download")

                # Download the file
                bytes_downloaded = 0

                async with client.stream('GET', url, headers=headers) as response:
                    response.raise_for_status()

                    # Get total size if available
                    total_size = None
                    if 'content-length' in response.headers:
                        total_size = int(response.headers['content-length'])

                        if validate_size and total_size > self.max_file_size:
                            raise ValidationError(
                                f"File size {total_size} bytes exceeds maximum allowed {self.max_file_size} bytes",
                                field="file_size",
                                value=total_size,
                                constraints={"maximum": self.max_file_size}
                            )

                    # Write to temporary file
                    async with aiofiles.open(temp_file, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                            await f.write(chunk)
                            bytes_downloaded += len(chunk)

                            # Check if we're exceeding size limit during download
                            if validate_size and bytes_downloaded > self.max_file_size:
                                raise ValidationError(
                                    f"Downloaded size exceeds maximum allowed {self.max_file_size} bytes",
                                    field="file_size",
                                    value=bytes_downloaded,
                                    constraints={"maximum": self.max_file_size}
                                )

                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(bytes_downloaded, total_size)

                # Move temporary file to final location
                shutil.move(str(temp_file), str(final_file))

                # Save metadata
                metadata = {
                    "url": url,
                    "download_time": datetime.now().isoformat(),
                    "file_size": bytes_downloaded,
                    "file_path": str(final_file)
                }

                metadata_path = self._get_metadata_path(final_file)
                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(json.dumps(metadata, indent=2))

                logger.info(f"Successfully downloaded {url} to {final_file}")
                return final_file

        except httpx.RequestError as e:
            # Clean up temporary file if it exists
            if temp_file.exists():
                temp_file.unlink()

            raise AGRConnectionError(
                f"Failed to download file from {url}: {str(e)}",
                host=httpx.URL(url).host,
                service="file_download"
            )
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file.exists():
                temp_file.unlink()

            # Re-raise validation errors and other AGR errors
            if isinstance(e, AGRMCPError):
                raise

            # Wrap other exceptions
            raise AGRMCPError(
                f"Unexpected error downloading file: {str(e)}",
                error_code="DOWNLOAD_ERROR",
                details={"url": url, "error": str(e)}
            )

    async def save_download(
        self,
        filename: str,
        content: Union[bytes, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save downloaded content to file with metadata.

        Args:
            filename: Name for the file
            content: File content
            metadata: Optional metadata to store

        Returns:
            Path to the saved file
        """
        file_path = self.base_dir / filename

        # Save content
        if isinstance(content, str):
            content = content.encode("utf-8")

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        logger.info(f"Saved file: {file_path}")

        # Save metadata
        if metadata:
            metadata["download_time"] = datetime.now().isoformat()
            metadata["file_size"] = len(content)
            metadata["file_path"] = str(file_path)

            metadata_path = self._get_metadata_path(file_path)
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(json.dumps(metadata, indent=2))

        return file_path

    async def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file information

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()

        info = {
            "name": file_path.name,
            "path": str(file_path),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }

        # Load metadata if available
        metadata_path = self._get_metadata_path(file_path)
        if metadata_path.exists():
            async with aiofiles.open(metadata_path, "r") as f:
                metadata = json.loads(await f.read())
                info["metadata"] = metadata

        return info

    async def list_files(
        self,
        pattern: str = "*",
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """List files in the download directory.

        Args:
            pattern: Glob pattern for filtering files
            include_metadata: Whether to include metadata

        Returns:
            List of file information dictionaries
        """
        files = []

        for file_path in self.base_dir.glob(pattern):
            if file_path.is_file() and file_path.parent != self.metadata_dir:
                try:
                    if include_metadata:
                        file_info = await self.get_file_info(file_path)
                    else:
                        stat = file_path.stat()
                        file_info = {
                            "name": file_path.name,
                            "path": str(file_path),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                    files.append(file_info)
                except Exception as e:
                    logger.error(f"Error getting info for {file_path}: {str(e)}")

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)

        return files

    async def cleanup_old_files(
        self,
        older_than_days: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Remove files older than specified days.

        Args:
            older_than_days: Remove files older than this many days
            filter_metadata: Optional metadata filters

        Returns:
            List of removed file paths
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_files = []

        for file_path in self.base_dir.glob("*"):
            if file_path.is_file() and file_path.parent != self.metadata_dir:
                try:
                    stat = file_path.stat()
                    modified_date = datetime.fromtimestamp(stat.st_mtime)

                    if modified_date < cutoff_date:
                        # Check metadata filters if provided
                        if filter_metadata:
                            metadata_path = self._get_metadata_path(file_path)
                            if metadata_path.exists():
                                async with aiofiles.open(metadata_path, "r") as f:
                                    metadata = json.loads(await f.read())

                                # Check if all filter criteria match
                                match = all(
                                    metadata.get(key) == value
                                    for key, value in filter_metadata.items()
                                )

                                if not match:
                                    continue

                        # Remove file and metadata
                        file_path.unlink()
                        removed_files.append(str(file_path))

                        metadata_path = self._get_metadata_path(file_path)
                        if metadata_path.exists():
                            metadata_path.unlink()

                        logger.info(f"Removed old file: {file_path}")

                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")

        return removed_files

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the download directory.

        Returns:
            Dictionary containing storage statistics
        """
        total_size = 0
        file_count = 0
        temp_size = 0
        temp_count = 0

        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size

                if file_path.parent == self.temp_dir:
                    temp_size += file_size
                    temp_count += 1
                elif file_path.parent != self.metadata_dir:
                    file_count += 1

        return {
            "directory": str(self.base_dir),
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
            "temp_files": temp_count,
            "temp_size_bytes": temp_size,
            "available_space_bytes": self._get_available_disk_space()
        }

    async def cleanup_temp_files(
        self,
        older_than_hours: Optional[int] = None
    ) -> List[str]:
        """Clean up temporary files.

        Args:
            older_than_hours: Remove temp files older than this many hours (default: all)

        Returns:
            List of removed file paths
        """
        removed_files = []

        if not self.temp_dir.exists():
            return removed_files

        cutoff_date = None
        if older_than_hours:
            cutoff_date = datetime.now() - timedelta(hours=older_than_hours)

        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                try:
                    # Check age if cutoff specified
                    if cutoff_date:
                        stat = file_path.stat()
                        modified_date = datetime.fromtimestamp(stat.st_mtime)
                        if modified_date >= cutoff_date:
                            continue

                    # Remove the file
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    logger.info(f"Removed temporary file: {file_path}")

                except Exception as e:
                    logger.error(f"Error removing temporary file {file_path}: {str(e)}")

        return removed_files

    async def validate_file_size(self, file_path: Union[str, Path]) -> None:
        """Validate that a file doesn't exceed size limits.

        Args:
            file_path: Path to the file to validate

        Raises:
            ValidationError: If file exceeds size limit
            ResourceNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ResourceNotFoundError(
                f"File not found: {file_path}",
                resource_type="file",
                resource_id=str(file_path)
            )

        file_size = file_path.stat().st_size

        if file_size > self.max_file_size:
            raise ValidationError(
                f"File size {file_size} bytes exceeds maximum allowed {self.max_file_size} bytes",
                field="file_size",
                value=file_size,
                constraints={"maximum": self.max_file_size}
            )

    def get_temp_file_path(self, prefix: str = "temp", suffix: str = "") -> Path:
        """Get a path for a new temporary file.

        Args:
            prefix: Prefix for the temp file name
            suffix: Suffix for the temp file name

        Returns:
            Path object for the temporary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}{suffix}"
        return self.temp_dir / filename
