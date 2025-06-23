"""Tests for file download functionality in AGR MCP Server.

This module contains test cases for the file download tools including
data file downloads in various formats.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import json
import tempfile
import os

from src.tools.file_download import download_data, list_available_files, get_download_status
from src.errors import ConnectionError, ValidationError, ToolExecutionError


class TestDownloadData:
    """Test cases for the download_data function."""

    @pytest.mark.asyncio
    async def test_download_data_json_format(self):
        """Test downloading data in JSON format."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client, \
             patch('src.tools.file_download.get_file_manager') as mock_get_manager:

            # Mock HTTP client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "AGR:101000", "symbol": "BRCA2", "species": "Homo sapiens"},
                    {"id": "AGR:102000", "symbol": "TP53", "species": "Homo sapiens"}
                ],
                "metadata": {"total": 2, "dataType": "genes"}
            }
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Mock file manager
            mock_manager = MagicMock()
            mock_manager.save_file.return_value = "/tmp/agr_genes_20241215.json"
            mock_get_manager.return_value = mock_manager

            # Test download
            result = await download_data("genes", format="json")

            # Verify results
            assert result["status"] == "success"
            assert result["format"] == "json"
            assert result["dataType"] == "genes"
            assert "filePath" in result
            assert result["recordCount"] == 2

    @pytest.mark.asyncio
    async def test_download_data_tsv_format(self):
        """Test downloading data in TSV format."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client, \
             patch('src.tools.file_download.get_file_manager') as mock_get_manager:

            # Mock HTTP client for TSV response
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "id\tsymbol\tspecies\nAGR:101000\tBRCA2\tHomo sapiens\nAGR:102000\tTP53\tHomo sapiens"
            mock_response.headers = {"content-type": "text/tab-separated-values"}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Mock file manager
            mock_manager = MagicMock()
            mock_manager.save_file.return_value = "/tmp/agr_genes_20241215.tsv"
            mock_get_manager.return_value = mock_manager

            # Test download
            result = await download_data("genes", format="tsv", species="Homo sapiens")

            # Verify results
            assert result["status"] == "success"
            assert result["format"] == "tsv"
            assert "filePath" in result

    @pytest.mark.asyncio
    async def test_download_data_csv_format(self):
        """Test downloading data in CSV format."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client, \
             patch('src.tools.file_download.get_file_manager') as mock_get_manager:

            # Mock HTTP client for CSV response
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "id,symbol,species\nAGR:101000,BRCA2,Homo sapiens\nAGR:102000,TP53,Homo sapiens"
            mock_response.headers = {"content-type": "text/csv"}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Mock file manager
            mock_manager = MagicMock()
            mock_manager.save_file.return_value = "/tmp/agr_genes_20241215.csv"
            mock_get_manager.return_value = mock_manager

            # Test download
            result = await download_data("genes", format="csv")

            # Verify results
            assert result["status"] == "success"
            assert result["format"] == "csv"

    @pytest.mark.asyncio
    async def test_download_data_invalid_data_type(self):
        """Test that invalid data type raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            await download_data("invalid_type", format="json")

        assert "Invalid data type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_data_invalid_format(self):
        """Test that invalid format raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            await download_data("genes", format="xml")

        assert "Invalid format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_data_with_species_filter(self):
        """Test downloading data with species filter."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client, \
             patch('src.tools.file_download.get_file_manager') as mock_get_manager:

            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [], "metadata": {"total": 0}}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            mock_manager = MagicMock()
            mock_manager.save_file.return_value = "/tmp/agr_data.json"
            mock_get_manager.return_value = mock_manager

            # Test with species filter
            await download_data("alleles", format="json", species="Mus musculus")

            # Verify API was called with species parameter
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert "species" in call_args[1]["params"]
            assert call_args[1]["params"]["species"] == "Mus musculus"

    @pytest.mark.asyncio
    async def test_download_data_api_error(self):
        """Test handling of API errors during download."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection timeout")
            mock_get_client.return_value = mock_client

            with pytest.raises(ToolExecutionError) as exc_info:
                await download_data("genes", format="json")

            assert "Download failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_data_large_file(self):
        """Test handling of large file downloads."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client, \
             patch('src.tools.file_download.get_file_manager') as mock_get_manager:

            # Mock streaming response
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {
                "content-length": "10485760",  # 10MB
                "content-type": "application/json"
            }
            mock_response.iter_bytes = AsyncMock(return_value=[
                b'{"data": [',
                b'{"id": "AGR:1", "symbol": "TEST1"},',
                b'{"id": "AGR:2", "symbol": "TEST2"}',
                b']}'
            ])
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Mock file manager
            mock_manager = MagicMock()
            mock_manager.save_stream.return_value = "/tmp/agr_large_data.json"
            mock_get_manager.return_value = mock_manager

            # Test large file download
            result = await download_data("expression", format="json")

            # Verify streaming was used
            assert result["status"] == "success"
            mock_manager.save_stream.assert_called_once()


class TestListAvailableFiles:
    """Test cases for listing available data files."""

    @pytest.mark.asyncio
    async def test_list_available_files_success(self):
        """Test successful listing of available files."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "files": [
                    {
                        "dataType": "genes",
                        "species": "all",
                        "formats": ["json", "tsv", "csv"],
                        "lastUpdated": "2024-12-15",
                        "size": "45MB"
                    },
                    {
                        "dataType": "alleles",
                        "species": "Drosophila melanogaster",
                        "formats": ["json", "tsv"],
                        "lastUpdated": "2024-12-14",
                        "size": "120MB"
                    }
                ]
            }
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test listing files
            result = await list_available_files()

            # Verify results
            assert len(result["files"]) == 2
            assert result["files"][0]["dataType"] == "genes"
            assert "json" in result["files"][0]["formats"]

    @pytest.mark.asyncio
    async def test_list_available_files_with_filter(self):
        """Test listing files with data type filter."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "files": [
                    {
                        "dataType": "disease",
                        "species": "all",
                        "formats": ["json", "tsv"],
                        "lastUpdated": "2024-12-15",
                        "size": "15MB"
                    }
                ]
            }
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test with filter
            result = await list_available_files(data_type="disease")

            # Verify filtered results
            assert len(result["files"]) == 1
            assert result["files"][0]["dataType"] == "disease"


class TestGetDownloadStatus:
    """Test cases for checking download status."""

    @pytest.mark.asyncio
    async def test_get_download_status_completed(self):
        """Test checking status of completed download."""
        with patch('src.tools.file_download.get_download_tracker') as mock_tracker:
            mock_tracker.get_status.return_value = {
                "downloadId": "dl_12345",
                "status": "completed",
                "progress": 100,
                "filePath": "/tmp/agr_genes.json",
                "startTime": "2024-12-15T10:00:00Z",
                "endTime": "2024-12-15T10:05:00Z",
                "fileSize": "45MB"
            }

            # Test status check
            result = await get_download_status("dl_12345")

            # Verify results
            assert result["status"] == "completed"
            assert result["progress"] == 100
            assert "filePath" in result

    @pytest.mark.asyncio
    async def test_get_download_status_in_progress(self):
        """Test checking status of ongoing download."""
        with patch('src.tools.file_download.get_download_tracker') as mock_tracker:
            mock_tracker.get_status.return_value = {
                "downloadId": "dl_12346",
                "status": "in_progress",
                "progress": 45,
                "startTime": "2024-12-15T10:00:00Z",
                "estimatedCompletion": "2024-12-15T10:10:00Z",
                "bytesDownloaded": 23592960,
                "totalBytes": 52428800
            }

            # Test status check
            result = await get_download_status("dl_12346")

            # Verify results
            assert result["status"] == "in_progress"
            assert result["progress"] == 45
            assert "estimatedCompletion" in result

    @pytest.mark.asyncio
    async def test_get_download_status_failed(self):
        """Test checking status of failed download."""
        with patch('src.tools.file_download.get_download_tracker') as mock_tracker:
            mock_tracker.get_status.return_value = {
                "downloadId": "dl_12347",
                "status": "failed",
                "error": "Connection timeout",
                "startTime": "2024-12-15T10:00:00Z",
                "endTime": "2024-12-15T10:01:00Z"
            }

            # Test status check
            result = await get_download_status("dl_12347")

            # Verify results
            assert result["status"] == "failed"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_get_download_status_not_found(self):
        """Test checking status of non-existent download."""
        with patch('src.tools.file_download.get_download_tracker') as mock_tracker:
            mock_tracker.get_status.return_value = None

            with pytest.raises(ValidationError) as exc_info:
                await get_download_status("dl_invalid")

            assert "Download not found" in str(exc_info.value)


# Test fixtures
@pytest.fixture
def temp_download_dir():
    """Create a temporary directory for download tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_download_config():
    """Provide mock download configuration."""
    return {
        "download_dir": "/tmp/agr_downloads",
        "max_file_size": 1073741824,  # 1GB
        "allowed_formats": ["json", "tsv", "csv"],
        "chunk_size": 8192
    }


@pytest.fixture
def sample_download_response():
    """Provide sample download response data."""
    return {
        "data": [
            {
                "id": "AGR:101000",
                "symbol": "BRCA2",
                "name": "breast cancer 2",
                "species": "Homo sapiens"
            },
            {
                "id": "AGR:102000",
                "symbol": "TP53",
                "name": "tumor protein p53",
                "species": "Homo sapiens"
            }
        ],
        "metadata": {
            "total": 2,
            "dataType": "genes",
            "species": "Homo sapiens",
            "version": "7.0.0",
            "generatedAt": "2024-12-15T00:00:00Z"
        }
    }
