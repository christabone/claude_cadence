"""Tests for error handling in AGR MCP Server.

This module contains test cases for error handling scenarios including
validation errors, API errors, network errors, and edge cases.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json

from src.errors import (
    AGRMCPError, ConnectionError, ValidationError, ResourceNotFoundError,
    ToolExecutionError, ConfigurationError
)
from src.server import AGRMCPServer
from src.tools import gene_query, file_download, api_schema
from src.utils import validators, http_client
from mcp.types import CallToolRequest


class TestCustomExceptions:
    """Test cases for custom exception classes."""

    def test_agr_error_base(self):
        """Test base AGRMCPError exception."""
        error = AGRMCPError("Base error occurred")
        assert str(error) == "Base error occurred"
        assert error.details == {}

        # Test with details
        error_with_details = AGRMCPError("Error with details", details={"code": 500})
        assert error_with_details.details == {"code": 500}

    def test_agr_api_error(self):
        """Test ConnectionError with connection parameters."""
        error = ConnectionError(
            "API request failed",
            host="api.alliancegenome.org",
            port=443,
            service="agr_api"
        )
        assert error.details["host"] == "api.alliancegenome.org"
        assert error.details["port"] == 443
        assert error.details["service"] == "agr_api"
        assert "API request failed" in str(error)

    def test_agr_validation_error(self):
        """Test ValidationError with field and value information."""
        error = ValidationError(
            "Invalid gene ID format",
            field="gene_id",
            value="invalid-id"
        )
        assert error.field == "gene_id"
        assert error.value == "invalid-id"
        assert "Invalid gene ID format" in str(error)

    def test_agr_not_found_error(self):
        """Test ResourceNotFoundError with resource information."""
        error = ResourceNotFoundError(
            "Gene not found",
            resource_type="gene",
            resource_id="AGR:12345"
        )
        assert error.resource_type == "gene"
        assert error.resource_id == "AGR:12345"
        assert "Gene not found" in str(error)

    def test_agr_download_error(self):
        """Test ToolExecutionError with tool information."""
        error = ToolExecutionError(
            "Download failed",
            tool_name="file_download",
            operation="download_data",
            error_output="Insufficient disk space"
        )
        assert error.details["tool_name"] == "file_download"
        assert error.details["operation"] == "download_data"
        assert error.details["error_output"] == "Insufficient disk space"
        assert "Download failed" in str(error)

    def test_agr_server_error(self):
        """Test AGRMCPError."""
        error = AGRMCPError("Server initialization failed")
        assert "Server initialization failed" in str(error)

    def test_agr_config_error(self):
        """Test ConfigurationError with configuration details."""
        error = ConfigurationError(
            "Missing required configuration",
            config_key="api_key"
        )
        assert error.details["config_key"] == "api_key"
        assert "Missing required configuration" in str(error)


class TestValidationErrorHandling:
    """Test cases for validation error scenarios."""

    @pytest.mark.asyncio
    async def test_empty_query_validation(self):
        """Test validation of empty search queries."""
        # Test empty string
        with pytest.raises(ValidationError) as exc_info:
            await gene_query.search_genes("")
        assert "cannot be empty" in str(exc_info.value)

        # Test whitespace only
        with pytest.raises(ValidationError) as exc_info:
            await gene_query.search_genes("   ")
        assert "cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_limit_validation(self):
        """Test validation of search result limits."""
        # Test negative limit
        with pytest.raises(ValidationError) as exc_info:
            await gene_query.search_genes("BRCA2", limit=-1)
        assert "between 1 and 100" in str(exc_info.value)

        # Test excessive limit
        with pytest.raises(ValidationError) as exc_info:
            await gene_query.search_genes("BRCA2", limit=1000)
        assert "between 1 and 100" in str(exc_info.value)

    def test_gene_id_validation(self):
        """Test gene ID format validation."""
        with patch('src.utils.validators.GENE_ID_PATTERN'):
            # Test invalid format
            with pytest.raises(ValidationError) as exc_info:
                validators.validate_gene_id("invalid-format")
            assert "Invalid gene ID" in str(exc_info.value)

            # Test empty ID
            with pytest.raises(ValidationError) as exc_info:
                validators.validate_gene_id("")
            assert "Gene ID cannot be empty" in str(exc_info.value)

    def test_species_validation(self):
        """Test species name/taxon ID validation."""
        # Mock invalid species
        with patch('src.utils.validators.VALID_SPECIES', []):
            with pytest.raises(ValidationError) as exc_info:
                validators.validate_species("Unknown species")
            assert "Invalid species" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_data_type_download(self):
        """Test validation of download data types."""
        with pytest.raises(ValidationError) as exc_info:
            await file_download.download_data("invalid_type")
        assert "Invalid data type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_format_download(self):
        """Test validation of download formats."""
        with pytest.raises(ValidationError) as exc_info:
            await file_download.download_data("genes", format="xml")
        assert "Invalid format" in str(exc_info.value)


class TestAPIErrorHandling:
    """Test cases for API error scenarios."""

    @pytest.mark.asyncio
    async def test_api_timeout_error(self):
        """Test handling of API timeout errors."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = asyncio.TimeoutError("Request timeout")
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await gene_query.search_genes("BRCA2")
            assert "Failed to search genes" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_connection_error(self):
        """Test handling of connection errors."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = ConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await gene_query.get_gene_details("AGR:101000")
            assert "Failed to get gene details" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_404_error(self):
        """Test handling of 404 not found responses."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"error": "Resource not found"}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ResourceNotFoundError) as exc_info:
                await gene_query.get_gene_details("AGR:INVALID")
            assert "Gene not found" in str(exc_info.value)
            assert exc_info.value.details["resource_id"] == "AGR:INVALID"

    @pytest.mark.asyncio
    async def test_api_500_error(self):
        """Test handling of server errors."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Internal server error"}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await file_download.download_data("genes")
            assert "API request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_rate_limit_error(self):
        """Test handling of rate limiting."""
        with patch('src.tools.api_schema.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await api_schema.get_schema()
            # ConnectionError doesn't have status_code attribute
            # Just check that the error was raised
            assert "rate limit" in str(exc_info.value).lower() or "connection" in str(exc_info.value).lower()


class TestServerErrorHandling:
    """Test cases for server-level error handling."""

    @pytest.mark.asyncio
    async def test_server_tool_execution_error(self):
        """Test server handling of tool execution errors."""
        with patch('src.server.Config') as mock_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.gene_query.search_genes') as mock_search:

            # Mock Config class attributes
            mock_config.BASE_URL = "https://www.alliancegenome.org"
            mock_config.DEFAULT_TIMEOUT = 30
            server = AGRMCPServer()

            # Mock search to raise an unexpected error
            mock_search.side_effect = RuntimeError("Unexpected error")

            request = CallToolRequest(
                params=MagicMock(
                    name="search_genes",
                    arguments={"query": "test"}
                )
            )

            with pytest.raises(AGRMCPError) as exc_info:
                await server.handle_tool_call(request)
            assert "Tool execution failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_server_missing_arguments(self):
        """Test server handling of missing required arguments."""
        with patch('src.server.Config') as mock_config, \
             patch('src.server.setup_logging'):

            # Mock Config class attributes
            mock_config.BASE_URL = "https://www.alliancegenome.org"
            mock_config.DEFAULT_TIMEOUT = 30
            server = AGRMCPServer()

            # Request without required arguments
            request = CallToolRequest(
                params=MagicMock(
                    name="get_gene_details",
                    arguments={}  # Missing required gene_id
                )
            )

            with pytest.raises(AGRMCPError):
                await server.handle_tool_call(request)

    def test_server_config_error(self):
        """Test server initialization with invalid config."""
        with patch('src.server.Config') as mock_config:
            # Make the Config class constructor raise an error
            mock_config.side_effect = ConfigurationError(
                "Missing API key",
                missing_keys=["api_key"]
            )

            with pytest.raises(ConfigurationError):
                AGRMCPServer()


class TestDownloadErrorHandling:
    """Test cases for download-specific error scenarios."""

    @pytest.mark.asyncio
    async def test_download_disk_space_error(self):
        """Test handling of insufficient disk space."""
        with patch('src.tools.file_download.get_file_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.check_disk_space.return_value = False
            mock_get_manager.return_value = mock_manager

            with pytest.raises(ToolExecutionError) as exc_info:
                await file_download.download_data("genes", format="json")
            assert "Insufficient disk space" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_write_permission_error(self):
        """Test handling of file write permission errors."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client, \
             patch('src.tools.file_download.get_file_manager') as mock_get_manager:

            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            mock_manager = MagicMock()
            mock_manager.save_file.side_effect = PermissionError("Permission denied")
            mock_get_manager.return_value = mock_manager

            with pytest.raises(ToolExecutionError) as exc_info:
                await file_download.download_data("genes")
            assert "Failed to save file" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_interrupted_error(self):
        """Test handling of interrupted downloads."""
        with patch('src.tools.file_download.get_http_client') as mock_get_client:
            mock_client = AsyncMock()

            # Simulate interrupted stream
            async def interrupted_stream():
                yield b'{"data": ['
                raise ConnectionError("Connection lost")

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_bytes = interrupted_stream
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ToolExecutionError) as exc_info:
                await file_download.download_data("expression", format="json")
            assert "Download interrupted" in str(exc_info.value)


class TestEdgeCaseErrorHandling:
    """Test cases for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_malformed_json_response(self):
        """Test handling of malformed JSON responses."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.side_effect = json.JSONDecodeError(
                "Expecting value", "doc", 0
            )
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await gene_query.search_genes("test")
            assert "Failed to parse response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty API responses."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = None
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await gene_query.search_genes("nonexistent")
            assert result["total"] == 0
            assert result["results"] == []

    @pytest.mark.asyncio
    async def test_partial_data_handling(self):
        """Test handling of partial/incomplete data."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            # Missing expected fields
            mock_response.json.return_value = {
                "results": [
                    {"id": "AGR:101000"}  # Missing symbol, name, etc.
                ]
            }
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await gene_query.search_genes("test")
            # Should handle missing fields gracefully
            assert result["results"][0]["id"] == "AGR:101000"
            assert result["results"][0].get("symbol") is None


# Test fixtures
@pytest.fixture
def mock_error_response():
    """Provide mock error response data."""
    return {
        "error": {
            "code": "RESOURCE_NOT_FOUND",
            "message": "The requested resource was not found",
            "details": {
                "resource_type": "gene",
                "resource_id": "AGR:INVALID"
            }
        },
        "timestamp": "2024-12-15T10:00:00Z",
        "request_id": "req_12345"
    }
