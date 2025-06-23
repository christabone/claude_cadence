"""Integration tests for AGR MCP server.

This module tests the full server functionality including
request handling and tool execution.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server.models import (
    InitializeRequest,
    ListToolsRequest,
    CallToolRequest,
    CallToolParams,
)

from agr_mcp.server import AGRMCPServer
from agr_mcp.models import Gene


@pytest.mark.asyncio
class TestAGRMCPServerIntegration:
    """Integration tests for AGR MCP server."""

    async def test_server_initialization(self, agr_server):
        """Test server initialization and configuration."""
        assert agr_server.config["api_url"] == "https://mock.alliancegenome.org/api"
        assert agr_server.client is not None
        assert agr_server.tools is not None

    async def test_handle_initialize(self, agr_server):
        """Test initialization request handling."""
        request = InitializeRequest(
            protocolVersion="0.1.0",
            capabilities={},
            clientInfo={"name": "test-client"},
        )

        result = await agr_server._handle_initialize(request)

        assert result.protocolVersion == "0.1.0"
        assert result.serverInfo.name == "agr-mcp-server"
        assert result.serverInfo.version == "0.1.0"
        assert result.capabilities.tools is True

    async def test_handle_list_tools(self, agr_server):
        """Test listing available tools."""
        request = ListToolsRequest()

        result = await agr_server._handle_list_tools(request)

        assert len(result.tools) > 0

        # Check a specific tool
        gene_tool = next(
            (t for t in result.tools if t.name == "get_gene"),
            None
        )
        assert gene_tool is not None
        assert gene_tool.description is not None
        assert gene_tool.inputSchema is not None

    async def test_handle_call_tool_success(self, agr_server, mock_gene_data):
        """Test successful tool invocation."""
        # Mock the client method
        mock_gene = Gene(**mock_gene_data)
        agr_server.client.get_gene = AsyncMock(return_value=mock_gene)

        request = CallToolRequest(
            params=CallToolParams(
                name="get_gene",
                arguments={"identifier": "HGNC:11998", "format": "json"},
            )
        )

        result = await agr_server._handle_call_tool(request)

        assert result.isError is False
        assert len(result.content) == 1

        # Parse the JSON response
        content = json.loads(result.content[0].text)
        assert content["id"] == "HGNC:11998"
        assert content["symbol"] == "TP53"

    async def test_handle_call_tool_unknown_tool(self, agr_server):
        """Test calling unknown tool."""
        request = CallToolRequest(
            params=CallToolParams(
                name="unknown_tool",
                arguments={},
            )
        )

        result = await agr_server._handle_call_tool(request)

        assert result.isError is True
        assert "Unknown tool: unknown_tool" in result.content[0].text

    async def test_handle_call_tool_error(self, agr_server):
        """Test tool invocation error handling."""
        # Mock the client to raise an error
        agr_server.client.get_gene = AsyncMock(
            side_effect=Exception("API connection failed")
        )

        request = CallToolRequest(
            params=CallToolParams(
                name="get_gene",
                arguments={"identifier": "HGNC:11998"},
            )
        )

        result = await agr_server._handle_call_tool(request)

        assert result.isError is True
        assert "Error executing tool get_gene" in result.content[0].text
        assert "API connection failed" in result.content[0].text

    async def test_get_server_info(self, agr_server):
        """Test getting server information."""
        info = agr_server.get_server_info()

        assert info["name"] == "agr-mcp-server"
        assert info["version"] == "0.1.0"
        assert info["api_url"] == agr_server.config["api_url"]
        assert info["cache_dir"] == agr_server.config["cache_dir"]
        assert info["tools_count"] > 0

    async def test_config_from_environment(self, monkeypatch, temp_cache_dir):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("AGR_API_URL", "https://env.api.com")
        monkeypatch.setenv("AGR_CACHE_DIR", str(temp_cache_dir))
        monkeypatch.setenv("AGR_CACHE_TTL", "7200")
        monkeypatch.setenv("AGR_LOG_LEVEL", "DEBUG")

        server = AGRMCPServer()

        assert server.config["api_url"] == "https://env.api.com"
        assert server.config["cache_dir"] == str(temp_cache_dir)
        assert server.config["cache_ttl"] == 7200
        assert server.config["log_level"] == "DEBUG"

    @patch("agr_mcp.server.stdio_server")
    async def test_server_run(self, mock_stdio_server, agr_server):
        """Test server run method."""
        # Mock stdio server context manager
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio_server.return_value.__aenter__.return_value = (
            mock_read,
            mock_write,
        )

        # Mock server.run to complete immediately
        agr_server.server.run = AsyncMock()

        # Run server
        await agr_server.run()

        # Verify server.run was called
        agr_server.server.run.assert_called_once()

    async def test_tool_integration_get_gene(self, agr_server, mock_gene_data):
        """Test full integration of get_gene tool."""
        # Mock the API response
        agr_server.client._request = AsyncMock(return_value=mock_gene_data)

        request = CallToolRequest(
            params=CallToolParams(
                name="get_gene",
                arguments={
                    "identifier": "HGNC:11998",
                    "include_orthologs": True,
                    "format": "text",
                },
            )
        )

        result = await agr_server._handle_call_tool(request)

        assert result.isError is False
        assert "Gene: TP53 (HGNC:11998)" in result.content[0].text
        assert "Species: Homo sapiens" in result.content[0].text
