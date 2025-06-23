"""Tests for AGR MCP Server functionality.

This module contains test cases for the main server implementation including
tool registration, request handling, and server lifecycle management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio

from src.server import AGRMCPServer
from src.errors import AGRMCPError, ConnectionError
from mcp.types import Tool, CallToolRequest, CallToolResult, TextContent


class TestAGRMCPServer:
    """Test cases for the AGRMCPServer class."""

    def test_server_initialization(self):
        """Test server initialization with default config."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging') as mock_setup_logging:

            mock_config = {
                "api_base_url": "https://api.alliancegenome.org",
                "logging": {"level": "INFO"}
            }
            mock_get_config.return_value = mock_config

            # Initialize server
            server = AGRMCPServer()

            # Verify initialization
            assert server.config == mock_config
            assert server.server is not None
            mock_setup_logging.assert_called_once_with({"level": "INFO"})

    def test_server_initialization_with_custom_config(self):
        """Test server initialization with custom config."""
        custom_config = {
            "api_base_url": "https://custom.api.com",
            "timeout": 60,
            "logging": {"level": "DEBUG"}
        }

        with patch('src.server.setup_logging') as mock_setup_logging:
            server = AGRMCPServer(config=custom_config)

            assert server.config == custom_config
            mock_setup_logging.assert_called_once_with({"level": "DEBUG"})

    def test_tool_registration(self):
        """Test that all required tools are registered."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'):

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the server's add_tool method to capture registrations
            registered_tools = []
            server.server.add_tool = MagicMock(
                side_effect=lambda tool: registered_tools.append(tool)
            )

            # Re-run tool setup
            server._setup_tools()

            # Verify all expected tools are registered
            tool_names = [tool.name for tool in registered_tools]
            assert "search_genes" in tool_names
            assert "get_gene_details" in tool_names
            assert "find_orthologs" in tool_names
            assert "download_data" in tool_names
            assert "get_api_schema" in tool_names
            assert len(registered_tools) == 5

            # Verify tool schemas
            search_tool = next(t for t in registered_tools if t.name == "search_genes")
            assert "query" in search_tool.inputSchema["properties"]
            assert search_tool.inputSchema["required"] == ["query"]

    @pytest.mark.asyncio
    async def test_handle_tool_call_search_genes(self):
        """Test handling search_genes tool call."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.gene_query.search_genes') as mock_search:

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the search function
            mock_search.return_value = {
                "query": "BRCA2",
                "total": 1,
                "results": [{"symbol": "BRCA2"}]
            }

            # Create request
            request = CallToolRequest(
                params=MagicMock(
                    name="search_genes",
                    arguments={"query": "BRCA2", "limit": 10}
                )
            )

            # Handle request
            response = await server.handle_tool_call(request)

            # Verify response
            assert isinstance(response, CallToolResult)
            assert len(response.content) == 1
            assert isinstance(response.content[0], TextContent)
            assert "BRCA2" in response.content[0].text
            mock_search.assert_called_once_with(query="BRCA2", limit=10)

    @pytest.mark.asyncio
    async def test_handle_tool_call_get_gene_details(self):
        """Test handling get_gene_details tool call."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.gene_query.get_gene_details') as mock_get_details:

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the get details function
            mock_get_details.return_value = {
                "id": "AGR:101000",
                "symbol": "BRCA2",
                "name": "breast cancer 2"
            }

            # Create request
            request = CallToolRequest(
                params=MagicMock(
                    name="get_gene_details",
                    arguments={"gene_id": "AGR:101000"}
                )
            )

            # Handle request
            response = await server.handle_tool_call(request)

            # Verify response
            assert isinstance(response, CallToolResult)
            mock_get_details.assert_called_once_with(gene_id="AGR:101000")

    @pytest.mark.asyncio
    async def test_handle_tool_call_find_orthologs(self):
        """Test handling find_orthologs tool call."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.gene_query.find_orthologs') as mock_find_orthologs:

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the find orthologs function
            mock_find_orthologs.return_value = {
                "sourceGene": {"id": "AGR:101000", "symbol": "BRCA2"},
                "orthologs": [{"symbol": "Brca2", "species": "Mus musculus"}],
                "total": 1
            }

            # Create request
            request = CallToolRequest(
                params=MagicMock(
                    name="find_orthologs",
                    arguments={"gene_id": "AGR:101000", "target_species": "Mus musculus"}
                )
            )

            # Handle request
            response = await server.handle_tool_call(request)

            # Verify response
            assert isinstance(response, CallToolResult)
            mock_find_orthologs.assert_called_once_with(
                gene_id="AGR:101000",
                target_species="Mus musculus"
            )

    @pytest.mark.asyncio
    async def test_handle_tool_call_download_data(self):
        """Test handling download_data tool call."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.file_download.download_data') as mock_download:

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the download function
            mock_download.return_value = {
                "status": "success",
                "filePath": "/tmp/agr_genes.json",
                "recordCount": 1000
            }

            # Create request
            request = CallToolRequest(
                params=MagicMock(
                    name="download_data",
                    arguments={"data_type": "genes", "format": "json"}
                )
            )

            # Handle request
            response = await server.handle_tool_call(request)

            # Verify response
            assert isinstance(response, CallToolResult)
            mock_download.assert_called_once_with(data_type="genes", format="json")

    @pytest.mark.asyncio
    async def test_handle_tool_call_get_api_schema(self):
        """Test handling get_api_schema tool call."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.api_schema.get_schema') as mock_get_schema:

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the get schema function
            mock_get_schema.return_value = {
                "endpoints": [
                    {"path": "/gene/{id}", "method": "GET", "description": "Get gene details"}
                ]
            }

            # Create request
            request = CallToolRequest(
                params=MagicMock(
                    name="get_api_schema",
                    arguments={"endpoint": "gene"}
                )
            )

            # Handle request
            response = await server.handle_tool_call(request)

            # Verify response
            assert isinstance(response, CallToolResult)
            mock_get_schema.assert_called_once_with(endpoint="gene")

    @pytest.mark.asyncio
    async def test_handle_tool_call_unknown_tool(self):
        """Test handling of unknown tool request."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'):

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Create request with unknown tool
            request = CallToolRequest(
                params=MagicMock(
                    name="unknown_tool",
                    arguments={}
                )
            )

            # Handle request should raise error
            with pytest.raises(AGRMCPError) as exc_info:
                await server.handle_tool_call(request)

            assert "Unknown tool: unknown_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_tool_call_with_error(self):
        """Test error handling in tool calls."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'), \
             patch('src.server.gene_query.search_genes') as mock_search:

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock search to raise an error
            mock_search.side_effect = ConnectionError("API connection failed")

            # Create request
            request = CallToolRequest(
                params=MagicMock(
                    name="search_genes",
                    arguments={"query": "BRCA2"}
                )
            )

            # Handle request should propagate error
            with pytest.raises(AGRMCPError) as exc_info:
                await server.handle_tool_call(request)

            assert "Tool execution failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_server_run(self):
        """Test server run method."""
        with patch('src.server.get_config') as mock_get_config, \
             patch('src.server.setup_logging'):

            mock_get_config.return_value = {}
            server = AGRMCPServer()

            # Mock the server's run method
            server.server.run = AsyncMock()
            server.server.call_tool = MagicMock()

            # Run server
            await server.run(host="0.0.0.0", port=8080)

            # Verify server was started
            server.server.run.assert_called_once()
            server.server.call_tool.assert_called_once()

    def test_main_entry_point(self):
        """Test the main entry point function."""
        with patch('src.server.AGRMCPServer') as mock_server_class, \
             patch('src.server.asyncio.run') as mock_asyncio_run:

            mock_server = MagicMock()
            mock_server_class.return_value = mock_server

            # Import and run main
            from src.server import main
            main()

            # Verify server was created and run
            mock_server_class.assert_called_once()
            mock_asyncio_run.assert_called_once()


# Test fixtures
@pytest.fixture
def mock_server_config():
    """Provide mock server configuration."""
    return {
        "api_base_url": "https://api.alliancegenome.org",
        "timeout": 30,
        "max_retries": 3,
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def sample_tool_request():
    """Provide sample tool request for testing."""
    return CallToolRequest(
        params=MagicMock(
            name="search_genes",
            arguments={
                "query": "p53",
                "species": "Homo sapiens",
                "limit": 5
            }
        )
    )


@pytest.fixture
async def mock_mcp_server():
    """Provide a mock MCP server instance."""
    with patch('src.server.get_config') as mock_get_config, \
         patch('src.server.setup_logging'):

        mock_get_config.return_value = {"logging": {"level": "INFO"}}
        server = AGRMCPServer()
        yield server
