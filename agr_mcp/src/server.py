"""MCP Server implementation for Alliance Genome Resources.

This module implements the main MCP server that handles tool registration,
request processing, and response formatting for AGR data access.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import Tool, CallToolRequest, CallToolResult, TextContent

from .config import Config
from .errors import AGRMCPError
from .tools import api_schema, file_download, gene_query, metadata_tools
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class AGRMCPServer:
    """Alliance Genome Resources MCP Server implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AGR MCP Server.

        Args:
            config: Optional configuration dictionary. If not provided,
                   loads from default configuration.
        """
        self.config = config or {"api_url": Config.BASE_URL, "timeout": Config.API_TIMEOUT}
        self.server = Server("agr-mcp-server")
        self._setup_tools()
        setup_logging(self.config.get("logging", {}))

    def _setup_tools(self) -> None:
        """Register all available tools with the MCP server."""
        tools = [
            Tool(
                name="search_genes",
                description="Search for genes by symbol, name, or identifier",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Gene symbol, name, or identifier"
                        },
                        "species": {
                            "type": "string",
                            "description": "Species name or taxon ID (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_gene_details",
                description="Get detailed information about a specific gene",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "gene_id": {
                            "type": "string",
                            "description": "AGR gene identifier"
                        }
                    },
                    "required": ["gene_id"]
                }
            ),
            Tool(
                name="find_orthologs",
                description="Find orthologous genes across species",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "gene_id": {
                            "type": "string",
                            "description": "AGR gene identifier"
                        },
                        "target_species": {
                            "type": "string",
                            "description": "Target species name or taxon ID (optional)"
                        }
                    },
                    "required": ["gene_id"]
                }
            ),
            Tool(
                name="download_data",
                description="Download AGR data files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_type": {
                            "type": "string",
                            "description": "Type of data to download",
                            "enum": ["genes", "alleles", "disease", "expression", "orthology"]
                        },
                        "species": {
                            "type": "string",
                            "description": "Species filter (optional)"
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format",
                            "enum": ["json", "tsv", "csv"]
                        }
                    },
                    "required": ["data_type"]
                }
            ),
            Tool(
                name="get_api_schema",
                description="Get AGR API schema information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "API endpoint name (optional)"
                        }
                    }
                }
            ),
            Tool(
                name="get_supported_species",
                description="Get list of supported species from Alliance Genome Resources",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            )
        ]

        for tool in tools:
            self.server.add_tool(tool)

    async def handle_tool_call(self, request: CallToolRequest) -> CallToolResult:
        """Handle incoming tool call requests.

        Args:
            request: The tool call request from the client

        Returns:
            Response containing the tool execution results

        Raises:
            AGRMCPError: If tool execution fails
        """
        try:
            tool_name = request.params.name
            arguments = request.params.arguments or {}

            logger.info(f"Handling tool call: {tool_name}")
            logger.debug(f"Arguments: {arguments}")

            if tool_name == "search_genes":
                result = await gene_query.search_genes(**arguments)
            elif tool_name == "get_gene_details":
                result = await gene_query.get_gene_details(**arguments)
            elif tool_name == "find_orthologs":
                result = await gene_query.find_orthologs(**arguments)
            elif tool_name == "download_data":
                result = await file_download.download_data(**arguments)
            elif tool_name == "get_api_schema":
                result = await api_schema.get_schema(**arguments)
            elif tool_name == "get_supported_species":
                result = await metadata_tools.get_supported_species(**arguments)
            else:
                raise AGRMCPError(f"Unknown tool: {tool_name}")

            return CallToolResult(
                content=[TextContent(type="text", text=str(result))]
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
            raise AGRMCPError(f"Tool execution failed: {str(e)}") from e

    async def run(self, host: str = "localhost", port: int = 8000) -> None:
        """Run the MCP server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        logger.info(f"Starting AGR MCP Server on {host}:{port}")

        @self.server.call_tool()
        async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
            return await self.handle_tool_call(request)

        await self.server.run()


def main():
    """Main entry point for the AGR MCP server."""
    import asyncio

    server = AGRMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
