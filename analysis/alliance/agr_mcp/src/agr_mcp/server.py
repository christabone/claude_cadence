"""Main MCP server implementation for AGR data access."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from mcp import Server, Tool
from mcp.types import TextContent, ToolResult

from agr_mcp.tools import (
    AlleleTools,
    DiseaseTools,
    ExpressionTools,
    GeneTools,
    XRefTools,
)
from agr_mcp.utils.api import AGRAPIClient
from agr_mcp.utils.cache import CacheManager
from agr_mcp.services.s3 import S3Service, S3Config
from agr_mcp.data import schemas

logger = logging.getLogger(__name__)


class AGRMCPServer:
    """AGR MCP Server implementation.

    Provides MCP interface to Alliance of Genome Resources data and tools.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ):
        """Initialize AGR MCP Server.

        Args:
            api_url: Alliance API URL (default from env or https://www.alliancegenome.org/api)
            cache_dir: Cache directory path (default from env or ~/.cache/agr-mcp)
            cache_ttl: Cache time-to-live in seconds (default from env or 3600)
        """
        self.api_url = api_url or os.getenv("AGR_API_URL", "https://www.alliancegenome.org/api")
        self.cache_dir = cache_dir or os.getenv("AGR_CACHE_DIR", os.path.expanduser("~/.cache/agr-mcp"))
        self.cache_ttl = cache_ttl or int(os.getenv("AGR_CACHE_TTL", "3600"))

        # Initialize components
        self.api_client = AGRAPIClient(base_url=self.api_url)
        self.cache_manager = CacheManager(cache_dir=self.cache_dir, ttl=self.cache_ttl)

        # Initialize S3 service for file downloads
        s3_config = S3Config(
            bucket=os.getenv("AGR_S3_BUCKET", "agr-data"),
            region=os.getenv("AGR_S3_REGION", "us-west-2"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.s3_service = S3Service(s3_config)

        # Initialize tool handlers
        self.gene_tools = GeneTools(self.api_client, self.cache_manager)
        self.disease_tools = DiseaseTools(self.api_client, self.cache_manager)
        self.expression_tools = ExpressionTools(self.api_client, self.cache_manager)
        self.allele_tools = AlleleTools(self.api_client, self.cache_manager)
        self.xref_tools = XRefTools(self.api_client, self.cache_manager)

        # Initialize MCP server
        self.server = Server("agr-mcp-server")
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Register all available tools with the MCP server."""
        tools = [
            # Gene tools
            self.gene_tools.get_gene_tool(),
            self.gene_tools.get_genes_batch_tool(),
            self.gene_tools.search_genes_tool(),

            # Disease tools
            self.disease_tools.get_disease_associations_tool(),
            self.disease_tools.get_disease_genes_tool(),

            # Expression tools
            self.expression_tools.get_expression_tool(),
            self.expression_tools.get_expression_summary_tool(),

            # Allele tools
            self.allele_tools.get_alleles_tool(),
            self.allele_tools.get_allele_phenotypes_tool(),

            # Cross-reference tools
            self.xref_tools.get_cross_references_tool(),
            self.xref_tools.map_identifiers_tool(),
        ]

        # Add file download tool
        download_tool = Tool(
            name="download_file",
            description="Download a file from AGR S3 repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "S3 object key/path to download"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Optional local destination path"
                    }
                },
                "required": ["key"]
            }
        )
        self.server.add_tool(download_tool, self._handle_download_file)

        # Add API schema tool
        schema_tool = Tool(
            name="get_api_schema",
            description="Get JSON schema for AGR data types",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_type": {
                        "type": "string",
                        "description": "Type of schema to retrieve",
                        "enum": ["gene", "allele", "disease", "expression"]
                    }
                },
                "required": ["schema_type"]
            }
        )
        self.server.add_tool(schema_tool, self._handle_get_api_schema)

        # Register other tools
        for tool in tools:
            self.server.add_tool(tool)

    async def run(self) -> None:
        """Run the MCP server."""
        logger.info(f"Starting AGR MCP Server (API: {self.api_url})")
        await self.server.run()

    async def _handle_download_file(self, key: str, destination: Optional[str] = None) -> str:
        """Handle file download from S3.

        Args:
            key: S3 object key
            destination: Optional local destination path

        Returns:
            JSON string with download result
        """
        try:
            if destination:
                dest_path = Path(destination)
            else:
                # Default to cache directory
                dest_path = Path(self.cache_dir) / "downloads" / Path(key).name

            downloaded_path = await self.s3_service.download_file(key, dest_path)

            result = {
                "success": True,
                "key": key,
                "local_path": str(downloaded_path),
                "size": downloaded_path.stat().st_size,
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Download failed for {key}: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "key": key,
            }, indent=2)

    async def _handle_get_api_schema(self, schema_type: str) -> str:
        """Handle API schema retrieval.

        Args:
            schema_type: Type of schema to retrieve (gene, allele, disease, expression)

        Returns:
            JSON string with schema definition
        """
        try:
            schema_map = {
                "gene": schemas.GeneSchema.schema,
                "allele": schemas.AlleleSchema.schema,
                "disease": schemas.DiseaseSchema.schema,
                "expression": schemas.ExpressionSchema.schema,
            }

            if schema_type not in schema_map:
                return json.dumps({
                    "error": f"Unknown schema type: {schema_type}",
                    "available_types": list(schema_map.keys()),
                }, indent=2)

            return json.dumps({
                "schema_type": schema_type,
                "schema": schema_map[schema_type],
                "description": f"JSON schema for {schema_type} data validation",
            }, indent=2)

        except Exception as e:
            logger.error(f"Failed to get schema for {schema_type}: {e}")
            return json.dumps({
                "error": str(e),
                "schema_type": schema_type,
            }, indent=2)

    def get_server_info(self) -> Dict[str, Any]:
        """Get server configuration information.

        Returns:
            Dictionary containing server configuration
        """
        return {
            "name": "agr-mcp-server",
            "version": "0.1.0",
            "api_url": self.api_url,
            "cache_dir": self.cache_dir,
            "cache_ttl": self.cache_ttl,
            "tools_count": len(self.server.tools),
        }


# Singleton instance
_server_instance: Optional[AGRMCPServer] = None


def get_server() -> AGRMCPServer:
    """Get or create the singleton server instance.

    Returns:
        The AGR MCP server instance
    """
    global _server_instance
    if _server_instance is None:
        _server_instance = AGRMCPServer()
    return _server_instance


async def start_server(
    api_url: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_ttl: Optional[int] = None,
) -> AGRMCPServer:
    """Start the AGR MCP server.

    Args:
        api_url: Optional API URL override
        cache_dir: Optional cache directory override
        cache_ttl: Optional cache TTL override

    Returns:
        The running server instance
    """
    global _server_instance

    # Create new instance with provided config
    _server_instance = AGRMCPServer(
        api_url=api_url,
        cache_dir=cache_dir,
        cache_ttl=cache_ttl,
    )

    # Start the server
    logger.info("Starting AGR MCP server...")
    asyncio.create_task(_server_instance.run())

    return _server_instance


async def stop_server() -> None:
    """Stop the AGR MCP server."""
    global _server_instance

    if _server_instance is not None:
        logger.info("Stopping AGR MCP server...")
        # Disconnect S3 service if connected
        if hasattr(_server_instance.s3_service, '_client') and _server_instance.s3_service._client:
            await _server_instance.s3_service.disconnect()

        # Clear the instance
        _server_instance = None
        logger.info("AGR MCP server stopped")


# Export singleton instance for direct access
agr_mcp_server = get_server()


@click.command()
@click.option(
    "--api-url",
    default=None,
    help="Alliance API URL",
    envvar="AGR_API_URL",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Cache directory path",
    envvar="AGR_CACHE_DIR",
)
@click.option(
    "--cache-ttl",
    default=None,
    type=int,
    help="Cache TTL in seconds",
    envvar="AGR_CACHE_TTL",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
    envvar="AGR_LOG_LEVEL",
)
def main(
    api_url: Optional[str],
    cache_dir: Optional[str],
    cache_ttl: Optional[int],
    log_level: str,
) -> None:
    """Run the AGR MCP Server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run server
    server = AGRMCPServer(
        api_url=api_url,
        cache_dir=cache_dir,
        cache_ttl=cache_ttl,
    )

    # Log server info
    info = server.get_server_info()
    logger.info(f"Server configuration: {info}")

    # Run server
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
