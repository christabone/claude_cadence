"""
Main MCP server implementation for AGR data access.

This module implements the Model Context Protocol server that provides
structured access to Alliance Genome Resource data and services.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp import Server
from mcp.server import Request
from mcp.types import Tool, ToolResult

from .config import Config
from .handlers import register_handlers

logger = logging.getLogger(__name__)


class AGRMCPServer:
    """
    Main MCP server for Alliance Genome Resource data access.

    This server provides tools for querying gene information, disease
    associations, expression data, and other genomic resources from the
    Alliance of Genome Resources.
    """

    def __init__(self, config: Config):
        """
        Initialize the AGR MCP server.

        Args:
            config: Server configuration object
        """
        self.config = config
        self.server = Server("agr-mcp-server")
        self._tools: Dict[str, Tool] = {}
        self._setup_server()

    def _setup_server(self) -> None:
        """Set up the MCP server with handlers and tools."""
        # Register request handlers
        register_handlers(self.server, self.config)

        # Register available tools
        self._register_tools()

        logger.info("AGR MCP server initialized successfully")

    def _register_tools(self) -> None:
        """Register all available tools with the server."""
        # Tool registration will be implemented by specific tool modules
        pass

    async def run(self) -> None:
        """Run the MCP server."""
        logger.info(f"Starting AGR MCP server on port {self.config.port}")
        try:
            await self.server.run()
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

    def get_tools(self) -> List[Tool]:
        """Get list of available tools."""
        return list(self._tools.values())


def create_server(config: Optional[Config] = None) -> AGRMCPServer:
    """
    Create and configure an AGR MCP server instance.

    Args:
        config: Optional configuration object. If not provided,
                default configuration will be loaded.

    Returns:
        Configured AGRMCPServer instance
    """
    if config is None:
        from .config import load_config
        config = load_config()

    return AGRMCPServer(config)


async def main() -> None:
    """Main entry point for running the server."""
    import sys
    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        config = load_config()
        server = create_server(config)
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
