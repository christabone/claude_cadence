"""AGR MCP Server source code.

This package provides the main entry point for the Alliance of Genome Resources
Model Context Protocol (MCP) server.
"""

from agr_mcp.server import start_server, stop_server, agr_mcp_server

__all__ = ["start_server", "stop_server", "agr_mcp_server"]
