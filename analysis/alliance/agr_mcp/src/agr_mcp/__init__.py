"""AGR MCP Server - Alliance of Genome Resources Model Context Protocol Server.

This package provides MCP server implementation for accessing Alliance data.
"""

__version__ = "0.1.0"
__author__ = "Alliance of Genome Resources"
__email__ = "help@alliancegenome.org"

from agr_mcp.server import AGRMCPServer, start_server, stop_server, agr_mcp_server

__all__ = ["AGRMCPServer", "start_server", "stop_server", "agr_mcp_server"]
