"""Alliance Genome Resources MCP Server.

A Model Context Protocol server providing access to Alliance Genome Resources data
including gene information, orthology relationships, disease associations, and more.
"""

__version__ = "0.1.0"
__author__ = "Alliance Genome Resources"
__email__ = "support@alliancegenome.org"

from . import config, errors, server, tools, utils

__all__ = ["config", "errors", "server", "tools", "utils", "__version__"]
