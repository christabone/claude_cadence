"""
Request handlers for the AGR MCP server.

This module implements handlers for various MCP protocol requests
and manages the routing of tool invocations.
"""

import logging
from typing import Any, Dict, List, Optional, Callable

from mcp import Server
from mcp.server import Request
from mcp.types import Tool, ToolResult, Error, ErrorCode

from .config import Config

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """Registry for tool handlers."""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool, handler: Callable) -> None:
        """Register a tool and its handler."""
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
        logger.debug(f"Registered tool: {tool.name}")

    def get_handler(self, tool_name: str) -> Optional[Callable]:
        """Get handler for a tool."""
        return self._handlers.get(tool_name)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool definition."""
        return self._tools.get(tool_name)

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())


# Global handler registry
_registry = HandlerRegistry()


def register_tool(tool: Tool) -> Callable:
    """
    Decorator to register a tool handler.

    Usage:
        @register_tool(Tool(name="my_tool", ...))
        async def handle_my_tool(arguments: Dict[str, Any]) -> Any:
            ...
    """
    def decorator(handler: Callable) -> Callable:
        _registry.register(tool, handler)
        return handler
    return decorator


async def handle_tool_request(request: Request) -> ToolResult:
    """
    Handle a tool invocation request.

    Args:
        request: The MCP tool request

    Returns:
        ToolResult with the execution result or error
    """
    tool_name = request.params.get("name")
    arguments = request.params.get("arguments", {})

    if not tool_name:
        return ToolResult(
            error=Error(
                code=ErrorCode.INVALID_PARAMS,
                message="Tool name is required"
            )
        )

    handler = _registry.get_handler(tool_name)
    if not handler:
        return ToolResult(
            error=Error(
                code=ErrorCode.METHOD_NOT_FOUND,
                message=f"Unknown tool: {tool_name}"
            )
        )

    try:
        logger.debug(f"Executing tool: {tool_name} with arguments: {arguments}")
        result = await handler(arguments)
        return ToolResult(result=result)
    except ValueError as e:
        logger.error(f"Validation error in {tool_name}: {e}")
        return ToolResult(
            error=Error(
                code=ErrorCode.INVALID_PARAMS,
                message=str(e)
            )
        )
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
        return ToolResult(
            error=Error(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Tool execution failed: {str(e)}"
            )
        )


def register_handlers(server: Server, config: Config) -> None:
    """
    Register all request handlers with the MCP server.

    Args:
        server: The MCP server instance
        config: Server configuration
    """
    # Register tool listing handler
    @server.list_tools()
    async def handle_list_tools() -> List[Tool]:
        """Handle list_tools request."""
        return _registry.get_all_tools()

    # Register tool invocation handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Any:
        """Handle tool invocation."""
        request = Request(
            id=None,
            method="tools/call",
            params={"name": name, "arguments": arguments}
        )
        result = await handle_tool_request(request)

        if result.error:
            raise Exception(result.error.message)

        return result.result

    logger.info("Request handlers registered successfully")


def get_available_tools() -> List[Tool]:
    """Get list of all available tools."""
    return _registry.get_all_tools()
