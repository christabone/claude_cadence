"""
Examples demonstrating async context manager usage for AGR MCP Server.

This module shows various patterns for using the async context managers
to properly manage resources in the AGR MCP Server application.
"""

import asyncio
import logging
from pathlib import Path

from agr_mcp.src.config import ConfigManager
from agr_mcp.src.utils.context_managers import (
    ApplicationResourceManager,
    ToolResourceManager,
    HttpClientManager,
    create_application_manager,
    create_tool_resource_manager
)
from agr_mcp.src.tools.file_download import FileDownloadTool
from agr_mcp.src.tools.gene_query import GeneQueryTool


async def example_basic_http_client_usage():
    """Example of basic HTTP client manager usage."""
    print("\n=== Basic HTTP Client Manager Usage ===")

    config_manager = ConfigManager.create()

    async with HttpClientManager(config_manager) as http_manager:
        # Get a client for API requests
        client = await http_manager.get_client(
            base_url="https://www.alliancegenome.org",
            timeout=15.0
        )

        try:
            # Use the client
            response = await client.get("/api/health")
            print(f"Health check status: {response.status_code}")
        finally:
            # Release client back to pool
            await http_manager.release_client(client)

    print("HTTP client manager cleanup complete")


async def example_tool_resource_manager():
    """Example of tool resource manager usage."""
    print("\n=== Tool Resource Manager Usage ===")

    config_manager = ConfigManager.create()

    async with ToolResourceManager(config_manager) as tool_manager:
        # Create tools and register them
        async with FileDownloadTool(config_manager=config_manager) as download_tool:
            tool_manager.register_tool(download_tool)

            async with GeneQueryTool() as gene_tool:
                tool_manager.register_tool(gene_tool)

                # Get HTTP client for tools to use
                client = await tool_manager.get_http_client(timeout=20.0)

                # Tools can now use shared resources
                print("Tools are ready to process requests")

                # Simulate some work
                await asyncio.sleep(0.1)

                # Release client
                await tool_manager.release_http_client(client)

                # Unregister tools (optional - cleanup will handle this)
                tool_manager.unregister_tool(gene_tool)
            tool_manager.unregister_tool(download_tool)

    print("Tool resource manager cleanup complete")


async def example_application_resource_manager():
    """Example of application-level resource management."""
    print("\n=== Application Resource Manager Usage ===")

    config_manager = ConfigManager.create()

    async with ApplicationResourceManager(config_manager) as app_manager:
        # Get tool manager from application manager
        tool_manager = await app_manager.get_tool_manager()

        # Create multiple tools that share resources
        tools = []

        # Create download tools
        for i in range(3):
            download_tool = FileDownloadTool(config_manager=config_manager)
            tools.append(download_tool)
            tool_manager.register_tool(download_tool)

        # Create gene query tools
        for i in range(2):
            gene_tool = GeneQueryTool()
            tools.append(gene_tool)
            tool_manager.register_tool(gene_tool)

        print(f"Created {len(tools)} tools sharing resources")

        # Get shared HTTP client
        shared_client = await tool_manager.get_http_client()

        # Simulate concurrent tool usage
        async def simulate_tool_work(tool, tool_id):
            print(f"Tool {tool_id} starting work")
            await asyncio.sleep(0.1)  # Simulate work
            print(f"Tool {tool_id} completed work")

        # Run tools concurrently
        tasks = [
            simulate_tool_work(tool, i)
            for i, tool in enumerate(tools)
        ]
        await asyncio.gather(*tasks)

        # Release shared client
        await tool_manager.release_http_client(shared_client)

        print("All tools completed work")

    print("Application resource manager cleanup complete")


async def example_error_handling():
    """Example of error handling with context managers."""
    print("\n=== Error Handling Example ===")

    config_manager = ConfigManager.create()

    try:
        async with create_application_manager(config_manager) as app_manager:
            tool_manager = await app_manager.get_tool_manager()

            # Intentionally cause an error
            async with FileDownloadTool(config_manager=config_manager) as download_tool:
                tool_manager.register_tool(download_tool)

                # Simulate an error occurring
                raise ValueError("Simulated error during tool operation")

    except ValueError as e:
        print(f"Caught expected error: {e}")
        print("Context managers still cleaned up properly")

    print("Error handling example complete")


async def example_file_manager_usage():
    """Example of file manager with context manager."""
    print("\n=== File Manager Context Usage ===")

    config_manager = ConfigManager.create()

    async with ToolResourceManager(config_manager) as tool_manager:
        # Get file manager instance
        file_manager = tool_manager.get_file_manager(
            base_dir=Path("/tmp/agr_test_downloads")
        )

        async with file_manager:
            # Use file manager with automatic cleanup
            stats = await file_manager.get_storage_stats()
            print(f"Storage stats: {stats}")

            # Create a test file
            test_content = b"Test file content"
            test_file = await file_manager.save_download(
                "test_file.txt",
                test_content,
                metadata={"source": "example", "type": "test"}
            )
            print(f"Created test file: {test_file}")

            # List files
            files = await file_manager.list_files()
            print(f"Found {len(files)} files")

    print("File manager context usage complete")


async def example_factory_patterns():
    """Example of using factory functions for context managers."""
    print("\n=== Factory Pattern Usage ===")

    config_manager = ConfigManager.create()

    # Use factory functions for cleaner code
    async with create_tool_resource_manager(config_manager) as tool_manager:
        print("Created tool manager via factory")

        # Get HTTP client
        client = await tool_manager.get_http_client(timeout=10.0)
        print("Retrieved HTTP client from tool manager")

        # Release client
        await tool_manager.release_http_client(client)
        print("Released HTTP client")

    print("Factory pattern usage complete")


async def run_all_examples():
    """Run all context manager examples."""
    print("Starting AGR MCP Context Manager Examples")
    print("=" * 50)

    examples = [
        example_basic_http_client_usage,
        example_tool_resource_manager,
        example_application_resource_manager,
        example_error_handling,
        example_file_manager_usage,
        example_factory_patterns
    ]

    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")

        # Small delay between examples
        await asyncio.sleep(0.5)

    print("\n" + "=" * 50)
    print("All context manager examples completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run examples
    asyncio.run(run_all_examples())
