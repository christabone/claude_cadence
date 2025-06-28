"""
Comprehensive examples demonstrating the dependency injection framework.

This module shows various patterns and use cases for the DI framework
including scoping, lifecycle management, and complex dependency graphs.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from agr_mcp.src.config import ConfigManager
from agr_mcp.src.utils.dependency_injection import (
    DIContainer, DependencyScope, get_container, inject, shutdown_container
)
from agr_mcp.src.utils.di_integration import (
    DIAwareServer, ToolFactory, managed_server, managed_tools,
    setup_default_container, get_configured_tool
)
from agr_mcp.src.tools.file_download import FileDownloadTool
from agr_mcp.src.tools.gene_query import GeneQueryTool
from agr_mcp.src.utils.context_managers import ApplicationResourceManager


async def example_basic_di_container():
    """Example of basic DI container usage."""
    print("\n=== Basic DI Container Usage ===")

    # Create and initialize container
    container = DIContainer()
    await container.initialize()

    try:
        # Get ConfigManager (automatically registered)
        config_manager = await container.get(ConfigManager)
        print(f"Got ConfigManager: {type(config_manager).__name__}")

        # Get ApplicationResourceManager (automatically registered as scoped)
        app_manager = await container.get(ApplicationResourceManager)
        print(f"Got ApplicationResourceManager: {type(app_manager).__name__}")

        # Get the same instance again (should be from scope cache)
        app_manager2 = await container.get(ApplicationResourceManager)
        print(f"Same instance: {app_manager is app_manager2}")

        # Health check
        health = await container.health_check()
        print(f"Container health: {health}")

    finally:
        await container.shutdown()

    print("Basic DI container example complete")


async def example_custom_registrations():
    """Example of custom dependency registrations."""
    print("\n=== Custom Registrations Example ===")

    container = DIContainer()
    await container.initialize()

    try:
        # Register a custom service as singleton
        class CustomService:
            def __init__(self, config: ConfigManager):
                self.config = config
                self.counter = 0

            def increment(self):
                self.counter += 1
                return self.counter

        container.register_singleton(
            CustomService,
            dependencies=[ConfigManager]
        )

        # Get instances
        service1 = await container.get(CustomService)
        service2 = await container.get(CustomService)

        print(f"Same singleton instance: {service1 is service2}")
        print(f"Counter: {service1.increment()}")
        print(f"Counter from second ref: {service2.increment()}")

        # Register transient service
        class TransientWorker:
            def __init__(self):
                self.id = id(self)

        container.register_transient(TransientWorker)

        worker1 = await container.get(TransientWorker)
        worker2 = await container.get(TransientWorker)

        print(f"Different transient instances: {worker1 is not worker2}")
        print(f"Worker IDs: {worker1.id}, {worker2.id}")

    finally:
        await container.shutdown()

    print("Custom registrations example complete")


async def example_dependency_scopes():
    """Example of dependency scopes."""
    print("\n=== Dependency Scopes Example ===")

    container = await get_container()

    # Create different scopes
    async with DependencyScope("scope1") as scope1:
        async with DependencyScope("scope2") as scope2:
            # Get scoped dependencies
            app_manager1 = await scope1.get(ApplicationResourceManager)
            app_manager2 = await scope2.get(ApplicationResourceManager)
            app_manager1_again = await scope1.get(ApplicationResourceManager)

            print(f"Different scopes, different instances: {app_manager1 is not app_manager2}")
            print(f"Same scope, same instance: {app_manager1 is app_manager1_again}")

    print("Scopes cleaned up automatically")
    print("Dependency scopes example complete")


@inject(ConfigManager)
async def example_decorator_injection(config_manager: ConfigManager, message: str):
    """Example function using the @inject decorator."""
    print(f"Injected ConfigManager: {type(config_manager).__name__}")
    print(f"Message: {message}")
    return f"Processed: {message}"


async def example_injection_decorator():
    """Example of using the @inject decorator."""
    print("\n=== Injection Decorator Example ===")

    # Initialize global container
    await setup_default_container()

    try:
        # Call function with automatic dependency injection
        result = await example_decorator_injection("Hello DI!")
        print(f"Result: {result}")

    finally:
        await shutdown_container()

    print("Injection decorator example complete")


async def example_di_aware_server():
    """Example of DI-aware server usage."""
    print("\n=== DI-Aware Server Example ===")

    config = {
        "api_url": "https://test.alliancegenome.org",
        "timeout": 15
    }

    # Use managed server context
    async with managed_server(config) as di_server:
        server = await di_server.get_server()
        print(f"Got server: {type(server).__name__}")

        # Get tools through DI
        file_tool = await di_server.get_tool(FileDownloadTool)
        gene_tool = await di_server.get_tool(GeneQueryTool)

        print(f"Got FileDownloadTool: {type(file_tool).__name__}")
        print(f"Got GeneQueryTool: {type(gene_tool).__name__}")

    print("DI-aware server example complete")


async def example_tool_factory():
    """Example of tool factory usage."""
    print("\n=== Tool Factory Example ===")

    # Use managed tools context
    async with managed_tools("example_scope") as factory:
        # Create tools with dependency injection
        file_tool = await factory.create_file_download_tool()
        gene_tool = await factory.create_gene_query_tool()
        api_tool = await factory.create_api_schema_tool()

        print(f"Created FileDownloadTool: {type(file_tool).__name__}")
        print(f"Created GeneQueryTool: {type(gene_tool).__name__}")
        print(f"Created APISchemaDocumentationTool: {type(api_tool).__name__}")

        # Tools are automatically cleaned up when scope exits

    print("Tool factory example complete")


async def example_configuration_overrides():
    """Example of configuration overrides with DI."""
    print("\n=== Configuration Overrides Example ===")

    await setup_default_container()

    try:
        # Get tool with default configuration
        default_tool = await get_configured_tool(FileDownloadTool)
        print(f"Default tool config: {default_tool.config_manager.base_url}")

        # Get tool with configuration override
        override_config = {
            "base_url": "https://override.example.com",
            "api_timeout": 10
        }
        custom_tool = await get_configured_tool(FileDownloadTool, override_config)
        print(f"Custom tool config: {custom_tool.config_manager.base_url}")

    finally:
        await shutdown_container()

    print("Configuration overrides example complete")


async def example_complex_dependency_graph():
    """Example of complex dependency resolution."""
    print("\n=== Complex Dependency Graph Example ===")

    container = DIContainer()
    await container.initialize()

    try:
        # Define complex dependency chain
        class DatabaseService:
            def __init__(self, config: ConfigManager):
                self.config = config
                print("DatabaseService created")

        class CacheService:
            def __init__(self, config: ConfigManager):
                self.config = config
                print("CacheService created")

        class BusinessService:
            def __init__(self, db: DatabaseService, cache: CacheService, config: ConfigManager):
                self.db = db
                self.cache = cache
                self.config = config
                print("BusinessService created")

        class APIController:
            def __init__(self, business: BusinessService, config: ConfigManager):
                self.business = business
                self.config = config
                print("APIController created")

        # Register dependencies
        container.register_singleton(DatabaseService, dependencies=[ConfigManager])
        container.register_singleton(CacheService, dependencies=[ConfigManager])
        container.register_scoped(BusinessService, dependencies=[DatabaseService, CacheService, ConfigManager])
        container.register_transient(APIController, dependencies=[BusinessService, ConfigManager])

        # Resolve complex dependency
        controller = await container.get(APIController)
        print(f"Successfully created: {type(controller).__name__}")

        # Verify dependency chain
        assert isinstance(controller.business, BusinessService)
        assert isinstance(controller.business.db, DatabaseService)
        assert isinstance(controller.business.cache, CacheService)
        print("Dependency chain verified")

    finally:
        await container.shutdown()

    print("Complex dependency graph example complete")


async def example_async_factories():
    """Example of async factory functions."""
    print("\n=== Async Factories Example ===")

    container = DIContainer()
    await container.initialize()

    try:
        # Define service with async initialization
        class AsyncService:
            def __init__(self):
                self.initialized = False

            async def initialize(self):
                await asyncio.sleep(0.1)  # Simulate async initialization
                self.initialized = True
                print("AsyncService initialized")

        # Async factory function
        async def create_async_service():
            service = AsyncService()
            await service.initialize()
            return service

        # Register with async factory
        container.register_singleton(
            AsyncService,
            factory=create_async_service,
            is_async=True
        )

        # Get service (factory will be awaited)
        service = await container.get(AsyncService)
        print(f"Service initialized: {service.initialized}")

    finally:
        await container.shutdown()

    print("Async factories example complete")


async def example_error_handling():
    """Example of error handling in DI framework."""
    print("\n=== Error Handling Example ===")

    container = DIContainer()
    await container.initialize()

    try:
        # Try to get unregistered dependency
        try:
            unknown = await container.get(str)  # str is not registered
            print("This should not print")
        except Exception as e:
            print(f"Expected error for unregistered dependency: {type(e).__name__}")

        # Register service with broken factory
        def broken_factory():
            raise ValueError("Factory intentionally broken")

        class BrokenService:
            pass

        container.register_transient(BrokenService, factory=broken_factory)

        try:
            broken = await container.get(BrokenService)
            print("This should not print")
        except Exception as e:
            print(f"Expected error for broken factory: {type(e).__name__}")

    finally:
        await container.shutdown()

    print("Error handling example complete")


async def run_all_examples():
    """Run all DI framework examples."""
    print("Starting Dependency Injection Framework Examples")
    print("=" * 60)

    examples = [
        example_basic_di_container,
        example_custom_registrations,
        example_dependency_scopes,
        example_injection_decorator,
        example_di_aware_server,
        example_tool_factory,
        example_configuration_overrides,
        example_complex_dependency_graph,
        example_async_factories,
        example_error_handling
    ]

    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

        # Small delay between examples
        await asyncio.sleep(0.5)

    print("\n" + "=" * 60)
    print("All dependency injection examples completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run examples
    asyncio.run(run_all_examples())
