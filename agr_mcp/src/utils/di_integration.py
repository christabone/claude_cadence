"""
Dependency Injection Integration for AGR MCP Server.

This module provides integration utilities for using the DI framework
with the existing AGR MCP Server components.
"""

import logging
from typing import Type, TypeVar, Optional, Any, Dict
from contextlib import asynccontextmanager

from .dependency_injection import DIContainer, DependencyScope, get_container
from ..config import ConfigManager
from ..tools.file_download import FileDownloadTool
from ..tools.gene_query import GeneQueryTool
from ..tools.api_schema import APISchemaDocumentationTool
from .context_managers import ApplicationResourceManager, ToolResourceManager
from ..server import AGRMCPServer

logger = logging.getLogger(__name__)
T = TypeVar('T')


class DIAwareServer:
    """AGR MCP Server with dependency injection support."""

    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize DI-aware server.

        Args:
            container: Optional DI container (uses global if not provided)
        """
        self._container = container
        self._server: Optional[AGRMCPServer] = None

    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the server with dependency injection.

        Args:
            config: Optional configuration override
        """
        if self._container is None:
            self._container = await get_container()

        # Configure additional tool registrations
        await self._register_tools()

        # Get ConfigManager from DI container
        config_manager = await self._container.get(ConfigManager)

        # Create server with injected configuration
        self._server = AGRMCPServer(config=config, config_manager=config_manager)

        logger.info("DI-aware server initialized")

    async def _register_tools(self):
        """Register tool types with the DI container."""
        if not self._container:
            return

        # Register FileDownloadTool
        self._container.register_transient(
            FileDownloadTool,
            factory=self._create_file_download_tool,
            dependencies=[ConfigManager],
            context_manager=True
        )

        # Register GeneQueryTool
        self._container.register_transient(
            GeneQueryTool,
            factory=lambda: GeneQueryTool(),
            context_manager=True
        )

        # Register APISchemaDocumentationTool
        self._container.register_transient(
            APISchemaDocumentationTool,
            factory=lambda: APISchemaDocumentationTool(),
            context_manager=True
        )

        logger.debug("Registered tool types with DI container")

    def _create_file_download_tool(self, config_manager: ConfigManager) -> FileDownloadTool:
        """Factory for FileDownloadTool."""
        return FileDownloadTool(config_manager=config_manager)

    async def get_server(self) -> AGRMCPServer:
        """Get the server instance.

        Returns:
            AGRMCPServer instance

        Raises:
            ValueError: If server not initialized
        """
        if self._server is None:
            raise ValueError("Server not initialized. Call initialize() first.")
        return self._server

    async def get_tool(self, tool_type: Type[T], scope_id: str = "default") -> T:
        """Get a tool instance from the DI container.

        Args:
            tool_type: Type of tool to get
            scope_id: Scope identifier

        Returns:
            Tool instance
        """
        if not self._container:
            raise ValueError("Container not initialized")
        return await self._container.get(tool_type, scope_id)

    async def shutdown(self):
        """Shutdown the server and clean up resources."""
        if self._container:
            await self._container.shutdown()
        logger.info("DI-aware server shutdown complete")


class ToolFactory:
    """Factory for creating tools with dependency injection."""

    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize tool factory.

        Args:
            container: Optional DI container
        """
        self._container = container

    async def create_file_download_tool(self,
                                      config_manager: Optional[ConfigManager] = None) -> FileDownloadTool:
        """Create FileDownloadTool with dependency injection.

        Args:
            config_manager: Optional config manager override

        Returns:
            FileDownloadTool instance
        """
        if self._container is None:
            self._container = await get_container()

        if config_manager is None:
            config_manager = await self._container.get(ConfigManager)

        return FileDownloadTool(config_manager=config_manager)

    async def create_gene_query_tool(self) -> GeneQueryTool:
        """Create GeneQueryTool with dependency injection.

        Returns:
            GeneQueryTool instance
        """
        if self._container is None:
            self._container = await get_container()

        return await self._container.get(GeneQueryTool)

    async def create_api_schema_tool(self) -> APISchemaDocumentationTool:
        """Create APISchemaDocumentationTool with dependency injection.

        Returns:
            APISchemaDocumentationTool instance
        """
        if self._container is None:
            self._container = await get_container()

        return await self._container.get(APISchemaDocumentationTool)


@asynccontextmanager
async def managed_server(config: Optional[Dict[str, Any]] = None):
    """Context manager for DI-aware server with automatic cleanup.

    Args:
        config: Optional configuration

    Yields:
        DIAwareServer instance
    """
    server = DIAwareServer()
    try:
        await server.initialize(config)
        yield server
    finally:
        await server.shutdown()


@asynccontextmanager
async def managed_tools(scope_id: str = "tools"):
    """Context manager for tools with dependency injection.

    Args:
        scope_id: Scope identifier for tool instances

    Yields:
        ToolFactory instance
    """
    container = await get_container()

    async with DependencyScope(scope_id) as scope:
        factory = ToolFactory(container)
        try:
            yield factory
        finally:
            # Scope cleanup is handled by DependencyScope context manager
            pass


async def setup_default_container() -> DIContainer:
    """Set up the default DI container with all standard registrations.

    Returns:
        Configured DIContainer instance
    """
    container = get_container()
    await container.initialize()

    # Register additional tools with proper DI
    from ..tools.file_download import FileDownloadTool
    from ..tools.gene_query import GeneQueryTool
    from ..tools.api_schema import APISchemaDocumentationTool
    from ..server import AGRMCPServer

    # Register tools as transient with ConfigManager dependency
    container.register_transient(
        FileDownloadTool,
        dependencies=[ConfigManager],
        factory=lambda config: FileDownloadTool(config_manager=config)
    )

    container.register_transient(
        GeneQueryTool,
        dependencies=[ConfigManager],
        factory=lambda config: GeneQueryTool(config_manager=config)
    )

    container.register_transient(
        APISchemaDocumentationTool,
        dependencies=[ConfigManager],
        factory=lambda config: APISchemaDocumentationTool()  # This tool doesn't need config yet
    )

    # Register server as singleton with ConfigManager dependency
    container.register_singleton(
        AGRMCPServer,
        dependencies=[ConfigManager],
        factory=lambda config: AGRMCPServer(config_manager=config)
    )

    logger.info("Default DI container setup complete with all tool registrations")
    return container


class DIMiddleware:
    """Middleware for request-scoped dependency injection."""

    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize DI middleware.

        Args:
            container: Optional DI container
        """
        self._container = container

    async def __call__(self, request_id: str, handler_func, *args, **kwargs):
        """Process request with dependency injection scope.

        Args:
            request_id: Unique request identifier
            handler_func: Handler function to call
            *args: Handler arguments
            **kwargs: Handler keyword arguments

        Returns:
            Handler result
        """
        if self._container is None:
            self._container = await get_container()

        # Create request-scoped dependency injection
        async with DependencyScope(f"request_{request_id}") as scope:
            # Inject scope into kwargs if handler expects it
            if 'di_scope' in handler_func.__code__.co_varnames:
                kwargs['di_scope'] = scope

            # Call handler
            return await handler_func(*args, **kwargs)


# Utility functions for common DI patterns

async def get_configured_tool(tool_type: Type[T],
                            config_override: Optional[Dict[str, Any]] = None) -> T:
    """Get a tool instance with optional configuration override.

    Args:
        tool_type: Type of tool to get
        config_override: Optional configuration values to override

    Returns:
        Tool instance
    """
    container = await get_container()

    if config_override:
        # Create custom ConfigManager with overrides
        base_config = await container.get(ConfigManager)
        custom_config = ConfigManager.create(
            config_type='custom',
            config_data=config_override
        )

        # For now, return tool with custom config
        # In a more advanced implementation, we could temporarily
        # register the custom config in the container
        if tool_type == FileDownloadTool:
            return FileDownloadTool(config_manager=custom_config)

    return await container.get(tool_type)


async def inject_dependencies(func, *extra_args, **extra_kwargs):
    """Inject dependencies into a function call.

    Args:
        func: Function to call with dependency injection
        *extra_args: Additional arguments
        **extra_kwargs: Additional keyword arguments

    Returns:
        Function result
    """
    container = await get_container()

    # Get function signature
    sig = inspect.signature(func)
    injected_args = []

    # Try to resolve parameters from DI container
    for param_name, param in sig.parameters.items():
        if param.annotation and param.annotation != inspect.Parameter.empty:
            if container.is_registered(param.annotation):
                dependency = await container.get(param.annotation)
                injected_args.append(dependency)

    # Call function with injected dependencies
    return await func(*injected_args, *extra_args, **extra_kwargs)

async def get_configured_file_download_tool() -> 'FileDownloadTool':
    """Get a properly configured FileDownloadTool instance.

    Returns:
        FileDownloadTool with injected dependencies
    """
    container = get_container()
    from ..tools.file_download import FileDownloadTool
    return await container.get(FileDownloadTool)


async def get_configured_gene_query_tool() -> 'GeneQueryTool':
    """Get a properly configured GeneQueryTool instance.

    Returns:
        GeneQueryTool with injected dependencies
    """
    container = get_container()
    from ..tools.gene_query import GeneQueryTool
    return await container.get(GeneQueryTool)


async def get_configured_server() -> 'AGRMCPServer':
    """Get a properly configured AGRMCPServer instance.

    Returns:
        AGRMCPServer with injected dependencies
    """
    container = get_container()
    from ..server import AGRMCPServer
    return await container.get(AGRMCPServer)


class DIToolFactory:
    """Factory for creating tools with dependency injection."""

    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize the tool factory.

        Args:
            container: Optional DI container to use
        """
        self._container = container

    async def _get_container(self) -> DIContainer:
        """Get the DI container."""
        if self._container is None:
            self._container = get_container()
            if not self._container._initialized:
                await self._container.initialize()
        return self._container

    async def create_file_download_tool(self,
                                      config_overrides: Optional[Dict[str, Any]] = None) -> 'FileDownloadTool':
        """Create FileDownloadTool with dependency injection.

        Args:
            config_overrides: Optional configuration overrides

        Returns:
            Configured FileDownloadTool instance
        """
        container = await self._get_container()

        if config_overrides:
            # Create custom ConfigManager with overrides
            from ..config import ConfigManager
            base_config = await container.get(ConfigManager)
            custom_config = ConfigManager.create(
                'data',
                config_data={**base_config.__dict__, **config_overrides}
            )

            # Create tool with custom config
            from ..tools.file_download import FileDownloadTool
            return FileDownloadTool(config_manager=custom_config)
        else:
            # Use standard DI
            from ..tools.file_download import FileDownloadTool
            return await container.get(FileDownloadTool)

    async def create_gene_query_tool(self,
                                   config_overrides: Optional[Dict[str, Any]] = None) -> 'GeneQueryTool':
        """Create GeneQueryTool with dependency injection.

        Args:
            config_overrides: Optional configuration overrides

        Returns:
            Configured GeneQueryTool instance
        """
        container = await self._get_container()

        if config_overrides:
            # Create custom ConfigManager with overrides
            from ..config import ConfigManager
            base_config = await container.get(ConfigManager)
            custom_config = ConfigManager.create(
                'data',
                config_data={**base_config.__dict__, **config_overrides}
            )

            # Create tool with custom config
            from ..tools.gene_query import GeneQueryTool
            return GeneQueryTool(config_manager=custom_config)
        else:
            # Use standard DI
            from ..tools.gene_query import GeneQueryTool
            return await container.get(GeneQueryTool)

    async def create_server(self,
                          config_overrides: Optional[Dict[str, Any]] = None) -> 'AGRMCPServer':
        """Create AGRMCPServer with dependency injection.

        Args:
            config_overrides: Optional configuration overrides

        Returns:
            Configured AGRMCPServer instance
        """
        container = await self._get_container()

        if config_overrides:
            # Create custom ConfigManager with overrides
            from ..config import ConfigManager
            base_config = await container.get(ConfigManager)
            custom_config = ConfigManager.create(
                'data',
                config_data={**base_config.__dict__, **config_overrides}
            )

            # Create server with custom config
            from ..server import AGRMCPServer
            return AGRMCPServer(config_manager=custom_config)
        else:
            # Use standard DI
            from ..server import AGRMCPServer
            return await container.get(AGRMCPServer)


class EnhancedToolFactory:
    """Enhanced tool factory with advanced configuration and lifecycle management."""

    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize enhanced tool factory.

        Args:
            container: Optional DI container
        """
        self._container = container
        self._composite_factory: Optional['CompositeFactory'] = None

    async def _get_composite_factory(self) -> 'CompositeFactory':
        """Get the composite factory instance."""
        if self._composite_factory is None:
            from .factory_patterns import get_global_factory
            self._composite_factory = await get_global_factory()
        return self._composite_factory

    async def create_file_download_tool(self,
                                      instance_id: str = "default",
                                      config_overrides: Optional[Dict[str, Any]] = None) -> 'FileDownloadTool':
        """Create FileDownloadTool with enhanced factory patterns.

        Args:
            instance_id: Unique identifier for the instance
            config_overrides: Optional configuration overrides

        Returns:
            FileDownloadTool instance
        """
        factory = await self._get_composite_factory()
        from ..tools.file_download import FileDownloadTool
        return await factory.create(
            FileDownloadTool,
            instance_id=instance_id,
            config_overrides=config_overrides
        )

    async def create_gene_query_tool(self,
                                   instance_id: str = "default",
                                   config_overrides: Optional[Dict[str, Any]] = None) -> 'GeneQueryTool':
        """Create GeneQueryTool with enhanced factory patterns.

        Args:
            instance_id: Unique identifier for the instance
            config_overrides: Optional configuration overrides

        Returns:
            GeneQueryTool instance
        """
        factory = await self._get_composite_factory()
        from ..tools.gene_query import GeneQueryTool
        return await factory.create(
            GeneQueryTool,
            instance_id=instance_id,
            config_overrides=config_overrides
        )

    async def create_api_schema_tool(self,
                                   instance_id: str = "default",
                                   config_overrides: Optional[Dict[str, Any]] = None) -> 'APISchemaDocumentationTool':
        """Create APISchemaDocumentationTool with enhanced factory patterns.

        Args:
            instance_id: Unique identifier for the instance
            config_overrides: Optional configuration overrides

        Returns:
            APISchemaDocumentationTool instance
        """
        factory = await self._get_composite_factory()
        from ..tools.api_schema import APISchemaDocumentationTool
        return await factory.create(
            APISchemaDocumentationTool,
            instance_id=instance_id,
            config_overrides=config_overrides
        )

    async def create_server(self,
                          instance_id: str = "default",
                          config_overrides: Optional[Dict[str, Any]] = None) -> 'AGRMCPServer':
        """Create AGRMCPServer with enhanced factory patterns.

        Args:
            instance_id: Unique identifier for the instance
            config_overrides: Optional configuration overrides

        Returns:
            AGRMCPServer instance
        """
        factory = await self._get_composite_factory()
        from ..server import AGRMCPServer
        return await factory.create(
            AGRMCPServer,
            instance_id=instance_id,
            config_overrides=config_overrides
        )

    async def create_custom_tool(self,
                               tool_type: Type[T],
                               instance_id: str = "default",
                               config_overrides: Optional[Dict[str, Any]] = None,
                               factory_config: Optional['FactoryConfiguration'] = None) -> T:
        """Create a custom tool with factory patterns.

        Args:
            tool_type: Type of tool to create
            instance_id: Unique identifier for the instance
            config_overrides: Optional configuration overrides
            factory_config: Optional factory configuration

        Returns:
            Tool instance
        """
        from .factory_patterns import create_tool_factory, FactoryConfiguration

        if factory_config is None:
            factory_config = FactoryConfiguration()

        if config_overrides:
            factory_config.config_overrides.update(config_overrides)

        tool_factory = await create_tool_factory(tool_type, factory_config, self._container)
        return await tool_factory.create(instance_id)

    async def shutdown(self):
        """Shutdown the enhanced tool factory."""
        if self._composite_factory:
            await self._composite_factory.shutdown()


# Enhanced factory functions
async def create_enhanced_tool_factory(container: Optional[DIContainer] = None) -> EnhancedToolFactory:
    """Create an enhanced tool factory instance.

    Args:
        container: Optional DI container

    Returns:
        EnhancedToolFactory instance
    """
    return EnhancedToolFactory(container)


async def managed_enhanced_tools(scope_id: str = "default",
                                container: Optional[DIContainer] = None) -> EnhancedToolFactory:
    """Create a managed enhanced tool factory context.

    Args:
        scope_id: Scope identifier for the tools
        container: Optional DI container

    Returns:
        EnhancedToolFactory instance
    """
    if container is None:
        container = get_container()
        if not container._initialized:
            await container.initialize()

    factory = EnhancedToolFactory(container)
    return factory


class FactoryRegistry:
    """Registry for managing multiple factory instances."""

    def __init__(self):
        self._factories: Dict[str, Any] = {}
        self._tool_factories: Dict[str, EnhancedToolFactory] = {}

    def register_factory(self, name: str, factory: Any):
        """Register a factory with a name.

        Args:
            name: Factory name
            factory: Factory instance
        """
        self._factories[name] = factory
        logger.debug(f"Registered factory: {name}")

    def register_tool_factory(self, name: str, factory: EnhancedToolFactory):
        """Register a tool factory with a name.

        Args:
            name: Factory name
            factory: EnhancedToolFactory instance
        """
        self._tool_factories[name] = factory
        logger.debug(f"Registered tool factory: {name}")

    def get_factory(self, name: str) -> Any:
        """Get a factory by name.

        Args:
            name: Factory name

        Returns:
            Factory instance

        Raises:
            ConfigurationError: If factory not found
        """
        if name not in self._factories:
            raise ConfigurationError(
                f"Factory '{name}' not registered",
                config_key="factory_registry"
            )
        return self._factories[name]

    def get_tool_factory(self, name: str) -> EnhancedToolFactory:
        """Get a tool factory by name.

        Args:
            name: Factory name

        Returns:
            EnhancedToolFactory instance

        Raises:
            ConfigurationError: If factory not found
        """
        if name not in self._tool_factories:
            raise ConfigurationError(
                f"Tool factory '{name}' not registered",
                config_key="factory_registry"
            )
        return self._tool_factories[name]

    async def shutdown_all(self):
        """Shutdown all registered factories."""
        for factory in self._tool_factories.values():
            await factory.shutdown()

        for factory in self._factories.values():
            if hasattr(factory, 'shutdown'):
                await factory.shutdown()

        self._factories.clear()
        self._tool_factories.clear()
        logger.info("All factories shut down")


# Global factory registry
_global_factory_registry: Optional[FactoryRegistry] = None


def get_factory_registry() -> FactoryRegistry:
    """Get the global factory registry.

    Returns:
        Global FactoryRegistry instance
    """
    global _global_factory_registry

    if _global_factory_registry is None:
        _global_factory_registry = FactoryRegistry()

    return _global_factory_registry


# Global tool factory instance
_tool_factory: Optional[DIToolFactory] = None


def get_tool_factory() -> DIToolFactory:
    """Get the global tool factory instance.

    Returns:
        DIToolFactory instance
    """
    global _tool_factory
    if _tool_factory is None:
        _tool_factory = DIToolFactory()
    return _tool_factory


# Example integration patterns

class DIIntegratedFileProcessor:
    """Example class showing DI integration patterns."""

    def __init__(self):
        """Initialize file processor."""
        self._container: Optional[DIContainer] = None

    async def initialize(self):
        """Initialize with DI container."""
        self._container = await get_container()

    async def process_file_download(self,
                                  file_url: str,
                                  output_dir: Optional[str] = None) -> str:
        """Process file download with DI-managed resources.

        Args:
            file_url: URL to download
            output_dir: Optional output directory

        Returns:
            Path to downloaded file
        """
        if not self._container:
            await self.initialize()

        # Get tool resource manager for shared resources
        async with self._container.get(ToolResourceManager) as tool_manager:
            # Get file manager
            file_manager = tool_manager.get_file_manager()

            async with file_manager:
                # Download file
                file_path = await file_manager.download_file(file_url)
                return str(file_path)

    async def query_gene_with_resources(self, gene_id: str) -> Dict[str, Any]:
        """Query gene information with shared resources.

        Args:
            gene_id: Gene identifier

        Returns:
            Gene information
        """
        if not self._container:
            await self.initialize()

        # Use dependency scope for this operation
        async with DependencyScope("gene_query") as scope:
            # Get gene query tool
            gene_tool = await scope.get(GeneQueryTool)

            async with gene_tool:
                # Query gene (implementation would depend on actual method)
                return {"gene_id": gene_id, "status": "processed"}


# Export key classes and functions
__all__ = [
    "DIAwareServer",
    "ToolFactory",
    "managed_server",
    "managed_tools",
    "setup_default_container",
    "DIMiddleware",
    "get_configured_tool",
    "inject_dependencies",
    "DIIntegratedFileProcessor",
    "get_configured_file_download_tool",
    "get_configured_gene_query_tool",
    "get_configured_server",
    "DIToolFactory",
    "get_tool_factory",
    "EnhancedToolFactory",
    "create_enhanced_tool_factory",
    "managed_enhanced_tools",
    "FactoryRegistry",
    "get_factory_registry"
]
