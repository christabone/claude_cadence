"""
Enhanced factory patterns for tool instantiation and component creation.

This module provides comprehensive factory patterns for creating various tools
and components with proper configuration injection and lifecycle management.
"""

import asyncio
from typing import TypeVar, Type, Any, Dict, Optional, List, Union, Generic, Protocol
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from enum import Enum

from .dependency_injection import DIContainer, get_container
from .logging_config import get_logger
from ..config import ConfigManager
from ..errors import ConfigurationError, ToolExecutionError

logger = get_logger(__name__)

T = TypeVar('T')
ConfigType = TypeVar('ConfigType')


class FactoryScope(Enum):
    """Factory scoping options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    PROTOTYPE = "prototype"


class ComponentLifecycle(Enum):
    """Component lifecycle management options."""
    MANAGED = "managed"       # Full lifecycle management
    UNMANAGED = "unmanaged"   # No lifecycle management
    CONTEXT = "context"       # Context manager lifecycle


class FactoryConfiguration:
    """Configuration for factory instances."""

    def __init__(self,
                 scope: FactoryScope = FactoryScope.TRANSIENT,
                 lifecycle: ComponentLifecycle = ComponentLifecycle.MANAGED,
                 cache_instances: bool = False,
                 max_instances: Optional[int] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        self.scope = scope
        self.lifecycle = lifecycle
        self.cache_instances = cache_instances
        self.max_instances = max_instances
        self.config_overrides = config_overrides or {}


class AsyncFactoryProtocol(Protocol[T]):
    """Protocol for async factory functions."""

    async def __call__(self, *args, **kwargs) -> T:
        ...


class ComponentFactory(Generic[T], ABC):
    """Abstract base class for component factories."""

    def __init__(self,
                 component_type: Type[T],
                 container: Optional[DIContainer] = None,
                 config: Optional[FactoryConfiguration] = None):
        self.component_type = component_type
        self._container = container
        self.config = config or FactoryConfiguration()
        self._instances: Dict[str, T] = {}
        self._instance_count = 0
        self._exit_stack: Optional[AsyncExitStack] = None

    async def _get_container(self) -> DIContainer:
        """Get the DI container."""
        if self._container is None:
            self._container = get_container()
            if not self._container._initialized:
                await self._container.initialize()
        return self._container

    @abstractmethod
    async def _create_instance(self, *args, **kwargs) -> T:
        """Create a new instance of the component."""
        pass

    async def create(self, instance_id: str = "default", *args, **kwargs) -> T:
        """Create or retrieve a component instance.

        Args:
            instance_id: Unique identifier for the instance
            *args: Positional arguments for component creation
            **kwargs: Keyword arguments for component creation

        Returns:
            Component instance

        Raises:
            ConfigurationError: If factory configuration is invalid
            ToolExecutionError: If component creation fails
        """
        try:
            # Check max instances limit
            if (self.config.max_instances is not None and
                self._instance_count >= self.config.max_instances):
                raise ConfigurationError(
                    f"Maximum instances ({self.config.max_instances}) exceeded for {self.component_type.__name__}",
                    config_key="factory_max_instances"
                )

            # Handle caching and scoping
            if self.config.cache_instances and instance_id in self._instances:
                return self._instances[instance_id]

            # Create new instance
            instance = await self._create_instance(*args, **kwargs)

            # Handle lifecycle management
            if self.config.lifecycle == ComponentLifecycle.CONTEXT:
                instance = await self._manage_context(instance)
            elif self.config.lifecycle == ComponentLifecycle.MANAGED:
                await self._setup_lifecycle_hooks(instance)

            # Cache if configured
            if self.config.cache_instances:
                self._instances[instance_id] = instance

            self._instance_count += 1
            logger.debug(f"Created {self.component_type.__name__} instance: {instance_id}")

            return instance

        except Exception as e:
            logger.error(f"Failed to create {self.component_type.__name__}: {str(e)}")
            raise ToolExecutionError(f"Component creation failed: {str(e)}")

    async def _manage_context(self, instance: T) -> T:
        """Manage context for context manager instances."""
        if not self._exit_stack:
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

        if hasattr(instance, '__aenter__'):
            instance = await self._exit_stack.enter_async_context(instance)

        return instance

    async def _setup_lifecycle_hooks(self, instance: T):
        """Set up lifecycle management hooks."""
        # Add any lifecycle setup logic here
        if hasattr(instance, 'initialize'):
            await instance.initialize()

    async def destroy(self, instance_id: str = "default"):
        """Destroy a specific instance.

        Args:
            instance_id: Identifier of instance to destroy
        """
        if instance_id in self._instances:
            instance = self._instances[instance_id]

            # Cleanup lifecycle
            if hasattr(instance, 'cleanup'):
                await instance.cleanup()

            del self._instances[instance_id]
            self._instance_count -= 1
            logger.debug(f"Destroyed {self.component_type.__name__} instance: {instance_id}")

    async def shutdown(self):
        """Shutdown the factory and clean up all instances."""
        try:
            # Cleanup all instances
            for instance_id in list(self._instances.keys()):
                await self.destroy(instance_id)

            # Cleanup exit stack
            if self._exit_stack:
                await self._exit_stack.__aexit__(None, None, None)
                self._exit_stack = None

            logger.info(f"Factory for {self.component_type.__name__} shutdown complete")

        except Exception as e:
            logger.error(f"Error during factory shutdown: {str(e)}")


class ToolFactory(ComponentFactory):
    """Factory for creating tools with dependency injection."""

    async def _create_instance(self, config_overrides: Optional[Dict[str, Any]] = None) -> T:
        """Create a tool instance with dependency injection."""
        container = await self._get_container()

        # Merge configuration overrides
        final_overrides = {**self.config.config_overrides}
        if config_overrides:
            final_overrides.update(config_overrides)

        if final_overrides:
            # Create custom ConfigManager with overrides
            base_config = await container.get(ConfigManager)
            custom_config = ConfigManager.create(
                'data',
                config_data={**base_config.__dict__, **final_overrides}
            )

            # Create tool with custom config
            if hasattr(self.component_type, '__init__'):
                return self.component_type(config_manager=custom_config)
            else:
                return self.component_type()
        else:
            # Use standard DI
            return await container.get(self.component_type)


class ServiceFactory(ComponentFactory):
    """Factory for creating services with dependency injection."""

    def __init__(self,
                 component_type: Type[T],
                 dependencies: Optional[List[Type]] = None,
                 container: Optional[DIContainer] = None,
                 config: Optional[FactoryConfiguration] = None):
        super().__init__(component_type, container, config)
        self.dependencies = dependencies or []

    async def _create_instance(self, *args, **kwargs) -> T:
        """Create a service instance with dependency resolution."""
        container = await self._get_container()

        # Resolve dependencies
        resolved_deps = []
        for dep_type in self.dependencies:
            dep_instance = await container.get(dep_type)
            resolved_deps.append(dep_instance)

        # Create service instance
        return self.component_type(*resolved_deps, *args, **kwargs)


class PrototypeFactory(ComponentFactory):
    """Factory for creating prototype instances (always new)."""

    def __init__(self,
                 component_type: Type[T],
                 prototype_factory: AsyncFactoryProtocol[T],
                 container: Optional[DIContainer] = None,
                 config: Optional[FactoryConfiguration] = None):
        super().__init__(component_type, container, config)
        self.prototype_factory = prototype_factory

    async def _create_instance(self, *args, **kwargs) -> T:
        """Create a new prototype instance."""
        return await self.prototype_factory(*args, **kwargs)


class CompositeFactory:
    """Factory that can create multiple types of components."""

    def __init__(self, container: Optional[DIContainer] = None):
        self._container = container
        self._factories: Dict[Type, ComponentFactory] = {}

    def register_factory(self, component_type: Type[T], factory: ComponentFactory[T]):
        """Register a factory for a component type.

        Args:
            component_type: Type of component
            factory: Factory instance for the component
        """
        self._factories[component_type] = factory
        logger.debug(f"Registered factory for {component_type.__name__}")

    async def create(self, component_type: Type[T], instance_id: str = "default", **kwargs) -> T:
        """Create a component using the registered factory.

        Args:
            component_type: Type of component to create
            instance_id: Unique identifier for the instance
            **kwargs: Additional arguments for component creation

        Returns:
            Component instance

        Raises:
            ConfigurationError: If no factory registered for component type
        """
        if component_type not in self._factories:
            raise ConfigurationError(
                f"No factory registered for {component_type.__name__}",
                config_key="composite_factory"
            )

        factory = self._factories[component_type]
        return await factory.create(instance_id, **kwargs)

    async def shutdown(self):
        """Shutdown all registered factories."""
        for factory in self._factories.values():
            await factory.shutdown()

        logger.info("CompositeFactory shutdown complete")


# Convenience factory creators
async def create_tool_factory(tool_type: Type[T],
                            config: Optional[FactoryConfiguration] = None,
                            container: Optional[DIContainer] = None) -> ToolFactory[T]:
    """Create a tool factory for the specified tool type.

    Args:
        tool_type: Type of tool to create factory for
        config: Factory configuration
        container: Optional DI container

    Returns:
        ToolFactory instance
    """
    return ToolFactory(tool_type, container, config)


async def create_service_factory(service_type: Type[T],
                                dependencies: Optional[List[Type]] = None,
                                config: Optional[FactoryConfiguration] = None,
                                container: Optional[DIContainer] = None) -> ServiceFactory[T]:
    """Create a service factory for the specified service type.

    Args:
        service_type: Type of service to create factory for
        dependencies: Dependencies required by the service
        config: Factory configuration
        container: Optional DI container

    Returns:
        ServiceFactory instance
    """
    return ServiceFactory(service_type, dependencies, container, config)


async def create_prototype_factory(component_type: Type[T],
                                 prototype_factory: AsyncFactoryProtocol[T],
                                 config: Optional[FactoryConfiguration] = None,
                                 container: Optional[DIContainer] = None) -> PrototypeFactory[T]:
    """Create a prototype factory for the specified component type.

    Args:
        component_type: Type of component to create factory for
        prototype_factory: Factory function that creates prototypes
        config: Factory configuration
        container: Optional DI container

    Returns:
        PrototypeFactory instance
    """
    return PrototypeFactory(component_type, prototype_factory, container, config)


# Global composite factory instance
_global_composite_factory: Optional[CompositeFactory] = None


async def get_global_factory() -> CompositeFactory:
    """Get the global composite factory instance.

    Returns:
        Global CompositeFactory instance
    """
    global _global_composite_factory

    if _global_composite_factory is None:
        _global_composite_factory = CompositeFactory()

        # Register default factories
        await _setup_default_factories(_global_composite_factory)

    return _global_composite_factory


async def _setup_default_factories(composite_factory: CompositeFactory):
    """Set up default factories in the composite factory."""
    from ..tools.file_download import FileDownloadTool
    from ..tools.gene_query import GeneQueryTool
    from ..tools.api_schema import APISchemaDocumentationTool
    from ..server import AGRMCPServer

    # Tool factories with singleton caching
    tool_config = FactoryConfiguration(
        scope=FactoryScope.SINGLETON,
        lifecycle=ComponentLifecycle.MANAGED,
        cache_instances=True
    )

    composite_factory.register_factory(
        FileDownloadTool,
        await create_tool_factory(FileDownloadTool, tool_config)
    )

    composite_factory.register_factory(
        GeneQueryTool,
        await create_tool_factory(GeneQueryTool, tool_config)
    )

    composite_factory.register_factory(
        APISchemaDocumentationTool,
        await create_tool_factory(APISchemaDocumentationTool, tool_config)
    )

    # Server factory with scoped lifecycle
    server_config = FactoryConfiguration(
        scope=FactoryScope.SCOPED,
        lifecycle=ComponentLifecycle.CONTEXT,
        cache_instances=False
    )

    composite_factory.register_factory(
        AGRMCPServer,
        await create_tool_factory(AGRMCPServer, server_config)
    )

    logger.info("Default factories registered")
