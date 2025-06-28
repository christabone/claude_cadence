"""
Comprehensive Dependency Injection Framework for AGR MCP Server.

This module provides a full-featured dependency injection container that manages
all application dependencies including tools, services, configuration, and resources.
Supports scoping, lifecycle management, and complex dependency graphs.
"""

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from enum import Enum
from typing import (
    Dict, Any, Optional, Type, TypeVar, Callable, List,
    Union, AsyncContextManager, Set, get_type_hints, get_origin, get_args,
    Generic
)
import weakref

from ..config import ConfigManager
from ..errors import ConfigurationError, AGRMCPError
from .context_managers import ApplicationResourceManager, ToolResourceManager, HttpClientManager
from .http_client import AGRHttpClient
from .file_manager import FileManager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LifecycleScope(Enum):
    """Defines the lifecycle scope for dependency instances."""
    SINGLETON = "singleton"      # One instance for entire application
    SCOPED = "scoped"           # One instance per scope (e.g., per request)
    TRANSIENT = "transient"     # New instance every time
    CONTEXT = "context"         # Managed by async context manager


class DependencyInfo:
    """Information about a registered dependency."""

    def __init__(self,
                 dependency_type: Type[T],
                 factory: Optional[Callable[..., T]] = None,
                 instance: Optional[T] = None,
                 scope: LifecycleScope = LifecycleScope.TRANSIENT,
                 dependencies: Optional[List[Type]] = None,
                 is_async: bool = False,
                 context_manager: bool = False):
        """Initialize dependency information.

        Args:
            dependency_type: The type of the dependency
            factory: Factory function to create instances
            instance: Pre-created instance (for singletons)
            scope: Lifecycle scope
            dependencies: List of dependency types this depends on
            is_async: Whether the factory is async
            context_manager: Whether the dependency is a context manager
        """
        self.dependency_type = dependency_type
        self.factory = factory
        self.instance = instance
        self.scope = scope
        self.dependencies = dependencies or []
        self.is_async = is_async
        self.context_manager = context_manager
        self._creation_lock = asyncio.Lock()


class DIContainer:
    """Comprehensive dependency injection container."""

    def __init__(self):
        """Initialize the DI container."""
        self._registrations: Dict[Type, DependencyInfo] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._context_managers: List[AsyncContextManager] = []
        self._exit_stack: Optional[AsyncExitStack] = None
        self._initialized = False
        self._creation_locks: Dict[Type, asyncio.Lock] = {}

    async def initialize(self):
        """Initialize the DI container."""
        if self._initialized:
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Register default dependencies
        await self._register_defaults()

        self._initialized = True
        logger.info("DI Container initialized")

    async def shutdown(self):
        """Shutdown the DI container and clean up resources."""
        if not self._initialized:
            return

        try:
            # Clean up context managers
            if self._exit_stack:
                await self._exit_stack.__aexit__(None, None, None)
                self._exit_stack = None

            # Clear all instances
            self._singletons.clear()
            self._scoped_instances.clear()
            self._context_managers.clear()

            self._initialized = False
            logger.info("DI Container shutdown complete")

        except Exception as e:
            logger.error(f"Error during DI container shutdown: {str(e)}")

    async def _register_defaults(self):
        """Register default dependencies."""
        # Register ConfigManager
        self.register_singleton(ConfigManager, factory=lambda: ConfigManager.create())

        # Register ApplicationResourceManager
        self.register_scoped(
            ApplicationResourceManager,
            factory=self._create_application_manager,
            dependencies=[ConfigManager],
            is_async=True,
            context_manager=True
        )

        # Register ToolResourceManager
        self.register_scoped(
            ToolResourceManager,
            factory=self._create_tool_manager,
            dependencies=[ApplicationResourceManager],
            is_async=True,
            context_manager=True
        )

        # Register HttpClientManager
        self.register_scoped(
            HttpClientManager,
            factory=self._create_http_manager,
            dependencies=[ConfigManager],
            is_async=True,
            context_manager=True
        )

        # Register FileManager
        self.register_transient(
            FileManager,
            factory=self._create_file_manager,
            dependencies=[ConfigManager],
            context_manager=True
        )

    async def _create_application_manager(self, config_manager: ConfigManager) -> ApplicationResourceManager:
        """Factory for ApplicationResourceManager."""
        return ApplicationResourceManager(config_manager)

    async def _create_tool_manager(self, app_manager: ApplicationResourceManager) -> ToolResourceManager:
        """Factory for ToolResourceManager."""
        return await app_manager.get_tool_manager()

    async def _create_http_manager(self, config_manager: ConfigManager) -> HttpClientManager:
        """Factory for HttpClientManager."""
        return HttpClientManager(config_manager)

    def _create_file_manager(self, config_manager: ConfigManager) -> FileManager:
        """Factory for FileManager."""
        from pathlib import Path
        return FileManager(Path(config_manager.default_download_dir))

    def register_singleton(self,
                          dependency_type: Type[T],
                          factory: Optional[Callable[..., T]] = None,
                          instance: Optional[T] = None,
                          dependencies: Optional[List[Type]] = None,
                          is_async: bool = False,
                          context_manager: bool = False):
        """Register a singleton dependency.

        Args:
            dependency_type: Type to register
            factory: Factory function (optional)
            instance: Pre-created instance (optional)
            dependencies: Dependencies this type requires
            is_async: Whether the factory is async
            context_manager: Whether it's a context manager
        """
        if factory is None and instance is None:
            factory = dependency_type

        self._registrations[dependency_type] = DependencyInfo(
            dependency_type=dependency_type,
            factory=factory,
            instance=instance,
            scope=LifecycleScope.SINGLETON,
            dependencies=dependencies or [],
            is_async=is_async,
            context_manager=context_manager
        )

        logger.debug(f"Registered singleton: {dependency_type.__name__}")

    def register_scoped(self,
                       dependency_type: Type[T],
                       factory: Optional[Callable[..., T]] = None,
                       instance: Optional[T] = None,
                       dependencies: Optional[List[Type]] = None,
                       is_async: bool = False,
                       context_manager: bool = False):
        """Register a scoped dependency.

        Args:
            dependency_type: Type to register
            factory: Factory function (optional)
            instance: Pre-created instance (optional)
            dependencies: Dependencies this type requires
            is_async: Whether the factory is async
            context_manager: Whether it's a context manager
        """
        if factory is None and instance is None:
            factory = dependency_type

        self._registrations[dependency_type] = DependencyInfo(
            dependency_type=dependency_type,
            factory=factory,
            instance=instance,
            scope=LifecycleScope.SCOPED,
            dependencies=dependencies or [],
            is_async=is_async,
            context_manager=context_manager
        )

        logger.debug(f"Registered scoped: {dependency_type.__name__}")

    def register_transient(self,
                          dependency_type: Type[T],
                          factory: Optional[Callable[..., T]] = None,
                          dependencies: Optional[List[Type]] = None,
                          is_async: bool = False,
                          context_manager: bool = False):
        """Register a transient dependency.

        Args:
            dependency_type: Type to register
            factory: Factory function (optional)
            dependencies: Dependencies this type requires
            is_async: Whether the factory is async
            context_manager: Whether it's a context manager
        """
        if factory is None:
            factory = dependency_type

        self._registrations[dependency_type] = DependencyInfo(
            dependency_type=dependency_type,
            factory=factory,
            scope=LifecycleScope.TRANSIENT,
            dependencies=dependencies or [],
            is_async=is_async,
            context_manager=context_manager
        )

        logger.debug(f"Registered transient: {dependency_type.__name__}")

    def register_context(self,
                         dependency_type: Type[T],
                         factory: Optional[Callable[..., T]] = None,
                         dependencies: Optional[List[Type]] = None,
                         is_async: bool = False):
        """Register a context-managed dependency.

        Args:
            dependency_type: Type to register
            factory: Factory function (optional)
            dependencies: Dependencies this type requires
            is_async: Whether the factory is async
        """
        if factory is None:
            factory = dependency_type

        self._registrations[dependency_type] = DependencyInfo(
            dependency_type=dependency_type,
            factory=factory,
            scope=LifecycleScope.CONTEXT,
            dependencies=dependencies or [],
            is_async=is_async,
            context_manager=True
        )

        logger.debug(f"Registered context: {dependency_type.__name__}")

    async def get(self, dependency_type: Type[T], scope_id: str = "default") -> T:
        """Get an instance of the specified dependency type.

        Args:
            dependency_type: Type to resolve
            scope_id: Scope identifier for scoped dependencies

        Returns:
            Instance of the requested type

        Raises:
            ConfigurationError: If dependency is not registered or cannot be resolved
        """
        if not self._initialized:
            await self.initialize()

        if dependency_type not in self._registrations:
            raise ConfigurationError(
                f"Dependency type {dependency_type.__name__} is not registered",
                config_key="dependency_injection"
            )

        dep_info = self._registrations[dependency_type]

        # Handle different scopes
        if dep_info.scope == LifecycleScope.SINGLETON:
            return await self._get_singleton(dependency_type, dep_info)
        elif dep_info.scope == LifecycleScope.SCOPED:
            return await self._get_scoped(dependency_type, dep_info, scope_id)
        elif dep_info.scope == LifecycleScope.CONTEXT:
            return await self._get_context(dependency_type, dep_info)
        else:  # TRANSIENT
            return await self._create_instance(dependency_type, dep_info)

    async def _get_singleton(self, dependency_type: Type[T], dep_info: DependencyInfo) -> T:
        """Get or create singleton instance."""
        if dependency_type in self._singletons:
            return self._singletons[dependency_type]

        # Use lock to prevent race conditions
        if dependency_type not in self._creation_locks:
            self._creation_locks[dependency_type] = asyncio.Lock()

        async with self._creation_locks[dependency_type]:
            # Double check in case another coroutine created it
            if dependency_type in self._singletons:
                return self._singletons[dependency_type]

            # Create instance
            instance = await self._create_instance(dependency_type, dep_info)

            # For context managers, enter them and track for cleanup
            if dep_info.context_manager and hasattr(instance, '__aenter__'):
                instance = await self._exit_stack.enter_async_context(instance)

            self._singletons[dependency_type] = instance
            logger.debug(f"Created singleton: {dependency_type.__name__}")
            return instance

    async def _get_scoped(self, dependency_type: Type[T], dep_info: DependencyInfo, scope_id: str) -> T:
        """Get or create scoped instance."""
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}

        scoped_cache = self._scoped_instances[scope_id]

        if dependency_type in scoped_cache:
            return scoped_cache[dependency_type]

        # Use lock to prevent race conditions
        lock_key = f"{dependency_type.__name__}_{scope_id}"
        if lock_key not in self._creation_locks:
            self._creation_locks[lock_key] = asyncio.Lock()

        async with self._creation_locks[lock_key]:
            # Double check
            if dependency_type in scoped_cache:
                return scoped_cache[dependency_type]

            # Create instance
            instance = await self._create_instance(dependency_type, dep_info)

            # For context managers, enter them
            if dep_info.context_manager and hasattr(instance, '__aenter__'):
                instance = await instance.__aenter__()
                # Store reference for cleanup
                self._context_managers.append(instance)

            scoped_cache[dependency_type] = instance
            logger.debug(f"Created scoped instance: {dependency_type.__name__} (scope: {scope_id})")
            return instance

    async def _get_context(self, dependency_type: Type[T], dep_info: DependencyInfo) -> T:
        """Get context-managed instance."""
        # Context instances are always created fresh
        instance = await self._create_instance(dependency_type, dep_info)

        # Enter context and track for cleanup
        if hasattr(instance, '__aenter__'):
            instance = await self._exit_stack.enter_async_context(instance)

        return instance

    async def _create_instance(self, dependency_type: Type[T], dep_info: DependencyInfo) -> T:
        """Create a new instance of the dependency."""
        # Resolve dependencies
        resolved_deps = []
        for dep_type in dep_info.dependencies:
            dep_instance = await self.get(dep_type)
            resolved_deps.append(dep_instance)

        # Create instance
        if dep_info.instance is not None:
            # Use pre-created instance
            return dep_info.instance
        elif dep_info.factory:
            # Use factory
            if dep_info.is_async:
                if asyncio.iscoroutinefunction(dep_info.factory):
                    instance = await dep_info.factory(*resolved_deps)
                else:
                    instance = dep_info.factory(*resolved_deps)
            else:
                instance = dep_info.factory(*resolved_deps)
        else:
            # Use constructor
            instance = dependency_type(*resolved_deps)

        logger.debug(f"Created instance: {dependency_type.__name__}")
        return instance

    def clear_scope(self, scope_id: str):
        """Clear all instances in a specific scope.

        Args:
            scope_id: Scope to clear
        """
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
            logger.debug(f"Cleared scope: {scope_id}")

    def is_registered(self, dependency_type: Type) -> bool:
        """Check if a dependency type is registered.

        Args:
            dependency_type: Type to check

        Returns:
            True if registered, False otherwise
        """
        return dependency_type in self._registrations

    def get_registration_info(self, dependency_type: Type) -> Optional[DependencyInfo]:
        """Get registration information for a dependency type.

        Args:
            dependency_type: Type to get info for

        Returns:
            DependencyInfo if registered, None otherwise
        """
        return self._registrations.get(dependency_type)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the DI container.

        Returns:
            Health check results
        """
        return {
            "initialized": self._initialized,
            "registered_types": len(self._registrations),
            "singleton_instances": len(self._singletons),
            "scoped_instances": sum(len(scope) for scope in self._scoped_instances.values()),
            "context_managers": len(self._context_managers)
        }


# Global DI container instance
_container: Optional[DIContainer] = None


async def get_container() -> DIContainer:
    """Get the global DI container instance.

    Returns:
        DIContainer instance
    """
    global _container
    if _container is None:
        _container = DIContainer()
        await _container.initialize()
    return _container


async def configure_container(container: DIContainer):
    """Configure the global DI container.

    Args:
        container: Container to set as global
    """
    global _container
    if _container is not None:
        await _container.shutdown()
    _container = container


async def shutdown_container():
    """Shutdown the global DI container."""
    global _container
    if _container is not None:
        await _container.shutdown()
        _container = None

# Enhanced DI integration
async def get_enhanced_container() -> 'EnhancedDIContainer':
    """Get an enhanced DI container with advanced features.

    Returns:
        Enhanced DI container instance
    """
    from .enhanced_di_framework import get_enhanced_container as _get_enhanced
    return await _get_enhanced()


async def configure_enhanced_di(config_manager: Optional['ConfigManager'] = None) -> 'EnhancedDIContainer':
    """Configure the enhanced DI system with modules and event handlers.

    Args:
        config_manager: Optional configuration manager

    Returns:
        Configured enhanced DI container
    """
    from .enhanced_di_framework import configure_enhanced_di as _configure_enhanced
    return await _configure_enhanced(config_manager)


def inject_dependencies(*dependency_types: Type):
    """Decorator for automatic dependency injection.

    Args:
        *dependency_types: Types to inject as function arguments

    Returns:
        Decorator function
    """
    from .enhanced_di_framework import inject
    return inject(*dependency_types)


def auto_register(dependency_type: Type[T], scope: LifecycleScope = LifecycleScope.TRANSIENT):
    """Class decorator for automatic DI registration.

    Args:
        dependency_type: Type to register class as
        scope: Lifecycle scope for the dependency

    Returns:
        Class decorator
    """
    from .enhanced_di_framework import injectable
    return injectable(dependency_type, scope)


class DIFeatures:
    """Feature toggles and utilities for DI system."""

    ENHANCED_FEATURES_AVAILABLE = True

    @staticmethod
    async def enable_enhanced_features():
        """Enable enhanced DI features."""
        global _container

        try:
            # Migrate existing container to enhanced version
            enhanced_container = await get_enhanced_container()

            # Copy existing registrations if any
            if _container and _container._initialized:
                # This would copy over existing registrations
                # For now, we'll start fresh with enhanced container
                await _container.shutdown()

            _container = enhanced_container
            logger.info("Enhanced DI features enabled")

        except Exception as e:
            logger.error(f"Failed to enable enhanced DI features: {str(e)}")

    @staticmethod
    async def get_di_metrics() -> Dict[str, Any]:
        """Get DI system metrics.

        Returns:
            Dictionary of DI metrics
        """
        container = get_container()

        if hasattr(container, 'health_check'):
            return await container.health_check()
        else:
            return {
                "initialized": container._initialized if hasattr(container, '_initialized') else False,
                "registered_types": len(container._registrations) if hasattr(container, '_registrations') else 0
            }

    @staticmethod
    async def validate_dependency_graph():
        """Validate the dependency graph for circular dependencies.

        Returns:
            Validation results
        """
        container = get_container()

        if hasattr(container, 'get_dependency_graph'):
            graph = container.get_dependency_graph()

            # Simple cycle detection
            visited = set()
            rec_stack = set()

            def has_cycle(node, graph, visited, rec_stack):
                visited.add(node)
                rec_stack.add(node)

                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        if has_cycle(neighbor, graph, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            cycles = []
            for node in graph:
                if node not in visited:
                    if has_cycle(node, graph, visited, rec_stack):
                        cycles.append(node)

            return {
                "has_cycles": len(cycles) > 0,
                "cycles": [n.__name__ for n in cycles],
                "total_dependencies": len(graph)
            }
        else:
            return {"status": "basic_container", "validation": "not_available"}


class DIScope:
    """Context manager for DI scoping."""

    def __init__(self, scope_id: str = "default"):
        """Initialize DI scope.

        Args:
            scope_id: Scope identifier
        """
        self.scope_id = scope_id
        self._container: Optional[DIContainer] = None

    async def __aenter__(self):
        """Enter the DI scope."""
        self._container = get_container()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the DI scope and clean up scoped instances."""
        if self._container and hasattr(self._container, 'clear_scope'):
            self._container.clear_scope(self.scope_id)

        return False

    async def get(self, dependency_type: Type[T]) -> T:
        """Get a dependency within this scope.

        Args:
            dependency_type: Type of dependency to get

        Returns:
            Dependency instance
        """
        if not self._container:
            raise AGRMCPError("DI scope not active")

        return await self._container.get(dependency_type, self.scope_id)


# Enhanced injection patterns
class LazyDependency(Generic[T]):
    """Lazy-loaded dependency wrapper."""

    def __init__(self, dependency_type: Type[T]):
        """Initialize lazy dependency.

        Args:
            dependency_type: Type of dependency
        """
        self.dependency_type = dependency_type
        self._instance: Optional[T] = None
        self._loaded = False

    async def get(self) -> T:
        """Get the dependency instance (lazy loaded).

        Returns:
            Dependency instance
        """
        if not self._loaded:
            container = get_container()
            self._instance = await container.get(self.dependency_type)
            self._loaded = True

        return self._instance

    def is_loaded(self) -> bool:
        """Check if dependency is loaded.

        Returns:
            True if loaded, False otherwise
        """
        return self._loaded


async def lazy_inject(dependency_type: Type[T]) -> LazyDependency[T]:
    """Create a lazy dependency injection.

    Args:
        dependency_type: Type of dependency

    Returns:
        Lazy dependency wrapper
    """
    return LazyDependency(dependency_type)


# Configuration-driven DI setup
async def setup_di_from_config(config_manager: 'ConfigManager'):
    """Set up DI container from configuration.

    Args:
        config_manager: Configuration manager instance
    """
    container = get_container()

    # Check if enhanced features should be enabled
    if hasattr(config_manager, 'enable_enhanced_di') and config_manager.enable_enhanced_di:
        await DIFeatures.enable_enhanced_features()
        container = get_container()  # Get the enhanced container

    # Configure based on settings
    if hasattr(config_manager, 'di_health_check_interval'):
        # Set health check interval if supported
        if hasattr(container, '_health_check_interval'):
            container._health_check_interval = config_manager.di_health_check_interval

    logger.info("DI system configured from config")


# Decorator for dependency injection
def inject(*dependency_types: Type):
    """Decorator for automatic dependency injection.

    Args:
        *dependency_types: Types to inject

    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            container = await get_container()

            # Resolve dependencies
            resolved_deps = []
            for dep_type in dependency_types:
                dep_instance = await container.get(dep_type)
                resolved_deps.append(dep_instance)

            # Call function with injected dependencies
            if asyncio.iscoroutinefunction(func):
                return await func(*resolved_deps, *args, **kwargs)
            else:
                return func(*resolved_deps, *args, **kwargs)

        return wrapper
    return decorator


# Context manager for scoped dependencies
class DependencyScope:
    """Context manager for scoped dependency injection."""

    def __init__(self, scope_id: str = None):
        """Initialize dependency scope.

        Args:
            scope_id: Unique identifier for this scope
        """
        self.scope_id = scope_id or f"scope_{id(self)}"
        self._container: Optional[DIContainer] = None

    async def __aenter__(self) -> 'DependencyScope':
        """Enter the dependency scope."""
        self._container = await get_container()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the dependency scope and clear instances."""
        if self._container:
            self._container.clear_scope(self.scope_id)

    async def get(self, dependency_type: Type[T]) -> T:
        """Get a dependency within this scope.

        Args:
            dependency_type: Type to resolve

        Returns:
            Instance of the requested type
        """
        if not self._container:
            raise AGRMCPError("DependencyScope not initialized")
        return await self._container.get(dependency_type, self.scope_id)
