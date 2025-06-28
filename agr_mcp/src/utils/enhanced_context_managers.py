"""
Enhanced async context managers for comprehensive resource management.

This module provides advanced async context managers with sophisticated lifecycle
control, error handling, and resource cleanup capabilities.
"""

import asyncio
import time
from typing import TypeVar, Type, Any, Dict, Optional, List, Union, Generic, Protocol, Callable, AsyncIterator
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from enum import Enum
from dataclasses import dataclass, field
import weakref

from .logging_config import get_logger
from ..config import ConfigManager
from ..errors import AGRMCPError, ConfigurationError, ResourceNotFoundError

logger = get_logger(__name__)

T = TypeVar('T')
ResourceType = TypeVar('ResourceType')


class ResourceState(Enum):
    """Resource lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    IN_USE = "in_use"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class CleanupStrategy(Enum):
    """Resource cleanup strategies."""
    IMMEDIATE = "immediate"     # Clean up resources immediately
    GRACEFUL = "graceful"       # Allow graceful shutdown with timeout
    LAZY = "lazy"               # Clean up when convenient
    PERSISTENT = "persistent"   # Keep resources alive across contexts


@dataclass
class ResourceMetrics:
    """Metrics for resource usage tracking."""
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    error_count: int = 0
    cleanup_time: Optional[float] = None
    lifetime_seconds: Optional[float] = None


class ResourceLifecycleProtocol(Protocol):
    """Protocol for resource lifecycle management."""

    async def initialize(self) -> None:
        """Initialize the resource."""
        ...

    async def cleanup(self) -> None:
        """Clean up the resource."""
        ...

    async def health_check(self) -> bool:
        """Check if resource is healthy."""
        ...


class ManagedResource(Generic[T]):
    """Wrapper for managed resources with lifecycle tracking."""

    def __init__(self,
                 resource: T,
                 resource_id: str,
                 cleanup_strategy: CleanupStrategy = CleanupStrategy.GRACEFUL,
                 cleanup_timeout: float = 30.0):
        self.resource = resource
        self.resource_id = resource_id
        self.cleanup_strategy = cleanup_strategy
        self.cleanup_timeout = cleanup_timeout
        self.state = ResourceState.CREATED
        self.metrics = ResourceMetrics()
        self._lock = asyncio.Lock()

    async def access(self) -> T:
        """Access the resource and update metrics."""
        async with self._lock:
            if self.state not in (ResourceState.READY, ResourceState.IN_USE):
                raise AGRMCPError(f"Resource {self.resource_id} not available: {self.state}")

            self.metrics.access_count += 1
            self.metrics.last_access_time = time.time()
            self.state = ResourceState.IN_USE

            return self.resource

    async def release(self):
        """Release the resource."""
        async with self._lock:
            if self.state == ResourceState.IN_USE:
                self.state = ResourceState.READY

    async def initialize(self):
        """Initialize the resource."""
        async with self._lock:
            if self.state != ResourceState.CREATED:
                return

            self.state = ResourceState.INITIALIZING

            try:
                if hasattr(self.resource, 'initialize'):
                    await self.resource.initialize()
                elif hasattr(self.resource, '__aenter__'):
                    await self.resource.__aenter__()

                self.state = ResourceState.READY
                logger.debug(f"Resource {self.resource_id} initialized")

            except Exception as e:
                self.state = ResourceState.ERROR
                self.metrics.error_count += 1
                logger.error(f"Failed to initialize resource {self.resource_id}: {str(e)}")
                raise

    async def cleanup(self):
        """Clean up the resource."""
        async with self._lock:
            if self.state in (ResourceState.CLOSING, ResourceState.CLOSED):
                return

            old_state = self.state
            self.state = ResourceState.CLOSING

            try:
                if self.cleanup_strategy == CleanupStrategy.IMMEDIATE:
                    await self._immediate_cleanup()
                elif self.cleanup_strategy == CleanupStrategy.GRACEFUL:
                    await self._graceful_cleanup()
                elif self.cleanup_strategy == CleanupStrategy.LAZY:
                    # Schedule for later cleanup
                    asyncio.create_task(self._lazy_cleanup())
                    return
                # PERSISTENT resources are not cleaned up

                self.state = ResourceState.CLOSED
                self.metrics.cleanup_time = time.time()
                self.metrics.lifetime_seconds = self.metrics.cleanup_time - self.metrics.creation_time

                logger.debug(f"Resource {self.resource_id} cleaned up")

            except Exception as e:
                self.state = ResourceState.ERROR
                self.metrics.error_count += 1
                logger.error(f"Failed to cleanup resource {self.resource_id}: {str(e)}")

    async def _immediate_cleanup(self):
        """Immediate cleanup without timeout."""
        if hasattr(self.resource, 'cleanup'):
            await self.resource.cleanup()
        elif hasattr(self.resource, '__aexit__'):
            await self.resource.__aexit__(None, None, None)
        elif hasattr(self.resource, 'close'):
            await self.resource.close()

    async def _graceful_cleanup(self):
        """Graceful cleanup with timeout."""
        try:
            await asyncio.wait_for(self._immediate_cleanup(), timeout=self.cleanup_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Resource {self.resource_id} cleanup timed out after {self.cleanup_timeout}s")
            # Force cleanup
            try:
                if hasattr(self.resource, 'close'):
                    await self.resource.close()
            except Exception as e:
                logger.error(f"Force cleanup failed for {self.resource_id}: {str(e)}")

    async def _lazy_cleanup(self):
        """Lazy cleanup - clean up when system is idle."""
        # Wait a bit to see if resource gets reused
        await asyncio.sleep(1.0)

        if self.state == ResourceState.CLOSING:
            await self._immediate_cleanup()
            self.state = ResourceState.CLOSED


class AdvancedResourceManager:
    """Advanced resource manager with sophisticated lifecycle control."""

    def __init__(self,
                 config_manager: Optional[ConfigManager] = None,
                 max_resources: int = 100,
                 default_cleanup_strategy: CleanupStrategy = CleanupStrategy.GRACEFUL,
                 health_check_interval: float = 60.0):
        self.config_manager = config_manager or ConfigManager.create()
        self.max_resources = max_resources
        self.default_cleanup_strategy = default_cleanup_strategy
        self.health_check_interval = health_check_interval

        self._resources: Dict[str, ManagedResource] = {}
        self._resource_factories: Dict[Type, Callable] = {}
        self._exit_stack: Optional[AsyncExitStack] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def register_factory(self, resource_type: Type[T], factory: Callable[..., T]):
        """Register a factory for a resource type.

        Args:
            resource_type: Type of resource
            factory: Factory function for creating resources
        """
        self._resource_factories[resource_type] = factory
        logger.debug(f"Registered factory for {resource_type.__name__}")

    async def get_resource(self,
                          resource_type: Type[T],
                          resource_id: str = "default",
                          cleanup_strategy: Optional[CleanupStrategy] = None,
                          **factory_kwargs) -> T:
        """Get a managed resource instance.

        Args:
            resource_type: Type of resource to get
            resource_id: Unique identifier for the resource
            cleanup_strategy: Optional cleanup strategy override
            **factory_kwargs: Arguments for resource factory

        Returns:
            Resource instance

        Raises:
            ConfigurationError: If no factory registered for resource type
            AGRMCPError: If resource creation fails
        """
        async with self._lock:
            # Check if resource already exists
            if resource_id in self._resources:
                managed_resource = self._resources[resource_id]
                return await managed_resource.access()

            # Check resource limits
            if len(self._resources) >= self.max_resources:
                await self._cleanup_stale_resources()

                if len(self._resources) >= self.max_resources:
                    raise AGRMCPError(f"Maximum resources ({self.max_resources}) exceeded")

            # Create new resource
            if resource_type not in self._resource_factories:
                raise ConfigurationError(
                    f"No factory registered for {resource_type.__name__}",
                    config_key="resource_factory"
                )

            factory = self._resource_factories[resource_type]

            try:
                # Create resource instance
                if asyncio.iscoroutinefunction(factory):
                    resource_instance = await factory(**factory_kwargs)
                else:
                    resource_instance = factory(**factory_kwargs)

                # Wrap in managed resource
                strategy = cleanup_strategy or self.default_cleanup_strategy
                managed_resource = ManagedResource(
                    resource_instance,
                    resource_id,
                    strategy
                )

                # Initialize resource
                await managed_resource.initialize()

                # Track resource
                self._resources[resource_id] = managed_resource

                logger.debug(f"Created managed resource: {resource_id} ({resource_type.__name__})")

                return await managed_resource.access()

            except Exception as e:
                logger.error(f"Failed to create resource {resource_id}: {str(e)}")
                raise AGRMCPError(f"Resource creation failed: {str(e)}")

    async def release_resource(self, resource_id: str):
        """Release a resource back to the manager.

        Args:
            resource_id: ID of resource to release
        """
        if resource_id in self._resources:
            await self._resources[resource_id].release()

    async def remove_resource(self, resource_id: str):
        """Remove and cleanup a resource.

        Args:
            resource_id: ID of resource to remove
        """
        async with self._lock:
            if resource_id in self._resources:
                managed_resource = self._resources[resource_id]
                await managed_resource.cleanup()
                del self._resources[resource_id]
                logger.debug(f"Removed resource: {resource_id}")

    async def _cleanup_stale_resources(self):
        """Clean up stale or unused resources."""
        current_time = time.time()
        stale_threshold = 300.0  # 5 minutes

        stale_resources = []
        for resource_id, managed_resource in self._resources.items():
            if (current_time - managed_resource.metrics.last_access_time > stale_threshold and
                managed_resource.state == ResourceState.READY):
                stale_resources.append(resource_id)

        for resource_id in stale_resources:
            await self.remove_resource(resource_id)

        if stale_resources:
            logger.info(f"Cleaned up {len(stale_resources)} stale resources")

    async def _health_check_loop(self):
        """Background health check for all resources."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                unhealthy_resources = []
                for resource_id, managed_resource in self._resources.items():
                    try:
                        if hasattr(managed_resource.resource, 'health_check'):
                            is_healthy = await managed_resource.resource.health_check()
                            if not is_healthy:
                                unhealthy_resources.append(resource_id)
                    except Exception as e:
                        logger.warning(f"Health check failed for {resource_id}: {str(e)}")
                        unhealthy_resources.append(resource_id)

                # Remove unhealthy resources
                for resource_id in unhealthy_resources:
                    await self.remove_resource(resource_id)

                if unhealthy_resources:
                    logger.info(f"Removed {len(unhealthy_resources)} unhealthy resources")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get resource metrics.

        Returns:
            Dictionary of resource metrics
        """
        total_resources = len(self._resources)
        total_access_count = sum(r.metrics.access_count for r in self._resources.values())
        total_error_count = sum(r.metrics.error_count for r in self._resources.values())

        state_counts = {}
        for state in ResourceState:
            state_counts[state.value] = sum(
                1 for r in self._resources.values() if r.state == state
            )

        return {
            "total_resources": total_resources,
            "max_resources": self.max_resources,
            "total_access_count": total_access_count,
            "total_error_count": total_error_count,
            "state_counts": state_counts,
            "factory_count": len(self._resource_factories)
        }

    async def __aenter__(self):
        """Async context manager entry."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("AdvancedResourceManager initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            # Stop health check
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Clean up all resources
            for resource_id in list(self._resources.keys()):
                await self.remove_resource(resource_id)

            # Clean up exit stack
            if self._exit_stack:
                await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
                self._exit_stack = None

            logger.info("AdvancedResourceManager shutdown complete")

        except Exception as e:
            logger.error(f"Error during AdvancedResourceManager cleanup: {str(e)}")

        return False


class DatabaseConnectionManager:
    """Specialized context manager for database connections."""

    def __init__(self,
                 config_manager: Optional[ConfigManager] = None,
                 pool_size: int = 10,
                 connection_timeout: float = 30.0):
        self.config_manager = config_manager or ConfigManager.create()
        self.pool_size = pool_size
        self.connection_timeout = connection_timeout

        self._connection_pool: Optional[Any] = None
        self._active_connections: weakref.WeakSet = weakref.WeakSet()

    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self._connection_pool:
            raise AGRMCPError("DatabaseConnectionManager not initialized as context manager")

        # This would be implemented with actual database driver
        # For now, this is a placeholder pattern
        pass

    async def execute_transaction(self, transaction_func: Callable):
        """Execute a function within a database transaction."""
        connection = await self.get_connection()

        try:
            # Begin transaction
            await connection.begin()

            try:
                result = await transaction_func(connection)
                await connection.commit()
                return result
            except Exception:
                await connection.rollback()
                raise

        finally:
            await self.release_connection(connection)

    async def release_connection(self, connection):
        """Release a connection back to the pool."""
        # Implementation would return connection to pool
        pass

    async def __aenter__(self):
        """Initialize database connection pool."""
        # This would initialize actual database connection pool
        logger.info("DatabaseConnectionManager initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close all database connections."""
        try:
            # Close all active connections
            for connection in list(self._active_connections):
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing database connection: {str(e)}")

            # Close connection pool
            if self._connection_pool:
                await self._connection_pool.close()

            logger.info("DatabaseConnectionManager cleanup complete")

        except Exception as e:
            logger.error(f"Error during DatabaseConnectionManager cleanup: {str(e)}")

        return False


@asynccontextmanager
async def managed_resource_context(resource_manager: AdvancedResourceManager,
                                 resource_type: Type[T],
                                 resource_id: str = "default",
                                 **kwargs) -> AsyncIterator[T]:
    """Context manager for individual resources.

    Args:
        resource_manager: Resource manager instance
        resource_type: Type of resource to manage
        resource_id: Unique identifier for the resource
        **kwargs: Additional arguments for resource creation

    Yields:
        Resource instance
    """
    resource = await resource_manager.get_resource(resource_type, resource_id, **kwargs)

    try:
        yield resource
    finally:
        await resource_manager.release_resource(resource_id)


@asynccontextmanager
async def scoped_resources(*managers) -> AsyncIterator[List[Any]]:
    """Context manager for multiple resource managers.

    Args:
        *managers: Resource manager instances

    Yields:
        List of initialized resource managers
    """
    initialized_managers = []

    try:
        for manager in managers:
            await manager.__aenter__()
            initialized_managers.append(manager)

        yield initialized_managers

    finally:
        # Clean up in reverse order
        for manager in reversed(initialized_managers):
            try:
                await manager.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up resource manager: {str(e)}")


# Convenience functions for creating common resource managers
async def create_http_resource_manager(config_manager: Optional[ConfigManager] = None) -> AdvancedResourceManager:
    """Create a resource manager configured for HTTP clients.

    Args:
        config_manager: Optional configuration manager

    Returns:
        Configured AdvancedResourceManager
    """
    from .http_client import AGRHttpClient

    manager = AdvancedResourceManager(
        config_manager=config_manager,
        max_resources=20,
        default_cleanup_strategy=CleanupStrategy.GRACEFUL
    )

    # Register HTTP client factory
    async def http_client_factory(base_url: Optional[str] = None, **kwargs):
        return AGRHttpClient(base_url=base_url, **kwargs)

    manager.register_factory(AGRHttpClient, http_client_factory)

    return manager


async def create_tool_resource_manager(config_manager: Optional[ConfigManager] = None) -> AdvancedResourceManager:
    """Create a resource manager configured for tools.

    Args:
        config_manager: Optional configuration manager

    Returns:
        Configured AdvancedResourceManager
    """
    manager = AdvancedResourceManager(
        config_manager=config_manager,
        max_resources=50,
        default_cleanup_strategy=CleanupStrategy.LAZY
    )

    # Register tool factories
    from ..tools.file_download import FileDownloadTool
    from ..tools.gene_query import GeneQueryTool
    from ..tools.api_schema import APISchemaDocumentationTool

    manager.register_factory(
        FileDownloadTool,
        lambda: FileDownloadTool(config_manager=config_manager)
    )

    manager.register_factory(
        GeneQueryTool,
        lambda: GeneQueryTool(config_manager=config_manager)
    )

    manager.register_factory(
        APISchemaDocumentationTool,
        lambda: APISchemaDocumentationTool(config_manager=config_manager)
    )

    return manager
