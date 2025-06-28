"""
Async context managers for resource management and lifecycle control.

This module provides comprehensive async context managers for managing
resources like HTTP clients, file managers, and tool instances.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional, List, Type
from pathlib import Path

from .http_client import AGRHttpClient
from .file_manager import FileManager
from ..config import ConfigManager
from ..errors import ConfigurationError, AGRMCPError

logger = logging.getLogger(__name__)


class ResourcePool:
    """Manages a pool of reusable resources with async context manager support."""

    def __init__(self, max_size: int = 10):
        """Initialize resource pool.

        Args:
            max_size: Maximum number of resources in the pool
        """
        self.max_size = max_size
        self._resources = asyncio.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self, factory):
        """Acquire a resource from the pool or create a new one."""
        try:
            # Try to get an existing resource
            resource = self._resources.get_nowait()
            logger.debug("Acquired existing resource from pool")
            return resource
        except asyncio.QueueEmpty:
            # Create new resource if under limit
            async with self._lock:
                if self._created_count < self.max_size:
                    resource = await factory()
                    self._created_count += 1
                    logger.debug(f"Created new resource (count: {self._created_count})")
                    return resource
                else:
                    # Wait for a resource to become available
                    logger.debug("Pool at capacity, waiting for available resource")
                    return await self._resources.get()

    async def release(self, resource):
        """Release a resource back to the pool."""
        try:
            self._resources.put_nowait(resource)
            logger.debug("Released resource back to pool")
        except asyncio.QueueFull:
            # Pool is full, close the resource
            if hasattr(resource, 'close'):
                await resource.close()
            elif hasattr(resource, '__aexit__'):
                await resource.__aexit__(None, None, None)
            logger.debug("Pool full, closed excess resource")

    async def close_all(self):
        """Close all resources in the pool."""
        while not self._resources.empty():
            try:
                resource = self._resources.get_nowait()
                if hasattr(resource, 'close'):
                    await resource.close()
                elif hasattr(resource, '__aexit__'):
                    await resource.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing resource: {str(e)}")
        self._created_count = 0
        logger.info("Closed all resources in pool")


class HttpClientManager:
    """Manages HTTP client resources with connection pooling."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize HTTP client manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager.create()
        self._client_pool = ResourcePool(max_size=5)
        self._default_client: Optional[AGRHttpClient] = None

    async def get_client(self,
                        base_url: Optional[str] = None,
                        timeout: float = 30.0,
                        max_retries: int = 3) -> AGRHttpClient:
        """Get an HTTP client instance from the pool.

        Args:
            base_url: Base URL for the client
            timeout: Request timeout
            max_retries: Maximum retry attempts

        Returns:
            AGRHttpClient instance
        """
        # Use default client if no specific config needed
        if (base_url is None and
            timeout == 30.0 and
            max_retries == 3 and
            self._default_client is not None):
            return self._default_client

        # Create factory for new client
        async def client_factory():
            return AGRHttpClient(
                base_url=base_url or self.config_manager.base_url,
                timeout=timeout,
                max_retries=max_retries
            )

        return await self._client_pool.acquire(client_factory)

    async def release_client(self, client: AGRHttpClient):
        """Release a client back to the pool.

        Args:
            client: Client to release
        """
        # Don't release the default client
        if client is not self._default_client:
            await self._client_pool.release(client)

    async def __aenter__(self):
        """Async context manager entry."""
        # Create default client
        self._default_client = AGRHttpClient(
            base_url=self.config_manager.base_url,
            timeout=30.0,
            max_retries=3
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            # Close default client
            if self._default_client:
                await self._default_client.close()
                self._default_client = None

            # Close all pooled clients
            await self._client_pool.close_all()

        except Exception as e:
            logger.warning(f"Error during HttpClientManager cleanup: {str(e)}")
        return False


class ToolResourceManager:
    """Manages resources for tool instances with lifecycle control."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize tool resource manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager.create()
        self._http_manager: Optional[HttpClientManager] = None
        self._file_manager: Optional[FileManager] = None
        self._active_tools: List[Any] = []

    async def get_http_client(self, **kwargs) -> AGRHttpClient:
        """Get HTTP client instance.

        Args:
            **kwargs: Arguments for HTTP client creation

        Returns:
            AGRHttpClient instance
        """
        if not self._http_manager:
            raise AGRMCPError("ToolResourceManager not initialized as context manager")
        return await self._http_manager.get_client(**kwargs)

    async def release_http_client(self, client: AGRHttpClient):
        """Release HTTP client instance.

        Args:
            client: Client to release
        """
        if self._http_manager:
            await self._http_manager.release_client(client)

    def get_file_manager(self,
                        base_dir: Optional[Path] = None,
                        **kwargs) -> FileManager:
        """Get file manager instance.

        Args:
            base_dir: Base directory for file operations
            **kwargs: Additional FileManager arguments

        Returns:
            FileManager instance
        """
        if base_dir is None:
            base_dir = Path(self.config_manager.default_download_dir)

        # Create new FileManager instance for each request to avoid conflicts
        return FileManager(base_dir, **kwargs)

    def register_tool(self, tool: Any):
        """Register a tool instance for cleanup tracking.

        Args:
            tool: Tool instance to track
        """
        self._active_tools.append(tool)
        logger.debug(f"Registered tool: {type(tool).__name__}")

    def unregister_tool(self, tool: Any):
        """Unregister a tool instance.

        Args:
            tool: Tool instance to stop tracking
        """
        if tool in self._active_tools:
            self._active_tools.remove(tool)
            logger.debug(f"Unregistered tool: {type(tool).__name__}")

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize HTTP client manager
        self._http_manager = HttpClientManager(self.config_manager)
        await self._http_manager.__aenter__()

        # Initialize file manager with cleanup
        self._file_manager = FileManager(
            base_dir=Path(self.config_manager.default_download_dir)
        )
        await self._file_manager.__aenter__()

        logger.info("Initialized ToolResourceManager")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            # Clean up active tools
            for tool in self._active_tools[:]:  # Create copy to avoid modification during iteration
                try:
                    if hasattr(tool, '__aexit__'):
                        await tool.__aexit__(exc_type, exc_val, exc_tb)
                    elif hasattr(tool, 'close'):
                        await tool.close()
                except Exception as e:
                    logger.warning(f"Error cleaning up tool {type(tool).__name__}: {str(e)}")
            self._active_tools.clear()

            # Clean up file manager
            if self._file_manager:
                await self._file_manager.__aexit__(exc_type, exc_val, exc_tb)
                self._file_manager = None

            # Clean up HTTP manager
            if self._http_manager:
                await self._http_manager.__aexit__(exc_type, exc_val, exc_tb)
                self._http_manager = None

            logger.info("Cleaned up ToolResourceManager")

        except Exception as e:
            logger.error(f"Error during ToolResourceManager cleanup: {str(e)}")
        return False


class ApplicationResourceManager:
    """Top-level resource manager for the entire application."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize application resource manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager.create()
        self._tool_manager: Optional[ToolResourceManager] = None
        self._exit_stack: Optional[AsyncExitStack] = None

    async def get_tool_manager(self) -> ToolResourceManager:
        """Get tool resource manager.

        Returns:
            ToolResourceManager instance

        Raises:
            AGRMCPError: If not initialized as context manager
        """
        if not self._tool_manager:
            raise AGRMCPError("ApplicationResourceManager not initialized as context manager")
        return self._tool_manager

    async def __aenter__(self):
        """Async context manager entry."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Initialize tool manager
        self._tool_manager = ToolResourceManager(self.config_manager)
        await self._exit_stack.enter_async_context(self._tool_manager)

        logger.info("Initialized ApplicationResourceManager")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            if self._exit_stack:
                await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
                self._exit_stack = None

            self._tool_manager = None
            logger.info("Cleaned up ApplicationResourceManager")

        except Exception as e:
            logger.error(f"Error during ApplicationResourceManager cleanup: {str(e)}")
        return False


# Convenience factory functions

async def create_http_client_manager(config_manager: Optional[ConfigManager] = None) -> HttpClientManager:
    """Create and initialize HTTP client manager.

    Args:
        config_manager: Optional configuration manager

    Returns:
        HttpClientManager instance
    """
    manager = HttpClientManager(config_manager)
    return manager


async def create_tool_resource_manager(config_manager: Optional[ConfigManager] = None) -> ToolResourceManager:
    """Create and initialize tool resource manager.

    Args:
        config_manager: Optional configuration manager

    Returns:
        ToolResourceManager instance
    """
    manager = ToolResourceManager(config_manager)
    return manager


async def create_application_manager(config_manager: Optional[ConfigManager] = None) -> ApplicationResourceManager:
    """Create and initialize application resource manager.

    Args:
        config_manager: Optional configuration manager

    Returns:
        ApplicationResourceManager instance
    """
    manager = ApplicationResourceManager(config_manager)
    return manager


class TransactionContextManager:
    """Context manager for transactional operations with rollback support."""

    def __init__(self, resource_manager: Optional['AdvancedResourceManager'] = None):
        """Initialize transaction context manager.

        Args:
            resource_manager: Optional resource manager for transaction resources
        """
        self.resource_manager = resource_manager
        self._operations: List[Callable] = []
        self._rollback_operations: List[Callable] = []
        self._committed = False

    def add_operation(self, operation: Callable, rollback_operation: Optional[Callable] = None):
        """Add an operation to the transaction.

        Args:
            operation: Operation to perform
            rollback_operation: Optional rollback operation
        """
        self._operations.append(operation)
        if rollback_operation:
            self._rollback_operations.append(rollback_operation)

    async def execute_operations(self):
        """Execute all registered operations."""
        for operation in self._operations:
            if asyncio.iscoroutinefunction(operation):
                await operation()
            else:
                operation()

    async def rollback_operations(self):
        """Rollback all operations in reverse order."""
        for rollback_op in reversed(self._rollback_operations):
            try:
                if asyncio.iscoroutinefunction(rollback_op):
                    await rollback_op()
                else:
                    rollback_op()
            except Exception as e:
                logger.warning(f"Rollback operation failed: {str(e)}")

    async def commit(self):
        """Commit the transaction."""
        await self.execute_operations()
        self._committed = True
        logger.debug("Transaction committed")

    async def __aenter__(self):
        """Transaction context entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Transaction context exit."""
        if exc_type is not None and not self._committed:
            # Exception occurred, rollback
            await self.rollback_operations()
            logger.info("Transaction rolled back due to exception")
        elif not self._committed:
            # No exception but not committed, rollback
            await self.rollback_operations()
            logger.info("Transaction rolled back (not committed)")

        return False


class TimedContextManager:
    """Context manager with timeout and timing capabilities."""

    def __init__(self,
                 timeout: Optional[float] = None,
                 warning_threshold: Optional[float] = None,
                 operation_name: str = "operation"):
        """Initialize timed context manager.

        Args:
            timeout: Optional timeout in seconds
            warning_threshold: Optional threshold for slow operation warnings
            operation_name: Name of operation for logging
        """
        self.timeout = timeout
        self.warning_threshold = warning_threshold
        self.operation_name = operation_name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._timeout_task: Optional[asyncio.Task] = None

    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return None

    async def __aenter__(self):
        """Timed context entry."""
        self._start_time = time.time()

        if self.timeout:
            self._timeout_task = asyncio.create_task(self._timeout_handler())

        logger.debug(f"Started {self.operation_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Timed context exit."""
        self._end_time = time.time()

        # Cancel timeout task
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

        duration = self.duration

        if duration:
            # Check for slow operations
            if self.warning_threshold and duration > self.warning_threshold:
                logger.warning(f"{self.operation_name} took {duration:.2f}s (threshold: {self.warning_threshold}s)")
            else:
                logger.debug(f"{self.operation_name} completed in {duration:.2f}s")

        return False

    async def _timeout_handler(self):
        """Handle timeout."""
        try:
            await asyncio.sleep(self.timeout)
            logger.error(f"{self.operation_name} timed out after {self.timeout}s")
            # In a real implementation, this would cancel the operation
        except asyncio.CancelledError:
            pass


class RetryContextManager:
    """Context manager with retry logic for transient failures."""

    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 retry_exceptions: tuple = (Exception,)):
        """Initialize retry context manager.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            backoff_factor: Exponential backoff factor
            retry_exceptions: Tuple of exceptions to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions
        self._attempt = 0
        self._last_exception: Optional[Exception] = None

    async def execute(self, operation: Callable):
        """Execute operation with retry logic.

        Args:
            operation: Async operation to execute

        Returns:
            Operation result

        Raises:
            Last exception if all retries exhausted
        """
        while self._attempt <= self.max_retries:
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()

            except self.retry_exceptions as e:
                self._last_exception = e
                self._attempt += 1

                if self._attempt <= self.max_retries:
                    delay = min(
                        self.base_delay * (self.backoff_factor ** (self._attempt - 1)),
                        self.max_delay
                    )

                    logger.warning(f"Operation failed (attempt {self._attempt}/{self.max_retries + 1}), retrying in {delay:.1f}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Operation failed after {self.max_retries + 1} attempts: {str(e)}")
                    raise e

        # Should not reach here
        if self._last_exception:
            raise self._last_exception

    async def __aenter__(self):
        """Retry context entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Retry context exit."""
        return False


class CompositeContextManager:
    """Context manager that manages multiple context managers."""

    def __init__(self, *context_managers):
        """Initialize composite context manager.

        Args:
            *context_managers: Context managers to manage
        """
        self.context_managers = context_managers
        self._entered_managers: List[Any] = []

    async def __aenter__(self):
        """Enter all context managers."""
        try:
            for manager in self.context_managers:
                result = await manager.__aenter__()
                self._entered_managers.append((manager, result))

            # Return list of entered manager results
            return [result for manager, result in self._entered_managers]

        except Exception as e:
            # Clean up any managers that were already entered
            await self._cleanup_entered_managers()
            raise e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit all context managers in reverse order."""
        await self._cleanup_entered_managers()
        return False

    async def _cleanup_entered_managers(self):
        """Clean up all entered managers."""
        # Exit in reverse order
        for manager, result in reversed(self._entered_managers):
            try:
                await manager.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error exiting context manager: {str(e)}")

        self._entered_managers.clear()


class ResourcePoolContextManager:
    """Context manager for resource pools with automatic scaling."""

    def __init__(self,
                 resource_factory: Callable,
                 min_size: int = 1,
                 max_size: int = 10,
                 scale_threshold: float = 0.8):
        """Initialize resource pool context manager.

        Args:
            resource_factory: Factory function for creating resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            scale_threshold: Threshold for scaling up (as fraction of pool usage)
        """
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.scale_threshold = scale_threshold

        self._pool: List[Any] = []
        self._available: List[Any] = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a resource from the pool."""
        async with self._lock:
            # Try to get available resource
            if self._available:
                resource = self._available.pop()
                self._in_use.add(resource)
                return resource

            # Check if we can create new resource
            if len(self._pool) < self.max_size:
                resource = await self._create_resource()
                self._pool.append(resource)
                self._in_use.add(resource)
                return resource

            # Pool is full, wait or raise error
            raise AGRMCPError("Resource pool exhausted")

    async def release(self, resource):
        """Release a resource back to the pool."""
        async with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                self._available.append(resource)

    async def _create_resource(self):
        """Create a new resource."""
        if asyncio.iscoroutinefunction(self.resource_factory):
            return await self.resource_factory()
        else:
            return self.resource_factory()

    async def _initialize_pool(self):
        """Initialize the pool with minimum resources."""
        for _ in range(self.min_size):
            resource = await self._create_resource()
            self._pool.append(resource)
            self._available.append(resource)

    async def _cleanup_pool(self):
        """Clean up all resources in the pool."""
        all_resources = self._pool[:]

        for resource in all_resources:
            try:
                if hasattr(resource, '__aexit__'):
                    await resource.__aexit__(None, None, None)
                elif hasattr(resource, 'close'):
                    await resource.close()
            except Exception as e:
                logger.warning(f"Error cleaning up pooled resource: {str(e)}")

        self._pool.clear()
        self._available.clear()
        self._in_use.clear()

    async def __aenter__(self):
        """Initialize the resource pool."""
        await self._initialize_pool()
        logger.debug(f"Resource pool initialized with {len(self._pool)} resources")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the resource pool."""
        await self._cleanup_pool()
        logger.debug("Resource pool cleaned up")
        return False


# Enhanced factory functions
async def create_enhanced_http_client_manager(config_manager: Optional[ConfigManager] = None) -> HttpClientManager:
    """Create an enhanced HTTP client manager with advanced features.

    Args:
        config_manager: Optional configuration manager

    Returns:
        Enhanced HttpClientManager instance
    """
    manager = HttpClientManager(config_manager)

    # Wrap with enhanced capabilities
    class EnhancedHttpClientManager(HttpClientManager):
        async def get_client_with_retry(self, max_retries: int = 3, **kwargs):
            """Get HTTP client with built-in retry logic."""
            async with RetryContextManager(max_retries=max_retries) as retry_ctx:
                return await retry_ctx.execute(lambda: self.get_client(**kwargs))

        async def get_client_with_timeout(self, timeout: float = 30.0, **kwargs):
            """Get HTTP client with timeout protection."""
            async with TimedContextManager(timeout=timeout, operation_name="HTTP client acquisition"):
                return await self.get_client(**kwargs)

    return EnhancedHttpClientManager(config_manager)


async def create_enhanced_application_manager(config_manager: Optional[ConfigManager] = None) -> ApplicationResourceManager:
    """Create an enhanced application resource manager.

    Args:
        config_manager: Optional configuration manager

    Returns:
        Enhanced ApplicationResourceManager instance
    """
    manager = ApplicationResourceManager(config_manager)

    # Add enhanced capabilities
    original_aenter = manager.__aenter__
    original_aexit = manager.__aexit__

    async def enhanced_aenter():
        """Enhanced context entry with timing and health checks."""
        async with TimedContextManager(
            warning_threshold=5.0,
            operation_name="ApplicationResourceManager initialization"
        ):
            return await original_aenter()

    async def enhanced_aexit(exc_type, exc_val, exc_tb):
        """Enhanced context exit with proper cleanup monitoring."""
        async with TimedContextManager(
            warning_threshold=10.0,
            operation_name="ApplicationResourceManager cleanup"
        ):
            return await original_aexit(exc_type, exc_val, exc_tb)

    manager.__aenter__ = enhanced_aenter
    manager.__aexit__ = enhanced_aexit

    return manager
