"""
Enhanced dependency injection framework with enterprise-level features.

This module extends the existing DI system with advanced capabilities including
decorator-based injection, module system, lifecycle events, metrics, and hot reloading.
"""

import asyncio
import inspect
import time
from typing import TypeVar, Type, Any, Dict, Optional, List, Callable, Union, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import weakref

from .dependency_injection import DIContainer, LifecycleScope, get_container
from .logging_config import get_logger
from ..config import ConfigManager
from ..errors import ConfigurationError, AGRMCPError

logger = get_logger(__name__)

T = TypeVar('T')


class DIEvent(Enum):
    """Dependency injection lifecycle events."""
    BEFORE_CREATE = "before_create"
    AFTER_CREATE = "after_create"
    BEFORE_DESTROY = "before_destroy"
    AFTER_DESTROY = "after_destroy"
    CONFIGURATION_CHANGED = "configuration_changed"
    HEALTH_CHECK_FAILED = "health_check_failed"


@dataclass
class DIEventData:
    """Event data for DI lifecycle events."""
    event_type: DIEvent
    dependency_type: Type
    instance: Optional[Any] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class DIEventHandler(ABC):
    """Abstract base class for DI event handlers."""

    @abstractmethod
    async def handle_event(self, event_data: DIEventData):
        """Handle a DI event.

        Args:
            event_data: Event data
        """
        pass


class LoggingEventHandler(DIEventHandler):
    """Event handler that logs DI events."""

    async def handle_event(self, event_data: DIEventData):
        """Log DI events."""
        if event_data.error:
            logger.error(f"DI Event {event_data.event_type.value} for {event_data.dependency_type.__name__}: {str(event_data.error)}")
        else:
            logger.debug(f"DI Event {event_data.event_type.value} for {event_data.dependency_type.__name__}")


class MetricsEventHandler(DIEventHandler):
    """Event handler that tracks DI metrics."""

    def __init__(self):
        self.creation_count: Dict[Type, int] = {}
        self.destruction_count: Dict[Type, int] = {}
        self.error_count: Dict[Type, int] = {}
        self.creation_times: Dict[Type, List[float]] = {}

    async def handle_event(self, event_data: DIEventData):
        """Track DI metrics."""
        dep_type = event_data.dependency_type

        if event_data.event_type == DIEvent.AFTER_CREATE:
            self.creation_count[dep_type] = self.creation_count.get(dep_type, 0) + 1
            if dep_type not in self.creation_times:
                self.creation_times[dep_type] = []
            self.creation_times[dep_type].append(event_data.timestamp)

        elif event_data.event_type == DIEvent.AFTER_DESTROY:
            self.destruction_count[dep_type] = self.destruction_count.get(dep_type, 0) + 1

        elif event_data.error:
            self.error_count[dep_type] = self.error_count.get(dep_type, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "creation_count": dict(self.creation_count),
            "destruction_count": dict(self.destruction_count),
            "error_count": dict(self.error_count),
            "avg_creation_rate": self._calculate_creation_rates()
        }

    def _calculate_creation_rates(self) -> Dict[str, float]:
        """Calculate average creation rates."""
        rates = {}
        current_time = time.time()
        window = 3600.0  # 1 hour window

        for dep_type, times in self.creation_times.items():
            recent_times = [t for t in times if current_time - t <= window]
            if recent_times:
                rates[dep_type.__name__] = len(recent_times) / window
            else:
                rates[dep_type.__name__] = 0.0

        return rates


class DIModule(ABC):
    """Abstract base class for DI modules."""

    @abstractmethod
    async def configure(self, container: DIContainer):
        """Configure dependencies in the container.

        Args:
            container: DI container to configure
        """
        pass

    @abstractmethod
    def get_dependencies(self) -> List[Type]:
        """Get list of dependency types provided by this module.

        Returns:
            List of dependency types
        """
        pass


class CoreModule(DIModule):
    """Core module with basic dependencies."""

    async def configure(self, container: DIContainer):
        """Configure core dependencies."""
        from ..config import ConfigManager
        from .http_client import AGRHttpClient
        from .file_manager import FileManager

        # Register ConfigManager if not already registered
        if not container.is_registered(ConfigManager):
            container.register_singleton(
                ConfigManager,
                factory=lambda: ConfigManager.create()
            )

        # Register core services
        container.register_singleton(
            AGRHttpClient,
            factory=self._create_http_client,
            dependencies=[ConfigManager],
            is_async=True
        )

        container.register_transient(
            FileManager,
            factory=self._create_file_manager,
            dependencies=[ConfigManager]
        )

    async def _create_http_client(self, config_manager: ConfigManager) -> AGRHttpClient:
        """Factory for HTTP client."""
        return AGRHttpClient(
            base_url=config_manager.base_url,
            timeout=getattr(config_manager, 'http_timeout', 30.0)
        )

    def _create_file_manager(self, config_manager: ConfigManager) -> FileManager:
        """Factory for file manager."""
        from pathlib import Path
        return FileManager(Path(config_manager.default_download_dir))

    def get_dependencies(self) -> List[Type]:
        """Get core dependencies."""
        from ..config import ConfigManager
        from .http_client import AGRHttpClient
        from .file_manager import FileManager

        return [ConfigManager, AGRHttpClient, FileManager]


class ToolsModule(DIModule):
    """Module for tool dependencies."""

    async def configure(self, container: DIContainer):
        """Configure tool dependencies."""
        from ..tools.file_download import FileDownloadTool
        from ..tools.gene_query import GeneQueryTool
        from ..tools.api_schema import APISchemaDocumentationTool
        from ..config import ConfigManager

        # Register tools with configuration injection
        container.register_scoped(
            FileDownloadTool,
            factory=lambda config: FileDownloadTool(config_manager=config),
            dependencies=[ConfigManager]
        )

        container.register_scoped(
            GeneQueryTool,
            factory=lambda config: GeneQueryTool(config_manager=config),
            dependencies=[ConfigManager]
        )

        container.register_scoped(
            APISchemaDocumentationTool,
            factory=lambda config: APISchemaDocumentationTool(config_manager=config),
            dependencies=[ConfigManager]
        )

    def get_dependencies(self) -> List[Type]:
        """Get tool dependencies."""
        from ..tools.file_download import FileDownloadTool
        from ..tools.gene_query import GeneQueryTool
        from ..tools.api_schema import APISchemaDocumentationTool

        return [FileDownloadTool, GeneQueryTool, APISchemaDocumentationTool]


class EnhancedDIContainer(DIContainer):
    """Enhanced DI container with events, modules, and advanced features."""

    def __init__(self):
        super().__init__()
        self._event_handlers: List[DIEventHandler] = []
        self._modules: Dict[str, DIModule] = {}
        self._hot_reload_enabled = False
        self._dependency_graph: Dict[Type, Set[Type]] = {}

        # Add default event handlers
        self.add_event_handler(LoggingEventHandler())
        self.add_event_handler(MetricsEventHandler())

    def add_event_handler(self, handler: DIEventHandler):
        """Add an event handler.

        Args:
            handler: Event handler to add
        """
        self._event_handlers.append(handler)
        logger.debug(f"Added event handler: {type(handler).__name__}")

    def remove_event_handler(self, handler: DIEventHandler):
        """Remove an event handler.

        Args:
            handler: Event handler to remove
        """
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
            logger.debug(f"Removed event handler: {type(handler).__name__}")

    async def _emit_event(self, event_type: DIEvent, dependency_type: Type, **kwargs):
        """Emit a DI event.

        Args:
            event_type: Type of event
            dependency_type: Dependency type
            **kwargs: Additional event data
        """
        event_data = DIEventData(
            event_type=event_type,
            dependency_type=dependency_type,
            **kwargs
        )

        for handler in self._event_handlers:
            try:
                await handler.handle_event(event_data)
            except Exception as e:
                logger.error(f"Error in event handler {type(handler).__name__}: {str(e)}")

    async def register_module(self, name: str, module: DIModule):
        """Register a DI module.

        Args:
            name: Module name
            module: Module instance
        """
        self._modules[name] = module
        await module.configure(self)

        # Build dependency graph
        for dep_type in module.get_dependencies():
            if dep_type not in self._dependency_graph:
                self._dependency_graph[dep_type] = set()

        logger.info(f"Registered DI module: {name}")

    def get_dependency_graph(self) -> Dict[Type, Set[Type]]:
        """Get the dependency graph.

        Returns:
            Dependency graph mapping
        """
        return dict(self._dependency_graph)

    async def _create_instance(self, dependency_type: Type[T], dep_info) -> T:
        """Enhanced instance creation with events."""
        await self._emit_event(DIEvent.BEFORE_CREATE, dependency_type)

        try:
            instance = await super()._create_instance(dependency_type, dep_info)
            await self._emit_event(DIEvent.AFTER_CREATE, dependency_type, instance=instance)
            return instance

        except Exception as e:
            await self._emit_event(DIEvent.BEFORE_CREATE, dependency_type, error=e)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with dependency validation."""
        basic_health = await super().health_check()

        # Add enhanced metrics
        metrics_handler = next(
            (h for h in self._event_handlers if isinstance(h, MetricsEventHandler)),
            None
        )

        enhanced_health = {
            **basic_health,
            "modules": list(self._modules.keys()),
            "event_handlers": len(self._event_handlers),
            "dependency_graph_size": len(self._dependency_graph)
        }

        if metrics_handler:
            enhanced_health["metrics"] = metrics_handler.get_metrics()

        return enhanced_health


# Decorator-based dependency injection
def inject(*dependency_types: Type):
    """Decorator for automatic dependency injection.

    Args:
        *dependency_types: Types to inject as function arguments

    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            container = get_container()

            # Resolve dependencies
            injected_args = []
            for dep_type in dependency_types:
                instance = await container.get(dep_type)
                injected_args.append(instance)

            # Call function with injected dependencies
            if asyncio.iscoroutinefunction(func):
                return await func(*args, *injected_args, **kwargs)
            else:
                return func(*args, *injected_args, **kwargs)

        return wrapper
    return decorator


def injectable(dependency_type: Type[T], scope: LifecycleScope = LifecycleScope.TRANSIENT):
    """Class decorator for automatic DI registration.

    Args:
        dependency_type: Type to register class as
        scope: Lifecycle scope for the dependency

    Returns:
        Class decorator
    """
    def decorator(cls):
        # Get or create container
        container = get_container()

        # Register class
        if scope == LifecycleScope.SINGLETON:
            container.register_singleton(dependency_type, factory=cls)
        elif scope == LifecycleScope.SCOPED:
            container.register_scoped(dependency_type, factory=cls)
        else:
            container.register_transient(dependency_type, factory=cls)

        logger.debug(f"Auto-registered {cls.__name__} as {dependency_type.__name__} with scope {scope.value}")

        return cls
    return decorator


class DIConfiguration:
    """Configuration system for DI container."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager.create()
        self._di_config: Dict[str, Any] = {}

    def load_configuration(self, config_section: str = "dependency_injection"):
        """Load DI configuration from config manager.

        Args:
            config_section: Configuration section name
        """
        try:
            # Try to get DI configuration
            if hasattr(self.config_manager, config_section):
                self._di_config = getattr(self.config_manager, config_section)
            else:
                # Use defaults
                self._di_config = {
                    "max_instances": 1000,
                    "health_check_interval": 60.0,
                    "enable_metrics": True,
                    "enable_hot_reload": False,
                    "default_scope": "transient"
                }

        except Exception as e:
            logger.warning(f"Failed to load DI configuration: {str(e)}")
            self._di_config = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._di_config.get(key, default)

    def set_config(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._di_config[key] = value


class DIBuilder:
    """Builder for creating and configuring DI containers."""

    def __init__(self):
        self._container: Optional[EnhancedDIContainer] = None
        self._modules: List[tuple] = []  # (name, module) pairs
        self._event_handlers: List[DIEventHandler] = []
        self._configuration: Optional[DIConfiguration] = None

    def with_container(self, container: EnhancedDIContainer) -> 'DIBuilder':
        """Set the container to use.

        Args:
            container: Container instance

        Returns:
            Builder instance for chaining
        """
        self._container = container
        return self

    def with_module(self, name: str, module: DIModule) -> 'DIBuilder':
        """Add a module to the container.

        Args:
            name: Module name
            module: Module instance

        Returns:
            Builder instance for chaining
        """
        self._modules.append((name, module))
        return self

    def with_event_handler(self, handler: DIEventHandler) -> 'DIBuilder':
        """Add an event handler.

        Args:
            handler: Event handler

        Returns:
            Builder instance for chaining
        """
        self._event_handlers.append(handler)
        return self

    def with_configuration(self, configuration: DIConfiguration) -> 'DIBuilder':
        """Set the configuration.

        Args:
            configuration: DI configuration

        Returns:
            Builder instance for chaining
        """
        self._configuration = configuration
        return self

    async def build(self) -> EnhancedDIContainer:
        """Build and configure the DI container.

        Returns:
            Configured container
        """
        if self._container is None:
            self._container = EnhancedDIContainer()

        # Add event handlers
        for handler in self._event_handlers:
            self._container.add_event_handler(handler)

        # Register modules
        for name, module in self._modules:
            await self._container.register_module(name, module)

        # Initialize container
        await self._container.initialize()

        logger.info("DI container built and configured")
        return self._container


# Global enhanced container instance
_enhanced_container: Optional[EnhancedDIContainer] = None


async def get_enhanced_container() -> EnhancedDIContainer:
    """Get the global enhanced DI container.

    Returns:
        Enhanced DI container instance
    """
    global _enhanced_container

    if _enhanced_container is None:
        builder = DIBuilder()
        _enhanced_container = await (builder
            .with_module("core", CoreModule())
            .with_module("tools", ToolsModule())
            .build())

    return _enhanced_container


async def configure_enhanced_di(config_manager: Optional[ConfigManager] = None) -> EnhancedDIContainer:
    """Configure the enhanced DI system.

    Args:
        config_manager: Optional configuration manager

    Returns:
        Configured enhanced DI container
    """
    di_config = DIConfiguration(config_manager)
    di_config.load_configuration()

    builder = (DIBuilder()
        .with_configuration(di_config)
        .with_module("core", CoreModule())
        .with_module("tools", ToolsModule()))

    # Add custom event handlers based on configuration
    if di_config.get_config("enable_metrics", True):
        builder.with_event_handler(MetricsEventHandler())

    container = await builder.build()

    global _enhanced_container
    _enhanced_container = container

    return container


# Convenience functions
async def reset_enhanced_container():
    """Reset the global enhanced container."""
    global _enhanced_container

    if _enhanced_container:
        await _enhanced_container.shutdown()
        _enhanced_container = None


# Example usage decorators
@inject(ConfigManager)
async def example_function_with_injection(config_manager: ConfigManager):
    """Example function using dependency injection."""
    logger.info(f"Using config manager: {config_manager.base_url}")


@injectable(ConfigManager, LifecycleScope.SINGLETON)
class ExampleService:
    """Example service with automatic DI registration."""

    def __init__(self):
        self.initialized = True
