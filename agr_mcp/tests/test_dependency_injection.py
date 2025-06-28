"""
Tests for the dependency injection framework.

This module contains comprehensive tests for the DI framework including
unit tests, integration tests, and lifecycle management tests.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Optional

from agr_mcp.src.config import ConfigManager
from agr_mcp.src.utils.dependency_injection import (
    DIContainer, DependencyScope, LifecycleScope, DependencyInfo,
    get_container, shutdown_container, inject
)
from agr_mcp.src.utils.di_integration import (
    DIAwareServer, ToolFactory, setup_default_container
)
from agr_mcp.src.tools.file_download import FileDownloadTool
from agr_mcp.src.tools.gene_query import GeneQueryTool


class MockService:
    """Mock service for testing."""

    def __init__(self, value: str = "default"):
        """Initialize mock service."""
        self.value = value
        self.calls = []

    def do_something(self, arg: str) -> str:
        """Mock method."""
        self.calls.append(arg)
        return f"{self.value}:{arg}"


class MockDependentService:
    """Mock service with dependencies."""

    def __init__(self, mock_service: MockService, config: ConfigManager):
        """Initialize with dependencies."""
        self.mock_service = mock_service
        self.config = config


@pytest.fixture
async def container():
    """Create test DI container."""
    container = DIContainer()
    await container.initialize()
    yield container
    await container.shutdown()


@pytest.fixture
async def clean_global_container():
    """Ensure clean global container state."""
    await shutdown_container()
    yield
    await shutdown_container()


class TestDIContainer:
    """Test cases for DIContainer."""

    async def test_singleton_registration_and_retrieval(self, container):
        """Test singleton registration and retrieval."""
        # Register singleton
        instance = MockService("singleton")
        container.register_singleton(MockService, instance=instance)

        # Retrieve multiple times
        retrieved1 = await container.get(MockService)
        retrieved2 = await container.get(MockService)

        # Should be same instance
        assert retrieved1 is instance
        assert retrieved2 is instance
        assert retrieved1 is retrieved2

    async def test_transient_registration_and_retrieval(self, container):
        """Test transient registration and retrieval."""
        # Register transient
        container.register_transient(MockService)

        # Retrieve multiple times
        retrieved1 = await container.get(MockService)
        retrieved2 = await container.get(MockService)

        # Should be different instances
        assert retrieved1 is not retrieved2
        assert isinstance(retrieved1, MockService)
        assert isinstance(retrieved2, MockService)

    async def test_scoped_registration_and_retrieval(self, container):
        """Test scoped registration and retrieval."""
        # Register scoped
        container.register_scoped(MockService)

        # Retrieve within same scope
        retrieved1 = await container.get(MockService, scope_id="scope1")
        retrieved2 = await container.get(MockService, scope_id="scope1")
        retrieved3 = await container.get(MockService, scope_id="scope2")

        # Same scope should return same instance
        assert retrieved1 is retrieved2
        # Different scope should return different instance
        assert retrieved1 is not retrieved3

    async def test_dependency_resolution(self, container):
        """Test dependency resolution."""
        # Register dependencies
        container.register_singleton(MockService)
        container.register_transient(
            MockDependentService,
            dependencies=[MockService, ConfigManager]
        )

        # Retrieve dependent service
        dependent = await container.get(MockDependentService)

        # Check dependencies were injected
        assert isinstance(dependent, MockDependentService)
        assert isinstance(dependent.mock_service, MockService)
        assert isinstance(dependent.config, ConfigManager)

    async def test_factory_function(self, container):
        """Test factory function registration."""
        def create_mock_service():
            return MockService("from_factory")

        container.register_singleton(MockService, factory=create_mock_service)

        service = await container.get(MockService)
        assert service.value == "from_factory"

    async def test_async_factory_function(self, container):
        """Test async factory function registration."""
        async def create_async_service():
            await asyncio.sleep(0.01)  # Simulate async work
            return MockService("async_factory")

        container.register_singleton(
            MockService,
            factory=create_async_service,
            is_async=True
        )

        service = await container.get(MockService)
        assert service.value == "async_factory"

    async def test_unregistered_dependency_error(self, container):
        """Test error when requesting unregistered dependency."""
        with pytest.raises(Exception):  # Should raise ConfigurationError
            await container.get(str)  # str is not registered

    async def test_circular_dependency_prevention(self, container):
        """Test circular dependency handling."""
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'):
                self.service_b = service_b

        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a

        container.register_singleton(ServiceA, dependencies=[ServiceB])
        container.register_singleton(ServiceB, dependencies=[ServiceA])

        # This should either resolve gracefully or raise a clear error
        # Implementation may vary based on how circular dependencies are handled
        with pytest.raises(Exception):
            await container.get(ServiceA)

    async def test_health_check(self, container):
        """Test container health check."""
        container.register_singleton(MockService)
        await container.get(MockService)

        health = await container.health_check()

        assert health["initialized"] is True
        assert health["registered_types"] > 0
        assert health["singleton_instances"] > 0

    async def test_scope_clearing(self, container):
        """Test scope clearing."""
        container.register_scoped(MockService)

        # Create instances in scope
        service1 = await container.get(MockService, scope_id="test_scope")
        service2 = await container.get(MockService, scope_id="test_scope")
        assert service1 is service2

        # Clear scope
        container.clear_scope("test_scope")

        # New instance should be created
        service3 = await container.get(MockService, scope_id="test_scope")
        assert service1 is not service3


class TestDependencyScope:
    """Test cases for DependencyScope."""

    async def test_scope_context_manager(self, container):
        """Test dependency scope context manager."""
        container.register_scoped(MockService)

        async with DependencyScope("test_scope") as scope:
            service1 = await scope.get(MockService)
            service2 = await scope.get(MockService)
            assert service1 is service2

        # After scope exit, new instance should be created
        async with DependencyScope("test_scope") as scope:
            service3 = await scope.get(MockService)
            assert service1 is not service3

    async def test_scope_without_context_manager_error(self):
        """Test error when using scope without context manager."""
        scope = DependencyScope("test")

        with pytest.raises(Exception):  # Should raise AGRMCPError
            await scope.get(MockService)


class TestInjectDecorator:
    """Test cases for @inject decorator."""

    async def test_inject_decorator(self, clean_global_container):
        """Test @inject decorator functionality."""
        # Set up global container
        container = await setup_default_container()
        container.register_singleton(MockService, instance=MockService("injected"))

        @inject(MockService)
        async def test_function(service: MockService, arg: str):
            return f"{service.value}:{arg}"

        result = await test_function("test")
        assert result == "injected:test"

    async def test_inject_multiple_dependencies(self, clean_global_container):
        """Test @inject with multiple dependencies."""
        container = await setup_default_container()
        container.register_singleton(MockService, instance=MockService("service1"))

        @inject(MockService, ConfigManager)
        async def test_function(service: MockService, config: ConfigManager, arg: str):
            return f"{service.value}:{type(config).__name__}:{arg}"

        result = await test_function("test")
        assert "service1" in result
        assert "ConfigManager" in result
        assert "test" in result


class TestDIIntegration:
    """Test cases for DI integration components."""

    async def test_di_aware_server_initialization(self):
        """Test DIAwareServer initialization."""
        server = DIAwareServer()
        await server.initialize()

        try:
            agr_server = await server.get_server()
            assert agr_server is not None
        finally:
            await server.shutdown()

    async def test_tool_factory(self, clean_global_container):
        """Test ToolFactory functionality."""
        await setup_default_container()

        factory = ToolFactory()

        file_tool = await factory.create_file_download_tool()
        gene_tool = await factory.create_gene_query_tool()

        assert isinstance(file_tool, FileDownloadTool)
        assert isinstance(gene_tool, GeneQueryTool)

    async def test_tool_factory_with_custom_config(self, clean_global_container):
        """Test ToolFactory with custom configuration."""
        await setup_default_container()

        custom_config = ConfigManager.create(
            config_type='custom',
            config_data={'base_url': 'https://custom.test.com'}
        )

        factory = ToolFactory()
        file_tool = await factory.create_file_download_tool(custom_config)

        assert file_tool.config_manager.base_url == 'https://custom.test.com'


class TestContextManagerIntegration:
    """Test cases for context manager integration."""

    async def test_context_manager_registration(self, container):
        """Test registration of context managers."""
        class MockContextManager:
            def __init__(self):
                self.entered = False
                self.exited = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.exited = True
                return False

        container.register_context(MockContextManager)

        # Get instance (should be entered automatically)
        instance = await container.get(MockContextManager)
        assert instance.entered is True

        # Shutdown should exit context managers
        await container.shutdown()
        # Note: In actual implementation, we'd need to track and exit context managers

    async def test_scoped_context_manager(self, container):
        """Test scoped context manager lifecycle."""
        class MockScopedManager:
            def __init__(self):
                self.active = False

            async def __aenter__(self):
                self.active = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.active = False
                return False

        container.register_scoped(
            MockScopedManager,
            context_manager=True
        )

        manager = await container.get(MockScopedManager, scope_id="test")
        assert manager.active is True


class TestErrorConditions:
    """Test cases for error conditions."""

    async def test_double_initialization(self, container):
        """Test double initialization of container."""
        # Should not raise error
        await container.initialize()

    async def test_get_before_initialization(self):
        """Test getting dependency before initialization."""
        container = DIContainer()
        # Should auto-initialize
        service = await container.get(ConfigManager)
        assert isinstance(service, ConfigManager)
        await container.shutdown()

    async def test_shutdown_before_initialization(self):
        """Test shutdown before initialization."""
        container = DIContainer()
        # Should not raise error
        await container.shutdown()

    async def test_registration_info(self, container):
        """Test registration info retrieval."""
        container.register_singleton(MockService)

        info = container.get_registration_info(MockService)
        assert info is not None
        assert info.dependency_type == MockService
        assert info.scope == LifecycleScope.SINGLETON

        assert container.is_registered(MockService) is True
        assert container.is_registered(str) is False


class TestPerformanceAndConcurrency:
    """Test cases for performance and concurrency."""

    async def test_concurrent_singleton_creation(self, container):
        """Test concurrent singleton creation doesn't create multiple instances."""
        container.register_singleton(MockService)

        # Create multiple concurrent requests
        tasks = [container.get(MockService) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should be the same instance
        first_instance = results[0]
        assert all(instance is first_instance for instance in results)

    async def test_concurrent_scoped_creation(self, container):
        """Test concurrent scoped creation within same scope."""
        container.register_scoped(MockService)

        scope_id = "concurrent_test"
        tasks = [container.get(MockService, scope_id) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should be the same instance within scope
        first_instance = results[0]
        assert all(instance is first_instance for instance in results)

    async def test_multiple_scope_isolation(self, container):
        """Test that multiple scopes are properly isolated."""
        container.register_scoped(MockService)

        async def get_service_in_scope(scope_id: str):
            return await container.get(MockService, scope_id)

        # Create services in different scopes concurrently
        tasks = [get_service_in_scope(f"scope_{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should be different instances
        for i, instance in enumerate(results):
            for j, other_instance in enumerate(results):
                if i != j:
                    assert instance is not other_instance


# Integration test
@pytest.mark.integration
async def test_full_di_workflow():
    """Test complete DI workflow from registration to cleanup."""
    container = DIContainer()
    await container.initialize()

    try:
        # Register complex dependency graph
        class DataService:
            def __init__(self, config: ConfigManager):
                self.config = config

        class BusinessLogic:
            def __init__(self, data_service: DataService):
                self.data_service = data_service

        class APIHandler:
            def __init__(self, business_logic: BusinessLogic, config: ConfigManager):
                self.business_logic = business_logic
                self.config = config

        container.register_singleton(DataService, dependencies=[ConfigManager])
        container.register_scoped(BusinessLogic, dependencies=[DataService])
        container.register_transient(APIHandler, dependencies=[BusinessLogic, ConfigManager])

        # Use within scopes
        async with DependencyScope("request1") as scope1:
            async with DependencyScope("request2") as scope2:
                handler1 = await scope1.get(APIHandler)
                handler2 = await scope2.get(APIHandler)

                # Handlers should be different (transient)
                assert handler1 is not handler2

                # But should share same data service (singleton)
                assert handler1.business_logic.data_service is handler2.business_logic.data_service

                # Business logic should be different (scoped)
                assert handler1.business_logic is not handler2.business_logic

    finally:
        await container.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
