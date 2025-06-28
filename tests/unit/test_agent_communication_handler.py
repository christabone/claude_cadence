"""
Unit tests for AgentCommunicationHandler
"""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from cadence.agent_communication_handler import (
    AgentCommunicationHandler, CallbackType, AgentOperation, CallbackEvent
)
from cadence.agent_dispatcher import AgentDispatcher
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, MessageContext,
    SuccessCriteria, CallbackInfo
)


class TestAgentCommunicationHandler:
    """Test cases for AgentCommunicationHandler"""

    @pytest.fixture
    def mock_dispatcher(self):
        """Create a mock AgentDispatcher"""
        dispatcher = Mock(spec=AgentDispatcher)
        dispatcher.dispatch_agent.return_value = "msg-123"
        dispatcher.cleanup.return_value = None
        return dispatcher

    @pytest.fixture
    def handler(self, mock_dispatcher):
        """Create AgentCommunicationHandler with mocked dispatcher"""
        return AgentCommunicationHandler(
            dispatcher=mock_dispatcher,
            default_timeout=1.0,  # Short timeout for tests
            enable_event_queue=False  # Disable queue for simpler testing
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample message context"""
        return MessageContext(
            task_id="test-task-123",
            parent_session="session-456",
            files_modified=["test.py", "config.py"],
            project_path="/test/project"
        )

    @pytest.fixture
    def sample_success_criteria(self):
        """Create sample success criteria"""
        return SuccessCriteria(
            expected_outcomes=["Fix applied", "Tests pass"],
            validation_steps=["Run tests", "Check syntax"]
        )

    def test_initialization(self):
        """Test handler initialization"""
        handler = AgentCommunicationHandler(default_timeout=5.0)

        assert handler.default_timeout == 5.0
        assert handler.enable_event_queue is True
        assert len(handler.active_operations) == 0
        assert handler.dispatcher is not None
        assert not handler.shutdown_event.is_set()

    def test_initialization_with_custom_dispatcher(self, mock_dispatcher):
        """Test initialization with custom dispatcher"""
        handler = AgentCommunicationHandler(dispatcher=mock_dispatcher)
        assert handler.dispatcher is mock_dispatcher

    def test_generate_agent_id(self, handler):
        """Test agent ID generation"""
        agent_id1 = handler.generate_agent_id()
        agent_id2 = handler.generate_agent_id()

        assert agent_id1.startswith("agent-")
        assert agent_id2.startswith("agent-")
        assert agent_id1 != agent_id2
        assert len(agent_id1.split("-")) == 3  # agent-uuid-timestamp

    def test_register_global_callback(self, handler):
        """Test global callback registration"""
        callback = Mock()

        handler.register_global_callback(CallbackType.ON_COMPLETE, callback)

        assert callback in handler.global_callbacks[CallbackType.ON_COMPLETE]
        assert len(handler.global_callbacks[CallbackType.ON_COMPLETE]) == 1

    def test_deregister_global_callback(self, handler):
        """Test global callback deregistration"""
        callback = Mock()

        # Register then deregister
        handler.register_global_callback(CallbackType.ON_ERROR, callback)
        assert handler.deregister_global_callback(CallbackType.ON_ERROR, callback) is True
        assert callback not in handler.global_callbacks[CallbackType.ON_ERROR]

        # Try to deregister non-existent callback
        assert handler.deregister_global_callback(CallbackType.ON_ERROR, callback) is False

    @pytest.mark.asyncio
    async def test_start_agent_operation_success(self, handler, mock_dispatcher, sample_context, sample_success_criteria):
        """Test successful agent operation start"""
        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria,
            timeout_seconds=2.0
        )

        assert agent_id.startswith("agent-")
        assert agent_id in handler.active_operations
        assert agent_id in handler.pending_timeouts

        operation = handler.active_operations[agent_id]
        assert operation.agent_type == AgentType.FIX
        assert operation.context == sample_context
        assert operation.timeout_seconds == 2.0

        # Verify dispatcher was called
        mock_dispatcher.dispatch_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_agent_operation_dispatch_failure(self, handler, mock_dispatcher, sample_context, sample_success_criteria):
        """Test agent operation start with dispatch failure"""
        mock_dispatcher.dispatch_agent.side_effect = Exception("Dispatch failed")

        error_callback = AsyncMock()
        handler.register_global_callback(CallbackType.ON_ERROR, error_callback)

        with pytest.raises(Exception, match="Dispatch failed"):
            await handler.start_agent_operation(
                agent_type=AgentType.REVIEW,
                context=sample_context,
                success_criteria=sample_success_criteria
            )

        # Should not have any active operations after failure
        assert len(handler.active_operations) == 0
        assert len(handler.pending_timeouts) == 0

        # Error callback should be triggered
        error_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_operation_callback(self, handler, sample_context, sample_success_criteria):
        """Test operation-specific callback registration"""
        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria
        )

        callback = Mock()
        result = handler.register_operation_callback(agent_id, CallbackType.ON_COMPLETE, callback)

        assert result is True
        operation = handler.active_operations[agent_id]
        assert callback in operation.callbacks[CallbackType.ON_COMPLETE]

        # Test registration for non-existent operation
        result = handler.register_operation_callback("invalid-id", CallbackType.ON_ERROR, callback)
        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_watchdog(self, handler, sample_context, sample_success_criteria):
        """Test timeout watchdog functionality"""
        timeout_callback = AsyncMock()
        handler.register_global_callback(CallbackType.ON_TIMEOUT, timeout_callback)

        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.REVIEW,
            context=sample_context,
            success_criteria=sample_success_criteria,
            timeout_seconds=0.1  # Very short timeout
        )

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Operation should be removed and callback triggered
        assert agent_id not in handler.active_operations
        assert agent_id not in handler.pending_timeouts
        timeout_callback.assert_called_once()

        # Verify callback arguments
        args = timeout_callback.call_args[0]
        assert args[0] == agent_id
        assert "timeout_seconds" in args[1]

    @pytest.mark.asyncio
    async def test_handle_agent_response_complete(self, handler, sample_context, sample_success_criteria):
        """Test handling TASK_COMPLETE response"""
        complete_callback = AsyncMock()
        handler.register_global_callback(CallbackType.ON_COMPLETE, complete_callback)

        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria
        )

        # Create completion message
        completion_message = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000),
            payload={"result": "success", "files_modified": ["test.py"]}
        )

        await handler._handle_agent_response(agent_id, completion_message)

        # Operation should be completed
        assert agent_id not in handler.active_operations
        assert agent_id not in handler.pending_timeouts
        complete_callback.assert_called_once()

        # Verify callback data
        args = complete_callback.call_args[0]
        assert args[0] == agent_id
        assert "message" in args[1]
        assert "duration" in args[1]

    @pytest.mark.asyncio
    async def test_handle_agent_response_error(self, handler, sample_context, sample_success_criteria):
        """Test handling ERROR response"""
        error_callback = AsyncMock()
        handler.register_global_callback(CallbackType.ON_ERROR, error_callback)

        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.REVIEW,
            context=sample_context,
            success_criteria=sample_success_criteria
        )

        # Create error message
        error_message = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.REVIEW,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000),
            payload={"error": "Agent failed", "error_type": "execution_error"}
        )

        await handler._handle_agent_response(agent_id, error_message)

        # Operation should be completed with error
        assert agent_id not in handler.active_operations
        assert agent_id not in handler.pending_timeouts
        error_callback.assert_called_once()

        # Verify callback data
        args = error_callback.call_args[0]
        assert args[0] == agent_id
        assert args[1]["error"] == "Agent failed"

    @pytest.mark.asyncio
    async def test_handle_agent_response_intermediate(self, handler, sample_context, sample_success_criteria):
        """Test handling intermediate AGENT_RESPONSE"""
        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria
        )

        # Create intermediate response
        response_message = AgentMessage(
            message_type=MessageType.AGENT_RESPONSE,
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000),
            payload={"status": "in_progress", "progress": 50}
        )

        await handler._handle_agent_response(agent_id, response_message)

        # Operation should still be active
        assert agent_id in handler.active_operations

    @pytest.mark.asyncio
    async def test_handle_unknown_agent_response(self, handler, sample_context, sample_success_criteria):
        """Test handling response for unknown agent"""
        # Create message for non-existent agent
        message = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000)
        )

        # Should not raise exception
        await handler._handle_agent_response("unknown-agent", message)

    @pytest.mark.asyncio
    async def test_callback_execution_order(self, handler, sample_context, sample_success_criteria):
        """Test that both global and operation callbacks are executed"""
        global_callback = AsyncMock()
        operation_callback = AsyncMock()

        # Register global callback
        handler.register_global_callback(CallbackType.ON_COMPLETE, global_callback)

        # Start operation and register operation callback
        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria
        )
        handler.register_operation_callback(agent_id, CallbackType.ON_COMPLETE, operation_callback)

        # Complete the operation
        completion_message = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000)
        )

        await handler._handle_agent_response(agent_id, completion_message)

        # Both callbacks should be called
        global_callback.assert_called_once()
        operation_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self, handler, sample_context, sample_success_criteria):
        """Test that callback exceptions don't break the system"""
        def failing_callback(agent_id, data):
            raise Exception("Callback failed")

        working_callback = AsyncMock()

        handler.register_global_callback(CallbackType.ON_ERROR, failing_callback)
        handler.register_global_callback(CallbackType.ON_ERROR, working_callback)

        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.REVIEW,
            context=sample_context,
            success_criteria=sample_success_criteria
        )

        # Trigger error
        error_message = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.REVIEW,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000),
            payload={"error": "Test error"}
        )

        # Should not raise exception despite failing callback
        await handler._handle_agent_response(agent_id, error_message)

        # Working callback should still be called
        working_callback.assert_called_once()

    def test_get_operation_status(self, handler, sample_context, sample_success_criteria):
        """Test operation status retrieval"""
        # Test non-existent operation
        status = handler.get_operation_status("invalid-id")
        assert status is None

        # Create operation and test status
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            agent_id = loop.run_until_complete(
                handler.start_agent_operation(
                    agent_type=AgentType.FIX,
                    context=sample_context,
                    success_criteria=sample_success_criteria,
                    timeout_seconds=10.0
                )
            )

            status = handler.get_operation_status(agent_id)
            assert status is not None
            assert status["agent_id"] == agent_id
            assert status["agent_type"] == "fix"
            assert status["timeout_seconds"] == 10.0
            assert "elapsed_seconds" in status
            assert status["context"]["task_id"] == "test-task-123"
        finally:
            loop.close()

    def test_list_active_operations(self, handler):
        """Test listing active operations"""
        assert handler.list_active_operations() == []

        # This test requires async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            context = MessageContext("task1", "session1", [], "/path")
            criteria = SuccessCriteria()

            agent_id = loop.run_until_complete(
                handler.start_agent_operation(AgentType.FIX, context, criteria)
            )

            active_ops = handler.list_active_operations()
            assert len(active_ops) == 1
            assert agent_id in active_ops
        finally:
            loop.close()

    def test_get_statistics(self, handler):
        """Test statistics retrieval"""
        # Register some global callbacks
        handler.register_global_callback(CallbackType.ON_COMPLETE, Mock())
        handler.register_global_callback(CallbackType.ON_ERROR, Mock())

        stats = handler.get_statistics()

        assert stats["active_operations"] == 0
        assert stats["pending_timeouts"] == 0
        assert stats["global_callbacks"]["on_complete"] == 1
        assert stats["global_callbacks"]["on_error"] == 1
        assert stats["event_queue_enabled"] is False
        assert stats["default_timeout"] == 1.0
        assert stats["is_shutdown"] is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, handler, mock_dispatcher, sample_context, sample_success_criteria):
        """Test graceful shutdown functionality"""
        # Start an operation
        agent_id = await handler.start_agent_operation(
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria
        )

        assert len(handler.active_operations) == 1
        assert len(handler.pending_timeouts) == 1

        # Shutdown
        await handler.shutdown()

        # Verify cleanup
        assert len(handler.active_operations) == 0
        assert len(handler.pending_timeouts) == 0
        assert handler.shutdown_event.is_set()
        mock_dispatcher.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_queue_processing(self):
        """Test event queue processing functionality"""
        # Create handler with event queue enabled
        mock_dispatcher = Mock(spec=AgentDispatcher)
        mock_dispatcher.dispatch_agent.return_value = "msg-123"

        handler = AgentCommunicationHandler(
            dispatcher=mock_dispatcher,
            enable_event_queue=True,
            default_timeout=1.0
        )

        callback = AsyncMock()
        handler.register_global_callback(CallbackType.ON_COMPLETE, callback)

        # Start event processor
        await handler.start_event_processor()

        try:
            # Start and complete an operation
            context = MessageContext("task1", "session1", [], "/path")
            criteria = SuccessCriteria()

            agent_id = await handler.start_agent_operation(
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=criteria
            )

            # Complete the operation
            completion_message = AgentMessage(
                message_type=MessageType.TASK_COMPLETE,
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=criteria,
                callback=CallbackInfo(handler="test", timeout_ms=30000)
            )

            await handler._handle_agent_response(agent_id, completion_message)

            # Wait for event processing
            await asyncio.sleep(0.1)

            # Callback should be executed
            callback.assert_called_once()

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_event_queue_with_operation_callbacks(self):
        """Test that operation-specific callbacks work with event queue enabled"""
        # Create handler with event queue enabled
        mock_dispatcher = Mock(spec=AgentDispatcher)
        mock_dispatcher.dispatch_agent.return_value = "msg-123"

        handler = AgentCommunicationHandler(
            dispatcher=mock_dispatcher,
            enable_event_queue=True,
            default_timeout=1.0
        )

        global_callback = AsyncMock()
        operation_callback = AsyncMock()

        # Register global callback
        handler.register_global_callback(CallbackType.ON_COMPLETE, global_callback)

        # Start event processor
        await handler.start_event_processor()

        try:
            # Start operation and register operation callback
            context = MessageContext("task1", "session1", [], "/path")
            criteria = SuccessCriteria()

            agent_id = await handler.start_agent_operation(
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=criteria
            )

            # Register operation-specific callback
            handler.register_operation_callback(agent_id, CallbackType.ON_COMPLETE, operation_callback)

            # Complete the operation
            completion_message = AgentMessage(
                message_type=MessageType.TASK_COMPLETE,
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=criteria,
                callback=CallbackInfo(handler="test", timeout_ms=30000)
            )

            await handler._handle_agent_response(agent_id, completion_message)

            # Wait for event processing
            await asyncio.sleep(0.1)

            # Both global and operation callbacks should be executed
            global_callback.assert_called_once()
            operation_callback.assert_called_once()

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, handler, sample_context, sample_success_criteria):
        """Test handling multiple concurrent operations"""
        # Start multiple operations
        agent_ids = []
        for i in range(3):
            context = MessageContext(f"task-{i}", "session", [], "/path")
            agent_id = await handler.start_agent_operation(
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=sample_success_criteria,
                timeout_seconds=5.0
            )
            agent_ids.append(agent_id)

        assert len(handler.active_operations) == 3
        assert len(handler.pending_timeouts) == 3

        # Complete one operation
        completion_message = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=sample_context,
            success_criteria=sample_success_criteria,
            callback=CallbackInfo(handler="test", timeout_ms=30000)
        )

        await handler._handle_agent_response(agent_ids[0], completion_message)

        assert len(handler.active_operations) == 2
        assert agent_ids[0] not in handler.active_operations
        assert agent_ids[1] in handler.active_operations
        assert agent_ids[2] in handler.active_operations


class TestCallbackEvent:
    """Test cases for CallbackEvent dataclass"""

    def test_callback_event_creation(self):
        """Test CallbackEvent creation"""
        event = CallbackEvent(
            callback_type=CallbackType.ON_COMPLETE,
            agent_id="agent-123",
            data={"result": "success"}
        )

        assert event.callback_type == CallbackType.ON_COMPLETE
        assert event.agent_id == "agent-123"
        assert event.data == {"result": "success"}
        assert isinstance(event.timestamp, datetime)


class TestAgentOperation:
    """Test cases for AgentOperation dataclass"""

    def test_agent_operation_creation(self):
        """Test AgentOperation creation"""
        context = MessageContext("task-1", "session-1", ["file.py"], "/project")
        criteria = SuccessCriteria(["outcome"], ["step"])

        operation = AgentOperation(
            agent_id="agent-123",
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=criteria,
            timeout_seconds=300.0
        )

        assert operation.agent_id == "agent-123"
        assert operation.agent_type == AgentType.FIX
        assert operation.context == context
        assert operation.success_criteria == criteria
        assert operation.timeout_seconds == 300.0
        assert isinstance(operation.start_time, datetime)
        assert operation.callbacks == {}
        assert operation.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__])
