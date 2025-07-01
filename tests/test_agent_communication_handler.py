import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

from cadence.agent_communication_handler import AgentCommunicationHandler, CallbackType, AgentOperation
from cadence.agent_messages import AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_dispatcher():
    """Fixture for a mocked AgentDispatcher."""
    dispatcher = MagicMock()
    dispatcher.dispatch_agent = MagicMock(return_value="test-message-id")
    dispatcher.cleanup = MagicMock()
    return dispatcher

@pytest_asyncio.fixture
async def handler(mock_dispatcher):
    """Fixture for an AgentCommunicationHandler instance."""
    h = AgentCommunicationHandler(dispatcher=mock_dispatcher, default_timeout=0.1)
    yield h
    # Ensure graceful shutdown after each test
    if not h.shutdown_event.is_set():
        await h.shutdown()

@pytest.fixture
def agent_params():
    """Fixture for standard agent operation parameters."""
    return {
        "agent_type": AgentType.REVIEW,
        "context": MessageContext(
            task_id="task-123",
            parent_session="test-session",
            files_modified=["test.py"],
            project_path="/dev/null"
        ),
        "success_criteria": SuccessCriteria(expected_outcomes=["test passed"]),
        "metadata": {"test_run": "true"}
    }

@pytest.fixture
def callback_info():
    """Fixture for standard callback info."""
    return CallbackInfo(handler="test_handler", timeout_ms=30000)

def create_agent_message(message_type, agent_type, context, success_criteria, callback_info, payload=None):
    """Helper function to create AgentMessage instances with all required fields."""
    return AgentMessage(
        message_type=message_type,
        agent_type=agent_type,
        context=context,
        success_criteria=success_criteria,
        callback=callback_info,
        payload=payload or {}
    )

class TestOperationLifecycle:
    """Tests for the core agent operation lifecycle: start, complete, error, timeout."""

    @pytest.mark.asyncio
    async def test_start_operation_success(self, handler, mock_dispatcher, agent_params):
        """
        Verify that starting an operation correctly initializes state and calls the dispatcher.
        """
        # Act
        agent_id = await handler.start_agent_operation(**agent_params, timeout_seconds=10)

        # Assert
        assert agent_id in handler.active_operations
        assert agent_id in handler.pending_timeouts
        assert handler.active_operations[agent_id].timeout_seconds == 10

        mock_dispatcher.dispatch_agent.assert_called_once()
        call_args = mock_dispatcher.dispatch_agent.call_args[1]
        assert call_args['agent_type'] == agent_params['agent_type']
        assert call_args['timeout_ms'] == 10000
        assert callable(call_args['callback_handler'])

    @pytest.mark.asyncio
    async def test_operation_completes_successfully(self, handler, agent_params, callback_info):
        """
        Verify that a successful agent response triggers ON_COMPLETE callbacks and cleans up state.
        """
        # Arrange
        callback_queue = asyncio.Queue()
        async def op_callback(aid, data):
            await callback_queue.put(("op", data))
        async def global_callback(aid, data):
            await callback_queue.put(("global", data))

        agent_id = await handler.start_agent_operation(**agent_params)
        handler.register_operation_callback(agent_id, CallbackType.ON_COMPLETE, op_callback)
        handler.register_global_callback(CallbackType.ON_COMPLETE, global_callback)

        # Act: Simulate agent response by invoking the captured handler
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        message = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info,
            {"result": "ok"}
        )
        await response_handler(message)

        # Assert
        assert agent_id not in handler.active_operations
        assert agent_id not in handler.pending_timeouts

        op_event = await asyncio.wait_for(callback_queue.get(), timeout=1)
        global_event = await asyncio.wait_for(callback_queue.get(), timeout=1)

        # Order isn't guaranteed, so check both
        events = {op_event[0]: op_event[1], global_event[0]: global_event[1]}
        assert "op" in events
        assert "global" in events
        assert events["op"]["result"] == {"result": "ok"}
        assert events["global"]["result"] == {"result": "ok"}
        assert "duration" in events["op"]

    @pytest.mark.asyncio
    async def test_operation_fails_with_error_message(self, handler, agent_params, callback_info):
        """
        Verify that an error response from an agent triggers ON_ERROR callbacks.
        """
        # Arrange
        error_callback = AsyncMock()
        agent_id = await handler.start_agent_operation(**agent_params)
        handler.register_global_callback(CallbackType.ON_ERROR, error_callback)

        # Act
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        error_payload = {"error": "something broke", "error_type": "test_error"}
        message = create_agent_message(
            MessageType.ERROR,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info,
            error_payload
        )
        await response_handler(message)

        # Assert
        assert agent_id not in handler.active_operations
        error_callback.assert_awaited_once()
        call_data = error_callback.call_args[0][1]
        assert call_data["error"] == "something broke"
        assert call_data["error_type"] == "test_error"

    @pytest.mark.asyncio
    async def test_operation_times_out_correctly(self, handler, agent_params):
        """
        Verify that an operation correctly times out and triggers ON_TIMEOUT callbacks.
        """
        # Arrange
        timeout_callback = AsyncMock()
        agent_id = await handler.start_agent_operation(**agent_params, timeout_seconds=0.01)
        handler.register_global_callback(CallbackType.ON_TIMEOUT, timeout_callback)

        # Act
        # The timeout is very short, so the watchdog task should fire quickly.
        await asyncio.sleep(0.05)

        # Assert
        assert agent_id not in handler.active_operations
        assert agent_id not in handler.pending_timeouts
        timeout_callback.assert_awaited_once_with(agent_id, {
            "timeout_seconds": 0.01,
            "duration": 0.01
        })

    @pytest.mark.asyncio
    async def test_start_operation_dispatch_fails_cleans_up(self, handler, mock_dispatcher, agent_params):
        """
        Verify that if dispatcher.dispatch_agent fails, the operation and timeout are cleaned up.
        """
        # Arrange
        mock_dispatcher.dispatch_agent.side_effect = RuntimeError("Dispatch failed")
        error_callback = AsyncMock()
        handler.register_global_callback(CallbackType.ON_ERROR, error_callback)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Dispatch failed"):
            await handler.start_agent_operation(**agent_params)

        assert not handler.active_operations
        assert not handler.pending_timeouts
        error_callback.assert_awaited_once()
        call_data = error_callback.call_args[0][1]
        assert "Failed to dispatch agent" in call_data["error"]
        assert call_data["error_type"] == "dispatch_error"


class TestConcurrencyAndEdgeCases:
    """Tests for race conditions and other edge cases."""

    @pytest.mark.asyncio
    async def test_race_condition_completion_beats_timeout(self, handler, agent_params, callback_info):
        """
        Test that if a completion message arrives just before a timeout, completion wins.
        """
        # Arrange
        complete_cb = AsyncMock()
        timeout_cb = AsyncMock()
        handler.register_global_callback(CallbackType.ON_COMPLETE, complete_cb)
        handler.register_global_callback(CallbackType.ON_TIMEOUT, timeout_cb)

        # Use a real sleep to simulate the race
        agent_id = await handler.start_agent_operation(**agent_params, timeout_seconds=0.05)

        # Act: Send completion response immediately, before timeout can fire
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        message = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info
        )
        await response_handler(message)

        # Wait long enough for the original timeout to have fired
        await asyncio.sleep(0.1)

        # Assert
        complete_cb.assert_awaited_once()
        timeout_cb.assert_not_awaited()
        assert agent_id not in handler.active_operations

    @pytest.mark.asyncio
    async def test_race_condition_timeout_beats_completion(self, handler, agent_params, callback_info):
        """
        Test that if a timeout fires first, a late completion message is ignored.
        """
        # Arrange
        complete_cb = AsyncMock()
        timeout_cb = AsyncMock()
        handler.register_global_callback(CallbackType.ON_COMPLETE, complete_cb)
        handler.register_global_callback(CallbackType.ON_TIMEOUT, timeout_cb)

        agent_id = await handler.start_agent_operation(**agent_params, timeout_seconds=0.01)

        # Act: Wait for timeout to fire, then send a late completion
        await asyncio.sleep(0.05)

        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        message = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info
        )
        await response_handler(message)

        # Assert
        timeout_cb.assert_awaited_once()
        complete_cb.assert_not_awaited()
        # The operation should have been removed by the timeout handler
        assert agent_id not in handler.active_operations

    @pytest.mark.asyncio
    async def test_failing_callback_does_not_stop_others(self, handler, agent_params, callback_info):
        """
        Verify that one failing callback doesn't prevent others from running.
        """
        # Arrange
        def failing_callback(agent_id, data):
            raise ValueError("Callback failed")

        successful_callback = AsyncMock()

        handler.register_global_callback(CallbackType.ON_COMPLETE, failing_callback)
        handler.register_global_callback(CallbackType.ON_COMPLETE, successful_callback)

        agent_id = await handler.start_agent_operation(**agent_params)

        # Act
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        with patch('cadence.agent_communication_handler.logger.error') as mock_log:
            message = create_agent_message(
                MessageType.TASK_COMPLETE,
                agent_params["agent_type"],
                agent_params["context"],
                agent_params["success_criteria"],
                callback_info
            )
            await response_handler(message)

            # Assert
            successful_callback.assert_awaited_once()
            mock_log.assert_called_once()
            # Logger uses single f-string argument, not multiple arguments
            log_message = mock_log.call_args[0][0]
            assert "Error in on_complete callback" in log_message
            assert "Callback failed" in log_message

class TestShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_operations(self, handler, mock_dispatcher, agent_params):
        """
        Verify that shutdown cancels all pending timeout tasks and cleans up state.
        """
        # Arrange - use longer timeout to prevent natural completion
        agent_id1 = await handler.start_agent_operation(**agent_params, timeout_seconds=10.0)
        agent_id2 = await handler.start_agent_operation(**agent_params, timeout_seconds=10.0)

        assert len(handler.active_operations) == 2
        assert len(handler.pending_timeouts) == 2

        timeout_task1 = handler.pending_timeouts[agent_id1]
        timeout_task2 = handler.pending_timeouts[agent_id2]

        # Act
        await handler.shutdown()

        # Assert
        assert handler.shutdown_event.is_set()
        assert not handler.active_operations
        assert not handler.pending_timeouts

        # Give tasks a moment to process cancellation
        await asyncio.sleep(0.01)

        # Tasks should be either cancelled or done (due to race conditions)
        assert timeout_task1.done()  # Either cancelled or completed naturally
        assert timeout_task2.done()

        mock_dispatcher.cleanup.assert_called_once()


class TestCallbackManagement:
    """Tests for callback registration and management."""

    @pytest.mark.asyncio
    async def test_register_and_unregister_global_callbacks(self, handler):
        """Test registering and unregistering global callbacks."""
        # Arrange
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Act & Assert - Register
        handler.register_global_callback(CallbackType.ON_COMPLETE, callback1)
        handler.register_global_callback(CallbackType.ON_COMPLETE, callback2)
        handler.register_global_callback(CallbackType.ON_ERROR, callback1)

        assert len(handler.global_callbacks[CallbackType.ON_COMPLETE]) == 2
        assert len(handler.global_callbacks[CallbackType.ON_ERROR]) == 1

        # Act & Assert - Unregister
        handler.deregister_global_callback(CallbackType.ON_COMPLETE, callback1)
        assert len(handler.global_callbacks[CallbackType.ON_COMPLETE]) == 1
        assert callback2 in handler.global_callbacks[CallbackType.ON_COMPLETE]

    @pytest.mark.asyncio
    async def test_register_operation_callbacks(self, handler, agent_params):
        """Test registering operation-specific callbacks."""
        # Arrange
        agent_id = await handler.start_agent_operation(**agent_params)
        op_callback = MagicMock()

        # Act
        handler.register_operation_callback(agent_id, CallbackType.ON_COMPLETE, op_callback)

        # Assert
        operation = handler.active_operations[agent_id]
        assert op_callback in operation.callbacks[CallbackType.ON_COMPLETE]

        # Test with non-existent operation - returns False, no warning logged
        result = handler.register_operation_callback("fake-id", CallbackType.ON_COMPLETE, op_callback)
        assert result is False

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_callbacks(self, handler, agent_params, callback_info):
        """Test that both sync and async callbacks work correctly."""
        # Arrange
        sync_data = []
        async_data = []

        def sync_callback(agent_id, data):
            sync_data.append(data)

        async def async_callback(agent_id, data):
            await asyncio.sleep(0.01)  # Simulate async work
            async_data.append(data)

        agent_id = await handler.start_agent_operation(**agent_params)
        handler.register_global_callback(CallbackType.ON_COMPLETE, sync_callback)
        handler.register_global_callback(CallbackType.ON_COMPLETE, async_callback)

        # Act
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        message = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info,
            {"test": "data"}
        )
        await response_handler(message)

        # Wait for async processing
        await asyncio.sleep(0.05)

        # Assert
        assert len(sync_data) == 1
        assert len(async_data) == 1
        assert sync_data[0]["result"] == {"test": "data"}
        assert async_data[0]["result"] == {"test": "data"}


class TestStatusAndMonitoring:
    """Tests for status reporting and monitoring methods."""

    @pytest.mark.asyncio
    async def test_get_status(self, handler, agent_params, callback_info):
        """Test the get_operation_status method returns correct operation status."""
        # Arrange
        agent_id = await handler.start_agent_operation(**agent_params)

        # Act & Assert - Active operation
        status = handler.get_operation_status(agent_id)
        assert status["agent_id"] == agent_id
        assert status["agent_type"] == AgentType.REVIEW.value
        assert status["start_time"] is not None
        assert status["elapsed_seconds"] >= 0

        # Complete the operation
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        message = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info
        )
        await response_handler(message)

        # Act & Assert - Completed operation
        status = handler.get_operation_status(agent_id)
        assert status is None

    @pytest.mark.asyncio
    async def test_list_active_operations(self, handler, agent_params):
        """Test listing all active operations."""
        # Arrange
        agent_id1 = await handler.start_agent_operation(**agent_params)
        agent_id2 = await handler.start_agent_operation(**agent_params)

        # Act
        active_ops = handler.list_active_operations()

        # Assert - list_active_operations returns List[str], not Dict
        assert len(active_ops) == 2
        assert agent_id1 in active_ops
        assert agent_id2 in active_ops
        assert isinstance(active_ops, list)


class TestEventProcessing:
    """Tests for the internal event processing mechanism."""

    @pytest.mark.asyncio
    async def test_event_queue_processing(self, handler, agent_params, callback_info):
        """Test that events are properly queued and processed."""
        # Arrange
        processed_events = []

        async def tracking_callback(agent_id, data):
            # Check actual callback data structure - complete callbacks have "result", error callbacks have "error"
            if "result" in data:
                processed_events.append((agent_id, "complete"))
            elif "error" in data:
                processed_events.append((agent_id, "error"))
            else:
                processed_events.append((agent_id, "unknown"))

        handler.register_global_callback(CallbackType.ON_COMPLETE, tracking_callback)
        handler.register_global_callback(CallbackType.ON_ERROR, tracking_callback)

        # Start multiple operations
        agent_id1 = await handler.start_agent_operation(**agent_params)
        agent_id2 = await handler.start_agent_operation(**agent_params)

        # Act - Generate multiple events rapidly
        response_handler1 = handler.dispatcher.dispatch_agent.call_args_list[0][1]['callback_handler']
        response_handler2 = handler.dispatcher.dispatch_agent.call_args_list[1][1]['callback_handler']

        message1 = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info,
            {"event_type": "complete1"}
        )
        message2 = create_agent_message(
            MessageType.ERROR,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info,
            {"event_type": "error2", "error": "test"}
        )
        await response_handler1(message1)
        await response_handler2(message2)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Assert
        assert len(processed_events) == 2
        assert (agent_id1, "complete") in processed_events
        assert (agent_id2, "error") in processed_events


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_handle_unexpected_message_type(self, handler, agent_params, callback_info):
        """Test handling of unexpected message types."""
        # Arrange
        agent_id = await handler.start_agent_operation(**agent_params)

        # Act
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']
        with patch('cadence.agent_communication_handler.logger.warning') as mock_log:
            # Send a message type that isn't TASK_COMPLETE or ERROR
            message = create_agent_message(
                MessageType.REVIEW_TRIGGERED,
                agent_params["agent_type"],
                agent_params["context"],
                agent_params["success_criteria"],
                callback_info,
                {"status": "processing"}
            )
            await response_handler(message)

        # Assert
        # Implementation silently ignores unknown message types, so no warning is logged
        mock_log.assert_not_called()
        # Operation should still be active
        assert agent_id in handler.active_operations

    @pytest.mark.asyncio
    async def test_handle_agent_response_for_unknown_operation(self, handler):
        """Test handling response for non-existent operation."""
        # Arrange
        fake_agent_id = "non-existent-id"

        # Act
        with patch('cadence.agent_communication_handler.logger.warning') as mock_log:
            message = create_agent_message(
                MessageType.TASK_COMPLETE,
                AgentType.REVIEW,
                MessageContext(task_id="fake", parent_session="fake", files_modified=[], project_path="/fake"),
                SuccessCriteria(expected_outcomes=[]),
                CallbackInfo(handler="fake")
            )
            await handler._handle_agent_response(fake_agent_id, message)

        # Assert
        mock_log.assert_called_once()
        assert f"Received response for unknown agent {fake_agent_id}" in mock_log.call_args[0][0]

    @pytest.mark.asyncio
    async def test_double_completion_ignored(self, handler, agent_params, callback_info):
        """Test that attempting to complete an operation twice is handled gracefully."""
        # Arrange
        complete_cb = AsyncMock()
        handler.register_global_callback(CallbackType.ON_COMPLETE, complete_cb)

        agent_id = await handler.start_agent_operation(**agent_params)
        response_handler = handler.dispatcher.dispatch_agent.call_args[1]['callback_handler']

        # Act - Complete once
        message1 = create_agent_message(
            MessageType.TASK_COMPLETE,
            agent_params["agent_type"],
            agent_params["context"],
            agent_params["success_criteria"],
            callback_info,
            {"first": True}
        )
        await response_handler(message1)

        # Try to complete again
        with patch('cadence.agent_communication_handler.logger.warning') as mock_log:
            message2 = create_agent_message(
                MessageType.TASK_COMPLETE,
                agent_params["agent_type"],
                agent_params["context"],
                agent_params["success_criteria"],
                callback_info,
                {"second": True}
            )
            await response_handler(message2)

        # Assert
        complete_cb.assert_awaited_once()  # Only called once
        call_data = complete_cb.call_args[0][1]
        assert call_data["result"]["first"] is True  # First completion data
        mock_log.assert_called_once()  # Warning for second attempt
