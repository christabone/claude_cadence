"""
Unit tests for AgentDispatcher
"""
import pytest
import time
from unittest.mock import Mock, patch
from cadence.agent_dispatcher import AgentDispatcher
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType,
    MessageContext, SuccessCriteria, CallbackInfo
)


class TestAgentDispatcher:
    """Test the AgentDispatcher class"""

    def test_generate_message_id(self):
        """Test message ID generation"""
        dispatcher = AgentDispatcher()

        id1 = dispatcher.generate_message_id()
        id2 = dispatcher.generate_message_id()

        assert id1 != id2
        # Should be a valid UUID
        import uuid
        uuid.UUID(id1)  # Will raise ValueError if not valid UUID
        uuid.UUID(id2)  # Will raise ValueError if not valid UUID

    def test_dispatch_agent(self):
        """Test dispatching an agent"""
        dispatcher = AgentDispatcher()

        # Create test data
        context = MessageContext(
            task_id="task-1",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/project"
        )

        criteria = SuccessCriteria(
            expected_outcomes=["Review complete"],
            validation_steps=["Check output"]
        )

        # Mock callback
        callback = Mock()
        callback.__name__ = "test_callback"

        # Dispatch agent
        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=criteria,
            callback_handler=callback,
            timeout_ms=60000
        )

        # Verify
        # message_id should be a valid UUID
        import uuid
        uuid.UUID(message_id)  # Will raise ValueError if not valid UUID
        assert message_id in dispatcher.pending_messages
        assert message_id in dispatcher.callbacks
        assert dispatcher.callbacks[message_id] == callback

        # Check message content
        message = dispatcher.pending_messages[message_id]
        assert message.message_type == MessageType.DISPATCH_AGENT
        assert message.agent_type == AgentType.REVIEW
        assert message.context.task_id == "task-1"
        assert message.callback.timeout_ms == 60000

    def test_receive_response_success(self):
        """Test receiving a successful response"""
        dispatcher = AgentDispatcher()

        # Setup: dispatch an agent first
        context = MessageContext(
            task_id="task-1",
            parent_session="session-1",
            files_modified=[],
            project_path="/project"
        )

        callback = Mock()
        callback.__name__ = "response_handler"

        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        # Create response
        response_data = {
            "message_type": "AGENT_RESPONSE",
            "agent_type": "fix",
            "context": {
                "task_id": "task-1",
                "parent_session": "session-1",
                "files_modified": ["fixed.py"],
                "project_path": "/project"
            },
            "success_criteria": {
                "expected_outcomes": [],
                "validation_steps": []
            },
            "callback": {
                "handler": "response_handler",
                "timeout_ms": 30000
            },
            "payload": {
                "original_message_id": message_id,
                "result": "success"
            }
        }

        # Process response
        result = dispatcher.receive_response(response_data)

        # Verify
        assert result is True
        assert callback.called
        assert message_id not in dispatcher.pending_messages
        assert message_id not in dispatcher.callbacks

    def test_receive_response_no_pending_message(self):
        """Test receiving response with no pending message"""
        dispatcher = AgentDispatcher()

        response_data = {
            "message_type": "AGENT_RESPONSE",
            "agent_type": "review",
            "context": {
                "task_id": "t1",
                "parent_session": "s1",
                "files_modified": [],
                "project_path": "/p"
            },
            "success_criteria": {
                "expected_outcomes": [],
                "validation_steps": []
            },
            "callback": {
                "handler": "handler",
                "timeout_ms": 30000
            },
            "payload": {
                "original_message_id": "non-existent"
            }
        }

        result = dispatcher.receive_response(response_data)
        assert result is False

    def test_create_error_response(self):
        """Test creating an error response"""
        dispatcher = AgentDispatcher()

        # Setup: dispatch an agent
        context = MessageContext(
            task_id="task-1",
            parent_session="session-1",
            files_modified=[],
            project_path="/project"
        )

        callback = Mock()
        callback.__name__ = "error_test_callback"

        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        # Create error response
        error_response = dispatcher.create_error_response(
            message_id,
            "Agent timeout"
        )

        # Verify
        assert error_response.message_type == MessageType.ERROR
        assert error_response.agent_type == AgentType.REVIEW
        assert error_response.payload["original_message_id"] == message_id
        assert error_response.payload["error"] == "Agent timeout"

    def test_get_pending_count(self):
        """Test getting pending message count"""
        dispatcher = AgentDispatcher()

        assert dispatcher.get_pending_count() == 0

        # Dispatch some agents
        for i in range(3):
            callback = Mock()
            callback.__name__ = f"callback_{i}"
            dispatcher.dispatch_agent(
                agent_type=AgentType.REVIEW,
                context=MessageContext(f"t{i}", f"s{i}", [], "/p"),
                success_criteria=SuccessCriteria(),
                callback_handler=callback
            )

        assert dispatcher.get_pending_count() == 3

    def test_queue_message(self):
        """Test message queuing"""
        dispatcher = AgentDispatcher()

        # Create a message
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=AgentType.REVIEW,
            context=MessageContext("t1", "s1", [], "/p"),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo("handler", 30000),
            message_id="550e8400-e29b-41d4-a716-446655440004"
        )

        # Queue it
        dispatcher.queue_message(message)

        # Verify we can get it back
        retrieved = dispatcher.get_next_message()
        assert retrieved is not None
        assert retrieved.message_id == "550e8400-e29b-41d4-a716-446655440004"

        # Queue should be empty now
        assert dispatcher.get_next_message() is None

    def test_dispatch_with_queue(self):
        """Test dispatching with queue option"""
        dispatcher = AgentDispatcher()

        context = MessageContext("t1", "s1", [], "/p")
        callback = Mock()
        callback.__name__ = "queue_callback"

        # Dispatch with queue
        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback,
            use_queue=True
        )

        # Message should be in queue
        queued = dispatcher.get_next_message()
        assert queued is not None
        assert queued.message_id == message_id
        assert queued.agent_type == AgentType.FIX

    def test_process_queue(self):
        """Test processing the message queue"""
        dispatcher = AgentDispatcher()

        # Queue multiple messages
        message_ids = []
        for i in range(3):
            context = MessageContext(f"t{i}", f"s{i}", [], "/p")
            callback = Mock()
            callback.__name__ = f"callback_{i}"

            msg_id = dispatcher.dispatch_agent(
                agent_type=AgentType.REVIEW,
                context=context,
                success_criteria=SuccessCriteria(),
                callback_handler=callback,
                use_queue=True
            )
            message_ids.append(msg_id)

        # Process queue
        processed = dispatcher.process_queue()

        # All messages should be processed
        assert len(processed) == 3
        assert set(processed) == set(message_ids)

        # Queue should be empty
        assert dispatcher.get_next_message() is None

    def test_response_queue(self):
        """Test response queuing and processing"""
        dispatcher = AgentDispatcher()

        # First dispatch an agent
        context = MessageContext("t1", "s1", [], "/p")
        callback = Mock()
        callback.__name__ = "resp_callback"

        msg_id = dispatcher.dispatch_agent(
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        # Create and queue a response
        response = AgentMessage(
            message_type=MessageType.AGENT_RESPONSE,
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo("resp_callback", 30000),
            message_id="550e8400-e29b-41d4-a716-446655440005",
            payload={"original_message_id": msg_id, "result": "success"}
        )

        dispatcher.queue_response(response)

        # Process queue
        processed = dispatcher.process_queue()

        # Response should be processed and callback called
        assert len(processed) == 1
        assert callback.called

    def test_timeout_handling(self):
        """Test message timeout handling"""
        dispatcher = AgentDispatcher()

        context = MessageContext("t1", "s1", [], "/p")
        callback = Mock()
        callback.__name__ = "timeout_callback"

        # Dispatch with short timeout
        msg_id = dispatcher.dispatch_agent(
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback,
            timeout_ms=100  # 100ms timeout
        )

        # Wait for timeout
        time.sleep(0.15)  # 150ms

        # Process the timeout error from response queue
        processed = dispatcher.process_queue()

        # Should have processed the timeout error
        assert len(processed) == 1

        # Callback should be called with error
        assert callback.called
        error_response = callback.call_args[0][0]
        assert error_response.message_type == MessageType.ERROR
        assert "timeout" in error_response.payload["error"].lower()

        # Cleanup
        dispatcher.cleanup()

    def test_cancel_timeout_on_response(self):
        """Test that timeout is cancelled when response received"""
        dispatcher = AgentDispatcher()

        context = MessageContext("t1", "s1", [], "/p")
        callback = Mock()
        callback.__name__ = "cancel_timeout_callback"

        # Dispatch with longer timeout
        msg_id = dispatcher.dispatch_agent(
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback,
            timeout_ms=1000  # 1 second timeout
        )

        # Send response before timeout
        response_data = {
            "message_type": "AGENT_RESPONSE",
            "agent_type": "fix",
            "context": context.__dict__,
            "success_criteria": {"expected_outcomes": [], "validation_steps": []},
            "callback": {"handler": "cancel_timeout_callback", "timeout_ms": 1000},
            "payload": {"original_message_id": msg_id, "result": "success"}
        }

        result = dispatcher.receive_response(response_data)
        assert result is True

        # Wait past original timeout
        time.sleep(0.1)

        # Process queue - should be empty (no timeout error)
        processed = dispatcher.process_queue()
        assert len(processed) == 0

        # Cleanup
        dispatcher.cleanup()

    def test_cleanup_timers(self):
        """Test cleanup of all timers"""
        dispatcher = AgentDispatcher()

        # Dispatch multiple agents with timeouts
        for i in range(3):
            context = MessageContext(f"t{i}", f"s{i}", [], "/p")
            callback = Mock()
            callback.__name__ = f"cleanup_callback_{i}"

            dispatcher.dispatch_agent(
                agent_type=AgentType.REVIEW,
                context=context,
                success_criteria=SuccessCriteria(),
                callback_handler=callback,
                timeout_ms=5000  # 5 second timeout
            )

        # Should have 3 timers
        assert len(dispatcher.timers) == 3

        # Cleanup all timers
        dispatcher.cleanup()

        # All timers should be cleared
        assert len(dispatcher.timers) == 0
