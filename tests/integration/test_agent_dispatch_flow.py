"""
Integration tests for complete agent dispatch flow
"""
import pytest
import time
from unittest.mock import Mock
from cadence.agent_dispatcher import AgentDispatcher
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType,
    MessageContext, SuccessCriteria, CallbackInfo
)


class TestAgentDispatchFlow:
    """Test complete message dispatch -> response -> callback flow"""

    def test_full_dispatch_flow(self):
        """Test complete dispatch and response flow"""
        dispatcher = AgentDispatcher()

        # Track callback execution
        callback_executed = False
        callback_response = None

        def review_callback(response: AgentMessage):
            nonlocal callback_executed, callback_response
            callback_executed = True
            callback_response = response

        # Dispatch an agent
        context = MessageContext(
            task_id="integration-test-1",
            parent_session="session-int-1",
            files_modified=["test1.py", "test2.py"],
            project_path="/test/project"
        )

        criteria = SuccessCriteria(
            expected_outcomes=["Code reviewed", "Issues found"],
            validation_steps=["Run linter", "Check tests"]
        )

        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=criteria,
            callback_handler=review_callback,
            timeout_ms=5000
        )

        # Simulate agent response
        agent_response = {
            "message_type": "AGENT_RESPONSE",
            "agent_type": "review",
            "context": {
                "task_id": "integration-test-1",
                "parent_session": "session-int-1",
                "files_modified": ["test1.py", "test2.py"],
                "project_path": "/test/project"
            },
            "success_criteria": {
                "expected_outcomes": ["Code reviewed", "Issues found"],
                "validation_steps": ["Run linter", "Check tests"]
            },
            "callback": {
                "handler": "review_callback",
                "timeout_ms": 5000
            },
            "message_id": "response-int-1",
            "payload": {
                "original_message_id": message_id,
                "result": "success",
                "issues_found": ["Missing docstring", "Long function"]
            }
        }

        # Process response
        result = dispatcher.receive_response(agent_response)

        # Verify
        assert result is True
        assert callback_executed is True
        assert callback_response is not None
        assert callback_response.message_type == MessageType.AGENT_RESPONSE
        assert callback_response.payload["result"] == "success"
        assert len(callback_response.payload["issues_found"]) == 2

        # Cleanup
        dispatcher.cleanup()

    def test_error_propagation(self):
        """Test error message propagation to callback"""
        dispatcher = AgentDispatcher()

        error_received = False
        error_message = None

        def error_callback(response: AgentMessage):
            nonlocal error_received, error_message
            error_received = True
            error_message = response.payload.get("error") if response.payload else None

        # Dispatch
        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.FIX,
            context=MessageContext("t1", "s1", [], "/p"),
            success_criteria=SuccessCriteria(),
            callback_handler=error_callback
        )

        # Send error response
        error_response = {
            "message_type": "ERROR",
            "agent_type": "fix",
            "context": {"task_id": "t1", "parent_session": "s1",
                       "files_modified": [], "project_path": "/p"},
            "success_criteria": {"expected_outcomes": [], "validation_steps": []},
            "callback": {"handler": "error_callback", "timeout_ms": 30000},
            "payload": {
                "original_message_id": message_id,
                "error": "Agent failed to process request"
            }
        }

        dispatcher.receive_response(error_response)

        assert error_received is True
        assert error_message == "Agent failed to process request"

        dispatcher.cleanup()

    def test_task_complete_flow(self):
        """Test TASK_COMPLETE message flow"""
        dispatcher = AgentDispatcher()

        complete_received = False

        def complete_callback(response: AgentMessage):
            nonlocal complete_received
            complete_received = response.message_type == MessageType.TASK_COMPLETE

        # Dispatch
        message_id = dispatcher.dispatch_agent(
            agent_type=AgentType.FIX,
            context=MessageContext("fix-1", "session-1", ["bug.py"], "/app"),
            success_criteria=SuccessCriteria(["Bug fixed"], ["Tests pass"]),
            callback_handler=complete_callback
        )

        # Send TASK_COMPLETE
        complete_response = {
            "message_type": "TASK_COMPLETE",
            "agent_type": "fix",
            "context": {"task_id": "fix-1", "parent_session": "session-1",
                       "files_modified": ["bug.py"], "project_path": "/app"},
            "success_criteria": {"expected_outcomes": ["Bug fixed"],
                               "validation_steps": ["Tests pass"]},
            "callback": {"handler": "complete_callback", "timeout_ms": 30000},
            "payload": {
                "original_message_id": message_id,
                "result": "success",
                "files_fixed": ["bug.py"],
                "tests_passed": True
            }
        }

        dispatcher.receive_response(complete_response)

        assert complete_received is True

        dispatcher.cleanup()

    def test_queued_dispatch_flow(self):
        """Test dispatch flow with message queuing"""
        dispatcher = AgentDispatcher()

        responses = []

        def queue_callback(response: AgentMessage):
            responses.append(response)

        # Dispatch multiple agents with queuing
        message_ids = []
        for i in range(3):
            msg_id = dispatcher.dispatch_agent(
                agent_type=AgentType.REVIEW,
                context=MessageContext(f"task-{i}", f"session-{i}", [], "/p"),
                success_criteria=SuccessCriteria(),
                callback_handler=queue_callback,
                use_queue=True
            )
            message_ids.append(msg_id)

        # Process outgoing queue
        processed_out = dispatcher.process_queue()
        assert len(processed_out) == 3

        # Queue responses
        for i, msg_id in enumerate(message_ids):
            response = AgentMessage(
                message_type=MessageType.AGENT_RESPONSE,
                agent_type=AgentType.REVIEW,
                context=MessageContext(f"task-{i}", f"session-{i}", [], "/p"),
                success_criteria=SuccessCriteria(),
                callback=CallbackInfo(handler="test_handler"),
                payload={"original_message_id": msg_id, "index": i}
            )
            dispatcher.queue_response(response)

        # Process response queue
        processed_in = dispatcher.process_queue()
        assert len(processed_in) == 3

        # Verify all callbacks executed
        assert len(responses) == 3
        for i, resp in enumerate(responses):
            assert resp.payload["index"] == i

        dispatcher.cleanup()
