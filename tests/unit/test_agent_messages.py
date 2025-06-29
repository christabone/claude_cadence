"""
Unit tests for agent message data classes
"""
import pytest
import json
from cadence.agent_messages import (
    MessageType, AgentType, Priority, MessageContext, SuccessCriteria,
    CallbackInfo, AgentMessage
)


class TestAgentMessages:
    """Test agent message data classes"""

    def test_message_type_enum(self):
        """Test MessageType enum values"""
        assert MessageType.DISPATCH_AGENT.value == "DISPATCH_AGENT"
        assert MessageType.AGENT_RESPONSE.value == "AGENT_RESPONSE"
        assert MessageType.TASK_COMPLETE.value == "TASK_COMPLETE"
        assert MessageType.ERROR.value == "ERROR"
        assert MessageType.REVIEW_TRIGGERED.value == "REVIEW_TRIGGERED"
        assert MessageType.FIX_REQUIRED.value == "FIX_REQUIRED"

    def test_agent_type_enum(self):
        """Test AgentType enum values"""
        assert AgentType.REVIEW.value == "review"
        assert AgentType.FIX.value == "fix"

    def test_priority_enum(self):
        """Test Priority enum values"""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_create_simple_message(self):
        """Test creating a simple agent message"""
        context = MessageContext(
            task_id="task-123",
            parent_session="session-456",
            files_modified=["file1.py", "file2.py"],
            project_path="/home/project"
        )

        criteria = SuccessCriteria(
            expected_outcomes=["Review complete"],
            validation_steps=["Check output"]
        )

        callback = CallbackInfo(
            handler="handle_review_response",
            timeout_ms=60000
        )

        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=criteria,
            callback=callback,
            message_id="msg-789"
        )

        assert message.message_type == MessageType.DISPATCH_AGENT
        assert message.agent_type == AgentType.REVIEW
        assert message.context.task_id == "task-123"
        assert len(message.context.files_modified) == 2
        assert message.callback.timeout_ms == 60000

    def test_message_to_dict(self):
        """Test converting message to dictionary"""
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="t1",
                parent_session="s1",
                files_modified=[],
                project_path="/path"
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="handle_fix"),
            message_id="m1"
        )

        result = message.to_dict()

        assert result["message_type"] == "DISPATCH_AGENT"
        assert result["agent_type"] == "fix"
        assert result["context"]["task_id"] == "t1"
        assert result["callback"]["handler"] == "handle_fix"
        assert result["callback"]["timeout_ms"] == 30000  # Default

    def test_message_from_dict(self):
        """Test creating message from dictionary"""
        data = {
            "message_type": "AGENT_RESPONSE",
            "agent_type": "review",
            "context": {
                "task_id": "task-1",
                "parent_session": "session-1",
                "files_modified": ["test.py"],
                "project_path": "/project"
            },
            "success_criteria": {
                "expected_outcomes": ["Done"],
                "validation_steps": ["Verify"]
            },
            "callback": {
                "handler": "callback_handler",
                "timeout_ms": 45000
            },
            "message_id": "550e8400-e29b-41d4-a716-446655440001",  # Valid UUID format
            "payload": {"result": "success"}
        }

        message = AgentMessage.from_dict(data)

        assert message.message_type == MessageType.AGENT_RESPONSE
        assert message.agent_type == AgentType.REVIEW
        assert message.context.task_id == "task-1"
        assert message.callback.timeout_ms == 45000
        assert message.payload["result"] == "success"

    def test_json_serialization(self):
        """Test JSON serialization roundtrip"""
        original = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.REVIEW,
            context=MessageContext(
                task_id="t1",
                parent_session="s1",
                files_modified=["a.py", "b.py"],
                project_path="/home"
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["No errors"],
                validation_steps=["Run tests"]
            ),
            callback=CallbackInfo(
                handler="error_handler",
                timeout_ms=15000
            )
        )

        # Convert to JSON and back
        json_str = json.dumps(original.to_dict())
        data = json.loads(json_str)
        restored = AgentMessage.from_dict(data)

        assert restored.message_type == original.message_type
        assert restored.agent_type == original.agent_type
        assert restored.context.task_id == original.context.task_id
        assert restored.context.files_modified == original.context.files_modified
        assert restored.callback.timeout_ms == original.callback.timeout_ms

    def test_from_dict_invalid_message_type(self):
        """Test from_dict with invalid message type"""
        data = {
            "message_type": "INVALID_TYPE",
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
                "timeout_ms": 1000
            }
        }

        with pytest.raises(ValueError) as excinfo:
            AgentMessage.from_dict(data)
        assert "Schema validation failed" in str(excinfo.value)

    def test_from_dict_invalid_agent_type(self):
        """Test from_dict with invalid agent type"""
        data = {
            "message_type": "DISPATCH_AGENT",
            "agent_type": "invalid_agent",
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
                "timeout_ms": 1000
            }
        }

        with pytest.raises(ValueError) as excinfo:
            AgentMessage.from_dict(data)
        assert "Schema validation failed" in str(excinfo.value)

    def test_from_dict_missing_required_fields(self):
        """Test from_dict with missing required fields"""
        base_data = {
            "message_type": "DISPATCH_AGENT",
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
                "timeout_ms": 1000
            }
        }

        # Test each required field
        required_fields = ["message_type", "agent_type", "context", "success_criteria", "callback"]
        for field in required_fields:
            incomplete_data = base_data.copy()
            del incomplete_data[field]

            with pytest.raises(ValueError) as excinfo:
                AgentMessage.from_dict(incomplete_data)
            assert "Schema validation failed" in str(excinfo.value)

    def test_from_dict_invalid_nested_structures(self):
        """Test from_dict with invalid nested structures"""
        # Test invalid context
        data = {
            "message_type": "DISPATCH_AGENT",
            "agent_type": "review",
            "context": "not a dict",
            "success_criteria": {
                "expected_outcomes": [],
                "validation_steps": []
            },
            "callback": {
                "handler": "handler",
                "timeout_ms": 1000
            }
        }
        with pytest.raises(ValueError) as excinfo:
            AgentMessage.from_dict(data)
        assert "Schema validation failed" in str(excinfo.value)

        # Test invalid success_criteria
        data["context"] = {
            "task_id": "t1",
            "parent_session": "s1",
            "files_modified": [],
            "project_path": "/p"
        }
        data["success_criteria"] = ["not", "a", "dict"]
        with pytest.raises(ValueError) as excinfo:
            AgentMessage.from_dict(data)
        assert "Schema validation failed" in str(excinfo.value)

        # Test invalid callback
        data["success_criteria"] = {
            "expected_outcomes": [],
            "validation_steps": []
        }
        data["callback"] = 123
        with pytest.raises(ValueError) as excinfo:
            AgentMessage.from_dict(data)
        assert "Schema validation failed" in str(excinfo.value)

    def test_enhanced_message_with_priority_and_defaults(self):
        """Test creating enhanced message with priority and auto-generated fields"""
        message = AgentMessage(
            message_type=MessageType.REVIEW_TRIGGERED,
            agent_type=AgentType.REVIEW,
            priority=Priority.HIGH,
            context=MessageContext(
                task_id="enhanced-test",
                parent_session="session-enhanced",
                files_modified=["enhanced.py"],
                project_path="/enhanced/path",
                file_paths=["additional.py"],
                modifications={"type": "enhancement"},
                scope={"max_changes": 5}
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["enhanced review"],
                validation_steps=["enhanced validation"]
            ),
            callback=CallbackInfo(handler="enhanced_handler", timeout_ms=45000)
        )

        # Test new fields
        assert message.priority == Priority.HIGH
        assert message.timestamp is not None
        assert message.session_id is not None
        assert message.message_id is not None
        assert message.context.file_paths == ["additional.py"]
        assert message.context.modifications == {"type": "enhancement"}
        assert message.context.scope == {"max_changes": 5}

    def test_enhanced_message_to_dict_with_optional_fields(self):
        """Test to_dict includes optional context fields when present"""
        message = AgentMessage(
            message_type=MessageType.FIX_REQUIRED,
            agent_type=AgentType.FIX,
            priority=Priority.CRITICAL,
            context=MessageContext(
                task_id="optional-test",
                parent_session="session-opt",
                files_modified=["test.py"],
                project_path="/test",
                file_paths=["extra1.py", "extra2.py"],
                modifications={"lines_changed": 20},
                scope={"affected_modules": ["module1", "module2"]}
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="opt_handler")
        )

        result = message.to_dict()

        # Check new top-level fields
        assert result["priority"] == "critical"
        assert "timestamp" in result
        assert "session_id" in result

        # Check optional context fields are included
        assert result["context"]["file_paths"] == ["extra1.py", "extra2.py"]
        assert result["context"]["modifications"] == {"lines_changed": 20}
        assert result["context"]["scope"] == {"affected_modules": ["module1", "module2"]}

    def test_enhanced_message_from_dict_with_new_fields(self):
        """Test from_dict handles new fields correctly"""
        data = {
            "message_type": "FIX_REQUIRED",
            "agent_type": "fix",
            "priority": "critical",
            "timestamp": "2025-06-27T17:45:00.000Z",
            "session_id": "550e8400-e29b-41d4-a716-446655440003",
            "context": {
                "task_id": "enhanced-from-dict",
                "parent_session": "session-efd",
                "files_modified": ["main.py"],
                "project_path": "/enhanced",
                "file_paths": ["utils.py"],
                "modifications": {"complexity": "high"},
                "scope": {"impact": "significant"}
            },
            "success_criteria": {
                "expected_outcomes": ["fix applied"],
                "validation_steps": ["regression test"]
            },
            "callback": {
                "handler": "enhanced_callback",
                "timeout_ms": 90000
            },
            "message_id": "550e8400-e29b-41d4-a716-446655440002",
            "payload": {"metadata": "enhanced"}
        }

        message = AgentMessage.from_dict(data)

        # Test new fields are properly parsed
        assert message.message_type == MessageType.FIX_REQUIRED
        assert message.agent_type == AgentType.FIX
        assert message.priority == Priority.CRITICAL
        assert message.timestamp == "2025-06-27T17:45:00.000Z"
        assert message.session_id == "550e8400-e29b-41d4-a716-446655440003"
        assert message.context.file_paths == ["utils.py"]
        assert message.context.modifications == {"complexity": "high"}
        assert message.context.scope == {"impact": "significant"}

    def test_from_dict_with_invalid_priority(self):
        """Test from_dict with invalid priority"""
        data = {
            "message_type": "DISPATCH_AGENT",
            "agent_type": "review",
            "priority": "invalid_priority",
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
                "timeout_ms": 1000
            }
        }

        with pytest.raises(ValueError) as excinfo:
            AgentMessage.from_dict(data)
        assert "Schema validation failed" in str(excinfo.value)
