"""
Comprehensive unit tests for AgentMessageSchema validation system
"""

import pytest
import json
import uuid
from datetime import datetime
from cadence.agent_messages import (
    AgentMessage, AgentMessageSchema, MessageType, AgentType, Priority,
    MessageContext, SuccessCriteria, CallbackInfo
)


class TestAgentMessageSchema:
    """Test cases for AgentMessageSchema validation"""

    @pytest.fixture
    def valid_message_data(self):
        """Sample valid message data for testing"""
        return {
            "message_type": "DISPATCH_AGENT",
            "agent_type": "review",
            "priority": "high",
            "timestamp": "2025-06-27T17:30:00.000Z",
            "session_id": str(uuid.uuid4()),
            "context": {
                "task_id": "task-123",
                "parent_session": "session-456",
                "files_modified": ["src/test.py", "tests/test_file.py"],
                "project_path": "/path/to/project",
                "file_paths": ["src/module.py"],
                "modifications": {"lines_added": 10},
                "scope": {"max_files": 5}
            },
            "success_criteria": {
                "expected_outcomes": ["code review completed", "issues identified"],
                "validation_steps": ["syntax check", "security scan"]
            },
            "callback": {
                "handler": "review_callback",
                "timeout_ms": 30000
            },
            "message_id": str(uuid.uuid4()),
            "payload": {"additional_data": "test"}
        }

    @pytest.fixture
    def minimal_valid_message_data(self):
        """Minimal valid message data for testing"""
        return {
            "message_type": "TASK_COMPLETE",
            "agent_type": "fix",
            "context": {
                "task_id": "task-123",
                "parent_session": "session-456",
                "files_modified": ["src/test.py"],
                "project_path": "/path/to/project"
            },
            "success_criteria": {
                "expected_outcomes": [],
                "validation_steps": []
            },
            "callback": {
                "handler": "completion_callback"
            }
        }

    def test_valid_message_validation(self, valid_message_data):
        """Test validation of a valid message"""
        result = AgentMessageSchema.validate_message(valid_message_data)

        assert result["message_type"] == "DISPATCH_AGENT"
        assert result["agent_type"] == "review"
        assert result["priority"] == "high"
        assert "timestamp" in result
        assert "session_id" in result
        assert result["context"]["task_id"] == "task-123"

    def test_minimal_valid_message_validation(self, minimal_valid_message_data):
        """Test validation of a minimal valid message"""
        result = AgentMessageSchema.validate_message(minimal_valid_message_data)

        assert result["message_type"] == "TASK_COMPLETE"
        assert result["agent_type"] == "fix"
        assert result["context"]["task_id"] == "task-123"

    def test_missing_required_fields(self):
        """Test validation fails for missing required fields"""
        invalid_data = {
            "message_type": "DISPATCH_AGENT",
            "agent_type": "review"
            # Missing context, success_criteria, callback
        }

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(invalid_data)

    def test_invalid_message_type(self, minimal_valid_message_data):
        """Test validation fails for invalid message type"""
        minimal_valid_message_data["message_type"] = "INVALID_TYPE"

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_invalid_agent_type(self, minimal_valid_message_data):
        """Test validation fails for invalid agent type"""
        minimal_valid_message_data["agent_type"] = "invalid_agent"

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_invalid_priority(self, minimal_valid_message_data):
        """Test validation fails for invalid priority"""
        minimal_valid_message_data["priority"] = "invalid_priority"

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_invalid_uuid_format(self, minimal_valid_message_data):
        """Test validation fails for invalid UUID format"""
        minimal_valid_message_data["session_id"] = "not-a-uuid"

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_invalid_timestamp_format(self, minimal_valid_message_data):
        """Test validation fails for invalid timestamp format"""
        minimal_valid_message_data["timestamp"] = "not-a-timestamp"

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_security_sanitization(self):
        """Test security sanitization removes dangerous characters"""
        malicious_data = {
            "message_type": "DISPATCH_AGENT",
            "agent_type": "review",
            "context": {
                "task_id": "task<script>alert('xss')</script>",
                "parent_session": 'session"; DROP TABLE users; --',
                "files_modified": ["../../../etc/passwd", "normal_file.py"],
                "project_path": "/path/to/project"
            },
            "success_criteria": {
                "expected_outcomes": [],
                "validation_steps": []
            },
            "callback": {
                "handler": "handler_name"
            }
        }

        result = AgentMessageSchema.validate_message(malicious_data)

        # Check that dangerous characters are removed
        assert "<script>" not in result["context"]["task_id"]
        assert "DROP TABLE" not in result["context"]["parent_session"]
        assert "../../../etc/passwd" != result["context"]["files_modified"][0]
        assert "normal_file.py" == result["context"]["files_modified"][1]

    def test_business_rule_review_triggered_validation(self, minimal_valid_message_data):
        """Test business rule: REVIEW_TRIGGERED must use review agent"""
        minimal_valid_message_data["message_type"] = "REVIEW_TRIGGERED"
        minimal_valid_message_data["agent_type"] = "fix"  # Wrong agent type

        with pytest.raises(ValueError, match="REVIEW_TRIGGERED messages must use 'review' agent_type"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_business_rule_fix_required_validation(self, minimal_valid_message_data):
        """Test business rule: FIX_REQUIRED must use fix agent"""
        minimal_valid_message_data["message_type"] = "FIX_REQUIRED"
        minimal_valid_message_data["agent_type"] = "review"  # Wrong agent type

        with pytest.raises(ValueError, match="FIX_REQUIRED messages must use 'fix' agent_type"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_business_rule_critical_priority_validation(self, minimal_valid_message_data):
        """Test business rule: critical priority requires files_modified"""
        minimal_valid_message_data["priority"] = "critical"
        minimal_valid_message_data["context"]["files_modified"] = []  # Empty files

        with pytest.raises(ValueError, match="Critical/high priority messages must specify files_modified"):
            AgentMessageSchema.validate_message(minimal_valid_message_data)

    def test_serialize_message_success(self):
        """Test successful message serialization"""
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=AgentType.REVIEW,
            priority=Priority.HIGH,
            context=MessageContext(
                task_id="test-123",
                parent_session="session-456",
                files_modified=["test.py"],
                project_path="/test/path"
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["review complete"],
                validation_steps=["syntax check"]
            ),
            callback=CallbackInfo(handler="test_handler", timeout_ms=30000)
        )

        json_str = AgentMessageSchema.serialize_message(message)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["message_type"] == "DISPATCH_AGENT"
        assert parsed["agent_type"] == "review"
        assert parsed["priority"] == "high"

    def test_deserialize_message_success(self, valid_message_data):
        """Test successful message deserialization"""
        json_str = json.dumps(valid_message_data)

        message = AgentMessageSchema.deserialize_message(json_str)

        assert isinstance(message, AgentMessage)
        assert message.message_type == MessageType.DISPATCH_AGENT
        assert message.agent_type == AgentType.REVIEW
        assert message.priority == Priority.HIGH
        assert message.context.task_id == "task-123"

    def test_deserialize_invalid_json(self):
        """Test deserialization fails with invalid JSON"""
        invalid_json = '{"invalid": json syntax}'

        with pytest.raises(ValueError, match="Invalid JSON format"):
            AgentMessageSchema.deserialize_message(invalid_json)

    def test_serialization_round_trip(self):
        """Test serialization/deserialization round trip"""
        original_message = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            priority=Priority.MEDIUM,
            context=MessageContext(
                task_id="round-trip-test",
                parent_session="session-rt",
                files_modified=["file1.py", "file2.py"],
                project_path="/rt/path",
                file_paths=["extra.py"],
                modifications={"test": "data"},
                scope={"limit": 10}
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["fix applied"],
                validation_steps=["test run"]
            ),
            callback=CallbackInfo(handler="rt_handler", timeout_ms=60000)
        )

        # Serialize
        json_str = AgentMessageSchema.serialize_message(original_message)

        # Deserialize
        restored_message = AgentMessageSchema.deserialize_message(json_str)

        # Compare key fields
        assert restored_message.message_type == original_message.message_type
        assert restored_message.agent_type == original_message.agent_type
        assert restored_message.priority == original_message.priority
        assert restored_message.context.task_id == original_message.context.task_id
        assert restored_message.context.file_paths == original_message.context.file_paths
        assert restored_message.context.modifications == original_message.context.modifications
        assert restored_message.context.scope == original_message.context.scope

    def test_schema_version(self):
        """Test schema version retrieval"""
        version = AgentMessageSchema.get_schema_version()
        assert version == "1.0"
        assert isinstance(version, str)

    def test_all_message_types_supported(self):
        """Test that all MessageType enum values are supported in schema"""
        schema_message_types = set(AgentMessageSchema.MESSAGE_SCHEMA["properties"]["message_type"]["enum"])
        enum_message_types = set(mt.value for mt in MessageType)

        assert schema_message_types == enum_message_types

    def test_all_agent_types_supported(self):
        """Test that all AgentType enum values are supported in schema"""
        schema_agent_types = set(AgentMessageSchema.MESSAGE_SCHEMA["properties"]["agent_type"]["enum"])
        enum_agent_types = set(at.value for at in AgentType)

        assert schema_agent_types == enum_agent_types

    def test_all_priority_types_supported(self):
        """Test that all Priority enum values are supported in schema"""
        schema_priorities = set(AgentMessageSchema.MESSAGE_SCHEMA["properties"]["priority"]["enum"])
        enum_priorities = set(p.value for p in Priority)

        assert schema_priorities == enum_priorities

    def test_context_field_limits(self, valid_message_data):
        """Test context field size limits"""
        # Test max files limit
        valid_message_data["context"]["files_modified"] = ["file.py"] * 101  # Over limit

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(valid_message_data)

    def test_success_criteria_limits(self, valid_message_data):
        """Test success criteria field limits"""
        # Test max outcomes limit
        valid_message_data["success_criteria"]["expected_outcomes"] = ["outcome"] * 51  # Over limit

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(valid_message_data)

    def test_timeout_range_validation(self, valid_message_data):
        """Test callback timeout range validation"""
        # Test timeout too low
        valid_message_data["callback"]["timeout_ms"] = 500  # Below minimum

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(valid_message_data)

        # Test timeout too high
        valid_message_data["callback"]["timeout_ms"] = 700000  # Above maximum

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(valid_message_data)

    def test_additional_properties_blocked(self, valid_message_data):
        """Test that additional properties are not allowed"""
        valid_message_data["unknown_field"] = "should_not_be_allowed"

        with pytest.raises(ValueError, match="Schema validation failed"):
            AgentMessageSchema.validate_message(valid_message_data)

    def test_enhanced_message_creation_with_defaults(self):
        """Test that AgentMessage creates proper defaults for new fields"""
        message = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.REVIEW,
            context=MessageContext(
                task_id="test-defaults",
                parent_session="session-def",
                files_modified=["test.py"],
                project_path="/test"
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="default_test")
        )

        # Check that defaults are set
        assert message.priority == Priority.MEDIUM
        assert message.timestamp is not None
        assert message.session_id is not None
        assert message.message_id is not None

        # Check UUID format
        uuid.UUID(message.session_id)  # Should not raise
        uuid.UUID(message.message_id)  # Should not raise

        # Check timestamp format
        datetime.fromisoformat(message.timestamp.replace('Z', '+00:00'))  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])
