"""
Enhanced message data classes for agent dispatch protocol with JSON schema validation
"""
import json
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

try:
    import jsonschema
except ImportError:
    jsonschema = None

from .config import MAX_TIMEOUT_MS


class MessageType(str, Enum):
    """Message type constants"""
    DISPATCH_AGENT = "DISPATCH_AGENT"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    TASK_COMPLETE = "TASK_COMPLETE"
    ERROR = "ERROR"
    REVIEW_TRIGGERED = "REVIEW_TRIGGERED"
    FIX_REQUIRED = "FIX_REQUIRED"
    ESCALATION_REQUIRED = "ESCALATION_REQUIRED"


class AgentType(str, Enum):
    """Agent type constants"""
    REVIEW = "review"
    FIX = "fix"
    VERIFY = "verify"


class Priority(str, Enum):
    """Priority levels for messages"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MessageContext:
    """Context information for agent messages"""
    task_id: str
    parent_session: str
    files_modified: List[str]
    project_path: str
    file_paths: Optional[List[str]] = None  # Enhanced for schema requirements
    modifications: Optional[Dict[str, Any]] = None
    scope: Optional[Dict[str, Any]] = None


@dataclass
class SuccessCriteria:
    """Success criteria for agent tasks"""
    expected_outcomes: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)


@dataclass
class CallbackInfo:
    """Callback information for response handling"""
    handler: str
    timeout_ms: int = 30000  # Default 30 seconds


@dataclass
class AgentMessage:
    """Main message structure for agent dispatch protocol"""
    message_type: MessageType
    agent_type: AgentType
    context: MessageContext
    success_criteria: SuccessCriteria
    callback: CallbackInfo
    priority: Priority = Priority.MEDIUM
    timestamp: Optional[str] = None  # ISO string format
    session_id: Optional[str] = None  # UUID format
    message_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values after creation"""
        if self.timestamp is None:
            # Use ISO 8601 format with 'Z' suffix for UTC timezone
            self.timestamp = datetime.now().isoformat() + 'Z'
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization"""
        context_dict = {
            "task_id": self.context.task_id,
            "parent_session": self.context.parent_session,
            "files_modified": self.context.files_modified,
            "project_path": self.context.project_path
        }

        # Add optional context fields if they exist
        if self.context.file_paths is not None:
            context_dict["file_paths"] = self.context.file_paths
        if self.context.modifications is not None:
            context_dict["modifications"] = self.context.modifications
        if self.context.scope is not None:
            context_dict["scope"] = self.context.scope

        return {
            "message_type": self.message_type.value,
            "agent_type": self.agent_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "context": context_dict,
            "success_criteria": {
                "expected_outcomes": self.success_criteria.expected_outcomes,
                "validation_steps": self.success_criteria.validation_steps
            },
            "callback": {
                "handler": self.callback.handler,
                "timeout_ms": self.callback.timeout_ms
            },
            "message_id": self.message_id,
            "payload": self.payload
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary with validation"""
        try:
            # Use AgentMessageSchema for comprehensive validation
            # This eliminates duplication and ensures consistent validation
            validated_data = AgentMessageSchema.validate_message(data)

            # Extract enum values from validated data
            message_type = MessageType(validated_data["message_type"])
            agent_type = AgentType(validated_data["agent_type"])
            priority = Priority(validated_data.get("priority", "medium"))

            return cls(
                message_type=message_type,
                agent_type=agent_type,
                priority=priority,
                timestamp=validated_data.get("timestamp"),
                session_id=validated_data.get("session_id"),
                context=MessageContext(**validated_data["context"]),
                success_criteria=SuccessCriteria(**validated_data["success_criteria"]),
                callback=CallbackInfo(**validated_data["callback"]),
                message_id=validated_data.get("message_id"),
                payload=validated_data.get("payload")
            )
        except Exception as e:
            # Re-raise as ValueError for consistency
            raise ValueError(f"Message creation failed: {str(e)}") from e


class AgentMessageSchema:
    """
    JSON Schema validation for agent messages with security and business rule validation.

    Provides comprehensive validation including:
    - Field type and format validation
    - Security sanitization to prevent injection attacks
    - Business rule validation
    - Schema versioning support
    """

    # Pre-compiled regex patterns for performance
    UUID_PATTERN = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
    DATETIME_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*Z$')
    FILE_PATH_PATTERN = re.compile(r'^[^<>:"|?*]*$')
    # Comprehensive security pattern covering major attack vectors
    SECURITY_SANITIZE_PATTERN = re.compile(r'''
        [<>"\';\\]|                                    # Basic injection chars
        script|javascript|                             # XSS script tags
        DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|     # SQL injection
        UPDATE\s+SET|CREATE\s+TABLE|ALTER\s+TABLE|     # SQL DDL/DML
        UNION\s+SELECT|                                # SQL union attacks
        \$where|\$regex|\$ne|\$gt|\$lt|                # NoSQL injection
        eval\s*\(|Function\s*\(|                      # Code injection
        setTimeout\s*\(.*["\']|setInterval\s*\(.*["\']|# Timer injection
        on\w+\s*=|                                     # HTML event handlers (onclick=, onload=, etc.)
        expression\s*\(|                               # CSS expression injection
        \/\*|\*\/|--\s|#                               # Comment injection
    ''', re.IGNORECASE | re.VERBOSE)
    PATH_TRAVERSAL_PATTERN = re.compile(r'\.\.[\\\/]')
    INVALID_PATH_CHARS_PATTERN = re.compile(r'[<>:"|?*]')

    # JSON Schema definition
    SCHEMA_VERSION = "1.0"

    MESSAGE_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["message_type", "agent_type", "context", "success_criteria", "callback"],
        "properties": {
            "message_type": {
                "type": "string",
                "enum": [mt.value for mt in MessageType]
            },
            "agent_type": {
                "type": "string",
                "enum": [at.value for at in AgentType]
            },
            "priority": {
                "type": "string",
                "enum": [p.value for p in Priority],
                "default": "medium"
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "pattern": DATETIME_PATTERN.pattern
            },
            "session_id": {
                "type": "string",
                "pattern": UUID_PATTERN.pattern
            },
            "message_id": {
                "type": ["string", "null"],
                "anyOf": [
                    {"type": "null"},
                    {"type": "string", "pattern": UUID_PATTERN.pattern}
                ]
            },
            "context": {
                "type": "object",
                "required": ["task_id", "parent_session", "files_modified", "project_path"],
                "properties": {
                    "task_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 255
                    },
                    "parent_session": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 255
                    },
                    "files_modified": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": FILE_PATH_PATTERN.pattern  # Prevent invalid file path characters
                        },
                        "maxItems": 100
                    },
                    "project_path": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000
                    },
                    "file_paths": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "string",
                            "pattern": FILE_PATH_PATTERN.pattern
                        },
                        "maxItems": 100
                    },
                    "modifications": {
                        "type": ["object", "null"]
                    },
                    "scope": {
                        "type": ["object", "null"]
                    }
                },
                "additionalProperties": False
            },
            "success_criteria": {
                "type": "object",
                "properties": {
                    "expected_outcomes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 50
                    },
                    "validation_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 50
                    }
                },
                "additionalProperties": False
            },
            "callback": {
                "type": "object",
                "required": ["handler"],
                "properties": {
                    "handler": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 255
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": MAX_TIMEOUT_MS  # Max timeout from config
                    }
                },
                "additionalProperties": False
            },
            "payload": {
                "type": ["object", "null"]
            }
        },
        "additionalProperties": False
    }

    @classmethod
    def validate_message(cls, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize a message dictionary.

        Args:
            message_data: Raw message dictionary

        Returns:
            Validated and sanitized message dictionary

        Raises:
            ValueError: If validation fails
        """
        try:
            # Check if jsonschema is available (imported at module level)
            if jsonschema is None:
                raise ImportError("jsonschema library is required for validation. Install with: pip install jsonschema")

            # Basic schema validation
            jsonschema.validate(message_data, cls.MESSAGE_SCHEMA)

            # Additional security validation
            sanitized_data = cls._sanitize_message(message_data)

            # Business rule validation
            cls._validate_business_rules(sanitized_data)

            return sanitized_data

        except jsonschema.ValidationError as e:
            raise ValueError(f"Schema validation failed: {e.message}")
        except Exception as e:
            raise ValueError(f"Message validation failed: {str(e)}")

    @classmethod
    def _sanitize_message(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize message data to prevent injection attacks"""
        import copy
        sanitized = copy.deepcopy(data)  # Deep copy to prevent nested mutations

        # Sanitize string fields to prevent injection - FIXED: Use correct nested paths
        string_fields = ["context.task_id", "context.parent_session", "context.project_path", "callback.handler"]

        for field_path in string_fields:
            value = cls._get_nested_value(sanitized, field_path)
            if value and isinstance(value, str):
                # Remove potentially dangerous characters using pre-compiled pattern
                sanitized_value = cls.SECURITY_SANITIZE_PATTERN.sub('', value)
                cls._set_nested_value(sanitized, field_path, sanitized_value)

        # Sanitize file paths
        for files_field in ["files_modified", "file_paths"]:
            files = sanitized.get("context", {}).get(files_field, [])
            if files:
                sanitized_files = []
                for file_path in files:
                    if isinstance(file_path, str):
                        # Remove path traversal attempts and dangerous characters
                        clean_path = cls.PATH_TRAVERSAL_PATTERN.sub('', file_path)
                        clean_path = cls.INVALID_PATH_CHARS_PATTERN.sub('', clean_path)
                        sanitized_files.append(clean_path)
                sanitized["context"][files_field] = sanitized_files

        return sanitized

    @classmethod
    def _validate_business_rules(cls, data: Dict[str, Any]) -> None:
        """Validate business-specific rules"""

        # Validate message type compatibility with agent type
        msg_type = data.get("message_type")
        agent_type = data.get("agent_type")

        if msg_type == "REVIEW_TRIGGERED" and agent_type != "review":
            raise ValueError("REVIEW_TRIGGERED messages must use 'review' agent_type")

        if msg_type == "FIX_REQUIRED" and agent_type != "fix":
            raise ValueError("FIX_REQUIRED messages must use 'fix' agent_type")

        # Validate timestamp format if provided
        timestamp = data.get("timestamp")
        if timestamp:
            try:
                # Handle both 'Z' suffix and explicit timezone formats
                if timestamp.endswith('Z'):
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    datetime.fromisoformat(timestamp)
            except ValueError:
                raise ValueError("Invalid timestamp format. Must be ISO 8601 format with 'Z' suffix or explicit timezone.")

        # Validate session_id format if provided using pre-compiled pattern
        session_id = data.get("session_id")
        if session_id:
            if not cls.UUID_PATTERN.match(session_id):
                raise ValueError("Invalid session_id format. Must be a valid UUID.")

        # Validate priority-based requirements
        priority = data.get("priority", "medium")
        if priority in ["critical", "high"]:
            context = data.get("context", {})
            if not context.get("files_modified"):
                raise ValueError("Critical/high priority messages must specify files_modified")

    @classmethod
    def _get_nested_value(cls, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    @classmethod
    def _set_nested_value(cls, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value in dictionary using dot notation"""
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    @classmethod
    def serialize_message(cls, message: AgentMessage) -> str:
        """
        Serialize an AgentMessage to JSON with validation.

        Args:
            message: AgentMessage instance

        Returns:
            JSON string representation

        Raises:
            ValueError: If serialization fails
        """
        try:
            message_dict = message.to_dict()
            validated_dict = cls.validate_message(message_dict)
            return json.dumps(validated_dict, indent=2)
        except Exception as e:
            raise ValueError(f"Message serialization failed: {str(e)}")

    @classmethod
    def deserialize_message(cls, json_str: str) -> AgentMessage:
        """
        Deserialize JSON string to AgentMessage with validation.

        Args:
            json_str: JSON string representation

        Returns:
            AgentMessage instance

        Raises:
            ValueError: If deserialization fails
        """
        try:
            message_dict = json.loads(json_str)
            validated_dict = cls.validate_message(message_dict)
            return AgentMessage.from_dict(validated_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Message deserialization failed: {str(e)}")

    @classmethod
    def get_schema_version(cls) -> str:
        """Get the current schema version"""
        return cls.SCHEMA_VERSION
