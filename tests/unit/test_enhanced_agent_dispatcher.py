"""
Unit tests for Enhanced Agent Dispatcher
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest

from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, Priority,
    MessageContext, SuccessCriteria, CallbackInfo
)
from cadence.fix_iteration_tracker import EscalationStrategy, PersistenceType


class TestDispatchConfig:
    """Test cases for DispatchConfig dataclass"""

    def test_default_initialization(self):
        """Test default DispatchConfig initialization"""
        config = DispatchConfig()

        assert config.max_concurrent_agents == 2
        assert config.default_timeout_ms == 600000
        assert config.enable_fix_tracking is True
        assert config.enable_escalation is True
        assert config.max_fix_iterations == 3
        assert config.escalation_strategy == "log_only"
        assert config.persistence_type == "memory"
        assert config.storage_path is None

    def test_custom_initialization(self):
        """Test DispatchConfig with custom values"""
        config = DispatchConfig(
            max_concurrent_agents=5,
            default_timeout_ms=300000,
            enable_fix_tracking=False,
            enable_escalation=False,
            max_fix_iterations=10,
            escalation_strategy="notify_supervisor",
            persistence_type="file",
            storage_path="/tmp/test.json"
        )

        assert config.max_concurrent_agents == 5
        assert config.default_timeout_ms == 300000
        assert config.enable_fix_tracking is False
        assert config.enable_escalation is False
        assert config.max_fix_iterations == 10
        assert config.escalation_strategy == "notify_supervisor"
        assert config.persistence_type == "file"
        assert config.storage_path == "/tmp/test.json"

    def test_to_dict_method(self):
        """Test DispatchConfig to_dict conversion"""
        config = DispatchConfig(
            max_concurrent_agents=3,
            default_timeout_ms=120000,
            escalation_strategy="escalate_immediately"
        )

        config_dict = config.to_dict()

        expected_keys = {
            "max_concurrent_agents", "default_timeout_ms", "enable_fix_tracking",
            "enable_escalation", "max_fix_iterations", "escalation_strategy",
            "persistence_type", "storage_path"
        }

        assert set(config_dict.keys()) == expected_keys
        assert config_dict["max_concurrent_agents"] == 3
        assert config_dict["default_timeout_ms"] == 120000
        assert config_dict["escalation_strategy"] == "escalate_immediately"
        assert config_dict["enable_fix_tracking"] is True  # default
        assert config_dict["storage_path"] is None  # default

    def test_to_dict_with_all_none_values(self):
        """Test to_dict includes None values"""
        config = DispatchConfig(storage_path=None)
        config_dict = config.to_dict()

        # None values should be included in the dictionary
        assert "storage_path" in config_dict
        assert config_dict["storage_path"] is None


class TestEnhancedAgentDispatcher:
    """Test cases for EnhancedAgentDispatcher"""

    def test_basic_initialization(self):
        """Test basic dispatcher initialization"""
        dispatcher = EnhancedAgentDispatcher(
            max_fix_iterations=3,
            escalation_strategy=EscalationStrategy.LOG_ONLY
        )

        assert dispatcher.max_fix_iterations == 3
        assert dispatcher.fix_manager is not None
        assert dispatcher.fix_manager.limit_enforcer.max_fix_iterations == 3

    def test_config_based_initialization(self):
        """Test initialization from configuration"""
        config = {
            'max_fix_iterations': 5,
            'escalation_strategy': 'notify_supervisor',
            'persistence_type': 'file',
            'storage_path': '/tmp/test_attempts.json'
        }

        dispatcher = EnhancedAgentDispatcher.from_config(config)

        assert dispatcher.max_fix_iterations == 5
        assert dispatcher.fix_manager.escalation_handler.escalation_strategy == EscalationStrategy.NOTIFY_SUPERVISOR

    def test_dispatch_fix_agent_success(self):
        """Test successful fix agent dispatch"""
        callback = Mock()
        dispatcher = EnhancedAgentDispatcher(max_fix_iterations=3)

        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        message_id = dispatcher.dispatch_fix_agent(
            task_id="test-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        assert message_id is not None
        assert dispatcher.get_task_fix_status("test-task")["attempt_count"] == 1

    def test_dispatch_fix_agent_blocked_by_limit(self):
        """Test fix agent dispatch blocked by attempt limit"""
        callback = Mock()
        dispatcher = EnhancedAgentDispatcher(max_fix_iterations=2)

        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        # Make maximum attempts
        for i in range(2):
            message_id = dispatcher.dispatch_fix_agent(
                task_id="test-task",
                context=context,
                success_criteria=SuccessCriteria(),
                callback_handler=callback
            )
            assert message_id is not None

            # Simulate failed response
            response = AgentMessage(
                message_type=MessageType.ERROR,
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=SuccessCriteria(),
                callback=CallbackInfo(handler="test"),
                payload={'error': f'Error {i+1}'}
            )

            # Get the enhanced callback and call it
            with dispatcher.lock:
                enhanced_callback = dispatcher.callbacks[message_id]
            enhanced_callback(response)

        # Third attempt should be blocked
        message_id = dispatcher.dispatch_fix_agent(
            task_id="test-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        assert message_id is None
        status = dispatcher.get_task_fix_status("test-task")
        assert status["is_escalated"] is True

    def test_dispatch_agent_with_tracking(self):
        """Test dispatching non-fix agents with tracking"""
        callback = Mock()
        dispatcher = EnhancedAgentDispatcher()

        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        message_id = dispatcher.dispatch_agent_with_tracking(
            agent_type=AgentType.REVIEW,
            task_id="test-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback,
            priority=Priority.HIGH
        )

        assert message_id is not None

        # Check message has tracking metadata
        with dispatcher.lock:
            message = dispatcher.pending_messages[message_id]
            assert message.payload is not None
            assert 'attempt_count' in message.payload
            assert 'max_attempts' in message.payload

    def test_fix_response_handling_success(self):
        """Test handling of successful fix response"""
        callback = Mock()
        dispatcher = EnhancedAgentDispatcher(max_fix_iterations=3)

        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        # Dispatch fix agent
        message_id = dispatcher.dispatch_fix_agent(
            task_id="test-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        # Simulate successful response
        response = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="test-task",
                parent_session="session-1",
                files_modified=["test.py", "fixed.py"],
                project_path="/test"
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="test")
        )

        # Get the enhanced callback and call it
        with dispatcher.lock:
            enhanced_callback = dispatcher.callbacks[message_id]
        enhanced_callback(response)

        # Check that original callback was called
        callback.assert_called_once_with(response)

        # Check that attempts were reset on success
        status = dispatcher.get_task_fix_status("test-task")
        assert status["attempt_count"] == 0
        assert not status["is_escalated"]

    def test_fix_response_handling_failure(self):
        """Test handling of failed fix response"""
        callback = Mock()
        dispatcher = EnhancedAgentDispatcher(max_fix_iterations=3)

        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        # Dispatch fix agent
        message_id = dispatcher.dispatch_fix_agent(
            task_id="test-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=callback
        )

        # Simulate error response
        response = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="test"),
            payload={'error': 'Fix failed due to syntax error'}
        )

        # Get the enhanced callback and call it
        with dispatcher.lock:
            enhanced_callback = dispatcher.callbacks[message_id]
        enhanced_callback(response)

        # Check that original callback was called
        callback.assert_called_once_with(response)

        # Check that attempt was recorded
        status = dispatcher.get_task_fix_status("test-task")
        assert status["attempt_count"] == 1
        assert not status["is_escalated"]  # Not escalated yet
        assert len(status["recent_attempts"]) == 1
        assert status["recent_attempts"][0]["error_message"] == "Fix failed due to syntax error"

    def test_escalation_workflow(self):
        """Test complete escalation workflow"""
        supervisor_callback = Mock()
        dispatcher = EnhancedAgentDispatcher(
            max_fix_iterations=2,
            escalation_strategy=EscalationStrategy.NOTIFY_SUPERVISOR,
            supervisor_callback=supervisor_callback
        )

        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        # Make failed attempts up to limit
        for i in range(2):
            message_id = dispatcher.dispatch_fix_agent(
                task_id="test-task",
                context=context,
                success_criteria=SuccessCriteria(),
                callback_handler=Mock()
            )

            # Simulate failed response
            response = AgentMessage(
                message_type=MessageType.ERROR,
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=SuccessCriteria(),
                callback=CallbackInfo(handler="test"),
                payload={'error': f'Error {i+1}'}
            )

            with dispatcher.lock:
                enhanced_callback = dispatcher.callbacks[message_id]
            enhanced_callback(response)

        # Check escalation occurred
        status = dispatcher.get_task_fix_status("test-task")
        assert status["is_escalated"] is True
        assert status["attempt_count"] == 2

        # Check supervisor was notified
        supervisor_callback.assert_called_once()
        call_args = supervisor_callback.call_args[0]
        assert call_args[0] == "test-task"
        assert "attempts" in call_args[1]
        assert call_args[2] == 2

    def test_escalation_message_creation(self):
        """Test creation of escalation messages"""
        dispatcher = EnhancedAgentDispatcher()

        escalation_msg = dispatcher.create_escalation_message(
            task_id="test-task",
            reason="Maximum attempts exceeded",
            attempt_count=3
        )

        assert escalation_msg.message_type == MessageType.ESCALATION_REQUIRED
        assert escalation_msg.agent_type == AgentType.FIX
        assert escalation_msg.priority == Priority.HIGH
        assert escalation_msg.context.task_id == "test-task"
        assert escalation_msg.payload['escalation_reason'] == "Maximum attempts exceeded"
        assert escalation_msg.payload['attempt_count'] == 3
        assert escalation_msg.payload['requires_manual_review'] is True

    def test_handle_escalation_message(self):
        """Test handling of escalation messages"""
        dispatcher = EnhancedAgentDispatcher()

        escalation_msg = AgentMessage(
            message_type=MessageType.ESCALATION_REQUIRED,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="test-task",
                parent_session="external",
                files_modified=[],
                project_path=""
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="escalation"),
            payload={
                'escalation_reason': 'External escalation',
                'requires_manual_review': True
            }
        )

        dispatcher.handle_escalation_message(escalation_msg)

        # Check task is marked as escalated
        status = dispatcher.get_task_fix_status("test-task")
        assert status["is_escalated"] is True
        assert status["escalation_reason"] == "External escalation"

    def test_reset_task_fix_attempts(self):
        """Test resetting task fix attempts"""
        dispatcher = EnhancedAgentDispatcher()

        # Make some attempts
        context = MessageContext(
            task_id="test-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        message_id = dispatcher.dispatch_fix_agent(
            task_id="test-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=Mock()
        )

        # Simulate failed response
        response = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="test"),
            payload={'error': 'Test error'}
        )

        with dispatcher.lock:
            enhanced_callback = dispatcher.callbacks[message_id]
        enhanced_callback(response)

        # Check attempt was recorded
        status = dispatcher.get_task_fix_status("test-task")
        assert status["attempt_count"] == 1

        # Reset attempts
        dispatcher.reset_task_fix_attempts("test-task")

        # Check attempts were reset
        status = dispatcher.get_task_fix_status("test-task")
        assert status["attempt_count"] == 0
        assert not status["is_escalated"]

    def test_automation_pause_and_resume(self):
        """Test automation pause and resume functionality"""
        dispatcher = EnhancedAgentDispatcher(
            escalation_strategy=EscalationStrategy.PAUSE_AUTOMATION
        )

        # Initially not paused
        assert not dispatcher.is_automation_paused()

        # Trigger escalation that pauses automation
        dispatcher.fix_manager.escalation_handler.handle_escalation(
            "test-task", "Test escalation", 3
        )

        # Should be paused
        assert dispatcher.is_automation_paused()

        # Resume automation
        dispatcher.resume_automation()

        # Should not be paused
        assert not dispatcher.is_automation_paused()

    def test_get_all_escalated_tasks(self):
        """Test getting all escalated tasks"""
        dispatcher = EnhancedAgentDispatcher()

        # No escalated tasks initially
        assert len(dispatcher.get_all_escalated_tasks()) == 0

        # Escalate some tasks
        dispatcher.fix_manager.iteration_tracker.mark_task_escalated("task1", "Reason 1")
        dispatcher.fix_manager.iteration_tracker.mark_task_escalated("task2", "Reason 2")

        escalated = dispatcher.get_all_escalated_tasks()
        assert len(escalated) == 2
        assert "task1" in escalated
        assert "task2" in escalated

    def test_persistence_integration(self):
        """Test persistence integration with enhanced dispatcher"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "dispatcher_test.json"

            # Create dispatcher with file persistence
            dispatcher1 = EnhancedAgentDispatcher(
                max_fix_iterations=2,
                persistence_type=PersistenceType.FILE,
                storage_path=str(storage_path)
            )

            context = MessageContext(
                task_id="persistent-task",
                parent_session="session-1",
                files_modified=["test.py"],
                project_path="/test"
            )

            # Make an attempt
            message_id = dispatcher1.dispatch_fix_agent(
                task_id="persistent-task",
                context=context,
                success_criteria=SuccessCriteria(),
                callback_handler=Mock()
            )

            # Simulate failed response
            response = AgentMessage(
                message_type=MessageType.ERROR,
                agent_type=AgentType.FIX,
                context=context,
                success_criteria=SuccessCriteria(),
                callback=CallbackInfo(handler="test"),
                payload={'error': 'Persistent error'}
            )

            with dispatcher1.lock:
                enhanced_callback = dispatcher1.callbacks[message_id]
            enhanced_callback(response)

            # Create new dispatcher with same storage
            dispatcher2 = EnhancedAgentDispatcher(
                max_fix_iterations=2,
                persistence_type=PersistenceType.FILE,
                storage_path=str(storage_path)
            )

            # Should load previous state
            status = dispatcher2.get_task_fix_status("persistent-task")
            assert status["attempt_count"] == 1
            assert len(status["recent_attempts"]) == 1
            assert status["recent_attempts"][0]["error_message"] == "Persistent error"

    def test_cleanup(self):
        """Test cleanup functionality"""
        dispatcher = EnhancedAgentDispatcher()

        # Add some state
        context = MessageContext(
            task_id="cleanup-task",
            parent_session="session-1",
            files_modified=["test.py"],
            project_path="/test"
        )

        dispatcher.dispatch_fix_agent(
            task_id="cleanup-task",
            context=context,
            success_criteria=SuccessCriteria(),
            callback_handler=Mock()
        )

        # Should have pending messages and timers
        assert len(dispatcher.pending_messages) > 0
        assert len(dispatcher.timers) > 0

        # Cleanup
        dispatcher.cleanup()

        # Should be cleaned up
        assert len(dispatcher.pending_messages) == 0
        assert len(dispatcher.timers) == 0

    def test_invalid_escalation_message_handling(self):
        """Test handling of invalid escalation messages"""
        dispatcher = EnhancedAgentDispatcher()

        # Create invalid escalation message (wrong type)
        invalid_msg = AgentMessage(
            message_type=MessageType.AGENT_RESPONSE,  # Wrong type
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="test-task",
                parent_session="session",
                files_modified=[],
                project_path=""
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="test")
        )

        with patch('cadence.enhanced_agent_dispatcher.logger') as mock_logger:
            dispatcher.handle_escalation_message(invalid_msg)

            # Should log warning about invalid message type
            mock_logger.warning.assert_called()
            args = mock_logger.warning.call_args[0]
            assert "Invalid escalation message type" in args[0]


if __name__ == "__main__":
    pytest.main([__file__])
