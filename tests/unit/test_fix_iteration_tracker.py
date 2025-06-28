"""
Comprehensive unit tests for Fix Iteration Tracker system
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest

from cadence.fix_iteration_tracker import (
    FixIterationTracker, FixAttemptLimitEnforcer, EscalationHandler,
    FixIterationManager, FixAttempt, TaskFixHistory,
    EscalationStrategy, PersistenceType
)
from cadence.agent_messages import AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo


class TestFixAttempt:
    """Test FixAttempt data class"""

    def test_create_fix_attempt(self):
        """Test creating a fix attempt"""
        attempt = FixAttempt(
            attempt_number=1,
            timestamp="2025-06-27T21:00:00Z",
            agent_id="agent-123",
            success=False,
            files_modified=["file1.py", "file2.py"]
        )

        assert attempt.attempt_number == 1
        assert attempt.timestamp == "2025-06-27T21:00:00Z"
        assert attempt.agent_id == "agent-123"
        assert attempt.success is False
        assert attempt.files_modified == ["file1.py", "file2.py"]

    def test_fix_attempt_serialization(self):
        """Test fix attempt serialization/deserialization"""
        attempt = FixAttempt(
            attempt_number=2,
            timestamp="2025-06-27T21:00:00Z",
            error_message="Test error",
            duration_seconds=30.5
        )

        # Serialize
        data = attempt.to_dict()
        assert data["attempt_number"] == 2
        assert data["error_message"] == "Test error"
        assert data["duration_seconds"] == 30.5

        # Deserialize
        restored = FixAttempt.from_dict(data)
        assert restored.attempt_number == attempt.attempt_number
        assert restored.error_message == attempt.error_message
        assert restored.duration_seconds == attempt.duration_seconds


class TestTaskFixHistory:
    """Test TaskFixHistory data class"""

    def test_create_task_history(self):
        """Test creating task history"""
        history = TaskFixHistory(task_id="test-task")

        assert history.task_id == "test-task"
        assert history.current_attempt_count == 0
        assert history.is_escalated is False
        assert len(history.attempts) == 0

    def test_add_attempt(self):
        """Test adding attempts to history"""
        history = TaskFixHistory(task_id="test-task")

        attempt1 = FixAttempt(1, "2025-06-27T21:00:00Z")
        attempt2 = FixAttempt(2, "2025-06-27T21:01:00Z")

        history.add_attempt(attempt1)
        assert history.current_attempt_count == 1
        assert len(history.attempts) == 1

        history.add_attempt(attempt2)
        assert history.current_attempt_count == 2
        assert len(history.attempts) == 2

    def test_mark_escalated(self):
        """Test marking history as escalated"""
        history = TaskFixHistory(task_id="test-task")

        history.mark_escalated("Too many attempts")

        assert history.is_escalated is True
        assert history.escalation_reason == "Too many attempts"
        assert history.escalation_timestamp is not None

    def test_reset_attempts(self):
        """Test resetting attempts"""
        history = TaskFixHistory(task_id="test-task")

        # Add attempts and escalate
        history.add_attempt(FixAttempt(1, "2025-06-27T21:00:00Z"))
        history.add_attempt(FixAttempt(2, "2025-06-27T21:01:00Z"))
        history.mark_escalated("Test escalation")

        # Reset
        history.reset_attempts()

        assert history.current_attempt_count == 0
        assert history.is_escalated is False
        assert history.escalation_reason is None
        assert len(history.attempts) == 0

    def test_history_serialization(self):
        """Test history serialization/deserialization"""
        history = TaskFixHistory(task_id="test-task")
        history.add_attempt(FixAttempt(1, "2025-06-27T21:00:00Z", success=True))
        history.mark_escalated("Test reason")

        # Serialize
        data = history.to_dict()
        assert data["task_id"] == "test-task"
        assert data["is_escalated"] is True
        assert len(data["attempts"]) == 1

        # Deserialize
        restored = TaskFixHistory.from_dict(data)
        assert restored.task_id == history.task_id
        assert restored.is_escalated == history.is_escalated
        assert len(restored.attempts) == len(history.attempts)


class TestFixIterationTracker:
    """Test FixIterationTracker class"""

    def test_memory_persistence(self):
        """Test memory-based persistence"""
        tracker = FixIterationTracker(persistence_type=PersistenceType.MEMORY)

        # Start attempts
        attempt1 = tracker.start_fix_attempt("task1", "agent1")
        attempt2 = tracker.start_fix_attempt("task1", "agent1")

        assert attempt1 == 1
        assert attempt2 == 2
        assert tracker.get_attempt_count("task1") == 2

    def test_file_persistence(self):
        """Test file-based persistence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test_attempts.json"

            # Create tracker and add attempts
            tracker1 = FixIterationTracker(
                persistence_type=PersistenceType.FILE,
                storage_path=str(storage_path)
            )

            tracker1.start_fix_attempt("task1", "agent1")
            tracker1.complete_fix_attempt("task1", False, "Test error")
            tracker1.start_fix_attempt("task1", "agent2")

            # Create new tracker with same storage
            tracker2 = FixIterationTracker(
                persistence_type=PersistenceType.FILE,
                storage_path=str(storage_path)
            )

            # Should load existing data
            assert tracker2.get_attempt_count("task1") == 2
            history = tracker2.get_task_history("task1")
            assert len(history.attempts) == 2
            assert history.attempts[0].error_message == "Test error"

    def test_complete_fix_attempt(self):
        """Test completing fix attempts"""
        tracker = FixIterationTracker()

        tracker.start_fix_attempt("task1", "agent1")
        tracker.complete_fix_attempt(
            "task1",
            success=False,
            error_message="Test error",
            duration_seconds=30.5,
            files_modified=["file1.py"]
        )

        history = tracker.get_task_history("task1")
        attempt = history.attempts[0]

        assert attempt.success is False
        assert attempt.error_message == "Test error"
        assert attempt.duration_seconds == 30.5
        assert attempt.files_modified == ["file1.py"]

    def test_reset_task_attempts(self):
        """Test resetting task attempts"""
        tracker = FixIterationTracker()

        tracker.start_fix_attempt("task1")
        tracker.start_fix_attempt("task1")
        assert tracker.get_attempt_count("task1") == 2

        tracker.reset_task_attempts("task1")
        assert tracker.get_attempt_count("task1") == 0
        assert not tracker.is_task_escalated("task1")

    def test_escalation_tracking(self):
        """Test escalation tracking"""
        tracker = FixIterationTracker()

        assert not tracker.is_task_escalated("task1")

        tracker.mark_task_escalated("task1", "Too many attempts")

        assert tracker.is_task_escalated("task1")
        assert "task1" in tracker.get_all_escalated_tasks()

    def test_thread_safety(self):
        """Test thread safety of tracker operations"""
        tracker = FixIterationTracker()
        results = []
        errors = []

        def worker(task_id, worker_id):
            try:
                for i in range(5):
                    attempt_num = tracker.start_fix_attempt(f"{task_id}", f"worker-{worker_id}")
                    results.append((worker_id, attempt_num))
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=("task1", i))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and correct attempt count
        assert len(errors) == 0
        assert tracker.get_attempt_count("task1") == 15  # 3 workers * 5 attempts


class TestFixAttemptLimitEnforcer:
    """Test FixAttemptLimitEnforcer class"""

    def test_basic_limit_enforcement(self):
        """Test basic limit enforcement"""
        tracker = FixIterationTracker()
        enforcer = FixAttemptLimitEnforcer(tracker, max_fix_iterations=2)

        # First two attempts should be allowed
        assert enforcer.can_attempt_fix("task1") is True
        tracker.start_fix_attempt("task1")

        assert enforcer.can_attempt_fix("task1") is True
        tracker.start_fix_attempt("task1")

        # Third attempt should be blocked
        assert enforcer.can_attempt_fix("task1") is False

    def test_escalation_blocking(self):
        """Test that escalated tasks are blocked"""
        tracker = FixIterationTracker()
        enforcer = FixAttemptLimitEnforcer(tracker, max_fix_iterations=5)

        # Mark task as escalated
        tracker.mark_task_escalated("task1", "Test escalation")

        # Should be blocked even under limit
        assert enforcer.can_attempt_fix("task1") is False

    def test_validate_fix_attempt_with_escalation(self):
        """Test validation with escalation handler"""
        tracker = FixIterationTracker()
        escalation_handler = Mock()
        enforcer = FixAttemptLimitEnforcer(
            tracker,
            max_fix_iterations=2,
            escalation_handler=escalation_handler
        )

        # Fill up attempts
        tracker.start_fix_attempt("task1")
        tracker.start_fix_attempt("task1")

        # This should trigger escalation
        result = enforcer.validate_fix_attempt("task1")

        assert result is False
        assert tracker.is_task_escalated("task1")
        escalation_handler.handle_escalation.assert_called_once()

    def test_check_and_escalate_after_failure(self):
        """Test escalation check after failed attempt"""
        tracker = FixIterationTracker()
        escalation_handler = Mock()
        enforcer = FixAttemptLimitEnforcer(
            tracker,
            max_fix_iterations=2,
            escalation_handler=escalation_handler
        )

        # Add attempts up to limit
        tracker.start_fix_attempt("task1")
        tracker.complete_fix_attempt("task1", success=False, error_message="Error 1")

        tracker.start_fix_attempt("task1")
        tracker.complete_fix_attempt("task1", success=False, error_message="Error 2")

        # Check for escalation
        escalated = enforcer.check_and_escalate_if_needed("task1")

        assert escalated is True
        assert tracker.is_task_escalated("task1")
        escalation_handler.handle_escalation.assert_called_once()


class TestEscalationHandler:
    """Test EscalationHandler class"""

    def test_log_only_strategy(self):
        """Test log-only escalation strategy"""
        handler = EscalationHandler(escalation_strategy=EscalationStrategy.LOG_ONLY)

        with patch('cadence.fix_iteration_tracker.logger') as mock_logger:
            handler.handle_escalation("task1", "Test reason", 3)

            # Should log the escalation
            mock_logger.error.assert_called()
            args = mock_logger.error.call_args[0]
            assert "task1" in args[0]
            assert "3 attempts" in args[0]

    def test_notify_supervisor_strategy(self):
        """Test supervisor notification strategy"""
        supervisor_callback = Mock()
        handler = EscalationHandler(
            escalation_strategy=EscalationStrategy.NOTIFY_SUPERVISOR,
            supervisor_callback=supervisor_callback
        )

        handler.handle_escalation("task1", "Test reason", 3)

        supervisor_callback.assert_called_once_with("task1", "Test reason", 3)

    def test_pause_automation_strategy(self):
        """Test automation pause strategy"""
        notification_callback = Mock()
        handler = EscalationHandler(
            escalation_strategy=EscalationStrategy.PAUSE_AUTOMATION,
            notification_callback=notification_callback
        )

        handler.handle_escalation("task1", "Test reason", 3)

        assert handler.is_automation_paused() is True
        notification_callback.assert_called_once()
        args = notification_callback.call_args[0]
        assert args[0] == "AUTOMATION_PAUSED"

    def test_manual_review_strategy(self):
        """Test manual review marking strategy"""
        notification_callback = Mock()
        handler = EscalationHandler(
            escalation_strategy=EscalationStrategy.MARK_FOR_MANUAL_REVIEW,
            notification_callback=notification_callback
        )

        handler.handle_escalation("task1", "Test reason", 3)

        notification_callback.assert_called_once()
        args = notification_callback.call_args[0]
        assert args[0] == "MANUAL_REVIEW_REQUIRED"

        # Parse the JSON message
        message_data = json.loads(args[1])
        assert message_data["message_type"] == "ESCALATION_REQUIRED"
        assert message_data["task_id"] == "task1"
        assert message_data["requires_manual_review"] is True

    def test_escalated_tasks_tracking(self):
        """Test tracking of escalated tasks"""
        handler = EscalationHandler()

        handler.handle_escalation("task1", "Reason 1", 3)
        handler.handle_escalation("task2", "Reason 2", 2)

        escalated = handler.get_escalated_tasks()
        assert len(escalated) == 2

        task_ids = [task["task_id"] for task in escalated]
        assert "task1" in task_ids
        assert "task2" in task_ids

    def test_clear_escalation(self):
        """Test clearing escalation"""
        handler = EscalationHandler()

        handler.handle_escalation("task1", "Test reason", 3)
        assert len(handler.get_escalated_tasks()) == 1

        cleared = handler.clear_escalation("task1")
        assert cleared is True
        assert len(handler.get_escalated_tasks()) == 0

        # Try to clear non-existent escalation
        cleared = handler.clear_escalation("task1")
        assert cleared is False

    def test_resume_automation(self):
        """Test resuming automation"""
        handler = EscalationHandler(escalation_strategy=EscalationStrategy.PAUSE_AUTOMATION)

        handler.handle_escalation("task1", "Test reason", 3)
        assert handler.is_automation_paused() is True

        handler.resume_automation()
        assert handler.is_automation_paused() is False


class TestFixIterationManager:
    """Test FixIterationManager integration"""

    def test_manager_initialization(self):
        """Test manager initialization with all components"""
        manager = FixIterationManager(
            max_fix_iterations=3,
            escalation_strategy=EscalationStrategy.LOG_ONLY,
            persistence_type=PersistenceType.MEMORY
        )

        assert manager.iteration_tracker is not None
        assert manager.escalation_handler is not None
        assert manager.limit_enforcer is not None

    def test_complete_fix_workflow(self):
        """Test complete fix workflow through manager"""
        supervisor_callback = Mock()
        manager = FixIterationManager(
            max_fix_iterations=2,
            escalation_strategy=EscalationStrategy.NOTIFY_SUPERVISOR,
            supervisor_callback=supervisor_callback
        )

        # Attempt 1 - should succeed
        attempt1 = manager.start_fix_attempt("task1", "agent1")
        assert attempt1 == 1

        # Complete with failure
        not_escalated = manager.complete_fix_attempt("task1", success=False, error_message="Error 1")
        assert not_escalated is True

        # Attempt 2 - should succeed
        attempt2 = manager.start_fix_attempt("task1", "agent1")
        assert attempt2 == 2

        # Complete with failure - should trigger escalation
        not_escalated = manager.complete_fix_attempt("task1", success=False, error_message="Error 2")
        assert not_escalated is False

        # Should not allow further attempts
        attempt3 = manager.start_fix_attempt("task1", "agent1")
        assert attempt3 is None

        # Supervisor should be notified
        supervisor_callback.assert_called_once()

    def test_successful_fix_resets_attempts(self):
        """Test that successful fix resets attempt count"""
        manager = FixIterationManager(max_fix_iterations=3)

        # Make some failed attempts
        manager.start_fix_attempt("task1")
        manager.complete_fix_attempt("task1", success=False, error_message="Error 1")

        manager.start_fix_attempt("task1")
        manager.complete_fix_attempt("task1", success=True)  # Success!

        # Should reset attempts
        status = manager.get_task_status("task1")
        assert status["attempt_count"] == 0
        assert not status["is_escalated"]

        # Should allow new attempts
        assert manager.can_attempt_fix("task1") is True

    def test_enhance_dispatch_message(self):
        """Test enhancing dispatch messages with attempt metadata"""
        manager = FixIterationManager(max_fix_iterations=3)

        # Make some attempts
        manager.start_fix_attempt("task1")
        manager.complete_fix_attempt("task1", success=False)

        # Create test message
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="task1",
                parent_session="session1",
                files_modified=["test.py"],
                project_path="/test"
            ),
            success_criteria=SuccessCriteria(),
            callback=CallbackInfo(handler="test_handler")
        )

        # Enhance message
        enhanced = manager.enhance_dispatch_message(message, "task1")

        assert enhanced.payload is not None
        assert enhanced.payload["attempt_count"] == 1
        assert enhanced.payload["is_escalated"] is False
        assert enhanced.payload["max_attempts"] == 3

    def test_get_task_status(self):
        """Test getting comprehensive task status"""
        manager = FixIterationManager(max_fix_iterations=2)

        # Make attempts
        manager.start_fix_attempt("task1", "agent1")
        manager.complete_fix_attempt("task1", success=False, error_message="Error 1", duration_seconds=10.5)

        manager.start_fix_attempt("task1", "agent2")
        manager.complete_fix_attempt("task1", success=False, error_message="Error 2", duration_seconds=15.2)

        status = manager.get_task_status("task1")

        assert status["task_id"] == "task1"
        assert status["attempt_count"] == 2
        assert status["max_attempts"] == 2
        assert status["is_escalated"] is True
        assert status["can_attempt_fix"] is False
        assert len(status["recent_attempts"]) == 2
        assert status["recent_attempts"][0]["error_message"] == "Error 1"
        assert status["recent_attempts"][1]["error_message"] == "Error 2"

    def test_concurrent_fix_attempts_different_tasks(self):
        """Test concurrent fix attempts for different tasks"""
        manager = FixIterationManager(max_fix_iterations=3)
        results = []
        errors = []

        def worker(task_id, attempts):
            try:
                for i in range(attempts):
                    attempt_num = manager.start_fix_attempt(task_id, f"worker-{task_id}")
                    if attempt_num:
                        manager.complete_fix_attempt(task_id, success=False, error_message=f"Error {i}")
                        results.append((task_id, attempt_num))
            except Exception as e:
                errors.append(e)

        # Start workers for different tasks
        threads = [
            threading.Thread(target=worker, args=("task1", 2)),
            threading.Thread(target=worker, args=("task2", 3)),
            threading.Thread(target=worker, args=("task3", 1))
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0
        assert len(results) == 6  # 2 + 3 + 1 attempts

        # Check individual task states
        assert manager.get_task_status("task1")["attempt_count"] == 2
        assert manager.get_task_status("task2")["attempt_count"] == 3
        assert manager.get_task_status("task3")["attempt_count"] == 1

    def test_malformed_task_id_handling(self):
        """Test handling of malformed task IDs"""
        manager = FixIterationManager()

        # Test with various malformed IDs
        test_ids = ["", None, "   ", "task/with/slashes", "task\nwith\nnewlines"]

        for task_id in test_ids:
            if task_id is not None:
                # Should not crash, should handle gracefully
                status = manager.get_task_status(task_id)
                assert status["task_id"] == task_id
                assert status["attempt_count"] == 0

    def test_recovery_scenarios(self):
        """Test recovery scenarios after system restart"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "recovery_test.json"

            # Create manager and make attempts
            manager1 = FixIterationManager(
                max_fix_iterations=2,
                persistence_type=PersistenceType.FILE,
                storage_path=str(storage_path)
            )

            manager1.start_fix_attempt("task1")
            manager1.complete_fix_attempt("task1", success=False, error_message="Pre-restart error")

            # Simulate system restart with new manager
            manager2 = FixIterationManager(
                max_fix_iterations=2,
                persistence_type=PersistenceType.FILE,
                storage_path=str(storage_path)
            )

            # Should load previous state
            status = manager2.get_task_status("task1")
            assert status["attempt_count"] == 1
            assert len(status["recent_attempts"]) == 1
            assert status["recent_attempts"][0]["error_message"] == "Pre-restart error"

            # Should allow one more attempt before escalation
            assert manager2.can_attempt_fix("task1") is True

            manager2.start_fix_attempt("task1")
            manager2.complete_fix_attempt("task1", success=False, error_message="Post-restart error")

            # Should now be escalated
            assert not manager2.can_attempt_fix("task1")


if __name__ == "__main__":
    pytest.main([__file__])
