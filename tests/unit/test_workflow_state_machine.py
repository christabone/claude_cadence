"""
Unit tests for WorkflowStateMachine
"""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from cadence.workflow_state_machine import (
    WorkflowStateMachine,
    WorkflowState,
    WorkflowContext,
    StateTransition,
    StateTransitionError,
    WorkflowStateManager
)


class TestWorkflowStateMachine:
    """Test cases for WorkflowStateMachine"""

    def test_initialization(self):
        """Test state machine initialization"""
        machine = WorkflowStateMachine()
        assert machine.current_state == WorkflowState.WORKING
        assert machine.context is not None
        assert len(machine.transition_history) == 0

    def test_initialization_with_custom_state(self):
        """Test initialization with custom initial state"""
        machine = WorkflowStateMachine(initial_state=WorkflowState.REVIEWING)
        assert machine.current_state == WorkflowState.REVIEWING

    def test_initialization_with_context(self):
        """Test initialization with custom context"""
        context = WorkflowContext(task_id="test-123", session_id="session-456")
        machine = WorkflowStateMachine(context=context)
        assert machine.context.task_id == "test-123"
        assert machine.context.session_id == "session-456"

    def test_valid_transitions(self):
        """Test valid state transitions"""
        machine = WorkflowStateMachine()

        # Test WORKING → REVIEW_TRIGGERED
        assert machine.can_transition_to(WorkflowState.REVIEW_TRIGGERED)
        result = machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "test_trigger")
        assert result is True
        assert machine.current_state == WorkflowState.REVIEW_TRIGGERED

        # Test REVIEW_TRIGGERED → REVIEWING
        assert machine.can_transition_to(WorkflowState.REVIEWING)
        result = machine.transition_to(WorkflowState.REVIEWING, "start_review")
        assert result is True
        assert machine.current_state == WorkflowState.REVIEWING

        # Test REVIEWING → FIX_REQUIRED
        result = machine.transition_to(WorkflowState.FIX_REQUIRED, "issues_found")
        assert result is True
        assert machine.current_state == WorkflowState.FIX_REQUIRED

        # Test FIX_REQUIRED → FIXING
        result = machine.transition_to(WorkflowState.FIXING, "start_fixing")
        assert result is True
        assert machine.current_state == WorkflowState.FIXING

        # Test FIXING → VERIFICATION
        result = machine.transition_to(WorkflowState.VERIFICATION, "fix_applied")
        assert result is True
        assert machine.current_state == WorkflowState.VERIFICATION

        # Test VERIFICATION → COMPLETE
        result = machine.transition_to(WorkflowState.COMPLETE, "verification_passed")
        assert result is True
        assert machine.current_state == WorkflowState.COMPLETE

    def test_invalid_transitions(self):
        """Test invalid state transitions are rejected"""
        machine = WorkflowStateMachine()

        # Test invalid transition WORKING → FIXING (must go through review first)
        assert not machine.can_transition_to(WorkflowState.FIXING)

        with pytest.raises(StateTransitionError, match="Invalid transition"):
            machine.transition_to(WorkflowState.FIXING, "invalid_trigger")

        # State should remain unchanged
        assert machine.current_state == WorkflowState.WORKING

    def test_error_state_transitions(self):
        """Test transitions to and from ERROR state"""
        machine = WorkflowStateMachine()

        # Any state can transition to ERROR
        result = machine.transition_to(WorkflowState.ERROR, "something_went_wrong")
        assert result is True
        assert machine.current_state == WorkflowState.ERROR

        # ERROR state can transition back to working states for recovery
        result = machine.transition_to(WorkflowState.WORKING, "recovery_attempt")
        assert result is True
        assert machine.current_state == WorkflowState.WORKING

    def test_complete_state_is_terminal(self):
        """Test that COMPLETE state doesn't allow further transitions"""
        machine = WorkflowStateMachine()

        # Get to COMPLETE state
        machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "trigger")
        machine.transition_to(WorkflowState.REVIEWING, "start")
        machine.transition_to(WorkflowState.COMPLETE, "no_issues_found")

        # COMPLETE state should have no valid transitions
        assert not machine.can_transition_to(WorkflowState.WORKING)
        assert not machine.can_transition_to(WorkflowState.ERROR)

        with pytest.raises(StateTransitionError):
            machine.transition_to(WorkflowState.WORKING, "invalid")

    def test_retry_transitions(self):
        """Test retry transitions (FIXING → FIX_REQUIRED, VERIFICATION → FIX_REQUIRED)"""
        machine = WorkflowStateMachine()

        # Get to FIXING state
        machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "trigger")
        machine.transition_to(WorkflowState.REVIEWING, "start")
        machine.transition_to(WorkflowState.FIX_REQUIRED, "issues")
        machine.transition_to(WorkflowState.FIXING, "fix_attempt")

        # Test retry from FIXING → FIX_REQUIRED
        result = machine.transition_to(WorkflowState.FIX_REQUIRED, "fix_failed")
        assert result is True
        assert machine.current_state == WorkflowState.FIX_REQUIRED

        # Test retry path: FIX_REQUIRED → FIXING → VERIFICATION → FIX_REQUIRED
        machine.transition_to(WorkflowState.FIXING, "retry_fix")
        machine.transition_to(WorkflowState.VERIFICATION, "fix_applied")
        result = machine.transition_to(WorkflowState.FIX_REQUIRED, "verification_failed")
        assert result is True
        assert machine.current_state == WorkflowState.FIX_REQUIRED

    def test_transition_history(self):
        """Test transition history tracking"""
        machine = WorkflowStateMachine()

        # Make some transitions
        machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "trigger1")
        machine.transition_to(WorkflowState.REVIEWING, "trigger2")

        history = machine.transition_history
        assert len(history) == 2

        # Check first transition
        assert history[0].from_state == WorkflowState.WORKING
        assert history[0].to_state == WorkflowState.REVIEW_TRIGGERED
        assert history[0].trigger == "trigger1"

        # Check second transition
        assert history[1].from_state == WorkflowState.REVIEW_TRIGGERED
        assert history[1].to_state == WorkflowState.REVIEWING
        assert history[1].trigger == "trigger2"

    def test_transition_with_metadata(self):
        """Test transitions with metadata"""
        machine = WorkflowStateMachine()

        metadata = {"test_key": "test_value", "count": 42}
        machine.transition_to(
            WorkflowState.REVIEW_TRIGGERED,
            "test_trigger",
            metadata=metadata
        )

        history = machine.transition_history
        assert len(history) == 1
        assert history[0].metadata == metadata

    def test_context_updates(self):
        """Test context updates"""
        machine = WorkflowStateMachine()

        # Test direct field updates
        machine.update_context(task_id="new-task", session_id="new-session")
        assert machine.context.task_id == "new-task"
        assert machine.context.session_id == "new-session"

        # Test metadata updates
        machine.update_context(custom_field="custom_value")
        assert machine.context.metadata["custom_field"] == "custom_value"

    def test_add_issue_and_fix(self):
        """Test adding issues and fixes to context"""
        machine = WorkflowStateMachine()

        # Add issue
        issue = {"severity": "high", "description": "Test issue"}
        machine.add_issue(issue)
        assert len(machine.context.issues_found) == 1
        assert machine.context.issues_found[0] == issue

        # Add fix
        fix = {"issue_id": "123", "description": "Test fix"}
        machine.add_fix(fix)
        assert len(machine.context.fixes_applied) == 1
        assert machine.context.fixes_applied[0] == fix

    def test_event_handlers(self):
        """Test state and transition event handlers"""
        machine = WorkflowStateMachine()

        # Mock handlers
        state_handler = Mock()
        transition_handler = Mock()

        # Register handlers
        machine.register_state_handler(WorkflowState.REVIEWING, state_handler)
        machine.register_transition_handler(
            WorkflowState.WORKING,
            WorkflowState.REVIEW_TRIGGERED,
            transition_handler
        )

        # Trigger transitions
        machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "trigger1")
        machine.transition_to(WorkflowState.REVIEWING, "trigger2")

        # Check handlers were called
        transition_handler.assert_called_once()
        state_handler.assert_called_once()

        # Check handler arguments
        transition_call_args = transition_handler.call_args[0][0]
        assert isinstance(transition_call_args, StateTransition)
        assert transition_call_args.from_state == WorkflowState.WORKING
        assert transition_call_args.to_state == WorkflowState.REVIEW_TRIGGERED

        state_call_args = state_handler.call_args[0]
        assert state_call_args[0] == WorkflowState.REVIEWING
        assert isinstance(state_call_args[1], StateTransition)

    def test_reset(self):
        """Test state machine reset"""
        machine = WorkflowStateMachine()

        # Make some changes
        machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "trigger")
        machine.update_context(task_id="test-task")
        machine.add_issue({"test": "issue"})

        # Reset
        machine.reset()

        # Verify reset
        assert machine.current_state == WorkflowState.WORKING
        assert len(machine.transition_history) == 0
        assert machine.context.task_id is None
        assert len(machine.context.issues_found) == 0

    def test_reset_with_custom_state(self):
        """Test reset with custom initial state"""
        machine = WorkflowStateMachine()
        machine.transition_to(WorkflowState.REVIEWING, "trigger")

        machine.reset(WorkflowState.ERROR)
        assert machine.current_state == WorkflowState.ERROR

    def test_get_state_summary(self):
        """Test state summary generation"""
        context = WorkflowContext(task_id="test-123")
        machine = WorkflowStateMachine(context=context)
        machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "test")

        summary = machine.get_state_summary()

        assert summary["current_state"] == "review_triggered"
        assert summary["context"]["task_id"] == "test-123"
        assert summary["transition_count"] == 1
        assert summary["last_transition"] is not None
        assert "working" in summary["valid_next_states"]  # ERROR is always valid

    def test_thread_safety(self):
        """Test thread safety of state machine operations"""
        import threading
        import time

        machine = WorkflowStateMachine()
        results = []
        errors = []

        def worker(worker_id):
            try:
                # Each worker tries to transition to a different state
                if worker_id % 2 == 0:
                    if machine.can_transition_to(WorkflowState.REVIEW_TRIGGERED):
                        result = machine.transition_to(WorkflowState.REVIEW_TRIGGERED, f"worker-{worker_id}")
                        results.append(result)
                else:
                    if machine.can_transition_to(WorkflowState.ERROR):
                        result = machine.transition_to(WorkflowState.ERROR, f"worker-{worker_id}")
                        results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have exactly one successful transition
        assert len([r for r in results if r]) == 1
        # No uncaught errors (StateTransitionError is expected)
        assert all(isinstance(e, StateTransitionError) for e in errors)


class TestWorkflowStatePersistence:
    """Test cases for state persistence"""

    def test_persistence_save_and_load(self):
        """Test saving and loading state"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "test_state.json"

            # Create machine with persistence
            context = WorkflowContext(task_id="test-123", session_id="session-456")
            machine = WorkflowStateMachine(
                context=context,
                persistence_path=persistence_path
            )

            # Make some changes
            machine.transition_to(WorkflowState.REVIEW_TRIGGERED, "test_trigger")
            machine.update_context(project_path="/test/path")
            machine.add_issue({"severity": "high", "desc": "test issue"})

            # Verify file was created
            assert persistence_path.exists()

            # Create new machine with same persistence path
            new_machine = WorkflowStateMachine(persistence_path=persistence_path)

            # Verify state was loaded
            assert new_machine.current_state == WorkflowState.REVIEW_TRIGGERED
            assert new_machine.context.task_id == "test-123"
            assert new_machine.context.session_id == "session-456"
            assert new_machine.context.project_path == "/test/path"
            assert len(new_machine.context.issues_found) == 1
            assert new_machine.context.issues_found[0]["severity"] == "high"
            assert len(new_machine.transition_history) == 1

    def test_persistence_file_creation(self):
        """Test that persistence directories are created automatically"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "nested" / "dir" / "state.json"

            machine = WorkflowStateMachine(persistence_path=persistence_path)
            machine.transition_to(WorkflowState.ERROR, "test")

            # Directory should be created automatically
            assert persistence_path.parent.exists()
            assert persistence_path.exists()

    @patch('cadence.workflow_state_machine.logger')
    def test_persistence_error_handling(self, mock_logger):
        """Test error handling in persistence operations"""
        # Test with invalid path
        invalid_path = Path("/invalid/path/that/should/not/exist/state.json")
        machine = WorkflowStateMachine(persistence_path=invalid_path)

        # This should not raise an exception
        machine.transition_to(WorkflowState.ERROR, "test")

        # Should log error
        mock_logger.error.assert_called()


class TestWorkflowStateManager:
    """Test cases for WorkflowStateManager"""

    def test_get_or_create_workflow(self):
        """Test getting or creating workflows"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowStateManager(persistence_dir=Path(temp_dir))

            # Create new workflow
            workflow1 = manager.get_or_create_workflow("test-1")
            assert workflow1.current_state == WorkflowState.WORKING

            # Get same workflow
            workflow1_again = manager.get_or_create_workflow("test-1")
            assert workflow1 is workflow1_again

            # Create different workflow
            context = WorkflowContext(task_id="task-2")
            workflow2 = manager.get_or_create_workflow(
                "test-2",
                initial_state=WorkflowState.REVIEWING,
                context=context
            )
            assert workflow2.current_state == WorkflowState.REVIEWING
            assert workflow2.context.task_id == "task-2"
            assert workflow1 is not workflow2

    def test_remove_workflow(self):
        """Test removing workflows"""
        manager = WorkflowStateManager()

        # Create workflow
        workflow = manager.get_or_create_workflow("test-remove")
        assert "test-remove" in manager.list_workflows()

        # Remove workflow
        result = manager.remove_workflow("test-remove")
        assert result is True
        assert "test-remove" not in manager.list_workflows()

        # Remove non-existent workflow
        result = manager.remove_workflow("non-existent")
        assert result is False

    def test_list_workflows(self):
        """Test listing workflows"""
        manager = WorkflowStateManager()

        assert manager.list_workflows() == []

        manager.get_or_create_workflow("workflow-1")
        manager.get_or_create_workflow("workflow-2")

        workflows = manager.list_workflows()
        assert len(workflows) == 2
        assert "workflow-1" in workflows
        assert "workflow-2" in workflows

    def test_get_workflow_summaries(self):
        """Test getting workflow summaries"""
        manager = WorkflowStateManager()

        # Create workflows with different states
        workflow1 = manager.get_or_create_workflow("test-1")
        workflow1.transition_to(WorkflowState.REVIEWING, "trigger")

        context2 = WorkflowContext(task_id="task-2")
        workflow2 = manager.get_or_create_workflow("test-2", context=context2)

        summaries = manager.get_workflow_summaries()

        assert len(summaries) == 2
        assert summaries["test-1"]["current_state"] == "reviewing"
        assert summaries["test-2"]["current_state"] == "working"
        assert summaries["test-2"]["context"]["task_id"] == "task-2"
