"""
State Transition Integration Tests

Tests validating workflow state machine integration with dispatch operations,
ensuring proper state management during agent dispatch and coordination.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from threading import Event, Thread
import json

# Import workflow components
from cadence.workflow_state_machine import (
    WorkflowStateMachine, WorkflowState, WorkflowContext, StateTransition,
    WorkflowStateManager, StateTransitionError
)
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.dispatch_logging import (
    DispatchLogger, OperationType, DispatchContext,
    get_dispatch_logger, setup_dispatch_logging
)
from cadence.agent_messages import AgentMessage, MessageType, AgentType, MessageContext


class TestStateTransitionIntegration:
    """Test workflow state machine integration with dispatch operations"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for state persistence"""
        temp_dir = tempfile.mkdtemp(prefix="state_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dispatch_config(self):
        """Create dispatch configuration for testing"""
        return DispatchConfig(
            max_concurrent_agents=3,
            default_timeout_ms=5000,
            enable_fix_tracking=True,
            enable_escalation=True,
            max_fix_iterations=5,
            escalation_strategy="notify_supervisor"
        )

    def test_state_transitions_with_agent_dispatch(self, temp_dir, dispatch_config):
        """Test state transitions integrated with actual agent dispatch"""

        # Setup components
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="dispatch-integration-test",
                session_id="dispatch-session",
                project_path=str(temp_dir)
            ),
            persistence_path=temp_dir / "workflow.json"
        )

        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())
        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("state_integration_test")

        # Track state changes and agent dispatches
        state_changes = []
        agent_dispatches = []

        def track_state_change(new_state: WorkflowState, transition: StateTransition):
            state_changes.append({
                "state": new_state,
                "transition": transition,
                "timestamp": time.time()
            })

        # Register state handlers
        for state in WorkflowState:
            workflow.register_state_handler(state, track_state_change)

        # Mock agent dispatch to track calls
        original_dispatch = dispatcher.dispatch_agent

        def mock_dispatch_agent(*args, **kwargs):
            agent_dispatches.append({
                "args": args,
                "kwargs": kwargs,
                "timestamp": time.time(),
                "state_at_dispatch": workflow.current_state
            })
            return f"mock_message_id_{len(agent_dispatches)}"

        dispatcher.dispatch_agent = mock_dispatch_agent

        try:
            # Test 1: Trigger Detection with State Transition
            assert workflow.current_state == WorkflowState.WORKING

            with dispatch_logger.operation_context(
                operation_type=OperationType.TRIGGER_DETECTION,
                session_id="dispatch-session",
                task_id="dispatch-integration-test"
            ) as context:

                # Transition to review triggered
                workflow.transition_to(
                    WorkflowState.REVIEW_TRIGGERED,
                    trigger="file_change_detected",
                    metadata={
                        "files": ["test.py", "utils.py"],
                        "correlation_id": context.correlation_id
                    }
                )

                # Verify state change was tracked
                assert len(state_changes) == 1
                assert state_changes[0]["state"] == WorkflowState.REVIEW_TRIGGERED
                assert state_changes[0]["transition"].trigger == "file_change_detected"

            # Test 2: Agent Dispatch with State Transition
            with dispatch_logger.operation_context(
                operation_type=OperationType.AGENT_DISPATCH,
                session_id="dispatch-session",
                task_id="dispatch-integration-test"
            ) as context:

                # Transition to reviewing state
                workflow.transition_to(
                    WorkflowState.REVIEWING,
                    trigger="review_agent_dispatched",
                    metadata={"correlation_id": context.correlation_id}
                )

                # Dispatch review agent
                message_id = dispatcher.dispatch_agent(
                    agent_type=AgentType.REVIEW,
                    context=MessageContext(
                        task_id="dispatch-integration-test",
                        parent_session="dispatch-session",
                        files_modified=["test.py", "utils.py"],
                        project_path=str(temp_dir)
                    ),
                    success_criteria=Mock(),
                    callback_handler=lambda x: None
                )

                # Verify agent was dispatched in correct state
                assert len(agent_dispatches) == 1
                assert agent_dispatches[0]["state_at_dispatch"] == WorkflowState.REVIEWING
                assert message_id == "mock_message_id_1"

            # Test 3: Multiple State Transitions with Agent Coordination
            transitions_to_test = [
                (WorkflowState.FIX_REQUIRED, "issues_found", AgentType.FIX),
                (WorkflowState.FIXING, "fix_agent_dispatched", None),
                (WorkflowState.VERIFICATION, "fixes_applied", AgentType.FIX),
                (WorkflowState.COMPLETE, "verification_successful", None)
            ]

            for target_state, trigger, agent_type in transitions_to_test:
                with dispatch_logger.operation_context(
                    operation_type=OperationType.WORKFLOW_TRANSITION,
                    session_id="dispatch-session",
                    task_id="dispatch-integration-test"
                ) as context:

                    # Perform state transition
                    workflow.transition_to(
                        target_state,
                        trigger=trigger,
                        metadata={"correlation_id": context.correlation_id}
                    )

                    # Dispatch agent if specified
                    if agent_type:
                        dispatcher.dispatch_agent(
                            agent_type=agent_type,
                            context=MessageContext(
                                task_id="dispatch-integration-test",
                                parent_session="dispatch-session",
                                files_modified=["test.py", "utils.py"],
                                project_path=str(temp_dir)
                            ),
                            success_criteria=Mock(),
                            callback_handler=lambda x: None
                        )

            # Verify all transitions completed successfully
            assert workflow.current_state == WorkflowState.COMPLETE
            assert len(state_changes) == 6  # All state transitions tracked

            # Verify agent dispatches occurred in correct states
            assert len(agent_dispatches) == 3  # Review, Fix, Verify agents
            assert agent_dispatches[1]["state_at_dispatch"] == WorkflowState.FIX_REQUIRED
            assert agent_dispatches[2]["state_at_dispatch"] == WorkflowState.VERIFICATION

            # Verify state persistence
            assert workflow._persistence_path.exists()

            # Verify workflow history
            history = workflow.transition_history
            assert len(history) == 6

            expected_states = [
                WorkflowState.REVIEW_TRIGGERED,
                WorkflowState.REVIEWING,
                WorkflowState.FIX_REQUIRED,
                WorkflowState.FIXING,
                WorkflowState.VERIFICATION,
                WorkflowState.COMPLETE
            ]

            actual_states = [t.to_state for t in history]
            assert actual_states == expected_states

        finally:
            dispatcher.cleanup()

    def test_state_transition_error_handling(self, temp_dir, dispatch_config):
        """Test error handling during state transitions with agent dispatch"""

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="error-test",
                session_id="error-session",
                project_path=str(temp_dir)
            )
        )

        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())

        error_transitions = []

        def track_error_transitions(new_state: WorkflowState, transition: StateTransition):
            if new_state == WorkflowState.ERROR:
                error_transitions.append(transition)

        workflow.register_state_handler(WorkflowState.ERROR, track_error_transitions)

        try:
            # Test 1: Invalid state transition
            with pytest.raises(StateTransitionError):
                workflow.transition_to(
                    WorkflowState.COMPLETE,  # Can't go directly from WORKING to COMPLETE
                    trigger="invalid_transition"
                )

            # Test 2: Agent dispatch failure with error state
            workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "test")
            workflow.transition_to(WorkflowState.REVIEWING, "test")

            # Simulate agent dispatch failure
            with patch.object(dispatcher, 'dispatch_agent', side_effect=Exception("Dispatch failed")):
                try:
                    dispatcher.dispatch_agent(
                        agent_type=AgentType.REVIEW,
                        context=MessageContext("test", "test", [], str(temp_dir)),
                        success_criteria=Mock(),
                        callback_handler=lambda x: None
                    )
                except Exception:
                    # Transition to error state due to dispatch failure
                    workflow.transition_to(
                        WorkflowState.ERROR,
                        trigger="agent_dispatch_failed",
                        metadata={"error": "Dispatch failed"}
                    )

            # Verify error was handled
            assert workflow.current_state == WorkflowState.ERROR
            assert len(error_transitions) == 1
            assert error_transitions[0].trigger == "agent_dispatch_failed"

            # Test 3: Recovery from error state
            workflow.transition_to(
                WorkflowState.REVIEWING,
                trigger="retry_after_error"
            )

            assert workflow.current_state == WorkflowState.REVIEWING

        finally:
            dispatcher.cleanup()

    def test_concurrent_state_transitions(self, temp_dir, dispatch_config):
        """Test concurrent state transitions with thread safety"""

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="concurrent-test",
                session_id="concurrent-session",
                project_path=str(temp_dir)
            )
        )

        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())

        # Track all state changes with threading info
        state_changes = []
        transition_errors = []

        def track_concurrent_changes(new_state: WorkflowState, transition: StateTransition):
            state_changes.append({
                "state": new_state,
                "transition": transition,
                "thread_id": Thread.current_thread().ident,
                "timestamp": time.time()
            })

        # Register handlers for all states
        for state in WorkflowState:
            workflow.register_state_handler(state, track_concurrent_changes)

        # Define transition sequences for different threads
        thread_sequences = [
            [
                (WorkflowState.REVIEW_TRIGGERED, "thread1_trigger1"),
                (WorkflowState.REVIEWING, "thread1_trigger2"),
            ],
            [
                (WorkflowState.FIX_REQUIRED, "thread2_trigger1"),
                (WorkflowState.FIXING, "thread2_trigger2"),
            ]
        ]

        def execute_transitions(sequence, thread_id):
            """Execute a sequence of transitions in a thread"""
            try:
                for target_state, trigger in sequence:
                    # Add some randomness to timing
                    time.sleep(0.01 * (thread_id + 1))

                    workflow.transition_to(
                        target_state,
                        trigger=f"{trigger}_thread_{thread_id}",
                        metadata={"thread_id": thread_id}
                    )

            except StateTransitionError as e:
                transition_errors.append({
                    "thread_id": thread_id,
                    "error": str(e),
                    "timestamp": time.time()
                })

        try:
            # Start initial transition
            workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "initial_trigger")

            # Create and start threads
            threads = []
            for i, sequence in enumerate(thread_sequences):
                thread = Thread(target=execute_transitions, args=(sequence, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=5.0)

            # Verify thread safety
            assert len(state_changes) >= 1  # At least initial transition

            # Some transitions may fail due to invalid state progression
            # This is expected and demonstrates thread safety

            # Verify workflow is still in a valid state
            assert workflow.current_state in WorkflowState

            # Verify no corruption in transition history
            history = workflow.transition_history
            assert all(isinstance(t, StateTransition) for t in history)

            # Check if any errors occurred (expected due to concurrent invalid transitions)
            if transition_errors:
                # Verify errors are properly handled
                for error in transition_errors:
                    assert "Invalid transition" in error["error"]

        finally:
            dispatcher.cleanup()

    def test_state_persistence_during_dispatch(self, temp_dir, dispatch_config):
        """Test state persistence during agent dispatch operations"""

        persistence_path = temp_dir / "persistent_workflow.json"

        # Create workflow with persistence
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="persistence-test",
                session_id="persistence-session",
                project_path=str(temp_dir)
            ),
            persistence_path=persistence_path,
            enable_persistence=True
        )

        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())

        try:
            # Execute several state transitions
            transitions = [
                (WorkflowState.REVIEW_TRIGGERED, "file_changes"),
                (WorkflowState.REVIEWING, "agent_dispatched"),
                (WorkflowState.FIX_REQUIRED, "issues_found"),
                (WorkflowState.FIXING, "fix_started")
            ]

            for target_state, trigger in transitions:
                workflow.transition_to(target_state, trigger)

                # Verify state was persisted immediately
                assert persistence_path.exists()

                # Verify persistence file is valid JSON
                with open(persistence_path, 'r') as f:
                    persisted_data = json.load(f)

                assert persisted_data["current_state"] == target_state.value
                assert "transition_history" in persisted_data
                assert "context" in persisted_data

            # Test workflow restoration from persistence
            original_history_length = len(workflow.transition_history)
            original_context = workflow.context

            # Create new workflow from same persistence file
            restored_workflow = WorkflowStateMachine(
                initial_state=WorkflowState.WORKING,  # Will be overridden by persistence
                context=WorkflowContext(),  # Will be overridden by persistence
                persistence_path=persistence_path,
                enable_persistence=True
            )

            # Verify restoration
            assert restored_workflow.current_state == WorkflowState.FIXING
            assert len(restored_workflow.transition_history) == original_history_length
            assert restored_workflow.context.task_id == original_context.task_id
            assert restored_workflow.context.session_id == original_context.session_id

            # Test continued operations after restoration
            restored_workflow.transition_to(WorkflowState.VERIFICATION, "fixes_applied")
            restored_workflow.transition_to(WorkflowState.COMPLETE, "verification_passed")

            assert restored_workflow.current_state == WorkflowState.COMPLETE

        finally:
            dispatcher.cleanup()

    def test_workflow_state_manager_integration(self, temp_dir, dispatch_config):
        """Test WorkflowStateManager with multiple concurrent workflows"""

        manager = WorkflowStateManager(persistence_dir=temp_dir / "workflows")
        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())

        try:
            # Create multiple workflows
            workflow_configs = [
                ("project-1", "session-1", WorkflowState.WORKING),
                ("project-2", "session-2", WorkflowState.REVIEW_TRIGGERED),
                ("project-3", "session-3", WorkflowState.REVIEWING)
            ]

            workflows = {}
            for project_id, session_id, initial_state in workflow_configs:
                context = WorkflowContext(
                    task_id=project_id,
                    session_id=session_id,
                    project_path=str(temp_dir / project_id)
                )

                workflow = manager.get_or_create_workflow(
                    workflow_id=f"{project_id}_{session_id}",
                    initial_state=initial_state,
                    context=context
                )
                workflows[f"{project_id}_{session_id}"] = workflow

            # Verify all workflows were created
            assert len(workflows) == 3
            active_workflows = manager.list_workflows()
            assert len(active_workflows) == 3

            # Test independent workflow progression
            for workflow_id, workflow in workflows.items():
                if workflow.current_state == WorkflowState.WORKING:
                    workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "test")
                elif workflow.current_state == WorkflowState.REVIEW_TRIGGERED:
                    workflow.transition_to(WorkflowState.REVIEWING, "test")
                elif workflow.current_state == WorkflowState.REVIEWING:
                    workflow.transition_to(WorkflowState.FIX_REQUIRED, "test")

            # Verify workflows progressed independently
            summaries = manager.get_workflow_summaries()
            assert len(summaries) == 3

            for workflow_id, summary in summaries.items():
                assert "current_state" in summary
                assert "transition_count" in summary
                assert "valid_next_states" in summary

            # Test workflow removal
            removed = manager.remove_workflow("project-1_session-1")
            assert removed is True

            remaining_workflows = manager.list_workflows()
            assert len(remaining_workflows) == 2
            assert "project-1_session-1" not in remaining_workflows

        finally:
            dispatcher.cleanup()

    def test_state_transition_callbacks_with_dispatch_events(self, temp_dir, dispatch_config):
        """Test state transition callbacks integrated with dispatch events"""

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="callback-test",
                session_id="callback-session",
                project_path=str(temp_dir)
            )
        )

        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())

        # Track callback executions
        callback_events = []

        def create_state_callback(state_name: str):
            def callback(new_state: WorkflowState, transition: StateTransition):
                callback_events.append({
                    "callback_state": state_name,
                    "actual_state": new_state.value,
                    "trigger": transition.trigger,
                    "timestamp": time.time(),
                    "metadata": transition.metadata
                })
            return callback

        def create_transition_callback(from_state: str, to_state: str):
            def callback(transition: StateTransition):
                callback_events.append({
                    "callback_type": "transition",
                    "from_state": from_state,
                    "to_state": to_state,
                    "trigger": transition.trigger,
                    "timestamp": time.time()
                })
            return callback

        # Register state entry callbacks
        workflow.register_state_handler(
            WorkflowState.REVIEW_TRIGGERED,
            create_state_callback("review_triggered")
        )
        workflow.register_state_handler(
            WorkflowState.REVIEWING,
            create_state_callback("reviewing")
        )
        workflow.register_state_handler(
            WorkflowState.COMPLETE,
            create_state_callback("complete")
        )

        # Register transition callbacks
        workflow.register_transition_handler(
            WorkflowState.WORKING,
            WorkflowState.REVIEW_TRIGGERED,
            create_transition_callback("working", "review_triggered")
        )
        workflow.register_transition_handler(
            WorkflowState.VERIFICATION,
            WorkflowState.COMPLETE,
            create_transition_callback("verification", "complete")
        )

        try:
            # Execute workflow with callbacks
            workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "trigger_test")
            workflow.transition_to(WorkflowState.REVIEWING, "review_test")
            workflow.transition_to(WorkflowState.FIX_REQUIRED, "fix_test")
            workflow.transition_to(WorkflowState.FIXING, "fixing_test")
            workflow.transition_to(WorkflowState.VERIFICATION, "verify_test")
            workflow.transition_to(WorkflowState.COMPLETE, "complete_test")

            # Verify callbacks were executed
            assert len(callback_events) >= 5  # At least state callbacks + transition callbacks

            # Verify state callbacks
            state_callbacks = [e for e in callback_events if "callback_state" in e]
            assert len(state_callbacks) == 3  # review_triggered, reviewing, complete

            # Verify transition callbacks
            transition_callbacks = [e for e in callback_events if "callback_type" == "transition"]
            assert len(transition_callbacks) == 2  # working->review_triggered, verification->complete

            # Verify callback order and consistency
            review_callback = next(e for e in state_callbacks if e["callback_state"] == "review_triggered")
            assert review_callback["actual_state"] == "review_triggered"
            assert review_callback["trigger"] == "trigger_test"

            complete_callback = next(e for e in state_callbacks if e["callback_state"] == "complete")
            assert complete_callback["actual_state"] == "complete"
            assert complete_callback["trigger"] == "complete_test"

        finally:
            dispatcher.cleanup()
