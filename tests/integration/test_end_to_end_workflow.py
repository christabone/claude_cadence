"""
Comprehensive End-to-End Workflow Integration Tests

Tests complete workflow from trigger detection → agent dispatch → review parsing →
fix application → verification cycle with real component integration.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import all workflow components
from cadence.orchestrator import SupervisorOrchestrator
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.workflow_state_machine import (
    WorkflowStateMachine, WorkflowState, WorkflowContext, StateTransition
)
from cadence.dispatch_logging import (
    DispatchLogger, OperationType, DispatchContext,
    get_dispatch_logger, setup_dispatch_logging
)
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo
)


@dataclass
class WorkflowTestResult:
    """Result of a complete workflow test"""
    success: bool
    states_visited: List[WorkflowState]
    agents_dispatched: List[str]
    issues_found: List[Dict[str, Any]]
    fixes_applied: List[Dict[str, Any]]
    duration_ms: float
    errors: List[str]
    logs_captured: List[Dict[str, Any]]


class MockAgent:
    """Mock agent for testing workflow integration"""

    def __init__(self, agent_type: AgentType, behavior: str = "success"):
        self.agent_type = agent_type
        self.behavior = behavior
        self.calls = []

    def process_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Process a mock agent request"""
        self.calls.append(message)

        if self.behavior == "error":
            return {
                "success": False,
                "error": f"Mock {self.agent_type.value} agent failed"
            }
        elif self.behavior == "timeout":
            time.sleep(0.1)  # Simulate timeout
            return {"success": False, "error": "Agent timed out"}
        elif self.agent_type == AgentType.REVIEW:
            return {
                "success": True,
                "issues_found": [
                    {"type": "security", "file": "test.py", "line": 10, "severity": "high"},
                    {"type": "performance", "file": "utils.py", "line": 25, "severity": "medium"}
                ],
                "requires_fixes": True
            }
        elif self.agent_type == AgentType.FIX:
            return {
                "success": True,
                "fixes_applied": [
                    {"file": "test.py", "line": 10, "fix": "Added input validation"},
                    {"file": "utils.py", "line": 25, "fix": "Optimized algorithm"}
                ],
                "verification_needed": True
            }
        elif self.agent_type == AgentType.FIX:
            return {
                "success": True,
                "verification_result": "all_fixes_verified",
                "tests_passed": True
            }
        else:
            return {"success": True, "result": f"Mock {self.agent_type.value} completed"}


class TestEndToEndWorkflowIntegration:
    """Test complete end-to-end workflow integration"""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        temp_dir = tempfile.mkdtemp(prefix="workflow_test_")
        project_path = Path(temp_dir) / "test_project"
        project_path.mkdir(parents=True)

        # Create sample files for testing
        (project_path / "test.py").write_text("""
def unsafe_function(user_input):
    # Security issue: no input validation
    return eval(user_input)
""")
        (project_path / "utils.py").write_text("""
def slow_function(data):
    # Performance issue: inefficient algorithm
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            result.append(data[i] + data[j])
    return result
""")

        yield project_path
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def workflow_config(self, temp_project_dir):
        """Create workflow configuration"""
        return {
            "project_path": temp_project_dir,
            "dispatch_config": DispatchConfig(
                max_concurrent_agents=2,
                default_timeout_ms=5000,
                enable_fix_tracking=True,
                enable_escalation=True,
                max_fix_iterations=3,
                escalation_strategy="log_only"
            ),
            "enable_logging": True,
            "log_level": "DEBUG"
        }

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        return {
            AgentType.REVIEW: MockAgent(AgentType.REVIEW, "success"),
            AgentType.FIX: MockAgent(AgentType.FIX, "success"),
            AgentType.FIX: MockAgent(AgentType.FIX, "success")
        }

    def test_complete_workflow_success_path(self, workflow_config, mock_agents, temp_project_dir):
        """Test complete successful workflow from trigger to verification"""

        # Setup dispatch logging
        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("workflow_test")

        # Setup workflow state machine
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="test-workflow-1",
                session_id="session-1",
                project_path=str(temp_project_dir)
            ),
            persistence_path=temp_project_dir / "workflow_state.json"
        )

        # Setup enhanced dispatcher
        dispatcher = EnhancedAgentDispatcher(config=workflow_config["dispatch_config"])

        # Track workflow execution
        states_visited = []
        agents_dispatched = []
        logs_captured = []

        def state_change_handler(new_state: WorkflowState, transition: StateTransition):
            states_visited.append(new_state)
            logs_captured.append({
                "type": "state_change",
                "from": transition.from_state.value,
                "to": transition.to_state.value,
                "trigger": transition.trigger
            })

        # Register state handlers
        for state in WorkflowState:
            workflow.register_state_handler(state, state_change_handler)

        start_time = time.time()

        try:
            # Step 1: Trigger Detection (WORKING → REVIEW_TRIGGERED)
            with dispatch_logger.operation_context(
                operation_type=OperationType.TRIGGER_DETECTION,
                session_id="session-1",
                task_id="test-workflow-1"
            ):
                workflow.transition_to(
                    WorkflowState.REVIEW_TRIGGERED,
                    trigger="file_changes_detected",
                    metadata={"files": ["test.py", "utils.py"], "change_count": 2}
                )

            # Step 2: Agent Dispatch (REVIEW_TRIGGERED → REVIEWING)
            with dispatch_logger.operation_context(
                operation_type=OperationType.AGENT_DISPATCH,
                session_id="session-1",
                task_id="test-workflow-1"
            ):
                workflow.transition_to(
                    WorkflowState.REVIEWING,
                    trigger="review_agent_dispatched"
                )

                # Mock agent dispatch
                review_message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=AgentType.REVIEW,
                    context=MessageContext(
                        task_id="test-workflow-1",
                        parent_session="session-1",
                        files_modified=["test.py", "utils.py"],
                        project_path=str(temp_project_dir)
                    ),
                    success_criteria=SuccessCriteria(
                        expected_outcomes=["Security issues identified", "Performance issues identified"],
                        validation_steps=["Static analysis", "Code quality check"]
                    ),
                    callback=CallbackInfo(handler="test_handler")
                )

                agents_dispatched.append("review_agent")
                review_result = mock_agents[AgentType.REVIEW].process_request(review_message)

            # Step 3: Review Parsing (REVIEWING → FIX_REQUIRED)
            with dispatch_logger.operation_context(
                operation_type=OperationType.REVIEW_PARSING,
                session_id="session-1",
                task_id="test-workflow-1"
            ):
                workflow.update_context(issues_found=review_result["issues_found"])
                workflow.transition_to(
                    WorkflowState.FIX_REQUIRED,
                    trigger="issues_found",
                    metadata={"issue_count": len(review_result["issues_found"])}
                )

            # Step 4: Fix Application (FIX_REQUIRED → FIXING → VERIFICATION)
            with dispatch_logger.operation_context(
                operation_type=OperationType.FIX_APPLICATION,
                session_id="session-1",
                task_id="test-workflow-1"
            ):
                workflow.transition_to(
                    WorkflowState.FIXING,
                    trigger="fix_agent_dispatched"
                )

                # Mock fix agent dispatch
                fix_message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=AgentType.FIX,
                    context=review_message.context,
                    success_criteria=SuccessCriteria(
                        expected_outcomes=["Security vulnerabilities fixed", "Performance optimized"],
                        validation_steps=["Apply fixes", "Verify syntax"]
                    ),
                    callback=CallbackInfo(handler="test_handler"),
                    payload={"issues_to_fix": review_result["issues_found"]}
                )

                agents_dispatched.append("fix_agent")
                fix_result = mock_agents[AgentType.FIX].process_request(fix_message)

                workflow.update_context(fixes_applied=fix_result["fixes_applied"])
                workflow.transition_to(
                    WorkflowState.VERIFICATION,
                    trigger="fixes_applied",
                    metadata={"fix_count": len(fix_result["fixes_applied"])}
                )

            # Step 5: Verification (VERIFICATION → COMPLETE)
            with dispatch_logger.operation_context(
                operation_type=OperationType.VERIFICATION,
                session_id="session-1",
                task_id="test-workflow-1"
            ):
                # Mock verification agent dispatch
                verify_message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=AgentType.FIX,
                    context=review_message.context,
                    success_criteria=SuccessCriteria(
                        expected_outcomes=["All fixes verified", "Tests pass"],
                        validation_steps=["Run tests", "Verify changes"]
                    ),
                    callback=CallbackInfo(handler="test_handler"),
                    payload={"fixes_to_verify": fix_result["fixes_applied"]}
                )

                agents_dispatched.append("verify_agent")
                verify_result = mock_agents[AgentType.FIX].process_request(verify_message)

                workflow.transition_to(
                    WorkflowState.COMPLETE,
                    trigger="verification_successful",
                    metadata={"verification_result": verify_result["verification_result"]}
                )

        except Exception as e:
            workflow.transition_to(
                WorkflowState.ERROR,
                trigger="workflow_error",
                metadata={"error": str(e)}
            )
            raise

        duration_ms = (time.time() - start_time) * 1000

        # Create test result
        result = WorkflowTestResult(
            success=workflow.current_state == WorkflowState.COMPLETE,
            states_visited=states_visited,
            agents_dispatched=agents_dispatched,
            issues_found=review_result["issues_found"],
            fixes_applied=fix_result["fixes_applied"],
            duration_ms=duration_ms,
            errors=[],
            logs_captured=logs_captured
        )

        # Assertions
        assert result.success, f"Workflow failed, final state: {workflow.current_state}"
        assert WorkflowState.REVIEW_TRIGGERED in states_visited
        assert WorkflowState.REVIEWING in states_visited
        assert WorkflowState.FIX_REQUIRED in states_visited
        assert WorkflowState.FIXING in states_visited
        assert WorkflowState.VERIFICATION in states_visited
        assert WorkflowState.COMPLETE in states_visited

        assert len(agents_dispatched) == 3
        assert "review_agent" in agents_dispatched
        assert "fix_agent" in agents_dispatched
        assert "verify_agent" in agents_dispatched

        assert len(result.issues_found) == 2
        assert len(result.fixes_applied) == 2
        assert result.duration_ms < 10000  # Should complete within 10 seconds

        # Verify workflow context
        context = workflow.context
        assert context.task_id == "test-workflow-1"
        assert context.session_id == "session-1"
        assert len(context.issues_found) == 2
        assert len(context.fixes_applied) == 2

        # Verify state persistence
        assert workflow._persistence_path.exists()

        # Cleanup
        dispatcher.cleanup()

    def test_workflow_with_error_recovery(self, workflow_config, temp_project_dir):
        """Test workflow error recovery and state transitions"""

        # Setup with error-prone mock agents
        error_agents = {
            AgentType.REVIEW: MockAgent(AgentType.REVIEW, "success"),
            AgentType.FIX: MockAgent(AgentType.FIX, "error"),  # Fix agent will fail
            AgentType.FIX: MockAgent(AgentType.FIX, "success")
        }

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="test-error-recovery",
                session_id="session-error",
                project_path=str(temp_project_dir)
            )
        )

        error_states = []

        def error_handler(new_state: WorkflowState, transition: StateTransition):
            if new_state == WorkflowState.ERROR:
                error_states.append(transition)

        workflow.register_state_handler(WorkflowState.ERROR, error_handler)

        try:
            # Progress through successful review
            workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "test_trigger")
            workflow.transition_to(WorkflowState.REVIEWING, "agent_dispatched")

            # Simulate review success
            review_result = error_agents[AgentType.REVIEW].process_request(Mock())
            workflow.update_context(issues_found=review_result["issues_found"])
            workflow.transition_to(WorkflowState.FIX_REQUIRED, "issues_found")
            workflow.transition_to(WorkflowState.FIXING, "fix_dispatched")

            # Simulate fix failure
            fix_result = error_agents[AgentType.FIX].process_request(Mock())
            assert not fix_result["success"], "Fix should fail for this test"

            # Transition to error state
            workflow.transition_to(
                WorkflowState.ERROR,
                "fix_agent_failed",
                metadata={"error": fix_result["error"]}
            )

            # Test recovery from error
            workflow.transition_to(
                WorkflowState.FIXING,
                "retry_fix",
                metadata={"retry_attempt": 1}
            )

        except Exception as e:
            pytest.fail(f"Error recovery test failed: {e}")

        # Verify error was handled
        assert len(error_states) == 1
        assert error_states[0].trigger == "fix_agent_failed"
        assert workflow.current_state == WorkflowState.FIXING  # Successfully recovered

    def test_workflow_performance_metrics(self, workflow_config, mock_agents, temp_project_dir):
        """Test performance metrics collection during workflow"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("performance_test")

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="performance-test",
                session_id="perf-session",
                project_path=str(temp_project_dir)
            )
        )

        performance_metrics = []

        # Measure each operation
        operations = [
            (WorkflowState.REVIEW_TRIGGERED, OperationType.TRIGGER_DETECTION),
            (WorkflowState.REVIEWING, OperationType.AGENT_DISPATCH),
            (WorkflowState.FIX_REQUIRED, OperationType.REVIEW_PARSING),
            (WorkflowState.FIXING, OperationType.FIX_APPLICATION),
            (WorkflowState.VERIFICATION, OperationType.VERIFICATION),
            (WorkflowState.COMPLETE, OperationType.WORKFLOW_TRANSITION)
        ]

        for target_state, operation_type in operations:
            start_time = time.time()

            with dispatch_logger.operation_context(
                operation_type=operation_type,
                session_id="perf-session",
                task_id="performance-test"
            ) as context:
                workflow.transition_to(
                    target_state,
                    f"{operation_type.value}_trigger"
                )

                # Simulate some work
                time.sleep(0.01)  # 10ms of work

            duration = (time.time() - start_time) * 1000
            performance_metrics.append({
                "operation": operation_type.value,
                "state": target_state.value,
                "duration_ms": duration,
                "correlation_id": context.correlation_id
            })

        # Verify performance tracking
        assert len(performance_metrics) == 6
        total_duration = sum(m["duration_ms"] for m in performance_metrics)
        assert total_duration < 1000  # Should complete within 1 second

        # Verify all operations have correlation IDs
        correlation_ids = [m["correlation_id"] for m in performance_metrics]
        assert len(set(correlation_ids)) == 6  # All unique correlation IDs

        # Verify operations are in correct order
        expected_operations = [op.value for _, op in operations]
        actual_operations = [m["operation"] for m in performance_metrics]
        assert actual_operations == expected_operations

    def test_workflow_with_multiple_fix_iterations(self, workflow_config, temp_project_dir):
        """Test workflow with multiple fix iterations and verification cycles"""

        # Create mock agents that require multiple iterations
        iteration_agents = {
            AgentType.REVIEW: MockAgent(AgentType.REVIEW, "success"),
            AgentType.FIX: MockAgent(AgentType.FIX, "success"),
            AgentType.FIX: MockAgent(AgentType.FIX, "success")
        }

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="multi-iteration-test",
                session_id="iteration-session",
                project_path=str(temp_project_dir)
            )
        )

        iteration_count = 0
        max_iterations = 3

        # Progress to fix required state
        workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "test")
        workflow.transition_to(WorkflowState.REVIEWING, "test")
        workflow.transition_to(WorkflowState.FIX_REQUIRED, "issues_found")

        # Simulate multiple fix/verification cycles
        while iteration_count < max_iterations:
            iteration_count += 1

            # Apply fixes
            workflow.transition_to(WorkflowState.FIXING, f"fix_attempt_{iteration_count}")
            fix_result = iteration_agents[AgentType.FIX].process_request(Mock())
            workflow.update_context(fixes_applied=fix_result["fixes_applied"])

            # Verify fixes
            workflow.transition_to(WorkflowState.VERIFICATION, f"verify_attempt_{iteration_count}")
            verify_result = iteration_agents[AgentType.FIX].process_request(Mock())

            # Simulate verification failure for first 2 iterations
            if iteration_count < 3:
                workflow.transition_to(
                    WorkflowState.FIX_REQUIRED,
                    f"verification_failed_{iteration_count}",
                    metadata={"remaining_issues": iteration_count}
                )
            else:
                # Final iteration succeeds
                workflow.transition_to(
                    WorkflowState.COMPLETE,
                    "verification_successful"
                )
                break

        # Verify multiple iterations were executed
        assert iteration_count == 3
        assert workflow.current_state == WorkflowState.COMPLETE

        # Verify transition history shows multiple cycles
        history = workflow.transition_history
        fix_transitions = [t for t in history if t.to_state == WorkflowState.FIXING]
        verify_transitions = [t for t in history if t.to_state == WorkflowState.VERIFICATION]

        assert len(fix_transitions) == 3
        assert len(verify_transitions) == 3

    def test_workflow_logging_integration(self, workflow_config, mock_agents, temp_project_dir):
        """Test comprehensive logging integration throughout workflow"""

        # Setup detailed logging
        setup_dispatch_logging(level="TRACE")
        dispatch_logger = get_dispatch_logger("logging_integration_test")

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="logging-test",
                session_id="log-session",
                project_path=str(temp_project_dir)
            )
        )

        logged_operations = []
        logged_states = []

        def capture_state_change(new_state: WorkflowState, transition: StateTransition):
            logged_states.append({
                "state": new_state.value,
                "transition": transition.trigger,
                "timestamp": transition.timestamp.isoformat(),
                "metadata": transition.metadata
            })

        # Register logging handlers
        for state in WorkflowState:
            workflow.register_state_handler(state, capture_state_change)

        # Execute workflow with comprehensive logging
        operations_to_test = [
            (OperationType.TRIGGER_DETECTION, WorkflowState.REVIEW_TRIGGERED),
            (OperationType.AGENT_DISPATCH, WorkflowState.REVIEWING),
            (OperationType.REVIEW_PARSING, WorkflowState.FIX_REQUIRED),
            (OperationType.FIX_APPLICATION, WorkflowState.FIXING),
            (OperationType.VERIFICATION, WorkflowState.VERIFICATION)
        ]

        for operation_type, target_state in operations_to_test:
            with dispatch_logger.operation_context(
                operation_type=operation_type,
                session_id="log-session",
                task_id="logging-test",
                file_paths=["test.py", "utils.py"]
            ) as context:

                # Log operation start
                dispatch_logger.log_operation_start(
                    operation_type=operation_type,
                    context_data={"state_transition": target_state.value}
                )

                # Perform state transition
                workflow.transition_to(
                    target_state,
                    f"{operation_type.value}_trigger",
                    metadata={"correlation_id": context.correlation_id}
                )

                # Log operation completion
                dispatch_logger.log_operation_completion(
                    operation_type=operation_type,
                    success=True,
                    result_data={"new_state": target_state.value}
                )

                logged_operations.append({
                    "operation": operation_type.value,
                    "state": target_state.value,
                    "correlation_id": context.correlation_id,
                    "session_id": context.session_id,
                    "task_id": context.task_id
                })

        # Complete workflow
        workflow.transition_to(WorkflowState.COMPLETE, "workflow_complete")

        # Verify comprehensive logging
        assert len(logged_operations) == 5
        assert len(logged_states) == 6  # Including COMPLETE state

        # Verify operation logging consistency
        for log_entry in logged_operations:
            assert log_entry["session_id"] == "log-session"
            assert log_entry["task_id"] == "logging-test"
            assert log_entry["correlation_id"] is not None

        # Verify state logging consistency
        expected_states = [
            "review_triggered", "reviewing", "fix_required",
            "fixing", "verification", "complete"
        ]
        actual_states = [entry["state"] for entry in logged_states]
        assert actual_states == expected_states

        # Verify correlation IDs are preserved in state metadata
        state_correlation_ids = [
            entry["metadata"].get("correlation_id")
            for entry in logged_states[:-1]  # Exclude final COMPLETE state
        ]
        operation_correlation_ids = [entry["correlation_id"] for entry in logged_operations]
        assert state_correlation_ids == operation_correlation_ids
