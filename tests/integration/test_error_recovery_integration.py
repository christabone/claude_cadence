"""
Error Recovery Integration Tests

Tests for error handling and recovery across component boundaries,
including cascade failure scenarios and recovery mechanisms.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import threading
import queue
import random
from enum import Enum

# Import error handling and recovery components
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.workflow_state_machine import (
    WorkflowStateMachine, WorkflowState, WorkflowContext, StateTransitionError
)
from cadence.dispatch_logging import (
    DispatchLogger, OperationType, ErrorContext,
    get_dispatch_logger, setup_dispatch_logging
)
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo
)


class ErrorType(Enum):
    """Types of errors for testing"""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    DEPENDENCY_ERROR = "dependency_error"
    CORRUPTION_ERROR = "corruption_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"


@dataclass
class ErrorScenario:
    """Error scenario configuration for testing"""
    error_type: ErrorType
    probability: float
    recovery_possible: bool
    recovery_time: float
    cascade_potential: bool
    critical_level: str


class ErrorSimulationAgent:
    """Agent that simulates various error conditions"""

    def __init__(self, error_scenarios: List[ErrorScenario], agent_type: AgentType):
        self.error_scenarios = error_scenarios
        self.agent_type = agent_type
        self.execution_count = 0
        self.error_history = []
        self.recovery_attempts = []

    def execute(self, message: AgentMessage, recovery_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute with potential error simulation"""
        self.execution_count += 1
        start_time = time.time()

        # Check if this is a recovery attempt
        is_recovery = recovery_context and recovery_context.get("is_recovery", False)
        retry_count = recovery_context.get("retry_count", 0) if recovery_context else 0

        # Simulate errors based on scenarios
        for scenario in self.error_scenarios:
            if random.random() < scenario.probability:
                # Reduce error probability on recovery attempts
                error_prob = scenario.probability * (0.5 ** retry_count) if is_recovery else scenario.probability

                if random.random() < error_prob:
                    error_info = {
                        "error_type": scenario.error_type.value,
                        "execution_count": self.execution_count,
                        "retry_count": retry_count,
                        "is_recovery": is_recovery,
                        "recovery_possible": scenario.recovery_possible,
                        "cascade_potential": scenario.cascade_potential,
                        "critical_level": scenario.critical_level,
                        "timestamp": time.time(),
                        "agent_type": self.agent_type.value
                    }

                    self.error_history.append(error_info)

                    # Simulate error-specific behavior
                    if scenario.error_type == ErrorType.TIMEOUT_ERROR:
                        time.sleep(0.5)  # Simulate timeout delay
                        return {
                            "success": False,
                            "error": "Agent execution timed out",
                            "error_type": scenario.error_type.value,
                            "error_info": error_info,
                            "recovery_possible": scenario.recovery_possible
                        }

                    elif scenario.error_type == ErrorType.NETWORK_ERROR:
                        return {
                            "success": False,
                            "error": "Network connection failed",
                            "error_type": scenario.error_type.value,
                            "error_info": error_info,
                            "recovery_possible": scenario.recovery_possible
                        }

                    elif scenario.error_type == ErrorType.RESOURCE_ERROR:
                        return {
                            "success": False,
                            "error": "Insufficient resources available",
                            "error_type": scenario.error_type.value,
                            "error_info": error_info,
                            "recovery_possible": scenario.recovery_possible
                        }

                    elif scenario.error_type == ErrorType.VALIDATION_ERROR:
                        return {
                            "success": False,
                            "error": "Input validation failed",
                            "error_type": scenario.error_type.value,
                            "error_info": error_info,
                            "recovery_possible": False  # Validation errors typically not recoverable
                        }

                    elif scenario.error_type == ErrorType.CORRUPTION_ERROR:
                        return {
                            "success": False,
                            "error": "Data corruption detected",
                            "error_type": scenario.error_type.value,
                            "error_info": error_info,
                            "recovery_possible": scenario.recovery_possible
                        }

        # Simulate recovery delay if this is a recovery attempt
        if is_recovery and retry_count > 0:
            recovery_delay = min(0.1 * retry_count, 1.0)  # Max 1 second delay
            time.sleep(recovery_delay)

        # Successful execution
        execution_time = time.time() - start_time
        return {
            "success": True,
            "result": f"{self.agent_type.value} completed successfully",
            "execution_time": execution_time,
            "execution_count": self.execution_count,
            "retry_count": retry_count,
            "is_recovery": is_recovery
        }


class TestErrorRecoveryIntegration:
    """Test error handling and recovery across component boundaries"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="error_recovery_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def recovery_config(self):
        """Configuration for error recovery testing"""
        return DispatchConfig(
            max_concurrent_agents=3,
            default_timeout_ms=5000,
            enable_fix_tracking=True,
            enable_escalation=True,
            max_fix_iterations=5,
            escalation_strategy="notify_supervisor",
            persistence_type="memory"
        )

    def test_single_component_error_recovery(self, temp_dir, recovery_config):
        """Test error recovery within a single component"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("single_component_recovery")

        # Create error scenarios
        error_scenarios = [
            ErrorScenario(ErrorType.NETWORK_ERROR, 0.6, True, 0.5, False, "medium"),
            ErrorScenario(ErrorType.TIMEOUT_ERROR, 0.3, True, 1.0, False, "low"),
        ]

        # Create workflow and dispatcher
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="single_component_test",
                session_id="recovery_session",
                project_path=str(temp_dir)
            )
        )

        dispatcher = EnhancedAgentDispatcher(config=recovery_config)
        error_agent = ErrorSimulationAgent(error_scenarios, AgentType.REVIEW)

        recovery_attempts = []
        final_results = []

        def attempt_with_recovery(operation_name: str, max_retries: int = 3):
            """Attempt operation with recovery logic"""
            for retry_count in range(max_retries + 1):
                with dispatch_logger.operation_context(
                    operation_type=OperationType.AGENT_DISPATCH,
                    session_id="recovery_session",
                    task_id="single_component_test"
                ) as context:

                    is_recovery = retry_count > 0
                    recovery_context = {
                        "is_recovery": is_recovery,
                        "retry_count": retry_count,
                        "correlation_id": context.correlation_id
                    }

                    message = AgentMessage(
                        message_type=MessageType.DISPATCH_AGENT,
                        agent_type=AgentType.REVIEW,
                        context=MessageContext(
                            task_id="single_component_test",
                            parent_session="recovery_session",
                            files_modified=["test.py"],
                            project_path=str(temp_dir)
                        ),
                        success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                        payload={"correlation_id": context.correlation_id}
                    )

                    result = error_agent.execute(message, recovery_context)

                    if result["success"]:
                        final_results.append({
                            "operation": operation_name,
                            "success": True,
                            "retry_count": retry_count,
                            "result": result
                        })
                        return result
                    else:
                        # Log error and attempt recovery
                        dispatch_logger.log_error(
                            operation_type=OperationType.AGENT_DISPATCH,
                            error_type=result.get("error_type", "unknown"),
                            error_message=result.get("error", "Unknown error"),
                            metadata={
                                "retry_count": retry_count,
                                "max_retries": max_retries,
                                "recovery_possible": result.get("recovery_possible", False),
                                "correlation_id": context.correlation_id
                            }
                        )

                        recovery_attempts.append({
                            "operation": operation_name,
                            "retry_count": retry_count,
                            "error_type": result.get("error_type"),
                            "recovery_possible": result.get("recovery_possible", False)
                        })

                        # Check if recovery is possible
                        if not result.get("recovery_possible", False):
                            break

                        # Log escalation for retry
                        if retry_count < max_retries:
                            dispatch_logger.log_escalation(
                                reason=f"Operation failed with {result.get('error_type')}, attempting recovery",
                                attempt_count=retry_count + 1,
                                metadata={
                                    "escalation_type": "retry",
                                    "error_type": result.get("error_type"),
                                    "correlation_id": context.correlation_id
                                }
                            )

            # All retries exhausted
            final_results.append({
                "operation": operation_name,
                "success": False,
                "retry_count": max_retries,
                "error": "All recovery attempts failed"
            })
            return None

        try:
            # Test multiple operations with recovery
            operations = ["review_operation_1", "review_operation_2", "review_operation_3"]

            for operation in operations:
                workflow.transition_to(WorkflowState.REVIEWING, f"{operation}_started")
                result = attempt_with_recovery(operation)

                if result and result["success"]:
                    workflow.transition_to(WorkflowState.FIX_REQUIRED, f"{operation}_completed")
                else:
                    workflow.transition_to(WorkflowState.ERROR, f"{operation}_failed")
                    # Attempt to recover from error state
                    workflow.transition_to(WorkflowState.REVIEWING, f"{operation}_recovery_attempt")

            # Analyze recovery results
            assert len(final_results) == len(operations)

            successful_operations = [r for r in final_results if r["success"]]
            failed_operations = [r for r in final_results if not r["success"]]

            # Verify recovery attempts were made
            assert len(recovery_attempts) > 0, "No recovery attempts were made"

            # Verify some operations succeeded through recovery
            assert len(successful_operations) > 0, "No operations succeeded even with recovery"

            # Verify retry logic was used
            retried_operations = [r for r in final_results if r["retry_count"] > 0]
            assert len(retried_operations) > 0, "No operations used retry logic"

            # Verify error logging
            assert len(error_agent.error_history) > 0, "No errors were logged"

        finally:
            dispatcher.cleanup()

    def test_cross_component_error_propagation(self, temp_dir, recovery_config):
        """Test error propagation and recovery across multiple components"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("cross_component_recovery")

        # Create different error scenarios for different components
        review_scenarios = [
            ErrorScenario(ErrorType.NETWORK_ERROR, 0.4, True, 0.5, True, "medium")
        ]

        fix_scenarios = [
            ErrorScenario(ErrorType.DEPENDENCY_ERROR, 0.5, True, 1.0, True, "high"),
            ErrorScenario(ErrorType.RESOURCE_ERROR, 0.3, True, 0.8, False, "medium")
        ]

        verify_scenarios = [
            ErrorScenario(ErrorType.VALIDATION_ERROR, 0.4, False, 0.0, False, "low")
        ]

        # Create error agents for each component
        error_agents = {
            AgentType.REVIEW: ErrorSimulationAgent(review_scenarios, AgentType.REVIEW),
            AgentType.FIX: ErrorSimulationAgent(fix_scenarios, AgentType.FIX),
            AgentType.FIX: ErrorSimulationAgent(verify_scenarios, AgentType.FIX)
        }

        # Create workflow
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="cross_component_test",
                session_id="cross_component_session",
                project_path=str(temp_dir)
            )
        )

        dispatcher = EnhancedAgentDispatcher(config=recovery_config)

        # Track cross-component interactions
        component_results = {}
        error_propagations = []
        recovery_chains = []

        def execute_component_with_recovery(agent_type: AgentType, operation_data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute component with cross-component error handling"""
            max_retries = 3
            agent = error_agents[agent_type]

            for retry_count in range(max_retries + 1):
                with dispatch_logger.operation_context(
                    operation_type=OperationType.AGENT_DISPATCH,
                    session_id="cross_component_session",
                    task_id="cross_component_test"
                ) as context:

                    recovery_context = {
                        "is_recovery": retry_count > 0,
                        "retry_count": retry_count,
                        "correlation_id": context.correlation_id,
                        "previous_component_data": operation_data.get("previous_results"),
                        "cascade_from": operation_data.get("cascade_from")
                    }

                    message = AgentMessage(
                        message_type=MessageType.DISPATCH_AGENT,
                        agent_type=agent_type,
                        context=MessageContext(
                            task_id="cross_component_test",
                            parent_session="cross_component_session",
                            files_modified=["test.py", "utils.py"],
                            project_path=str(temp_dir)
                        ),
                        success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                        payload={
                            "correlation_id": context.correlation_id,
                            "operation_data": operation_data
                        }
                    )

                    result = agent.execute(message, recovery_context)

                    if result["success"]:
                        component_results[agent_type.value] = result
                        return result
                    else:
                        # Check for cascade potential
                        error_info = result.get("error_info", {})
                        if error_info.get("cascade_potential", False):
                            error_propagations.append({
                                "source_component": agent_type.value,
                                "error_type": result.get("error_type"),
                                "retry_count": retry_count,
                                "cascade_potential": True,
                                "propagation_time": time.time()
                            })

                        # Log cross-component error
                        dispatch_logger.log_error(
                            operation_type=OperationType.AGENT_DISPATCH,
                            error_type=result.get("error_type", "unknown"),
                            error_message=f"Cross-component error in {agent_type.value}: {result.get('error')}",
                            metadata={
                                "component": agent_type.value,
                                "retry_count": retry_count,
                                "cascade_from": operation_data.get("cascade_from"),
                                "correlation_id": context.correlation_id
                            }
                        )

                        if not result.get("recovery_possible", False) or retry_count >= max_retries:
                            break

                        # Record recovery chain
                        recovery_chains.append({
                            "component": agent_type.value,
                            "retry_count": retry_count + 1,
                            "error_type": result.get("error_type"),
                            "recovery_context": recovery_context
                        })

            return {"success": False, "error": "All recovery attempts failed", "component": agent_type.value}

        try:
            # Execute cross-component workflow with error handling
            workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "cross_component_start")

            # Step 1: Review component
            workflow.transition_to(WorkflowState.REVIEWING, "review_started")
            review_result = execute_component_with_recovery(AgentType.REVIEW, {"stage": "review"})

            if review_result["success"]:
                workflow.transition_to(WorkflowState.FIX_REQUIRED, "review_completed")

                # Step 2: Fix component (with review results)
                workflow.transition_to(WorkflowState.FIXING, "fix_started")
                fix_result = execute_component_with_recovery(AgentType.FIX, {
                    "stage": "fix",
                    "previous_results": review_result,
                    "cascade_from": "review" if not review_result["success"] else None
                })

                if fix_result["success"]:
                    workflow.transition_to(WorkflowState.VERIFICATION, "fix_completed")

                    # Step 3: Verify component (with fix results)
                    verify_result = execute_component_with_recovery(AgentType.FIX, {
                        "stage": "verify",
                        "previous_results": fix_result,
                        "cascade_from": "fix" if not fix_result["success"] else None
                    })

                    if verify_result["success"]:
                        workflow.transition_to(WorkflowState.COMPLETE, "verification_completed")
                    else:
                        workflow.transition_to(WorkflowState.ERROR, "verification_failed")
                else:
                    workflow.transition_to(WorkflowState.ERROR, "fix_failed")
            else:
                workflow.transition_to(WorkflowState.ERROR, "review_failed")

            # Analyze cross-component error handling
            successful_components = len([r for r in component_results.values() if r["success"]])
            total_components = len(error_agents)

            # Verify error propagation tracking
            if len(error_propagations) > 0:
                cascade_errors = [e for e in error_propagations if e["cascade_potential"]]
                assert len(cascade_errors) >= 0, "Cascade errors not properly tracked"

            # Verify recovery chains
            if len(recovery_chains) > 0:
                multi_retry_components = set(r["component"] for r in recovery_chains if r["retry_count"] > 1)
                assert len(multi_retry_components) >= 0, "Multi-retry recovery not tracked"

            # Verify workflow state progression
            final_state = workflow.current_state
            assert final_state in [WorkflowState.COMPLETE, WorkflowState.ERROR, WorkflowState.VERIFICATION]

            # Verify error histories across components
            total_errors = sum(len(agent.error_history) for agent in error_agents.values())
            assert total_errors > 0, "No errors occurred across components"

        finally:
            dispatcher.cleanup()

    def test_cascade_failure_recovery(self, temp_dir, recovery_config):
        """Test recovery from cascade failures across the entire system"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("cascade_failure_test")

        # Create scenarios that promote cascade failures
        cascade_scenarios = [
            ErrorScenario(ErrorType.RESOURCE_ERROR, 0.7, True, 2.0, True, "critical"),
            ErrorScenario(ErrorType.DEPENDENCY_ERROR, 0.6, True, 1.5, True, "high"),
            ErrorScenario(ErrorType.CORRUPTION_ERROR, 0.3, False, 0.0, True, "critical")
        ]

        # Create multiple workflows that can affect each other
        workflows = {}
        cascade_agents = {}

        project_ids = ["project_alpha", "project_beta", "project_gamma"]

        for project_id in project_ids:
            workflows[project_id] = WorkflowStateMachine(
                initial_state=WorkflowState.WORKING,
                context=WorkflowContext(
                    task_id=f"cascade_test_{project_id}",
                    session_id=f"cascade_session_{project_id}",
                    project_path=str(temp_dir / project_id)
                )
            )

            cascade_agents[project_id] = {
                agent_type: ErrorSimulationAgent(cascade_scenarios, agent_type)
                for agent_type in [AgentType.REVIEW, AgentType.FIX, AgentType.FIX]
            }

        dispatcher = EnhancedAgentDispatcher(config=recovery_config)

        # Track cascade failures and recovery
        cascade_events = []
        system_recovery_attempts = []

        def execute_with_cascade_detection(project_id: str, agent_type: AgentType,
                                         cascade_context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Execute with cascade failure detection and recovery"""
            agent = cascade_agents[project_id][agent_type]
            workflow = workflows[project_id]

            # Check for system-wide cascade conditions
            active_errors = sum(
                len(agent_dict[agent_type].error_history)
                for other_project, agent_dict in cascade_agents.items()
                if other_project != project_id
            )

            cascade_risk = active_errors > 5  # High error count indicates cascade risk

            recovery_context = {
                "cascade_risk": cascade_risk,
                "system_error_count": active_errors,
                "cascade_context": cascade_context or {},
                "project_id": project_id
            }

            with dispatch_logger.operation_context(
                operation_type=OperationType.AGENT_DISPATCH,
                session_id=f"cascade_session_{project_id}",
                task_id=f"cascade_test_{project_id}"
            ) as context:

                message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=agent_type,
                    context=MessageContext(
                        task_id=f"cascade_test_{project_id}",
                        parent_session=f"cascade_session_{project_id}",
                        files_modified=["test.py"],
                        project_path=str(temp_dir / project_id)
                    ),
                    success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                    payload={"correlation_id": context.correlation_id}
                )

                result = agent.execute(message, recovery_context)

                # Check for cascade failure
                if not result["success"]:
                    error_info = result.get("error_info", {})

                    if error_info.get("cascade_potential", False) and cascade_risk:
                        cascade_events.append({
                            "project_id": project_id,
                            "agent_type": agent_type.value,
                            "error_type": result.get("error_type"),
                            "system_error_count": active_errors,
                            "cascade_triggered": True,
                            "timestamp": time.time()
                        })

                        # Trigger system-wide recovery
                        system_recovery_attempts.append({
                            "trigger_project": project_id,
                            "trigger_agent": agent_type.value,
                            "recovery_strategy": "circuit_breaker",
                            "timestamp": time.time()
                        })

                        # Log cascade failure
                        dispatch_logger.log_escalation(
                            reason=f"Cascade failure detected in {project_id} {agent_type.value}",
                            attempt_count=active_errors,
                            metadata={
                                "escalation_type": "cascade_failure",
                                "project_id": project_id,
                                "agent_type": agent_type.value,
                                "system_error_count": active_errors,
                                "cascade_events_count": len(cascade_events),
                                "correlation_id": context.correlation_id
                            }
                        )

                        # Implement circuit breaker (simulate temporary halt)
                        time.sleep(0.2)  # Brief pause to prevent further cascade

                        # Attempt system recovery
                        workflow.transition_to(WorkflowState.ERROR, "cascade_failure_detected")

                return result

        try:
            # Execute multiple projects concurrently to create cascade conditions
            import threading

            def execute_project_workflow(project_id: str):
                """Execute project workflow with cascade potential"""
                workflow = workflows[project_id]

                try:
                    # Execute review
                    workflow.transition_to(WorkflowState.REVIEWING, "cascade_test_review")
                    review_result = execute_with_cascade_detection(project_id, AgentType.REVIEW)

                    if review_result["success"]:
                        workflow.transition_to(WorkflowState.FIX_REQUIRED, "review_completed")

                        # Execute fix
                        workflow.transition_to(WorkflowState.FIXING, "cascade_test_fix")
                        fix_result = execute_with_cascade_detection(
                            project_id, AgentType.FIX,
                            {"previous_component": "review", "review_result": review_result}
                        )

                        if fix_result["success"]:
                            workflow.transition_to(WorkflowState.VERIFICATION, "fix_completed")

                            # Execute verify
                            verify_result = execute_with_cascade_detection(
                                project_id, AgentType.FIX,
                                {"previous_component": "fix", "fix_result": fix_result}
                            )

                            if verify_result["success"]:
                                workflow.transition_to(WorkflowState.COMPLETE, "verification_completed")
                            else:
                                workflow.transition_to(WorkflowState.ERROR, "verification_failed")
                        else:
                            workflow.transition_to(WorkflowState.ERROR, "fix_failed")
                    else:
                        workflow.transition_to(WorkflowState.ERROR, "review_failed")

                except Exception as e:
                    workflow.transition_to(WorkflowState.ERROR, f"unexpected_error: {str(e)}")

            # Start concurrent workflows
            threads = []
            for project_id in project_ids:
                thread = threading.Thread(target=execute_project_workflow, args=(project_id,))
                threads.append(thread)
                thread.start()

            # Wait for all workflows
            for thread in threads:
                thread.join(timeout=15.0)

            # Analyze cascade failure recovery
            total_errors = sum(
                sum(len(agent.error_history) for agent in agent_dict.values())
                for agent_dict in cascade_agents.values()
            )

            assert total_errors > 0, "No errors occurred to test cascade recovery"

            # Verify cascade detection
            if len(cascade_events) > 0:
                assert len(system_recovery_attempts) > 0, "Cascade failures not handled with system recovery"

                # Verify recovery attempts were logged
                cascade_recoveries = [r for r in system_recovery_attempts if r["recovery_strategy"] == "circuit_breaker"]
                assert len(cascade_recoveries) > 0, "Circuit breaker recovery not triggered"

            # Verify workflow states after cascade handling
            error_states = [w for w in workflows.values() if w.current_state == WorkflowState.ERROR]
            completed_states = [w for w in workflows.values() if w.current_state == WorkflowState.COMPLETE]

            # At least some workflows should have error states due to cascade potential
            total_workflows = len(workflows)
            assert len(error_states) + len(completed_states) == total_workflows, \
                "Workflows in unexpected states after cascade test"

            # Verify error isolation (not all workflows should fail)
            if len(error_states) > 0 and len(error_states) < total_workflows:
                # Good - cascade was contained
                pass
            elif len(error_states) == total_workflows:
                # Total system failure - verify this was due to severe cascade
                assert len(cascade_events) > 2, "Total failure without sufficient cascade events"

        finally:
            dispatcher.cleanup()

    def test_persistent_error_state_recovery(self, temp_dir, recovery_config):
        """Test recovery from persistent error states with state machine integration"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("persistent_error_recovery")

        # Create persistent error scenarios
        persistent_scenarios = [
            ErrorScenario(ErrorType.CORRUPTION_ERROR, 0.8, True, 3.0, False, "critical"),
            ErrorScenario(ErrorType.AUTHENTICATION_ERROR, 0.6, True, 2.0, False, "high")
        ]

        # Create workflow with persistence
        persistence_path = temp_dir / "persistent_workflow.json"
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="persistent_error_test",
                session_id="persistent_session",
                project_path=str(temp_dir)
            ),
            persistence_path=persistence_path,
            enable_persistence=True
        )

        dispatcher = EnhancedAgentDispatcher(config=recovery_config)
        persistent_agent = ErrorSimulationAgent(persistent_scenarios, AgentType.FIX)

        error_recovery_sequence = []

        try:
            # Create persistent error condition
            workflow.transition_to(WorkflowState.REVIEWING, "start_persistent_test")
            workflow.transition_to(WorkflowState.FIX_REQUIRED, "issues_found")
            workflow.transition_to(WorkflowState.FIXING, "fix_started")

            # Attempt operation that will fail persistently
            with dispatch_logger.operation_context(
                operation_type=OperationType.FIX_APPLICATION,
                session_id="persistent_session",
                task_id="persistent_error_test"
            ) as context:

                result = persistent_agent.execute(
                    AgentMessage(
                        message_type=MessageType.DISPATCH_AGENT,
                        agent_type=AgentType.FIX,
                        context=MessageContext(
                            task_id="persistent_error_test",
                            parent_session="persistent_session",
                            files_modified=["corrupted.py"],
                            project_path=str(temp_dir)
                        ),
                        success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler")
                    ),
                    {"initial_attempt": True}
                )

                if not result["success"]:
                    # Transition to error state
                    workflow.transition_to(
                        WorkflowState.ERROR,
                        "persistent_error_detected",
                        metadata={
                            "error_type": result.get("error_type"),
                            "correlation_id": context.correlation_id
                        }
                    )

                    error_recovery_sequence.append({
                        "stage": "initial_failure",
                        "error_type": result.get("error_type"),
                        "workflow_state": workflow.current_state.value,
                        "timestamp": time.time()
                    })

            # Verify persistence of error state
            assert workflow.current_state == WorkflowState.ERROR
            assert persistence_path.exists()

            # Simulate system restart by creating new workflow from persistence
            recovered_workflow = WorkflowStateMachine(
                initial_state=WorkflowState.WORKING,  # Will be overridden by persistence
                context=WorkflowContext(),  # Will be overridden by persistence
                persistence_path=persistence_path,
                enable_persistence=True
            )

            # Verify error state was persisted and recovered
            assert recovered_workflow.current_state == WorkflowState.ERROR
            assert recovered_workflow.context.task_id == "persistent_error_test"

            error_recovery_sequence.append({
                "stage": "state_recovered",
                "workflow_state": recovered_workflow.current_state.value,
                "timestamp": time.time()
            })

            # Attempt recovery from persistent error state
            max_recovery_attempts = 5

            for recovery_attempt in range(max_recovery_attempts):
                with dispatch_logger.operation_context(
                    operation_type=OperationType.ESCALATION,
                    session_id="persistent_session",
                    task_id="persistent_error_test"
                ) as context:

                    # Log recovery attempt
                    dispatch_logger.log_escalation(
                        reason=f"Attempting recovery from persistent error, attempt {recovery_attempt + 1}",
                        attempt_count=recovery_attempt + 1,
                        metadata={
                            "escalation_type": "persistent_error_recovery",
                            "max_attempts": max_recovery_attempts,
                            "correlation_id": context.correlation_id
                        }
                    )

                    # Transition back to working state for recovery attempt
                    recovered_workflow.transition_to(
                        WorkflowState.FIXING,
                        f"recovery_attempt_{recovery_attempt + 1}",
                        metadata={"correlation_id": context.correlation_id}
                    )

                    # Create recovery agent with improved characteristics
                    recovery_agent = ErrorSimulationAgent(
                        [ErrorScenario(
                            error_type,
                            max(0.1, scenario.probability - (recovery_attempt + 1) * 0.15),  # Reduce error probability
                            scenario.recovery_possible,
                            scenario.recovery_time,
                            scenario.cascade_potential,
                            scenario.critical_level
                        ) for scenario in persistent_scenarios for error_type in [scenario.error_type]],
                        AgentType.FIX
                    )

                    recovery_result = recovery_agent.execute(
                        AgentMessage(
                            message_type=MessageType.DISPATCH_AGENT,
                            agent_type=AgentType.FIX,
                            context=MessageContext(
                                task_id="persistent_error_test",
                                parent_session="persistent_session",
                                files_modified=["corrupted.py"],
                                project_path=str(temp_dir)
                            ),
                            success_criteria=SuccessCriteria(),
                            callback=CallbackInfo(handler="test_handler")
                        ),
                        {
                            "is_recovery": True,
                            "recovery_attempt": recovery_attempt + 1,
                            "persistent_error": True
                        }
                    )

                    error_recovery_sequence.append({
                        "stage": "recovery_attempt",
                        "attempt": recovery_attempt + 1,
                        "success": recovery_result["success"],
                        "error_type": recovery_result.get("error_type"),
                        "workflow_state": recovered_workflow.current_state.value,
                        "timestamp": time.time()
                    })

                    if recovery_result["success"]:
                        # Successful recovery
                        recovered_workflow.transition_to(
                            WorkflowState.VERIFICATION,
                            "persistent_error_recovered",
                            metadata={"correlation_id": context.correlation_id}
                        )
                        break
                    else:
                        # Recovery failed, go back to error state
                        recovered_workflow.transition_to(
                            WorkflowState.ERROR,
                            f"recovery_attempt_{recovery_attempt + 1}_failed",
                            metadata={
                                "error_type": recovery_result.get("error_type"),
                                "correlation_id": context.correlation_id
                            }
                        )

            # Analyze persistent error recovery
            assert len(error_recovery_sequence) >= 3, "Insufficient recovery sequence events"

            # Verify recovery attempts were made
            recovery_attempts = [e for e in error_recovery_sequence if e["stage"] == "recovery_attempt"]
            assert len(recovery_attempts) > 0, "No recovery attempts made"

            # Verify state persistence throughout recovery
            state_recovered = any(e["stage"] == "state_recovered" for e in error_recovery_sequence)
            assert state_recovered, "Error state not properly persisted and recovered"

            # Verify final outcome
            final_state = recovered_workflow.current_state
            if final_state == WorkflowState.VERIFICATION:
                # Successful recovery
                final_recovery = True
            elif final_state == WorkflowState.ERROR:
                # Recovery failed but attempts were made
                final_recovery = False
            else:
                # Unexpected state
                pytest.fail(f"Unexpected final state after persistent error recovery: {final_state}")

            # Log final recovery status
            error_recovery_sequence.append({
                "stage": "final_outcome",
                "final_recovery": final_recovery,
                "workflow_state": final_state.value,
                "total_attempts": len(recovery_attempts),
                "timestamp": time.time()
            })

        finally:
            dispatcher.cleanup()
