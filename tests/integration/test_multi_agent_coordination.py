"""
Multi-Agent Coordination Integration Tests

Tests for concurrent agent operations, agent handoffs, resource contention,
and coordination scenarios across the dispatch system.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from threading import Thread, Event, Lock
import concurrent.futures
import queue

# Import coordination components
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo
)
from cadence.workflow_state_machine import (
    WorkflowStateMachine, WorkflowState, WorkflowContext
)
from cadence.dispatch_logging import (
    DispatchLogger, OperationType, get_dispatch_logger, setup_dispatch_logging
)


@dataclass
class AgentExecutionResult:
    """Result of agent execution for coordination testing"""
    agent_id: str
    agent_type: AgentType
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    result_data: Dict[str, Any]
    errors: List[str]
    correlation_id: str


class MockCoordinatedAgent:
    """Mock agent that simulates realistic execution timing and coordination"""

    def __init__(self, agent_type: AgentType, execution_time: float = 1.0,
                 success_rate: float = 1.0, coordination_delay: float = 0.1):
        self.agent_type = agent_type
        self.execution_time = execution_time
        self.success_rate = success_rate
        self.coordination_delay = coordination_delay
        self.execution_count = 0
        self.coordination_events = []

    def execute(self, message: AgentMessage, coordination_context: Dict[str, Any] = None) -> AgentExecutionResult:
        """Execute agent with coordination awareness"""
        self.execution_count += 1
        agent_id = f"{self.agent_type.value}_{self.execution_count}"
        start_time = time.time()

        # Simulate coordination delay
        if coordination_context and coordination_context.get("wait_for_previous"):
            time.sleep(self.coordination_delay)

        # Simulate execution work
        time.sleep(self.execution_time)

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Determine success
        import random
        success = random.random() < self.success_rate

        # Generate realistic result data
        result_data = self._generate_result_data(success, coordination_context)
        errors = [] if success else ["Mock execution failed"]

        # Record coordination event
        self.coordination_events.append({
            "agent_id": agent_id,
            "start_time": start_time,
            "end_time": end_time,
            "coordination_context": coordination_context,
            "success": success
        })

        return AgentExecutionResult(
            agent_id=agent_id,
            agent_type=self.agent_type,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            result_data=result_data,
            errors=errors,
            correlation_id=message.context.task_id if message else "no_correlation"
        )

    def _generate_result_data(self, success: bool, coordination_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic result data based on agent type"""
        if not success:
            return {"error": "Execution failed"}

        if self.agent_type == AgentType.REVIEW:
            return {
                "issues_found": [
                    {"type": "security", "severity": "high", "file": "test.py", "line": 10},
                    {"type": "performance", "severity": "medium", "file": "utils.py", "line": 25}
                ],
                "files_reviewed": ["test.py", "utils.py"],
                "review_score": 7.5,
                "coordination_data": coordination_context
            }
        elif self.agent_type == AgentType.FIX:
            return {
                "fixes_applied": [
                    {"file": "test.py", "line": 10, "fix": "Added input validation"},
                    {"file": "utils.py", "line": 25, "fix": "Optimized algorithm"}
                ],
                "files_modified": ["test.py", "utils.py"],
                "fix_success_rate": 0.95,
                "coordination_data": coordination_context
            }
        elif self.agent_type == AgentType.FIX:
            return {
                "verification_result": "passed",
                "tests_run": 15,
                "tests_passed": 15,
                "verification_score": 9.2,
                "coordination_data": coordination_context
            }
        else:
            return {
                "operation": f"{self.agent_type.value}_completed",
                "coordination_data": coordination_context
            }


class TestMultiAgentCoordination:
    """Test multi-agent coordination and concurrent operations"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="coordination_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def coordination_config(self):
        """Create configuration for coordination testing"""
        return DispatchConfig(
            max_concurrent_agents=4,  # Allow multiple concurrent agents
            default_timeout_ms=10000,
            enable_fix_tracking=True,
            enable_escalation=True,
            max_fix_iterations=3,
            escalation_strategy="notify_supervisor"
        )

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents with different execution characteristics"""
        return {
            AgentType.REVIEW: MockCoordinatedAgent(AgentType.REVIEW, execution_time=0.5, success_rate=0.9),
            AgentType.FIX: MockCoordinatedAgent(AgentType.FIX, execution_time=1.0, success_rate=0.8),
            AgentType.VERIFY: MockCoordinatedAgent(AgentType.VERIFY, execution_time=0.3, success_rate=0.95),
        }

    def test_concurrent_agent_dispatch_and_execution(self, temp_dir, coordination_config, mock_agents):
        """Test concurrent dispatch and execution of multiple agents"""

        dispatcher = EnhancedAgentDispatcher(config=coordination_config.to_dict())
        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("coordination_test")

        # Track concurrent executions with queue for thread-safe communication
        execution_results_queue = queue.Queue()

        def agent_callback(agent_type: AgentType, result: AgentExecutionResult):
            """Callback to track agent execution results"""
            execution_results_queue.put(result)

        # Create agent execution tasks
        agent_tasks = [
            (AgentType.REVIEW, "project_1", ["file1.py", "file2.py"]),
            (AgentType.REVIEW, "project_2", ["file3.py", "file4.py"]),
            (AgentType.FIX, "project_1", ["file1.py"]),
            (AgentType.FIX, "project_3", ["file5.py"]),
        ]

        message_ids = []
        threads = []  # Track threads for proper cleanup
        start_time = time.time()

        try:
            # Dispatch all agents concurrently
            for agent_type, project_id, files in agent_tasks:
                with dispatch_logger.operation_context(
                    operation_type=OperationType.AGENT_DISPATCH,
                    session_id=f"session_{project_id}",
                    task_id=f"task_{project_id}",
                    file_paths=files
                ) as context:

                    message = AgentMessage(
                        message_type=MessageType.DISPATCH_AGENT,
                        agent_type=agent_type,
                        context=MessageContext(
                            task_id=f"task_{project_id}",
                            parent_session=f"session_{project_id}",
                            files_modified=files,
                            project_path=str(temp_dir / project_id)
                        ),
                        success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                        payload={"coordination_context": {"correlation_id": context.correlation_id}}
                    )

                    # Mock agent execution in separate thread
                    def execute_agent(agent_type, message, callback):
                        agent = mock_agents[agent_type]
                        result = agent.execute(message, {"correlation_id": context.correlation_id})
                        callback(agent_type, result)

                    # Start agent execution thread
                    thread = Thread(
                        target=execute_agent,
                        args=(agent_type, message, agent_callback)
                    )
                    threads.append(thread)  # Track thread for cleanup
                    thread.start()

                    message_ids.append(f"msg_{len(message_ids)}")

            # Wait for all agents to complete - collect results from queue
            execution_results = []
            for _ in range(len(agent_tasks)):
                try:
                    result = execution_results_queue.get(timeout=15.0)
                    execution_results.append(result)
                except queue.Empty:
                    break  # Timeout occurred

            total_duration = time.time() - start_time

            # Verify concurrent execution
            assert len(execution_results) == len(agent_tasks), \
                f"Expected {len(agent_tasks)} results, got {len(execution_results)}"

            # Verify execution happened concurrently (should be faster than sequential)
            sequential_duration = sum(agent.execution_time for agent in mock_agents.values())
            assert total_duration < sequential_duration, \
                f"Execution not concurrent: {total_duration}s >= {sequential_duration}s"

            # Verify all agents executed successfully
            successful_executions = [r for r in execution_results if r.success]
            assert len(successful_executions) >= len(agent_tasks) * 0.8, \
                "Too many agent executions failed"

            # Verify timing overlap (concurrent execution)
            execution_times = [(r.start_time, r.end_time) for r in execution_results]
            overlaps = 0

            for i, (start1, end1) in enumerate(execution_times):
                for j, (start2, end2) in enumerate(execution_times[i+1:], i+1):
                    # Check if time ranges overlap
                    if start1 < end2 and start2 < end1:
                        overlaps += 1

            assert overlaps > 0, "No overlapping execution times found - agents not running concurrently"

        finally:
            # Wait for all threads to complete before cleanup
            for thread in threads:
                try:
                    thread.join(timeout=5.0)  # 5 second timeout for thread cleanup
                except Exception:
                    pass  # Continue cleanup even if thread join fails
            dispatcher.cleanup()

    def test_agent_handoff_coordination(self, temp_dir, coordination_config, mock_agents):
        """Test coordinated handoff between agents (review → fix → verify)"""

        dispatcher = EnhancedAgentDispatcher(config=coordination_config.to_dict())
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="handoff-test",
                session_id="handoff-session",
                project_path=str(temp_dir)
            )
        )

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("handoff_test")

        # Track handoff sequence with thread-safe queue
        handoff_queue = queue.Queue()
        handoff_data = {}
        completion_events = {
            "review": Event(),
            "fix": Event(),
            "verify": Event()
        }
        handoff_threads = []  # Track threads for proper cleanup

        def create_handoff_callback(stage: str):
            def callback(result: AgentExecutionResult):
                handoff_queue.put(stage)
                handoff_data[stage] = result
                completion_events[stage].set()

                # Trigger next stage based on result
                if stage == "review" and result.success:
                    # Hand off to fix agent with review results
                    start_fix_stage(result.result_data)
                elif stage == "fix" and result.success:
                    # Hand off to verify agent with fix results
                    start_verify_stage(result.result_data)
            return callback

        def start_fix_stage(review_data: Dict[str, Any]):
            """Start fix stage with review results"""
            with dispatch_logger.operation_context(
                operation_type=OperationType.FIX_APPLICATION,
                session_id="handoff-session",
                task_id="handoff-test"
            ) as context:

                workflow.transition_to(WorkflowState.FIXING, "fix_agent_started")

                fix_message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=AgentType.FIX,
                    context=MessageContext(
                        task_id="handoff-test",
                        parent_session="handoff-session",
                        files_modified=review_data.get("files_reviewed", []),
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                    payload={
                        "issues_to_fix": review_data.get("issues_found", []),
                        "handoff_from": "review",
                        "correlation_id": context.correlation_id
                    }
                )

                # Execute fix agent
                def execute_fix():
                    agent = mock_agents[AgentType.FIX]
                    result = agent.execute(fix_message, {
                        "handoff_from": "review",
                        "review_data": review_data,
                        "wait_for_previous": True
                    })
                    create_handoff_callback("fix")(result)

                fix_thread = Thread(target=execute_fix)
                handoff_threads.append(fix_thread)
                fix_thread.start()

        def start_verify_stage(fix_data: Dict[str, Any]):
            """Start verify stage with fix results"""
            with dispatch_logger.operation_context(
                operation_type=OperationType.VERIFICATION,
                session_id="handoff-session",
                task_id="handoff-test"
            ) as context:

                workflow.transition_to(WorkflowState.VERIFICATION, "verify_agent_started")

                verify_message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=AgentType.VERIFY,
                    context=MessageContext(
                        task_id="handoff-test",
                        parent_session="handoff-session",
                        files_modified=fix_data.get("files_modified", []),
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                    payload={
                        "fixes_to_verify": fix_data.get("fixes_applied", []),
                        "handoff_from": "fix",
                        "correlation_id": context.correlation_id
                    }
                )

                # Execute verify agent
                def execute_verify():
                    agent = mock_agents[AgentType.VERIFY]
                    result = agent.execute(verify_message, {
                        "handoff_from": "fix",
                        "fix_data": fix_data,
                        "wait_for_previous": True
                    })
                    create_handoff_callback("verify")(result)

                verify_thread = Thread(target=execute_verify)
                handoff_threads.append(verify_thread)
                verify_thread.start()

        try:
            # Start the handoff sequence with review agent
            with dispatch_logger.operation_context(
                operation_type=OperationType.REVIEW_PARSING,
                session_id="handoff-session",
                task_id="handoff-test"
            ) as context:

                workflow.transition_to(WorkflowState.REVIEWING, "review_agent_started")

                review_message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=AgentType.REVIEW,
                    context=MessageContext(
                        task_id="handoff-test",
                        parent_session="handoff-session",
                        files_modified=["test.py", "utils.py"],
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                    payload={"correlation_id": context.correlation_id}
                )

                # Execute review agent
                def execute_review():
                    agent = mock_agents[AgentType.REVIEW]
                    result = agent.execute(review_message, {"correlation_id": context.correlation_id})
                    create_handoff_callback("review")(result)

                review_thread = Thread(target=execute_review)
                handoff_threads.append(review_thread)
                review_thread.start()

            # Wait for complete handoff sequence using events
            handoff_sequence = []
            stages_to_wait = ["review", "fix", "verify"]

            for stage in stages_to_wait:
                if completion_events[stage].wait(timeout=20.0):
                    handoff_sequence.append(stage)
                else:
                    break  # Timeout occurred

            # Verify handoff sequence
            assert len(handoff_sequence) >= 2, f"Handoff sequence incomplete: {handoff_sequence}"
            assert handoff_sequence[0] == "review", "First stage should be review"

            if len(handoff_sequence) >= 2:
                assert handoff_sequence[1] == "fix", "Second stage should be fix"

            if len(handoff_sequence) >= 3:
                assert handoff_sequence[2] == "verify", "Third stage should be verify"

            # Verify data handoff between stages
            if "review" in handoff_data and "fix" in handoff_data:
                review_result = handoff_data["review"]
                fix_result = handoff_data["fix"]

                # Verify review data was passed to fix
                fix_coordination = fix_result.result_data.get("coordination_data", {})
                assert "review_data" in fix_coordination, "Review data not passed to fix stage"

            if "fix" in handoff_data and "verify" in handoff_data:
                fix_result = handoff_data["fix"]
                verify_result = handoff_data["verify"]

                # Verify fix data was passed to verify
                verify_coordination = verify_result.result_data.get("coordination_data", {})
                assert "fix_data" in verify_coordination, "Fix data not passed to verify stage"

            # Verify workflow state progression
            if len(handoff_sequence) >= 3:
                assert workflow.current_state == WorkflowState.VERIFICATION
            elif len(handoff_sequence) >= 2:
                assert workflow.current_state == WorkflowState.FIXING
            else:
                assert workflow.current_state == WorkflowState.REVIEWING

        finally:
            # Wait for all handoff threads to complete before cleanup
            for thread in handoff_threads:
                try:
                    thread.join(timeout=5.0)  # 5 second timeout for thread cleanup
                except Exception:
                    pass  # Continue cleanup even if thread join fails
            dispatcher.cleanup()

    def test_resource_contention_handling(self, temp_dir, coordination_config):
        """Test handling of resource contention between agents"""

        # Create configuration with limited resources
        limited_config = DispatchConfig(
            max_concurrent_agents=2,  # Limit to 2 concurrent agents
            default_timeout_ms=5000,
            enable_fix_tracking=True,
            enable_escalation=True
        )

        dispatcher = EnhancedAgentDispatcher(config=limited_config.to_dict())
        setup_dispatch_logging(level="DEBUG")

        # Create agents that compete for resources
        resource_contention_agents = {
            AgentType.REVIEW: MockCoordinatedAgent(AgentType.REVIEW, execution_time=1.0),
            AgentType.FIX: MockCoordinatedAgent(AgentType.FIX, execution_time=1.5),
        }

        # Track resource usage
        active_agents = []
        max_concurrent = 0
        resource_lock = Lock()

        def track_resource_usage(agent_id: str, action: str):
            """Track agent resource usage"""
            with resource_lock:
                if action == "start":
                    active_agents.append(agent_id)
                elif action == "end":
                    if agent_id in active_agents:
                        active_agents.remove(agent_id)

                nonlocal max_concurrent
                max_concurrent = max(max_concurrent, len(active_agents))

        def create_resource_aware_agent(base_agent: MockCoordinatedAgent, agent_id: str):
            """Create agent wrapper that tracks resource usage"""
            def execute_with_tracking(message, coordination_context=None):
                track_resource_usage(agent_id, "start")
                try:
                    result = base_agent.execute(message, coordination_context)
                    return result
                finally:
                    track_resource_usage(agent_id, "end")

            return execute_with_tracking

        try:
            # Create multiple agents that will compete for resources
            agent_executions_queue = queue.Queue()
            contention_threads = []  # Track threads for proper cleanup

            for i in range(5):  # 5 agents competing for 2 slots
                agent_type = AgentType.REVIEW if i % 2 == 0 else AgentType.FIX
                agent_id = f"{agent_type.value}_{i}"

                message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=agent_type,
                    context=MessageContext(
                        task_id=f"contention_task_{i}",
                        parent_session=f"contention_session_{i}",
                        files_modified=[f"file_{i}.py"],
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria()
                )

                # Create resource-aware agent executor
                base_agent = resource_contention_agents[agent_type]
                executor = create_resource_aware_agent(base_agent, agent_id)

                # Execute in separate thread
                def run_agent(executor, message, agent_id):
                    result = executor(message)
                    agent_executions_queue.put((agent_id, result))

                thread = Thread(target=run_agent, args=(executor, message, agent_id))
                contention_threads.append(thread)  # Track thread for cleanup
                thread.start()

            # Wait for all executions to complete - collect from queue
            agent_executions = []
            for _ in range(5):
                try:
                    execution = agent_executions_queue.get(timeout=15.0)
                    agent_executions.append(execution)
                except queue.Empty:
                    break  # Timeout occurred

            # Verify resource limits were respected
            assert max_concurrent <= limited_config.max_concurrent_agents, \
                f"Resource limit exceeded: {max_concurrent} > {limited_config.max_concurrent_agents}"

            # Verify all agents eventually executed
            assert len(agent_executions) >= 4, \
                f"Not enough agents executed: {len(agent_executions)} < 4"

            # Verify successful executions
            successful_executions = [
                (agent_id, result) for agent_id, result in agent_executions
                if result.success
            ]
            assert len(successful_executions) >= 3, \
                "Too many executions failed due to resource contention"

        finally:
            # Wait for all contention threads to complete before cleanup
            for thread in contention_threads:
                try:
                    thread.join(timeout=5.0)  # 5 second timeout for thread cleanup
                except Exception:
                    pass  # Continue cleanup even if thread join fails
            dispatcher.cleanup()

    def test_agent_coordination_failure_recovery(self, temp_dir, coordination_config, mock_agents):
        """Test recovery from coordination failures between agents"""

        dispatcher = EnhancedAgentDispatcher(config=coordination_config.to_dict())
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="coordination-failure-test",
                session_id="failure-session",
                project_path=str(temp_dir)
            )
        )

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("failure_recovery_test")

        # Create agents with failure scenarios
        failing_agents = {
            AgentType.REVIEW: MockCoordinatedAgent(AgentType.REVIEW, success_rate=0.3),  # High failure rate
            AgentType.FIX: MockCoordinatedAgent(AgentType.FIX, success_rate=0.6),
            AgentType.VERIFY: MockCoordinatedAgent(AgentType.VERIFY, success_rate=0.9),
        }

        coordination_attempts = []
        recovery_actions = []

        def attempt_coordination(stage: str, agent_type: AgentType, retry_count: int = 0):
            """Attempt agent coordination with retry logic"""
            max_retries = 3

            with dispatch_logger.operation_context(
                operation_type=OperationType.AGENT_DISPATCH,
                session_id="failure-session",
                task_id="coordination-failure-test"
            ) as context:

                message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=agent_type,
                    context=MessageContext(
                        task_id="coordination-failure-test",
                        parent_session="failure-session",
                        files_modified=["test.py"],
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                    payload={
                        "stage": stage,
                        "retry_count": retry_count,
                        "correlation_id": context.correlation_id
                    }
                )

                agent = failing_agents[agent_type]
                result = agent.execute(message, {
                    "stage": stage,
                    "retry_count": retry_count,
                    "correlation_id": context.correlation_id
                })

                coordination_attempts.append({
                    "stage": stage,
                    "agent_type": agent_type,
                    "retry_count": retry_count,
                    "success": result.success,
                    "correlation_id": context.correlation_id
                })

                if not result.success and retry_count < max_retries:
                    # Log recovery action
                    dispatch_logger.log_escalation(
                        reason=f"{stage} coordination failed, retrying",
                        attempt_count=retry_count + 1,
                        metadata={
                            "escalation_type": "retry",
                            "stage": stage,
                            "max_retries": max_retries,
                            "correlation_id": context.correlation_id
                        }
                    )

                    recovery_actions.append({
                        "stage": stage,
                        "action": "retry",
                        "retry_count": retry_count + 1,
                        "correlation_id": context.correlation_id
                    })

                    # Retry with backoff
                    time.sleep(0.1 * (retry_count + 1))
                    return attempt_coordination(stage, agent_type, retry_count + 1)

                return result

        try:
            # Test coordination failure and recovery sequence
            stages = [
                ("review", AgentType.REVIEW),
                ("fix", AgentType.FIX),
                ("verify", AgentType.VERIFY)
            ]

            final_results = []

            for stage, agent_type in stages:
                result = attempt_coordination(stage, agent_type)
                final_results.append((stage, result))

                # Update workflow state based on result
                if stage == "review" and result.success:
                    workflow.transition_to(WorkflowState.FIX_REQUIRED, "review_completed")
                elif stage == "fix" and result.success:
                    workflow.transition_to(WorkflowState.VERIFICATION, "fix_completed")
                elif stage == "verify" and result.success:
                    workflow.transition_to(WorkflowState.COMPLETE, "verification_completed")

            # Verify coordination attempts and recovery
            assert len(coordination_attempts) >= 3, "Not enough coordination attempts"

            # Verify retry logic was triggered for failed attempts
            failed_attempts = [a for a in coordination_attempts if not a["success"]]
            if failed_attempts:
                assert len(recovery_actions) > 0, "No recovery actions taken for failed attempts"

                # Verify retry attempts for same stage
                retry_attempts = [a for a in coordination_attempts if a["retry_count"] > 0]
                assert len(retry_attempts) > 0, "No retry attempts found"

            # Verify eventual success or reasonable failure handling
            final_success_count = sum(1 for stage, result in final_results if result.success)
            assert final_success_count >= 1, "All coordination attempts failed"

            # Verify workflow progressed appropriately
            if final_success_count == 3:
                assert workflow.current_state == WorkflowState.COMPLETE
            elif final_success_count >= 1:
                assert workflow.current_state != WorkflowState.WORKING

        finally:
            dispatcher.cleanup()

    def test_complex_multi_agent_workflow(self, temp_dir, coordination_config, mock_agents):
        """Test complex workflow with multiple concurrent agent types and coordination"""

        dispatcher = EnhancedAgentDispatcher(config=coordination_config.to_dict())
        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("complex_workflow_test")

        # Create multiple workflows running concurrently
        workflows = {}
        workflow_results = {}

        project_configs = [
            ("project_alpha", ["alpha1.py", "alpha2.py", "alpha3.py"]),
            ("project_beta", ["beta1.py", "beta2.py"]),
            ("project_gamma", ["gamma1.py", "gamma2.py", "gamma3.py", "gamma4.py"])
        ]

        def execute_project_workflow(project_id: str, files: List[str]):
            """Execute complete workflow for a project"""
            workflow = WorkflowStateMachine(
                initial_state=WorkflowState.WORKING,
                context=WorkflowContext(
                    task_id=f"task_{project_id}",
                    session_id=f"session_{project_id}",
                    project_path=str(temp_dir / project_id)
                )
            )
            workflows[project_id] = workflow

            project_results = []

            try:
                # Step 1: Concurrent review of different file groups
                file_groups = [files[i:i+2] for i in range(0, len(files), 2)]
                review_results = []

                for i, file_group in enumerate(file_groups):
                    with dispatch_logger.operation_context(
                        operation_type=OperationType.REVIEW_PARSING,
                        session_id=f"session_{project_id}",
                        task_id=f"task_{project_id}_review_{i}",
                        file_paths=file_group
                    ) as context:

                        message = AgentMessage(
                            message_type=MessageType.DISPATCH_AGENT,
                            agent_type=AgentType.REVIEW,
                            context=MessageContext(
                                task_id=f"task_{project_id}_review_{i}",
                                parent_session=f"session_{project_id}",
                                files_modified=file_group,
                                project_path=str(temp_dir / project_id)
                            ),
                            success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                            payload={"correlation_id": context.correlation_id}
                        )

                        agent = mock_agents[AgentType.REVIEW]
                        result = agent.execute(message, {
                            "project_id": project_id,
                            "file_group": i,
                            "correlation_id": context.correlation_id
                        })
                        review_results.append(result)

                workflow.transition_to(WorkflowState.REVIEWING, "reviews_started")
                project_results.extend(review_results)

                # Step 2: Coordinated fix application
                if all(r.success for r in review_results):
                    workflow.transition_to(WorkflowState.FIX_REQUIRED, "reviews_completed")

                    with dispatch_logger.operation_context(
                        operation_type=OperationType.FIX_APPLICATION,
                        session_id=f"session_{project_id}",
                        task_id=f"task_{project_id}_fix",
                        file_paths=files
                    ) as context:

                        # Aggregate review results for fix agent
                        all_issues = []
                        for review_result in review_results:
                            issues = review_result.result_data.get("issues_found", [])
                            all_issues.extend(issues)

                        fix_message = AgentMessage(
                            message_type=MessageType.DISPATCH_AGENT,
                            agent_type=AgentType.FIX,
                            context=MessageContext(
                                task_id=f"task_{project_id}_fix",
                                parent_session=f"session_{project_id}",
                                files_modified=files,
                                project_path=str(temp_dir / project_id)
                            ),
                            success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                            payload={
                                "issues_to_fix": all_issues,
                                "review_results": [r.result_data for r in review_results],
                                "correlation_id": context.correlation_id
                            }
                        )

                        agent = mock_agents[AgentType.FIX]
                        fix_result = agent.execute(fix_message, {
                            "project_id": project_id,
                            "aggregated_reviews": True,
                            "correlation_id": context.correlation_id
                        })
                        project_results.append(fix_result)

                        if fix_result.success:
                            workflow.transition_to(WorkflowState.VERIFICATION, "fixes_applied")

                # Step 3: Final verification
                if workflow.current_state == WorkflowState.VERIFICATION:
                    with dispatch_logger.operation_context(
                        operation_type=OperationType.VERIFICATION,
                        session_id=f"session_{project_id}",
                        task_id=f"task_{project_id}_verify",
                        file_paths=files
                    ) as context:

                        verify_message = AgentMessage(
                            message_type=MessageType.DISPATCH_AGENT,
                            agent_type=AgentType.VERIFY,
                            context=MessageContext(
                                task_id=f"task_{project_id}_verify",
                                parent_session=f"session_{project_id}",
                                files_modified=files,
                                project_path=str(temp_dir / project_id)
                            ),
                            success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler"),
                            payload={"correlation_id": context.correlation_id}
                        )

                        agent = mock_agents[AgentType.VERIFY]
                        verify_result = agent.execute(verify_message, {
                            "project_id": project_id,
                            "final_verification": True,
                            "correlation_id": context.correlation_id
                        })
                        project_results.append(verify_result)

                        if verify_result.success:
                            workflow.transition_to(WorkflowState.COMPLETE, "verification_completed")

            except Exception as e:
                workflow.transition_to(WorkflowState.ERROR, "workflow_error", metadata={"error": str(e)})

            workflow_results[project_id] = project_results

        try:
            # Execute all project workflows concurrently
            threads = []
            for project_id, files in project_configs:
                thread = Thread(target=execute_project_workflow, args=(project_id, files))
                threads.append(thread)
                thread.start()

            # Wait for all workflows to complete
            for thread in threads:
                thread.join(timeout=30.0)

            # Verify all workflows completed
            assert len(workflow_results) == len(project_configs), \
                f"Not all workflows completed: {len(workflow_results)} != {len(project_configs)}"

            # Verify workflow progression
            completed_workflows = 0
            for project_id, workflow in workflows.items():
                if workflow.current_state in [WorkflowState.COMPLETE, WorkflowState.VERIFICATION]:
                    completed_workflows += 1

            assert completed_workflows >= len(project_configs) * 0.6, \
                f"Too few workflows completed successfully: {completed_workflows}"

            # Verify agent coordination across projects
            total_executions = sum(len(results) for results in workflow_results.values())
            assert total_executions >= len(project_configs) * 2, \
                f"Not enough agent executions: {total_executions}"

            # Verify concurrent execution efficiency
            all_results = [result for results in workflow_results.values() for result in results]
            if len(all_results) > 1:
                execution_times = [(r.start_time, r.end_time) for r in all_results]

                # Check for overlapping execution times (indicates concurrency)
                overlaps = 0
                for i, (start1, end1) in enumerate(execution_times):
                    for j, (start2, end2) in enumerate(execution_times[i+1:], i+1):
                        if start1 < end2 and start2 < end1:
                            overlaps += 1

                assert overlaps > 0, "No concurrent agent execution detected in complex workflow"

        finally:
            dispatcher.cleanup()
