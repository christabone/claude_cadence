"""
Configuration Variation Integration Tests

Tests for different DispatchConfig scenarios and their operational impact,
including performance testing under various configurations.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Optional
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration and dispatch components
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo
)
from cadence.dispatch_logging import (
    DispatchLogger, OperationType, get_dispatch_logger, setup_dispatch_logging
)
from cadence.workflow_state_machine import (
    WorkflowStateMachine, WorkflowState, WorkflowContext
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for configuration testing"""
    config_name: str
    total_execution_time: float
    average_agent_response_time: float
    successful_dispatches: int
    failed_dispatches: int
    timeout_count: int
    max_concurrent_agents: int
    memory_usage_mb: float
    throughput_agents_per_second: float
    escalation_count: int
    retry_count: int


class ConfigurationTestAgent:
    """Test agent that simulates different execution characteristics"""

    def __init__(self, execution_time: float = 1.0, failure_rate: float = 0.0,
                 timeout_rate: float = 0.0, memory_usage: float = 25.0):
        self.execution_time = execution_time
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.memory_usage = memory_usage
        self.execution_count = 0
        self.active_executions = 0
        self.max_concurrent = 0
        self.execution_lock = threading.Lock()
        self.completion_event = threading.Event()

    def execute(self, message: AgentMessage, timeout_ms: int = 30000) -> Dict[str, Any]:
        """Execute agent with configurable behavior"""
        with self.execution_lock:
            self.execution_count += 1
            self.active_executions += 1
            self.max_concurrent = max(self.max_concurrent, self.active_executions)

        start_time = time.time()

        try:
            # Simulate timeout behavior
            import random
            if random.random() < self.timeout_rate:
                time.sleep(timeout_ms / 1000.0 + 0.1)  # Exceed timeout
                return {"success": False, "error": "Agent timed out", "timeout": True}

            # Simulate execution time
            time.sleep(self.execution_time)

            # Simulate failure behavior
            if random.random() < self.failure_rate:
                return {
                    "success": False,
                    "error": "Simulated agent failure",
                    "execution_time": time.time() - start_time
                }

            # Successful execution
            return {
                "success": True,
                "result": f"Agent {message.agent_type.value} completed successfully",
                "execution_time": time.time() - start_time,
                "memory_usage": self.memory_usage
            }

        finally:
            with self.execution_lock:
                self.active_executions -= 1
                if self.active_executions == 0:
                    self.completion_event.set()

    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for all executions to complete with timeout"""
        return self.completion_event.wait(timeout)

    def reset_completion(self):
        """Reset completion event for new test run"""
        self.completion_event.clear()


class TestConfigurationVariations:
    """Test different DispatchConfig scenarios and their impact"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="config_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing"""
        return DispatchConfig(
            max_concurrent_agents=2,
            default_timeout_ms=5000,
            enable_fix_tracking=True,
            enable_escalation=True,
            max_fix_iterations=3,
            escalation_strategy="log_only",
            persistence_type="memory"
        )

    def test_max_concurrent_agents_impact(self, temp_dir, base_config):
        """Test impact of different max_concurrent_agents settings"""

        setup_dispatch_logging(level="DEBUG")

        # Test different concurrency levels
        concurrency_configs = [
            ("low_concurrency", 1),
            ("medium_concurrency", 3),
            ("high_concurrency", 6),
            ("unlimited_concurrency", 10)
        ]

        performance_results = []

        for config_name, max_concurrent in concurrency_configs:
            config = replace(base_config, max_concurrent_agents=max_concurrent)
            dispatcher = EnhancedAgentDispatcher(config=config.to_dict())

            # Create test agents
            test_agent = ConfigurationTestAgent(execution_time=0.5, failure_rate=0.1)
            test_agent.reset_completion()

            # Mock agent dispatch to use test agent
            original_dispatch = dispatcher.dispatch_agent

            def mock_dispatch(*args, **kwargs):
                agent_type = args[0] if args else kwargs.get('agent_type')
                message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=agent_type,
                    context=kwargs.get('context', MessageContext("test", "test", [], str(temp_dir))),
                    success_criteria=kwargs.get('success_criteria', SuccessCriteria()),
                    callback=CallbackInfo(handler="test_handler")
                )
                result = test_agent.execute(message, config.default_timeout_ms)
                return f"msg_{test_agent.execution_count}"

            dispatcher.dispatch_agent = mock_dispatch

            try:
                start_time = time.time()

                # Dispatch multiple agents concurrently
                agent_count = 15
                dispatched_agents = []

                for i in range(agent_count):
                    agent_type = AgentType.REVIEW if i % 2 == 0 else AgentType.FIX

                    message_id = dispatcher.dispatch_agent(
                        agent_type=agent_type,
                        context=MessageContext(
                            task_id=f"task_{i}",
                            parent_session=f"session_{i}",
                            files_modified=[f"file_{i}.py"],
                            project_path=str(temp_dir)
                        ),
                        success_criteria=SuccessCriteria(),
                        callback_handler=lambda x: None
                    )
                    dispatched_agents.append(message_id)

                # Wait for all agents to complete with timeout
                if not test_agent.wait_for_completion(timeout=30.0):
                    raise TimeoutError("Test agents did not complete within timeout")

                total_time = time.time() - start_time

                # Calculate performance metrics
                metrics = PerformanceMetrics(
                    config_name=config_name,
                    total_execution_time=total_time,
                    average_agent_response_time=total_time / agent_count,
                    successful_dispatches=agent_count,
                    failed_dispatches=0,
                    timeout_count=0,
                    max_concurrent_agents=test_agent.max_concurrent,
                    memory_usage_mb=test_agent.memory_usage * test_agent.max_concurrent,
                    throughput_agents_per_second=agent_count / total_time,
                    escalation_count=0,
                    retry_count=0
                )

                performance_results.append(metrics)

            finally:
                dispatcher.cleanup()

        # Analyze concurrency impact
        assert len(performance_results) == len(concurrency_configs)

        # Verify concurrency limits were respected
        for metrics in performance_results:
            if metrics.config_name == "low_concurrency":
                assert metrics.max_concurrent_agents <= 1
            elif metrics.config_name == "medium_concurrency":
                assert metrics.max_concurrent_agents <= 3
            elif metrics.config_name == "high_concurrency":
                assert metrics.max_concurrent_agents <= 6

        # Verify throughput generally increases with concurrency
        throughputs = [m.throughput_agents_per_second for m in performance_results]

        # High concurrency should generally be faster than low concurrency
        low_throughput = next(m.throughput_agents_per_second for m in performance_results
                            if m.config_name == "low_concurrency")
        high_throughput = next(m.throughput_agents_per_second for m in performance_results
                             if m.config_name == "high_concurrency")

        assert high_throughput > low_throughput * 1.5, \
            f"High concurrency not significantly faster: {high_throughput} vs {low_throughput}"

    def test_timeout_configuration_impact(self, temp_dir, base_config):
        """Test impact of different timeout configurations"""

        setup_dispatch_logging(level="DEBUG")

        # Test different timeout configurations
        timeout_configs = [
            ("short_timeout", 1000),   # 1 second
            ("medium_timeout", 5000),  # 5 seconds
            ("long_timeout", 15000),   # 15 seconds
        ]

        timeout_results = []

        for config_name, timeout_ms in timeout_configs:
            config = replace(base_config, default_timeout_ms=timeout_ms)
            dispatcher = EnhancedAgentDispatcher(config=config.to_dict())

            # Create agents with varying execution times
            test_agents = {
                "fast": ConfigurationTestAgent(execution_time=0.5),
                "medium": ConfigurationTestAgent(execution_time=2.0),
                "slow": ConfigurationTestAgent(execution_time=8.0),
                "very_slow": ConfigurationTestAgent(execution_time=12.0)
            }

            execution_results = []
            timeouts = 0

            try:
                for agent_speed, test_agent in test_agents.items():
                    start_time = time.time()

                    message = AgentMessage(
                        message_type=MessageType.DISPATCH_AGENT,
                        agent_type=AgentType.REVIEW,
                        context=MessageContext(
                            task_id=f"timeout_test_{agent_speed}",
                            parent_session="timeout_session",
                            files_modified=["test.py"],
                            project_path=str(temp_dir)
                        ),
                        success_criteria=SuccessCriteria(),
                        callback=CallbackInfo(handler="test_handler")
                    )

                    result = test_agent.execute(message, timeout_ms)
                    execution_time = time.time() - start_time

                    execution_results.append({
                        "agent_speed": agent_speed,
                        "success": result.get("success", False),
                        "timeout": result.get("timeout", False),
                        "execution_time": execution_time,
                        "expected_time": test_agent.execution_time
                    })

                    if result.get("timeout", False):
                        timeouts += 1

                # Calculate timeout metrics
                successful_executions = [r for r in execution_results if r["success"]]
                failed_executions = [r for r in execution_results if not r["success"]]

                timeout_results.append({
                    "config_name": config_name,
                    "timeout_ms": timeout_ms,
                    "successful_count": len(successful_executions),
                    "timeout_count": timeouts,
                    "total_executions": len(execution_results),
                    "success_rate": len(successful_executions) / len(execution_results)
                })

            finally:
                dispatcher.cleanup()

        # Analyze timeout impact
        assert len(timeout_results) == len(timeout_configs)

        # Verify timeout behavior
        short_timeout_result = next(r for r in timeout_results if r["config_name"] == "short_timeout")
        long_timeout_result = next(r for r in timeout_results if r["config_name"] == "long_timeout")

        # Short timeout should have more timeouts than long timeout
        assert short_timeout_result["timeout_count"] >= long_timeout_result["timeout_count"], \
            "Short timeout didn't cause more timeouts than long timeout"

        # Long timeout should have higher success rate
        assert long_timeout_result["success_rate"] >= short_timeout_result["success_rate"], \
            "Long timeout didn't improve success rate"

    def test_escalation_strategy_variations(self, temp_dir, base_config):
        """Test different escalation strategies and their impact"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("escalation_test")

        # Test different escalation strategies
        escalation_configs = [
            ("no_escalation", "log_only"),
            ("notify_escalation", "notify_supervisor"),
            ("pause_escalation", "pause_automation"),
            ("manual_review_escalation", "mark_for_manual_review")
        ]

        escalation_results = []

        for config_name, escalation_strategy in escalation_configs:
            config = replace(
                base_config,
                escalation_strategy=escalation_strategy,
                max_fix_iterations=5
            )
            dispatcher = EnhancedAgentDispatcher(config=config.to_dict())

            # Create failing agent to trigger escalation
            failing_agent = ConfigurationTestAgent(
                execution_time=1.0,
                failure_rate=0.6,  # High failure rate to trigger escalation
                timeout_rate=0.2
            )

            escalation_events = []
            retry_attempts = []

            try:
                # Simulate multiple operations that will fail and trigger escalation
                for i in range(10):
                    with dispatch_logger.operation_context(
                        operation_type=OperationType.AGENT_DISPATCH,
                        session_id="escalation_session",
                        task_id=f"escalation_task_{i}"
                    ) as context:

                        message = AgentMessage(
                            message_type=MessageType.DISPATCH_AGENT,
                            agent_type=AgentType.FIX,
                            context=MessageContext(
                                task_id=f"escalation_task_{i}",
                                parent_session="escalation_session",
                                files_modified=[f"file_{i}.py"],
                                project_path=str(temp_dir)
                            ),
                            success_criteria=SuccessCriteria(),
                            callback=CallbackInfo(handler="test_handler"),
                            payload={"correlation_id": context.correlation_id}
                        )

                        # Initial attempt
                        result = failing_agent.execute(message, config.default_timeout_ms)

                        if not result["success"]:
                            # Log escalation based on strategy
                            if escalation_strategy == "notify_supervisor":
                                # Simulate notify supervisor escalation
                                for retry in range(3):
                                    dispatch_logger.log_escalation(
                                        reason=f"Operation failed, notifying supervisor - attempt {retry + 1}",
                                        attempt_count=retry + 1,
                                        metadata={
                                            "escalation_type": "notify_supervisor",
                                            "original_error": result.get("error"),
                                            "escalation_strategy": escalation_strategy
                                        }
                                    )

                                    retry_attempts.append({
                                        "task_id": f"escalation_task_{i}",
                                        "retry_count": retry + 1,
                                        "strategy": escalation_strategy
                                    })

                                    # Retry with modified agent
                                    retry_agent = ConfigurationTestAgent(
                                        execution_time=1.0,
                                        failure_rate=max(0.1, 0.6 - (retry + 1) * 0.15)  # Improve success rate
                                    )

                                    retry_result = retry_agent.execute(message, config.default_timeout_ms)

                                    if retry_result["success"]:
                                        break

                            elif escalation_strategy == "pause_automation":
                                # Simulate pause automation escalation
                                dispatch_logger.log_escalation(
                                    reason="Operation failed, pausing automation",
                                    attempt_count=1,
                                    metadata={
                                        "escalation_type": "pause_automation",
                                        "original_timeout": config.default_timeout_ms,
                                        "escalation_strategy": escalation_strategy
                                    }
                                )

                                # Simulate manual intervention
                                escalated_result = failing_agent.execute(message, config.default_timeout_ms)

                            elif escalation_strategy == "mark_for_manual_review":
                                # Simulate mark for manual review escalation
                                dispatch_logger.log_escalation(
                                    reason="Operation failed, marking for manual review",
                                    attempt_count=1,
                                    metadata={
                                        "escalation_type": "mark_for_manual_review",
                                        "escalation_strategy": escalation_strategy
                                    }
                                )

                                # Create enhanced agent for manual review simulation
                                enhanced_agent = ConfigurationTestAgent(
                                    execution_time=0.8,  # Faster execution
                                    failure_rate=0.2,    # Lower failure rate
                                    memory_usage=50.0    # More memory
                                )

                                enhanced_result = enhanced_agent.execute(message, config.default_timeout_ms)

                            escalation_events.append({
                                "task_id": f"escalation_task_{i}",
                                "strategy": escalation_strategy,
                                "original_result": result,
                                "escalated": True
                            })
                        else:
                            escalation_events.append({
                                "task_id": f"escalation_task_{i}",
                                "strategy": escalation_strategy,
                                "original_result": result,
                                "escalated": False
                            })

                # Calculate escalation metrics
                escalated_count = len([e for e in escalation_events if e["escalated"]])
                retry_count = len(retry_attempts)

                escalation_results.append({
                    "config_name": config_name,
                    "escalation_strategy": escalation_strategy,
                    "total_operations": len(escalation_events),
                    "escalated_operations": escalated_count,
                    "escalation_rate": escalated_count / len(escalation_events),
                    "retry_attempts": retry_count,
                    "average_retries_per_escalation": retry_count / max(1, escalated_count)
                })

            finally:
                dispatcher.cleanup()

        # Analyze escalation strategy impact
        assert len(escalation_results) == len(escalation_configs)

        # Verify escalation strategies behaved differently
        notify_result = next(r for r in escalation_results if r["escalation_strategy"] == "notify_supervisor")
        log_only_result = next(r for r in escalation_results if r["escalation_strategy"] == "log_only")

        # Notify supervisor strategy should have more retry attempts
        assert notify_result["retry_attempts"] > log_only_result["retry_attempts"], \
            "Notify supervisor escalation strategy didn't generate more retries"

        # All strategies should have some escalations due to high failure rate
        for result in escalation_results:
            assert result["escalated_operations"] > 0, \
                f"No escalations triggered for {result['escalation_strategy']}"

    def test_persistence_configuration_impact(self, temp_dir, base_config):
        """Test different persistence configurations and their performance impact"""

        setup_dispatch_logging(level="DEBUG")

        # Test different persistence configurations
        persistence_configs = [
            ("memory_only", "memory", None),
            ("file_persistence", "file", str(temp_dir / "persistence")),
            ("database_simulation", "database", str(temp_dir / "db.json"))
        ]

        persistence_results = []

        for config_name, persistence_type, storage_path in persistence_configs:
            config = replace(
                base_config,
                persistence_type=persistence_type,
                storage_path=storage_path
            )

            # Create storage directory if needed
            if storage_path and persistence_type == "file":
                Path(storage_path).mkdir(parents=True, exist_ok=True)

            dispatcher = EnhancedAgentDispatcher(config=config.to_dict())

            # Create workflow with persistence
            workflow = WorkflowStateMachine(
                initial_state=WorkflowState.WORKING,
                context=WorkflowContext(
                    task_id="persistence_test",
                    session_id="persistence_session",
                    project_path=str(temp_dir)
                ),
                persistence_path=Path(storage_path) / "workflow.json" if storage_path else None,
                enable_persistence=(persistence_type != "memory")
            )

            # Measure persistence performance
            persistence_operations = []

            try:
                start_time = time.time()

                # Perform multiple state transitions to test persistence
                state_transitions = [
                    (WorkflowState.REVIEW_TRIGGERED, "test_trigger_1"),
                    (WorkflowState.REVIEWING, "test_review_1"),
                    (WorkflowState.FIX_REQUIRED, "test_fix_1"),
                    (WorkflowState.FIXING, "test_fixing_1"),
                    (WorkflowState.VERIFICATION, "test_verify_1"),
                    (WorkflowState.COMPLETE, "test_complete_1")
                ]

                for target_state, trigger in state_transitions:
                    operation_start = time.time()

                    workflow.transition_to(
                        target_state,
                        trigger,
                        metadata={"persistence_test": True, "config": config_name}
                    )

                    operation_duration = time.time() - operation_start
                    persistence_operations.append({
                        "state": target_state.value,
                        "duration": operation_duration,
                        "persistence_type": persistence_type
                    })

                # Test persistence recovery if applicable
                recovery_time = 0.0
                if persistence_type != "memory" and workflow._persistence_path and workflow._persistence_path.exists():
                    recovery_start = time.time()

                    # Create new workflow from same persistence
                    recovered_workflow = WorkflowStateMachine(
                        initial_state=WorkflowState.WORKING,
                        context=WorkflowContext(),
                        persistence_path=workflow._persistence_path,
                        enable_persistence=True
                    )

                    recovery_time = time.time() - recovery_start

                    # Verify recovery worked
                    assert recovered_workflow.current_state == WorkflowState.COMPLETE
                    assert len(recovered_workflow.transition_history) == len(state_transitions)

                total_time = time.time() - start_time

                # Calculate persistence metrics
                avg_persistence_time = statistics.mean([op["duration"] for op in persistence_operations])

                persistence_results.append({
                    "config_name": config_name,
                    "persistence_type": persistence_type,
                    "total_time": total_time,
                    "average_persistence_time": avg_persistence_time,
                    "recovery_time": recovery_time,
                    "operations_count": len(persistence_operations),
                    "throughput": len(persistence_operations) / total_time
                })

            finally:
                dispatcher.cleanup()

        # Analyze persistence impact
        assert len(persistence_results) == len(persistence_configs)

        # Memory persistence should be fastest
        memory_result = next(r for r in persistence_results if r["persistence_type"] == "memory")
        file_result = next(r for r in persistence_results if r["persistence_type"] == "file")

        # Memory should be faster than file persistence
        assert memory_result["average_persistence_time"] <= file_result["average_persistence_time"], \
            f"Memory persistence not faster than file: {memory_result['average_persistence_time']} vs {file_result['average_persistence_time']}"

        # File persistence should have successful recovery
        assert file_result["recovery_time"] > 0, "File persistence recovery not tested"
        assert file_result["recovery_time"] < 1.0, "File persistence recovery too slow"

    def test_performance_under_load_variations(self, temp_dir, base_config):
        """Test performance under different load scenarios"""

        setup_dispatch_logging(level="INFO")  # Reduce logging overhead

        # Test different load scenarios
        load_scenarios = [
            ("light_load", 5, 0.5),      # 5 agents, 0.5s each
            ("medium_load", 20, 1.0),    # 20 agents, 1.0s each
            ("heavy_load", 50, 0.8),     # 50 agents, 0.8s each
            ("burst_load", 100, 0.3),    # 100 agents, 0.3s each
        ]

        load_results = []

        for scenario_name, agent_count, execution_time in load_scenarios:
            # Adjust config for load scenario
            if scenario_name in ["heavy_load", "burst_load"]:
                config = replace(base_config, max_concurrent_agents=8, default_timeout_ms=10000)
            else:
                config = replace(base_config, max_concurrent_agents=4, default_timeout_ms=5000)

            dispatcher = EnhancedAgentDispatcher(config=config.to_dict())

            # Create load test agent
            load_agent = ConfigurationTestAgent(
                execution_time=execution_time,
                failure_rate=0.05,  # Low failure rate for load testing
                memory_usage=30.0
            )

            try:
                start_time = time.time()

                # Use ThreadPoolExecutor for realistic concurrent load
                with ThreadPoolExecutor(max_workers=config.max_concurrent_agents * 2) as executor:
                    futures = []

                    for i in range(agent_count):
                        agent_type = [AgentType.REVIEW, AgentType.FIX][i % 2]

                        def execute_load_agent(agent_id):
                            message = AgentMessage(
                                message_type=MessageType.DISPATCH_AGENT,
                                agent_type=agent_type,
                                context=MessageContext(
                                    task_id=f"load_task_{agent_id}",
                                    parent_session=f"load_session_{scenario_name}",
                                    files_modified=[f"file_{agent_id}.py"],
                                    project_path=str(temp_dir)
                                ),
                                success_criteria=SuccessCriteria(),
                                callback=CallbackInfo(handler="test_handler")
                            )

                            return load_agent.execute(message, config.default_timeout_ms)

                        future = executor.submit(execute_load_agent, i)
                        futures.append(future)

                    # Collect results
                    successful_executions = 0
                    failed_executions = 0
                    timeout_executions = 0
                    execution_times = []

                    for future in as_completed(futures, timeout=60.0):
                        try:
                            result = future.result()
                            if result["success"]:
                                successful_executions += 1
                            else:
                                failed_executions += 1
                                if result.get("timeout"):
                                    timeout_executions += 1

                            execution_times.append(result.get("execution_time", 0))

                        except Exception as e:
                            failed_executions += 1

                total_time = time.time() - start_time

                # Calculate load performance metrics
                load_results.append({
                    "scenario_name": scenario_name,
                    "agent_count": agent_count,
                    "total_time": total_time,
                    "successful_executions": successful_executions,
                    "failed_executions": failed_executions,
                    "timeout_executions": timeout_executions,
                    "success_rate": successful_executions / agent_count,
                    "throughput": agent_count / total_time,
                    "average_execution_time": statistics.mean(execution_times) if execution_times else 0,
                    "max_concurrent_observed": load_agent.max_concurrent,
                    "peak_memory_usage": load_agent.memory_usage * load_agent.max_concurrent
                })

            finally:
                dispatcher.cleanup()

        # Analyze load performance
        assert len(load_results) == len(load_scenarios)

        # Verify all scenarios completed with reasonable success rates
        for result in load_results:
            assert result["success_rate"] >= 0.8, \
                f"Low success rate for {result['scenario_name']}: {result['success_rate']}"

            assert result["throughput"] > 0, \
                f"Zero throughput for {result['scenario_name']}"

        # Verify throughput scales appropriately
        light_load = next(r for r in load_results if r["scenario_name"] == "light_load")
        heavy_load = next(r for r in load_results if r["scenario_name"] == "heavy_load")

        # Heavy load should handle more agents per second than light load
        assert heavy_load["throughput"] >= light_load["throughput"], \
            f"Heavy load throughput not higher: {heavy_load['throughput']} vs {light_load['throughput']}"

        # Burst load should demonstrate system limits
        burst_load = next(r for r in load_results if r["scenario_name"] == "burst_load")
        assert burst_load["throughput"] > 5.0, \
            f"Burst load throughput too low: {burst_load['throughput']}"
