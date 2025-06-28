"""
Dispatch Logging Integration Tests

Tests ensuring dispatch logging works correctly across the complete workflow,
with proper correlation ID tracking, performance metrics, and error handling.
"""

import pytest
import tempfile
import shutil
import time
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from threading import Thread
import io
import sys

# Import dispatch and logging components
from cadence.dispatch_logging import (
    DispatchLogger, OperationType, DispatchContext, PerformanceMetrics,
    get_dispatch_logger, setup_dispatch_logging,
    get_global_dispatch_logger
)
from cadence.workflow_state_machine import (
    WorkflowStateMachine, WorkflowState, WorkflowContext
)
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.agent_messages import AgentMessage, MessageType, AgentType, MessageContext


class LogCapture:
    """Utility class to capture and analyze log output"""

    def __init__(self):
        self.logs = []
        self.handler = None

    def __enter__(self):
        # Create custom handler to capture logs
        self.handler = logging.Handler()
        self.handler.emit = self._capture_log

        # Add to all dispatch loggers
        dispatch_logger = get_dispatch_logger("test")
        dispatch_logger.logger.addHandler(self.handler)

        global_logger = get_global_dispatch_logger()
        global_logger.logger.addHandler(self.handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            dispatch_logger = get_dispatch_logger("test")
            dispatch_logger.logger.removeHandler(self.handler)

            global_logger = get_global_dispatch_logger()
            global_logger.logger.removeHandler(self.handler)

    def _capture_log(self, record):
        """Capture log record"""
        self.logs.append({
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "timestamp": record.created,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "extra": getattr(record, '__dict__', {})
        })

    def get_logs_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get logs filtered by level"""
        return [log for log in self.logs if log["level"] == level]

    def get_logs_containing(self, text: str) -> List[Dict[str, Any]]:
        """Get logs containing specific text"""
        return [log for log in self.logs if text in log["message"]]

    def get_correlation_ids(self) -> List[str]:
        """Extract correlation IDs from logs"""
        correlation_ids = []
        for log in self.logs:
            message = log["message"]
            # Extract correlation IDs from log messages
            if "correlation_id=" in message:
                start = message.find("correlation_id=") + len("correlation_id=")
                end = message.find(" ", start)
                if end == -1:
                    end = len(message)
                correlation_id = message[start:end].strip()
                correlation_ids.append(correlation_id)
        return correlation_ids


class TestDispatchLoggingIntegration:
    """Test dispatch logging integration across workflow components"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="logging_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dispatch_config(self):
        """Create dispatch configuration"""
        return DispatchConfig(
            max_concurrent_agents=2,
            default_timeout_ms=3000,
            enable_fix_tracking=True,
            enable_escalation=True
        )

    def test_end_to_end_logging_correlation(self, temp_dir, dispatch_config):
        """Test correlation ID tracking through complete workflow"""

        # Setup logging
        setup_dispatch_logging(level="DEBUG")

        # Create components
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="logging-correlation-test",
                session_id="log-session",
                project_path=str(temp_dir)
            )
        )

        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())
        dispatch_logger = get_dispatch_logger("correlation_test")

        correlation_ids_used = []
        operation_contexts = []

        with LogCapture() as log_capture:
            try:
                # Step 1: Trigger Detection
                with dispatch_logger.operation_context(
                    operation_type=OperationType.TRIGGER_DETECTION,
                    session_id="log-session",
                    task_id="logging-correlation-test",
                    file_paths=["test.py", "utils.py"]
                ) as context:
                    correlation_ids_used.append(context.correlation_id)
                    operation_contexts.append(("trigger_detection", context))

                    dispatch_logger.log_trigger_detection(
                        trigger_type="file_change",
                        file_paths=["test.py", "utils.py"],
                        metadata={"change_count": 2}
                    )

                    workflow.transition_to(
                        WorkflowState.REVIEW_TRIGGERED,
                        "file_changes_detected",
                        metadata={"correlation_id": context.correlation_id}
                    )

                # Step 2: Agent Dispatch
                with dispatch_logger.operation_context(
                    operation_type=OperationType.AGENT_DISPATCH,
                    session_id="log-session",
                    task_id="logging-correlation-test",
                    parent_correlation_id=correlation_ids_used[0]
                ) as context:
                    correlation_ids_used.append(context.correlation_id)
                    operation_contexts.append(("agent_dispatch", context))

                    dispatch_logger.log_agent_dispatch(
                        agent_type="review",
                        agent_id="test_agent_1",
                        message_type="review_request"
                    )

                    workflow.transition_to(
                        WorkflowState.REVIEWING,
                        "review_agent_dispatched",
                        metadata={"correlation_id": context.correlation_id}
                    )

                # Step 3: Review Parsing
                with dispatch_logger.operation_context(
                    operation_type=OperationType.REVIEW_PARSING,
                    session_id="log-session",
                    task_id="logging-correlation-test",
                    parent_correlation_id=correlation_ids_used[1]
                ) as context:
                    correlation_ids_used.append(context.correlation_id)
                    operation_contexts.append(("review_parsing", context))

                    dispatch_logger.log_review_parsing(
                        issues_found=2,
                        processing_time_ms=1500
                    )

                    workflow.transition_to(
                        WorkflowState.FIX_REQUIRED,
                        "issues_found",
                        metadata={"correlation_id": context.correlation_id}
                    )

                # Step 4: Fix Application
                with dispatch_logger.operation_context(
                    operation_type=OperationType.FIX_APPLICATION,
                    session_id="log-session",
                    task_id="logging-correlation-test",
                    parent_correlation_id=correlation_ids_used[2]
                ) as context:
                    correlation_ids_used.append(context.correlation_id)
                    operation_contexts.append(("fix_application", context))

                    dispatch_logger.log_fix_application(
                        fixes_applied=2,
                        verification_status="success"
                    )

                    workflow.transition_to(
                        WorkflowState.VERIFICATION,
                        "fixes_applied",
                        metadata={"correlation_id": context.correlation_id}
                    )

                # Step 5: Verification
                with dispatch_logger.operation_context(
                    operation_type=OperationType.VERIFICATION,
                    session_id="log-session",
                    task_id="logging-correlation-test",
                    parent_correlation_id=correlation_ids_used[3]
                ) as context:
                    correlation_ids_used.append(context.correlation_id)
                    operation_contexts.append(("verification", context))

                    # Using log_operation_complete for verification since there's no specific verification method
                    dispatch_logger.log_operation_complete(context, 3000)

                    workflow.transition_to(
                        WorkflowState.COMPLETE,
                        "verification_successful",
                        metadata={"correlation_id": context.correlation_id}
                    )

            finally:
                dispatcher.cleanup()

        # Analyze captured logs
        all_logs = log_capture.logs
        assert len(all_logs) > 0, "No logs were captured"

        # Verify correlation ID usage
        assert len(correlation_ids_used) == 5
        assert len(set(correlation_ids_used)) == 5  # All unique

        # Verify parent-child correlation relationships
        for i, (operation_name, context) in enumerate(operation_contexts[1:], 1):
            assert context.parent_correlation_id == correlation_ids_used[i-1]

        # Verify correlation IDs appear in logs
        logged_correlation_ids = log_capture.get_correlation_ids()
        for correlation_id in correlation_ids_used:
            assert correlation_id in logged_correlation_ids, f"Correlation ID {correlation_id} not found in logs"

        # Verify operation-specific logs
        trigger_logs = log_capture.get_logs_containing("TRIGGER_DETECTION")
        agent_logs = log_capture.get_logs_containing("AGENT_DISPATCH")
        review_logs = log_capture.get_logs_containing("REVIEW_PARSING")
        fix_logs = log_capture.get_logs_containing("FIX_APPLICATION")
        verify_logs = log_capture.get_logs_containing("VERIFICATION")

        assert len(trigger_logs) > 0, "No trigger detection logs found"
        assert len(agent_logs) > 0, "No agent dispatch logs found"
        assert len(review_logs) > 0, "No review parsing logs found"
        assert len(fix_logs) > 0, "No fix application logs found"
        assert len(verify_logs) > 0, "No verification logs found"

    def test_performance_metrics_logging_integration(self, temp_dir, dispatch_config):
        """Test performance metrics collection and logging throughout workflow"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("performance_test")

        # Track performance metrics
        performance_data = []

        with LogCapture() as log_capture:
            operations = [
                (OperationType.TRIGGER_DETECTION, 0.1),
                (OperationType.AGENT_DISPATCH, 0.05),
                (OperationType.REVIEW_PARSING, 1.5),
                (OperationType.FIX_APPLICATION, 2.0),
                (OperationType.VERIFICATION, 3.0)
            ]

            for operation_type, duration_seconds in operations:
                with dispatch_logger.operation_context(
                    operation_type=operation_type,
                    session_id="perf-session",
                    task_id="performance-test"
                ) as context:

                    start_time = time.time()

                    # Simulate operation work
                    time.sleep(duration_seconds)

                    end_time = time.time()
                    actual_duration_ms = (end_time - start_time) * 1000

                    # Log performance metrics
                    metrics = PerformanceMetrics(
                        operation_type=operation_type,
                        duration_ms=actual_duration_ms,
                        memory_usage_mb=50.5,
                        cpu_usage_percent=25.0,
                        correlation_id=context.correlation_id
                    )
                    dispatch_logger.log_performance_metrics(metrics)

                    performance_data.append({
                        "operation": operation_type.value,
                        "expected_ms": duration_seconds * 1000,
                        "actual_ms": actual_duration_ms,
                        "correlation_id": context.correlation_id
                    })

        # Analyze performance logs
        performance_logs = log_capture.get_logs_containing("PERFORMANCE")
        assert len(performance_logs) >= 5, f"Expected at least 5 performance logs, got {len(performance_logs)}"

        # Verify timing accuracy
        for perf_data in performance_data:
            expected = perf_data["expected_ms"]
            actual = perf_data["actual_ms"]
            tolerance = 50  # 50ms tolerance

            assert abs(actual - expected) < tolerance, \
                f"Performance timing off for {perf_data['operation']}: expected ~{expected}ms, got {actual}ms"

        # Verify performance data in logs
        for log in performance_logs:
            assert "duration_ms=" in log["message"]
            assert "memory_usage_mb=" in log["message"]
            assert "cpu_usage_percent=" in log["message"]

    def test_error_logging_integration(self, temp_dir, dispatch_config):
        """Test error logging and recovery throughout workflow"""

        setup_dispatch_logging(level="DEBUG")
        dispatch_logger = get_dispatch_logger("error_test")

        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="error-test",
                session_id="error-session",
                project_path=str(temp_dir)
            )
        )

        with LogCapture() as log_capture:
            # Test 1: Agent dispatch error
            with dispatch_logger.operation_context(
                operation_type=OperationType.AGENT_DISPATCH,
                session_id="error-session",
                task_id="error-test"
            ) as context:

                # Log dispatch error
                dispatch_error = Exception("Failed to connect to agent service")
                dispatch_logger.log_error(dispatch_error, context)

                workflow.transition_to(
                    WorkflowState.ERROR,
                    "agent_dispatch_failed",
                    metadata={
                        "error": "Failed to connect to agent service",
                        "correlation_id": context.correlation_id
                    }
                )

            # Test 2: Error recovery
            with dispatch_logger.operation_context(
                operation_type=OperationType.ESCALATION,
                session_id="error-session",
                task_id="error-test"
            ) as context:

                dispatch_logger.log_escalation(
                    reason="Agent dispatch failed, retrying with different configuration",
                    attempt_count=1,
                    metadata={
                        "escalation_type": "retry",
                        "original_error": "Failed to connect to agent service",
                        "correlation_id": context.correlation_id
                    }
                )

                # Recovery transition
                workflow.transition_to(
                    WorkflowState.REVIEW_TRIGGERED,
                    "retry_after_error",
                    metadata={"correlation_id": context.correlation_id}
                )

            # Test 3: Fix application error
            workflow.transition_to(WorkflowState.REVIEWING, "retry_successful")
            workflow.transition_to(WorkflowState.FIX_REQUIRED, "issues_found")

            with dispatch_logger.operation_context(
                operation_type=OperationType.FIX_APPLICATION,
                session_id="error-session",
                task_id="error-test"
            ) as context:

                fix_error = SyntaxError("Invalid syntax in generated fix")
                dispatch_logger.log_error(fix_error, context)

                workflow.transition_to(
                    WorkflowState.FIX_REQUIRED,  # Go back to fix required
                    "fix_failed",
                    metadata={
                        "error": "Fix introduced syntax error",
                        "correlation_id": context.correlation_id
                    }
                )

        # Analyze error logs
        error_logs = log_capture.get_logs_by_level("ERROR")
        assert len(error_logs) >= 2, f"Expected at least 2 error logs, got {len(error_logs)}"

        # Verify specific error types
        dispatch_errors = log_capture.get_logs_containing("AgentDispatchError")
        fix_errors = log_capture.get_logs_containing("SyntaxError")
        escalation_logs = log_capture.get_logs_containing("ESCALATION")

        assert len(dispatch_errors) > 0, "No agent dispatch errors logged"
        assert len(fix_errors) > 0, "No fix errors logged"
        assert len(escalation_logs) > 0, "No escalation logs found"

        # Verify error recovery workflow
        assert workflow.current_state == WorkflowState.FIX_REQUIRED

        # Verify error metadata preservation
        for error_log in error_logs:
            message = error_log["message"]
            assert "correlation_id=" in message or "traceback" in message.lower()

    def test_concurrent_logging_thread_safety(self, temp_dir, dispatch_config):
        """Test thread safety of dispatch logging with concurrent operations"""

        setup_dispatch_logging(level="DEBUG")

        # Shared data structures
        thread_logs = {}
        correlation_ids = {}

        def worker_thread(thread_id: int, operations_count: int):
            """Worker thread that performs logging operations"""
            dispatch_logger = get_dispatch_logger(f"thread_{thread_id}")
            thread_logs[thread_id] = []
            correlation_ids[thread_id] = []

            for i in range(operations_count):
                with dispatch_logger.operation_context(
                    operation_type=OperationType.AGENT_DISPATCH,
                    session_id=f"thread_{thread_id}_session",
                    task_id=f"thread_{thread_id}_task_{i}"
                ) as context:

                    correlation_ids[thread_id].append(context.correlation_id)

                    # Perform various logging operations
                    dispatch_logger.log_agent_dispatch(
                        agent_type="review",
                        agent_id=f"agent_{thread_id}_{i}",
                        message_type="agent_request"
                    )

                    # Add some timing variation
                    time.sleep(0.01 * (thread_id + 1))

                    # Create performance metrics object first
                    metrics = PerformanceMetrics(
                        operation_type=OperationType.AGENT_DISPATCH,
                        duration_ms=10.0 * (thread_id + 1),
                        memory_usage_mb=25.0,
                        cpu_usage_percent=15.0,
                        correlation_id=context.correlation_id
                    )
                    dispatch_logger.log_performance_metrics(metrics)

                    thread_logs[thread_id].append({
                        "operation": i,
                        "correlation_id": context.correlation_id,
                        "thread_id": thread_id
                    })

        with LogCapture() as log_capture:
            # Start multiple worker threads
            threads = []
            thread_count = 3
            operations_per_thread = 5

            for thread_id in range(thread_count):
                thread = Thread(
                    target=worker_thread,
                    args=(thread_id, operations_per_thread)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10.0)

        # Verify thread safety
        total_expected_logs = thread_count * operations_per_thread

        # Verify all threads completed
        assert len(thread_logs) == thread_count
        assert len(correlation_ids) == thread_count

        # Verify each thread logged expected number of operations
        for thread_id in range(thread_count):
            assert len(thread_logs[thread_id]) == operations_per_thread
            assert len(correlation_ids[thread_id]) == operations_per_thread

        # Verify correlation ID uniqueness across threads
        all_correlation_ids = []
        for thread_id in range(thread_count):
            all_correlation_ids.extend(correlation_ids[thread_id])

        assert len(set(all_correlation_ids)) == len(all_correlation_ids), \
            "Duplicate correlation IDs found across threads"

        # Verify logs were captured
        captured_logs = log_capture.logs
        assert len(captured_logs) >= total_expected_logs * 2, \
            f"Expected at least {total_expected_logs * 2} logs, got {len(captured_logs)}"

        # Verify thread-specific data integrity
        for thread_id in range(thread_count):
            thread_specific_logs = [
                log for log in captured_logs
                if f"thread_{thread_id}" in log["message"]
            ]
            assert len(thread_specific_logs) > 0, f"No logs found for thread {thread_id}"

    def test_logging_configuration_and_levels(self, temp_dir):
        """Test logging configuration and level filtering"""

        # Test different log levels
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level in log_levels:
            setup_dispatch_logging(level=level)
            dispatch_logger = get_dispatch_logger(f"level_test_{level.lower()}")

            with LogCapture() as log_capture:
                with dispatch_logger.operation_context(
                    operation_type=OperationType.TRIGGER_DETECTION,
                    session_id="level_test_session",
                    task_id="level_test_task"
                ) as context:

                    # Log at different levels
                    dispatch_logger.logger.debug(f"Debug message for {level}")
                    dispatch_logger.logger.info(f"Info message for {level}")
                    dispatch_logger.logger.warning(f"Warning message for {level}")
                    dispatch_logger.logger.error(f"Error message for {level}")

                    # Use specific logging methods
                    dispatch_logger.log_trigger_detection(
                        trigger_type="test",
                        file_paths=["test.py"],
                        metadata={"test_level": level}
                    )

            # Verify level filtering
            captured_logs = log_capture.logs

            if level == "DEBUG":
                assert len(captured_logs) >= 4  # All levels should be captured
            elif level == "INFO":
                debug_logs = [log for log in captured_logs if log["level"] == "DEBUG"]
                assert len(debug_logs) == 0  # Debug should be filtered out
            elif level == "WARNING":
                low_level_logs = [
                    log for log in captured_logs
                    if log["level"] in ["DEBUG", "INFO"]
                ]
                assert len(low_level_logs) == 0  # Debug and Info should be filtered
            elif level == "ERROR":
                error_only_logs = [
                    log for log in captured_logs
                    if log["level"] == "ERROR"
                ]
                assert len(error_only_logs) > 0  # Only error logs should remain
