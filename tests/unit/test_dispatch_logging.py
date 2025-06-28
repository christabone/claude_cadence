"""
Unit tests for Dispatch Logging System
"""

import pytest
import time
import uuid
import logging
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cadence.dispatch_logging import (
    DispatchLogger, OperationType, DispatchContext, PerformanceMetrics, ErrorContext,
    setup_dispatch_logging, get_dispatch_logger, get_global_dispatch_logger,
    log_operation_start, log_performance, log_error_with_context
)


class TestOperationType:
    """Test OperationType enum"""

    def test_operation_types(self):
        """Test operation type values"""
        assert OperationType.TRIGGER_DETECTION == "trigger_detection"
        assert OperationType.AGENT_DISPATCH == "agent_dispatch"
        assert OperationType.REVIEW_PARSING == "review_parsing"
        assert OperationType.FIX_APPLICATION == "fix_application"
        assert OperationType.VERIFICATION == "verification"
        assert OperationType.ESCALATION == "escalation"
        assert OperationType.WORKFLOW_TRANSITION == "workflow_transition"
        assert OperationType.ORCHESTRATION == "orchestration"


class TestDispatchContext:
    """Test DispatchContext dataclass"""

    def test_default_context(self):
        """Test default context creation"""
        context = DispatchContext()

        assert context.correlation_id is not None
        assert len(context.correlation_id) == 36  # UUID length
        assert context.operation_type is None
        assert context.session_id is None
        assert context.task_id is None
        assert context.agent_id is None
        assert context.file_paths == []
        assert context.metadata == {}
        assert context.start_time is None
        assert context.parent_correlation_id is None

    def test_context_with_values(self):
        """Test context with specific values"""
        file_paths = ["file1.py", "file2.py"]
        metadata = {"key": "value"}

        context = DispatchContext(
            operation_type=OperationType.REVIEW_PARSING,
            session_id="session-123",
            task_id="task-456",
            agent_id="agent-789",
            file_paths=file_paths,
            metadata=metadata,
            start_time=time.time()
        )

        assert context.operation_type == OperationType.REVIEW_PARSING
        assert context.session_id == "session-123"
        assert context.task_id == "task-456"
        assert context.agent_id == "agent-789"
        assert context.file_paths == file_paths
        assert context.metadata == metadata
        assert context.start_time is not None

    def test_context_to_dict(self):
        """Test context to dictionary conversion"""
        context = DispatchContext(
            operation_type=OperationType.AGENT_DISPATCH,
            session_id="session-123",
            file_paths=["test.py"]
        )

        dict_repr = context.to_dict()

        assert dict_repr["operation_type"] == OperationType.AGENT_DISPATCH
        assert dict_repr["session_id"] == "session-123"
        assert dict_repr["file_paths"] == ["test.py"]
        assert "correlation_id" in dict_repr
        # None values should be excluded
        assert "task_id" not in dict_repr
        assert "agent_id" not in dict_repr


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_basic_metrics(self):
        """Test basic performance metrics"""
        metrics = PerformanceMetrics(
            operation_type=OperationType.REVIEW_PARSING,
            duration_ms=150.5,
            files_processed=3,
            issues_found=5
        )

        assert metrics.operation_type == OperationType.REVIEW_PARSING
        assert metrics.duration_ms == 150.5
        assert metrics.files_processed == 3
        assert metrics.issues_found == 5
        assert metrics.memory_usage_mb is None
        assert metrics.cpu_usage_percent is None
        assert metrics.fixes_applied == 0
        assert metrics.correlation_id == ""

    def test_metrics_to_dict(self):
        """Test metrics to dictionary conversion"""
        metrics = PerformanceMetrics(
            operation_type=OperationType.AGENT_DISPATCH,
            duration_ms=250.0,
            memory_usage_mb=128.5,
            cpu_usage_percent=45.2,
            correlation_id="test-correlation-id"
        )

        dict_repr = metrics.to_dict()

        assert dict_repr["operation_type"] == OperationType.AGENT_DISPATCH
        assert dict_repr["duration_ms"] == 250.0
        assert dict_repr["memory_usage_mb"] == 128.5
        assert dict_repr["cpu_usage_percent"] == 45.2
        assert dict_repr["correlation_id"] == "test-correlation-id"


class TestErrorContext:
    """Test ErrorContext dataclass"""

    def test_basic_error_context(self):
        """Test basic error context"""
        error_context = ErrorContext(
            error_type="ValueError",
            error_message="Test error message",
            operation_type=OperationType.FIX_APPLICATION,
            correlation_id="error-correlation-id"
        )

        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test error message"
        assert error_context.operation_type == OperationType.FIX_APPLICATION
        assert error_context.correlation_id == "error-correlation-id"
        assert error_context.file_paths == []
        assert error_context.stack_trace is None
        assert error_context.recovery_attempted is False
        assert error_context.recovery_successful is False
        assert error_context.metadata == {}

    def test_error_context_to_dict(self):
        """Test error context to dictionary conversion"""
        error_context = ErrorContext(
            error_type="RuntimeError",
            error_message="Runtime error occurred",
            operation_type=OperationType.ESCALATION,
            correlation_id="error-id",
            file_paths=["error_file.py"],
            stack_trace="Traceback...",
            recovery_attempted=True,
            recovery_successful=False,
            metadata={"attempt": 3}
        )

        dict_repr = error_context.to_dict()

        assert dict_repr["error_type"] == "RuntimeError"
        assert dict_repr["error_message"] == "Runtime error occurred"
        assert dict_repr["operation_type"] == OperationType.ESCALATION
        assert dict_repr["correlation_id"] == "error-id"
        assert dict_repr["file_paths"] == ["error_file.py"]
        assert dict_repr["stack_trace"] == "Traceback..."
        assert dict_repr["recovery_attempted"] is True
        assert dict_repr["recovery_successful"] is False
        assert dict_repr["metadata"] == {"attempt": 3}


class TestDispatchLogger:
    """Test DispatchLogger class"""

    def setup_method(self):
        """Set up test environment"""
        self.mock_logger = Mock(spec=logging.Logger)
        self.dispatch_logger = DispatchLogger("test", self.mock_logger)

    def test_logger_initialization(self):
        """Test logger initialization"""
        assert self.dispatch_logger.name == "test"
        assert self.dispatch_logger.logger == self.mock_logger
        assert len(self.dispatch_logger.operation_times) == 0
        assert len(self.dispatch_logger.operation_counts) == 0
        assert len(self.dispatch_logger.error_counts) == 0
        assert len(self.dispatch_logger.recent_errors) == 0

    def test_context_management(self):
        """Test context setting and clearing"""
        context = DispatchContext(
            operation_type=OperationType.TRIGGER_DETECTION,
            session_id="test-session"
        )

        # Initially no context
        assert self.dispatch_logger.current_context is None

        # Set context
        self.dispatch_logger.set_context(context)
        assert self.dispatch_logger.current_context == context

        # Clear context
        self.dispatch_logger.clear_context()
        assert self.dispatch_logger.current_context is None

    def test_operation_context_manager(self):
        """Test operation context manager"""
        file_paths = ["test1.py", "test2.py"]

        with self.dispatch_logger.operation_context(
            operation_type=OperationType.REVIEW_PARSING,
            session_id="test-session",
            task_id="test-task",
            file_paths=file_paths,
            custom_metadata="test"
        ) as context:

            # Context should be set
            assert self.dispatch_logger.current_context == context
            assert context.operation_type == OperationType.REVIEW_PARSING
            assert context.session_id == "test-session"
            assert context.task_id == "test-task"
            assert context.file_paths == file_paths
            assert context.metadata["custom_metadata"] == "test"
            assert context.start_time is not None

            # Operation count should be tracked
            assert self.dispatch_logger.operation_counts[OperationType.REVIEW_PARSING] == 1

        # Context should be cleared after exiting
        assert self.dispatch_logger.current_context is None

    def test_operation_context_with_exception(self):
        """Test operation context manager with exception"""
        try:
            with self.dispatch_logger.operation_context(
                operation_type=OperationType.AGENT_DISPATCH,
                session_id="test-session"
            ) as context:

                assert self.dispatch_logger.current_context == context

                # Raise an exception
                raise ValueError("Test exception")

        except ValueError:
            pass  # Expected

        # Context should still be cleared even after exception
        assert self.dispatch_logger.current_context is None

        # Operation should still be counted
        assert self.dispatch_logger.operation_counts[OperationType.AGENT_DISPATCH] == 1

    def test_format_message_without_context(self):
        """Test message formatting without context"""
        message = self.dispatch_logger._format_message("Test message")
        assert message == "Test message"

    def test_format_message_with_context(self):
        """Test message formatting with context"""
        context = DispatchContext(
            operation_type=OperationType.VERIFICATION,
            session_id="test-session-12345678",
            task_id="test-task",
            agent_id="agent-12345678",
            file_paths=["file1.py", "file2.py"]
        )

        self.dispatch_logger.set_context(context)

        message = self.dispatch_logger._format_message("Test message")

        # Should contain context information
        assert "correlation_id=" in message
        assert "op=verification" in message
        assert "session=test-ses" in message  # Truncated
        assert "task=test-task" in message
        assert "agent=agent-12" in message  # Truncated
        assert "files=2" in message
        assert "Test message" in message

    def test_format_message_with_extra_data(self):
        """Test message formatting with extra data"""
        context = DispatchContext(operation_type=OperationType.ESCALATION)
        self.dispatch_logger.set_context(context)

        extra_data = {"duration_ms": "150.5", "issues_found": 3}

        message = self.dispatch_logger._format_message("Test message", extra_data)

        assert "duration_ms=150.5" in message
        assert "issues_found=3" in message
        assert "Test message" in message

    def test_log_trigger_detection(self):
        """Test trigger detection logging"""
        file_paths = ["file1.py", "file2.py"]
        metadata = {"source": "test", "confidence": 0.9}

        self.dispatch_logger.log_trigger_detection("code_review", file_paths, metadata)

        # Should have called info log
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "Detected code_review trigger for 2 files" in call_args

    def test_log_agent_dispatch(self):
        """Test agent dispatch logging"""
        self.dispatch_logger.log_agent_dispatch("fix", "agent-123", "FIX_REQUIRED")

        # Should have called info log
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "Dispatching fix agent with FIX_REQUIRED message" in call_args

    def test_log_review_parsing(self):
        """Test review parsing logging"""
        self.dispatch_logger.log_review_parsing(5, 150.5)

        # Should have called info log
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "Parsed review results: 5 issues found" in call_args

    def test_log_fix_application(self):
        """Test fix application logging"""
        self.dispatch_logger.log_fix_application(3, "success")

        # Should have called info log
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "Applied 3 fixes, verification: success" in call_args

    def test_log_workflow_transition(self):
        """Test workflow transition logging"""
        # Mock the integration log method
        self.mock_logger.integration = Mock()

        self.dispatch_logger.log_workflow_transition("WORKING", "REVIEWING", "code_review_triggered")

        # Should have called integration log
        self.mock_logger.integration.assert_called_once()
        call_args = self.mock_logger.integration.call_args[0][0]
        assert "Workflow transition: WORKING -> REVIEWING (code_review_triggered)" in call_args

    def test_log_escalation(self):
        """Test escalation logging"""
        metadata = {"task_id": "test-task", "attempts": 5}

        self.dispatch_logger.log_escalation("Max attempts exceeded", 5, metadata)

        # Should have called warning log
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]
        assert "Escalating after 5 attempts: Max attempts exceeded" in call_args

    def test_log_error_tracking(self):
        """Test error logging and tracking"""
        test_error = ValueError("Test error")
        context = DispatchContext(
            operation_type=OperationType.FIX_APPLICATION,
            correlation_id="test-correlation"
        )

        # Initially no errors
        assert len(self.dispatch_logger.recent_errors) == 0
        assert len(self.dispatch_logger.error_counts) == 0

        # Log error
        self.dispatch_logger.log_error(test_error, context)

        # Should track error
        assert len(self.dispatch_logger.recent_errors) == 1
        assert self.dispatch_logger.error_counts["ValueError"] == 1

        error_context = self.dispatch_logger.recent_errors[0]
        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test error"
        assert error_context.operation_type == OperationType.FIX_APPLICATION
        assert error_context.correlation_id == "test-correlation"

        # Should have called error log
        self.mock_logger.error.assert_called_once()

    def test_log_performance_metrics(self):
        """Test performance metrics logging"""
        # Mock the performance log method
        self.mock_logger.performance = Mock()

        metrics = PerformanceMetrics(
            operation_type=OperationType.AGENT_DISPATCH,
            duration_ms=250.5,
            files_processed=3,
            issues_found=5,
            correlation_id="test-correlation"
        )

        self.dispatch_logger.log_performance_metrics(metrics)

        # Should have called performance log
        self.mock_logger.performance.assert_called_once()
        call_args = self.mock_logger.performance.call_args[0][0]
        assert "Performance: agent_dispatch took 250.50ms" in call_args

    def test_debug_state(self):
        """Test debug state logging"""
        # Mock the trace log method
        self.mock_logger.trace = Mock()

        state_data = {
            "active_agents": 2,
            "max_concurrent": 5,
            "complex_data": {"nested": {"data": "value"}},
            "long_string": "x" * 150  # Test truncation
        }

        self.dispatch_logger.debug_state("agent_dispatcher", state_data)

        # Should have called trace log
        self.mock_logger.trace.assert_called_once()
        call_args = self.mock_logger.trace.call_args[0][0]
        assert "Debug state for agent_dispatcher" in call_args

    def test_get_operation_statistics(self):
        """Test operation statistics"""
        # Add some operations and errors
        self.dispatch_logger.operation_counts[OperationType.REVIEW_PARSING] = 5
        self.dispatch_logger.operation_counts[OperationType.AGENT_DISPATCH] = 3
        self.dispatch_logger.error_counts["ValueError"] = 2
        self.dispatch_logger.error_counts["RuntimeError"] = 1

        # Add some recent errors
        for i in range(3):
            error_context = ErrorContext(
                error_type="TestError",
                error_message=f"Error {i}",
                operation_type=OperationType.VERIFICATION,
                correlation_id=f"correlation-{i}"
            )
            self.dispatch_logger.recent_errors.append(error_context)

        stats = self.dispatch_logger.get_operation_statistics()

        assert stats["operation_counts"][OperationType.REVIEW_PARSING] == 5
        assert stats["operation_counts"][OperationType.AGENT_DISPATCH] == 3
        assert stats["error_counts"]["ValueError"] == 2
        assert stats["error_counts"]["RuntimeError"] == 1
        assert stats["total_operations"] == 8
        assert stats["total_errors"] == 3
        assert len(stats["recent_errors"]) == 3


class TestModuleFunctions:
    """Test module-level convenience functions"""

    def test_setup_dispatch_logging(self):
        """Test dispatch logging setup"""
        logger = setup_dispatch_logging(level=logging.INFO)

        assert isinstance(logger, DispatchLogger)
        assert logger.name == "cadence.dispatch"

    def test_get_dispatch_logger(self):
        """Test getting component-specific dispatch logger"""
        logger = get_dispatch_logger("test_component")

        assert isinstance(logger, DispatchLogger)
        assert logger.name == "cadence.dispatch.test_component"

    def test_get_global_dispatch_logger(self):
        """Test getting global dispatch logger"""
        logger1 = get_global_dispatch_logger()
        logger2 = get_global_dispatch_logger()

        # Should return the same instance
        assert logger1 is logger2
        assert isinstance(logger1, DispatchLogger)

    @patch('cadence.dispatch_logging.get_global_dispatch_logger')
    def test_log_operation_start(self, mock_get_logger):
        """Test convenience function for logging operation start"""
        mock_logger = Mock()
        # Mock the context manager
        mock_context = Mock()
        mock_logger.operation_context.return_value.__enter__ = Mock(return_value=mock_context)
        mock_logger.operation_context.return_value.__exit__ = Mock(return_value=None)
        mock_get_logger.return_value = mock_logger

        log_operation_start(OperationType.TRIGGER_DETECTION, session_id="test")

        mock_get_logger.assert_called_once()
        mock_logger.operation_context.assert_called_once()

    @patch('cadence.dispatch_logging.get_global_dispatch_logger')
    def test_log_performance(self, mock_get_logger):
        """Test convenience function for logging performance"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_performance(OperationType.REVIEW_PARSING, 150.5, files_processed=3)

        mock_get_logger.assert_called_once()
        mock_logger.log_performance_metrics.assert_called_once()

    @patch('cadence.dispatch_logging.get_global_dispatch_logger')
    def test_log_error_with_context(self, mock_get_logger):
        """Test convenience function for logging error with context"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        test_error = ValueError("Test error")

        log_error_with_context(test_error, OperationType.ESCALATION, task_id="test")

        mock_get_logger.assert_called_once()
        mock_logger.set_context.assert_called_once()
        mock_logger.log_error.assert_called_once()
        mock_logger.clear_context.assert_called_once()


class TestCustomLogLevels:
    """Test custom log levels"""

    def test_custom_levels_added(self):
        """Test that custom log levels are properly added"""
        mock_logger = Mock(spec=logging.Logger)
        dispatch_logger = DispatchLogger("test", mock_logger)

        # Check that custom methods are added
        assert hasattr(mock_logger, 'trace')
        assert hasattr(mock_logger, 'performance')
        assert hasattr(mock_logger, 'operation')
        assert hasattr(mock_logger, 'integration')

    def test_custom_level_constants(self):
        """Test custom log level constants"""
        # These should be added to the logging module
        assert logging.getLevelName(5) == "TRACE"
        assert logging.getLevelName(25) == "PERFORMANCE"
        assert logging.getLevelName(22) == "OPERATION"
        assert logging.getLevelName(35) == "INTEGRATION"


class TestThreadSafety:
    """Test thread safety of dispatch logging"""

    def test_thread_local_context(self):
        """Test that context is thread-local"""
        import threading

        logger = DispatchLogger("test")

        context1 = DispatchContext(operation_type=OperationType.TRIGGER_DETECTION)
        context2 = DispatchContext(operation_type=OperationType.AGENT_DISPATCH)

        contexts = {}

        def set_and_get_context(context, thread_id):
            logger.set_context(context)
            contexts[thread_id] = logger.current_context

        # Create two threads with different contexts
        thread1 = threading.Thread(target=set_and_get_context, args=(context1, 1))
        thread2 = threading.Thread(target=set_and_get_context, args=(context2, 2))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Each thread should have its own context
        assert contexts[1].operation_type == OperationType.TRIGGER_DETECTION
        assert contexts[2].operation_type == OperationType.AGENT_DISPATCH
        assert contexts[1] != contexts[2]


class TestErrorRecovery:
    """Test error recovery and handling"""

    def test_error_count_limits(self):
        """Test error count limits"""
        logger = DispatchLogger("test")

        # Add more than 100 errors
        for i in range(105):
            error = ValueError(f"Error {i}")
            context = DispatchContext(operation_type=OperationType.VERIFICATION)
            logger.log_error(error, context)

        # Should only keep last 100 errors
        assert len(logger.recent_errors) == 100

        # Should still count all errors
        assert logger.error_counts["ValueError"] == 105

        # Recent errors should be the last ones
        assert logger.recent_errors[-1].error_message == "Error 104"
        assert logger.recent_errors[0].error_message == "Error 5"
