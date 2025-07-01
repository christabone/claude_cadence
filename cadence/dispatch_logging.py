"""
Comprehensive Logging and Monitoring for Claude Cadence Dispatch System

This module provides structured logging with correlation IDs, performance metrics,
and specialized logging for dispatch system operations including trigger detection,
agent dispatch, review parsing, fix application, and verification.
"""

import logging
import time
import uuid
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum

from .log_utils import ColoredFormatter, Colors


class OperationType(str, Enum):
    """Types of dispatch operations for structured logging"""
    TRIGGER_DETECTION = "trigger_detection"
    AGENT_DISPATCH = "agent_dispatch"
    REVIEW_PARSING = "review_parsing"
    FIX_APPLICATION = "fix_application"
    VERIFICATION = "verification"
    ESCALATION = "escalation"
    WORKFLOW_TRANSITION = "workflow_transition"
    ORCHESTRATION = "orchestration"


class LogLevel(str, Enum):
    """Custom log levels for dispatch operations"""
    TRACE = "TRACE"  # Detailed debug info
    PERFORMANCE = "PERFORMANCE"  # Performance metrics
    OPERATION = "OPERATION"  # High-level operations
    INTEGRATION = "INTEGRATION"  # System integration points


@dataclass
class DispatchContext:
    """Context information for dispatch operations"""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: Optional[OperationType] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    file_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    parent_correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_type: OperationType
    duration_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    files_processed: int = 0
    issues_found: int = 0
    fixes_applied: int = 0
    correlation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)


@dataclass
class ErrorContext:
    """Error context for detailed error tracking"""
    error_type: str
    error_message: str
    operation_type: OperationType
    correlation_id: str
    file_paths: List[str] = field(default_factory=list)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)


class DispatchLogger:
    """
    Enhanced logger for dispatch system with structured logging, correlation IDs,
    and performance metrics tracking.
    """

    # Thread-local storage for context
    _local = threading.local()

    def __init__(self, name: str, base_logger: Optional[logging.Logger] = None):
        """Initialize dispatch logger"""
        self.name = name
        self.logger = base_logger or logging.getLogger(name)

        # Add custom log levels
        self._add_custom_levels()

        # Performance tracking
        self.operation_times: Dict[str, float] = {}
        self.operation_counts: Dict[OperationType, int] = {}

        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[ErrorContext] = []

    def _add_custom_levels(self):
        """Add custom log levels to the logger"""
        # Add TRACE level (lower than DEBUG)
        TRACE_LEVEL = 5
        logging.addLevelName(TRACE_LEVEL, "TRACE")

        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(TRACE_LEVEL):
                self._log(TRACE_LEVEL, message, args, **kwargs)

        # Add PERFORMANCE level (between INFO and WARNING)
        PERFORMANCE_LEVEL = 25
        logging.addLevelName(PERFORMANCE_LEVEL, "PERFORMANCE")

        def performance(self, message, *args, **kwargs):
            if self.isEnabledFor(PERFORMANCE_LEVEL):
                self._log(PERFORMANCE_LEVEL, message, args, **kwargs)

        # Add OPERATION level (between INFO and WARNING)
        OPERATION_LEVEL = 22
        logging.addLevelName(OPERATION_LEVEL, "OPERATION")

        def operation(self, message, *args, **kwargs):
            if self.isEnabledFor(OPERATION_LEVEL):
                self._log(OPERATION_LEVEL, message, args, **kwargs)

        # Add INTEGRATION level (between WARNING and ERROR)
        INTEGRATION_LEVEL = 35
        logging.addLevelName(INTEGRATION_LEVEL, "INTEGRATION")

        def integration(self, message, *args, **kwargs):
            if self.isEnabledFor(INTEGRATION_LEVEL):
                self._log(INTEGRATION_LEVEL, message, args, **kwargs)

        # Bind methods to logger
        import types
        self.logger.trace = types.MethodType(trace, self.logger)
        self.logger.performance = types.MethodType(performance, self.logger)
        self.logger.operation = types.MethodType(operation, self.logger)
        self.logger.integration = types.MethodType(integration, self.logger)

    @property
    def current_context(self) -> Optional[DispatchContext]:
        """Get current context from thread-local storage"""
        return getattr(self._local, 'context', None)

    @current_context.setter
    def current_context(self, context: Optional[DispatchContext]):
        """Set current context in thread-local storage"""
        self._local.context = context

    def set_context(self, context: DispatchContext):
        """Set dispatch context for current thread"""
        self.current_context = context

    def clear_context(self):
        """Clear dispatch context for current thread"""
        self.current_context = None

    @contextmanager
    def operation_context(self,
                         operation_type: OperationType,
                         session_id: Optional[str] = None,
                         task_id: Optional[str] = None,
                         agent_id: Optional[str] = None,
                         file_paths: Optional[List[str]] = None,
                         parent_context: Optional[DispatchContext] = None,
                         **metadata):
        """Context manager for dispatch operations with automatic timing"""

        # Create new context
        context = DispatchContext(
            operation_type=operation_type,
            session_id=session_id,
            task_id=task_id,
            agent_id=agent_id,
            file_paths=file_paths or [],
            metadata=metadata,
            start_time=time.time(),
            parent_correlation_id=parent_context.correlation_id if parent_context else None
        )

        # Set context
        old_context = self.current_context
        self.current_context = context

        # Track operation start
        self.operation_counts[operation_type] = self.operation_counts.get(operation_type, 0) + 1

        try:
            self.log_operation_start(context)
            yield context

        except Exception as e:
            self.log_error(e, context)
            raise

        finally:
            # Calculate duration and log completion
            if context.start_time:
                duration_ms = (time.time() - context.start_time) * 1000
                self.log_operation_complete(context, duration_ms)

                # Track performance metrics
                metrics = PerformanceMetrics(
                    operation_type=operation_type,
                    duration_ms=duration_ms,
                    files_processed=len(context.file_paths),
                    correlation_id=context.correlation_id
                )
                self.log_performance_metrics(metrics)

            # Restore previous context
            self.current_context = old_context

    def _format_message(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context information"""
        context = self.current_context
        if not context:
            return message

        # Build context string
        context_parts = []
        context_parts.append(f"correlation_id={context.correlation_id[:8]}")

        if context.operation_type:
            context_parts.append(f"op={context.operation_type.value}")

        if context.session_id:
            context_parts.append(f"session={context.session_id[:8]}")

        if context.task_id:
            context_parts.append(f"task={context.task_id}")

        if context.agent_id:
            context_parts.append(f"agent={context.agent_id[:8]}")

        if context.file_paths:
            context_parts.append(f"files={len(context.file_paths)}")

        context_str = " | ".join(context_parts)

        # Add extra data if provided
        if extra_data:
            extra_str = " | ".join(f"{k}={v}" for k, v in extra_data.items())
            context_str = f"{context_str} | {extra_str}"

        return f"[{context_str}] {message}"

    def log_operation_start(self, context: DispatchContext):
        """Log the start of an operation"""
        message = f"Starting {context.operation_type.value}"
        if context.file_paths:
            message += f" for {len(context.file_paths)} files"

        self.logger.operation(self._format_message(message))

    def log_operation_complete(self, context: DispatchContext, duration_ms: float):
        """Log the completion of an operation"""
        message = f"Completed {context.operation_type.value} in {duration_ms:.2f}ms"
        extra_data = {"duration_ms": f"{duration_ms:.2f}"}

        self.logger.operation(self._format_message(message, extra_data))

    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        message = (f"Performance: {metrics.operation_type.value} "
                  f"took {metrics.duration_ms:.2f}ms")

        extra_data = {
            "duration_ms": f"{metrics.duration_ms:.2f}",
            "files_processed": metrics.files_processed
        }

        if metrics.issues_found > 0:
            extra_data["issues_found"] = metrics.issues_found

        if metrics.fixes_applied > 0:
            extra_data["fixes_applied"] = metrics.fixes_applied

        if metrics.memory_usage_mb:
            extra_data["memory_mb"] = f"{metrics.memory_usage_mb:.2f}"

        if metrics.cpu_usage_percent:
            extra_data["cpu_percent"] = f"{metrics.cpu_usage_percent:.1f}"

        self.logger.performance(self._format_message(message, extra_data))

    def log_error(self, error: Exception, context: Optional[DispatchContext] = None):
        """Log an error with context"""
        context = context or self.current_context

        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            operation_type=context.operation_type if context else OperationType.ORCHESTRATION,
            correlation_id=context.correlation_id if context else "unknown",
            file_paths=context.file_paths if context else []
        )

        # Track error
        self.error_counts[error_context.error_type] = self.error_counts.get(error_context.error_type, 0) + 1
        self.recent_errors.append(error_context)

        # Keep only recent errors (last 100)
        if len(self.recent_errors) > 100:
            self.recent_errors = self.recent_errors[-100:]

        message = f"Error in {error_context.operation_type.value}: {error_context.error_message}"
        extra_data = {
            "error_type": error_context.error_type,
            "error_count": self.error_counts[error_context.error_type]
        }

        self.logger.error(self._format_message(message, extra_data))

    def log_trigger_detection(self, trigger_type: str, file_paths: List[str], metadata: Dict[str, Any]):
        """Log trigger detection events"""
        message = f"Detected {trigger_type} trigger for {len(file_paths)} files"
        extra_data = {
            "trigger_type": trigger_type,
            "file_count": len(file_paths)
        }
        extra_data.update(metadata)

        self.logger.info(self._format_message(message, extra_data))

    def log_agent_dispatch(self, agent_type: str, agent_id: str, message_type: str):
        """Log agent dispatch events"""
        message = f"Dispatching {agent_type} agent with {message_type} message"
        extra_data = {
            "agent_type": agent_type,
            "agent_id": agent_id[:8],
            "message_type": message_type
        }

        self.logger.info(self._format_message(message, extra_data))

    def log_review_parsing(self, issues_found: int, processing_time_ms: float):
        """Log review result parsing"""
        message = f"Parsed review results: {issues_found} issues found"
        extra_data = {
            "issues_found": issues_found,
            "processing_time_ms": f"{processing_time_ms:.2f}"
        }

        self.logger.info(self._format_message(message, extra_data))

    def log_fix_application(self, fixes_applied: int, verification_status: str):
        """Log fix application results"""
        message = f"Applied {fixes_applied} fixes, verification: {verification_status}"
        extra_data = {
            "fixes_applied": fixes_applied,
            "verification_status": verification_status
        }

        self.logger.info(self._format_message(message, extra_data))

    def log_workflow_transition(self, from_state: str, to_state: str, trigger: str):
        """Log workflow state transitions"""
        message = f"Workflow transition: {from_state} -> {to_state} ({trigger})"
        extra_data = {
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger
        }

        self.logger.integration(self._format_message(message, extra_data))

    def log_escalation(self, reason: str, attempt_count: int, metadata: Dict[str, Any]):
        """Log escalation events"""
        message = f"Escalating after {attempt_count} attempts: {reason}"
        extra_data = {
            "reason": reason,
            "attempt_count": attempt_count
        }
        extra_data.update(metadata)

        self.logger.warning(self._format_message(message, extra_data))

    def debug_state(self, component: str, state_data: Dict[str, Any]):
        """Log debug information about component state"""
        message = f"Debug state for {component}"

        # Format state data for readability
        formatted_state = {}
        for key, value in state_data.items():
            if isinstance(value, (dict, list)):
                formatted_state[key] = json.dumps(value, indent=2)[:200] + "..." if len(str(value)) > 200 else json.dumps(value, indent=2)
            else:
                formatted_state[key] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)

        self.logger.trace(self._format_message(message, formatted_state))

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics"""
        return {
            "operation_counts": dict(self.operation_counts),
            "error_counts": dict(self.error_counts),
            "recent_errors": [error.to_dict() for error in self.recent_errors[-10:]],  # Last 10 errors
            "total_operations": sum(self.operation_counts.values()),
            "total_errors": sum(self.error_counts.values())
        }


class DispatchLoggerFormatter(ColoredFormatter):
    """Enhanced formatter for dispatch logger with custom colors"""

    # Enhanced colors for dispatch operations
    OPERATION_COLORS = {
        'TRACE': Colors.CYAN,
        'PERFORMANCE': Colors.BOLD_CYAN,
        'OPERATION': Colors.BOLD_GREEN,
        'INTEGRATION': Colors.BOLD_MAGENTA,
        'trigger_detection': Colors.YELLOW,
        'agent_dispatch': Colors.BLUE,
        'review_parsing': Colors.MAGENTA,
        'fix_application': Colors.GREEN,
        'verification': Colors.CYAN,
        'escalation': Colors.RED,
        'workflow_transition': Colors.BOLD_BLUE,
        'correlation_id': Colors.BOLD_WHITE,
    }

    def format(self, record):
        # Get base formatted message
        msg = super().format(record)

        if not self.use_color:
            return msg

        # Apply operation-specific colors
        for operation, color in self.OPERATION_COLORS.items():
            if operation in msg:
                msg = msg.replace(operation, f"{color}{operation}{Colors.RESET}")

        # Highlight correlation IDs
        import re
        correlation_pattern = r'correlation_id=([a-f0-9]{8})'
        msg = re.sub(correlation_pattern, rf'correlation_id={Colors.BOLD_WHITE}\1{Colors.RESET}', msg)

        return msg


def setup_dispatch_logging(level=logging.INFO,
                          enable_performance_logging: bool = True,
                          enable_trace_logging: bool = False) -> DispatchLogger:
    """
    Set up comprehensive dispatch logging system.

    Args:
        level: Base logging level
        enable_performance_logging: Enable performance metrics logging
        enable_trace_logging: Enable trace-level debugging

    Returns:
        Configured DispatchLogger instance
    """
    # Create console handler with enhanced formatter
    console_handler = logging.StreamHandler()

    # Set level based on options
    if enable_trace_logging:
        console_handler.setLevel(5)  # TRACE level
    elif enable_performance_logging:
        console_handler.setLevel(22)  # OPERATION level
    else:
        console_handler.setLevel(level)

    # Create enhanced formatter
    formatter = DispatchLoggerFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Create root dispatch logger
    root_logger = logging.getLogger('cadence.dispatch')

    # Remove any existing handlers to ensure idempotency
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)
    root_logger.setLevel(5 if enable_trace_logging else level)

    # Create main dispatch logger
    dispatch_logger = DispatchLogger('cadence.dispatch', root_logger)

    return dispatch_logger


def get_dispatch_logger(name: str) -> DispatchLogger:
    """Get a dispatch logger for a specific component"""
    logger_name = f"cadence.dispatch.{name}"
    base_logger = logging.getLogger(logger_name)
    return DispatchLogger(logger_name, base_logger)


# Global dispatch logger instance
_global_dispatch_logger = None


def get_global_dispatch_logger() -> DispatchLogger:
    """Get the global dispatch logger instance"""
    global _global_dispatch_logger
    if _global_dispatch_logger is None:
        _global_dispatch_logger = setup_dispatch_logging()
    return _global_dispatch_logger


# Convenience functions for common logging operations
def log_operation_start(operation_type: OperationType, **context):
    """Log operation start with global logger"""
    logger = get_global_dispatch_logger()
    with logger.operation_context(operation_type, **context) as ctx:
        pass


def log_performance(operation_type: OperationType, duration_ms: float, **metrics):
    """Log performance metrics with global logger"""
    logger = get_global_dispatch_logger()
    perf_metrics = PerformanceMetrics(
        operation_type=operation_type,
        duration_ms=duration_ms,
        **metrics
    )
    logger.log_performance_metrics(perf_metrics)


def log_error_with_context(error: Exception, operation_type: OperationType, **context):
    """Log error with context using global logger"""
    logger = get_global_dispatch_logger()
    ctx = DispatchContext(operation_type=operation_type, **context)
    logger.set_context(ctx)
    logger.log_error(error, ctx)
    logger.clear_context()
