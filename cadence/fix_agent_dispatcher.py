"""
Fix Agent Dispatcher for handling code review issue remediation.

This module dispatches fix agents to address critical and high-priority issues
identified during code reviews, with context preservation and iteration limits.

REFACTORED VERSION: Now extends EnhancedAgentDispatcher for cleaner architecture.
"""

import json
import logging
import threading
import time
import heapq
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta

from .agent_messages import AgentMessage, MessageType, AgentType, Priority
from .enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from .config import FixAgentDispatcherConfig
from .fix_iteration_tracker import EscalationStrategy, PersistenceType


class FixAttemptStatus(Enum):
    """Status of a fix attempt"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    EXCEEDED_RETRIES = "exceeded_retries"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class FixAttempt:
    """Represents a single fix attempt"""
    attempt_number: int
    status: FixAttemptStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    files_modified: List[str] = field(default_factory=list)
    fix_agent_id: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None


@dataclass
class IssueContext:
    """Context for an issue that needs fixing"""
    issue_id: str
    severity: str  # critical, high, medium, low
    issue_type: str  # bug, security, performance, etc.
    description: str
    file_path: str
    line_numbers: Optional[List[int]] = None
    code_review_findings: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None


class FixAgentDispatcher(EnhancedAgentDispatcher):
    """
    Manages the dispatch and coordination of fix agents for code issues.

    Now extends EnhancedAgentDispatcher for better code reuse and cleaner architecture.

    Features:
    - Inherits fix iteration tracking from EnhancedAgentDispatcher
    - Adds advanced context preservation between attempts
    - Circular dependency detection
    - Issue classification and prioritization
    - Verification workflow integration
    - Smart retry scheduling with exponential backoff
    """

    def __init__(self, config: Optional[FixAgentDispatcherConfig] = None):
        """
        Initialize the FixAgentDispatcher.

        Args:
            config: Configuration object for the dispatcher. If None, uses defaults.
        """
        if config is None:
            config = FixAgentDispatcherConfig()

        # Convert FixAgentDispatcherConfig to DispatchConfig for parent
        dispatch_config = {
            'max_fix_iterations': config.max_attempts,
            'default_timeout_ms': config.timeout_ms,
            'enable_fix_tracking': True,
            'enable_escalation': config.enable_auto_fix,
            'escalation_strategy': 'log_only',  # Can be enhanced based on config
            'persistence_type': 'memory',  # Can be made configurable
        }

        # Initialize parent with converted config
        super().__init__(
            max_fix_iterations=config.max_attempts,
            escalation_strategy=EscalationStrategy.LOG_ONLY,
            persistence_type=PersistenceType.MEMORY,
            config=dispatch_config
        )

        # Store fix-specific configuration
        self.max_attempts = config.max_attempts
        self.timeout_ms = config.timeout_ms
        self.enable_auto_fix = config.enable_auto_fix
        self.severity_threshold = config.severity_threshold
        self.enable_verification = config.enable_verification
        self.verification_timeout_ms = config.verification_timeout_ms
        self._max_file_modifications = config.circular_dependency.max_file_modifications
        self._min_attempts_before_check = config.circular_dependency.min_attempts_before_check

        # Initialize logging
        self.logger = logging.getLogger(f"cadence.fix_dispatcher.{id(self)}")
        self.logger.info(f"FixAgentDispatcher initialized with max_attempts={config.max_attempts}, timeout={config.timeout_ms}ms")

        # Additional tracking structures specific to fix dispatcher
        self.active_fixes: Dict[str, IssueContext] = {}  # issue_id -> context
        self.fix_history: Dict[str, List[FixAttempt]] = {}  # issue_id -> attempts

        # Thread safety (parent has its own lock, but we need one for our structures)
        self.fix_lock = threading.Lock()

        # Callbacks for events
        self.on_fix_complete: Optional[Callable] = None
        self.on_fix_failed: Optional[Callable] = None
        self.on_max_retries: Optional[Callable] = None
        self.on_verification_failed: Optional[Callable] = None

        # Retry scheduling
        self.retry_queue: List[Tuple[datetime, IssueContext]] = []
        self.retry_timer: Optional[threading.Timer] = None

        # Fix verification
        self.fix_verifier: Optional[Callable] = None

        # Issue contexts for preserving state between attempts
        self._issue_contexts: Dict[str, Dict[str, Any]] = {}

        self.logger.info("FixAgentDispatcher ready")

    def should_dispatch_fix(self, issue: IssueContext) -> bool:
        """
        Determine if a fix agent should be dispatched for an issue.

        Args:
            issue: The issue context to evaluate

        Returns:
            bool: True if fix agent should be dispatched
        """
        if not self.enable_auto_fix:
            self.logger.debug("Auto-fix disabled, skipping dispatch")
            return False

        # Check severity threshold
        severity_levels = ["low", "medium", "high", "critical"]
        threshold_index = severity_levels.index(self.severity_threshold)
        issue_index = severity_levels.index(issue.severity.lower())

        if issue_index < threshold_index:
            self.logger.debug(f"Issue severity {issue.severity} below threshold {self.severity_threshold}")
            return False

        # Check if already being fixed or exceeded retries
        with self.fix_lock:
            if issue.issue_id in self.active_fixes:
                self.logger.debug(f"Issue {issue.issue_id} already being fixed")
                return False

            # Check our own history first
            attempts = self.fix_history.get(issue.issue_id, [])
            if len(attempts) >= self.max_attempts:
                self.logger.warning(f"Issue {issue.issue_id} exceeded max attempts ({self.max_attempts})")
                return False

            # Also check parent's fix manager
            if not self.fix_manager.can_attempt_fix(issue.issue_id):
                self.logger.warning(f"Issue {issue.issue_id} exceeded max attempts or is escalated (parent tracker)")
                return False

        return True

    def get_fix_status(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a fix attempt.

        Args:
            issue_id: The issue identifier

        Returns:
            Status dictionary or None if not found
        """
        with self.fix_lock:
            if issue_id not in self.active_fixes and issue_id not in self.fix_history:
                return None

            status = {
                "issue_id": issue_id,
                "is_active": issue_id in self.active_fixes,
                "attempts": []
            }

            # Include attempts from our detailed history
            if issue_id in self.fix_history:
                for attempt in self.fix_history[issue_id]:
                    status["attempts"].append({
                        "number": attempt.attempt_number,
                        "status": attempt.status.value,
                        "start_time": attempt.start_time.isoformat(),
                        "end_time": attempt.end_time.isoformat() if attempt.end_time else None,
                        "error": attempt.error_message,
                        "files_modified": attempt.files_modified
                    })

            # Also include info from parent's fix manager
            parent_status = self.fix_manager.get_task_status(issue_id)
            if parent_status:
                status["parent_tracking"] = parent_status

            return status

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up FixAgentDispatcher")
        with self.fix_lock:
            self.active_fixes.clear()
            self.fix_history.clear()
            if hasattr(self, '_issue_contexts'):
                self._issue_contexts.clear()
            self.retry_queue.clear()
            if self.retry_timer:
                self.retry_timer.cancel()
                self.retry_timer = None

    def classify_issue_type(self, issue: IssueContext) -> str:
        """
        Classify the issue type based on description and other context.

        Args:
            issue: The issue context

        Returns:
            str: Classified issue type
        """
        description_lower = issue.description.lower()

        # Security issues
        security_keywords = ["security", "vulnerability", "injection", "xss", "csrf",
                           "authentication", "authorization", "encryption", "password",
                           "token", "credential", "privilege"]
        if any(keyword in description_lower for keyword in security_keywords):
            return "security"

        # Performance issues
        performance_keywords = ["performance", "slow", "optimization", "memory leak",
                              "cpu", "latency", "throughput", "bottleneck"]
        if any(keyword in description_lower for keyword in performance_keywords):
            return "performance"

        # Bug/error issues
        bug_keywords = ["error", "exception", "crash", "fail", "bug", "broken",
                       "incorrect", "wrong", "invalid", "null", "undefined"]
        if any(keyword in description_lower for keyword in bug_keywords):
            return "bug"

        # Code quality issues
        quality_keywords = ["refactor", "cleanup", "duplicate", "complexity",
                          "maintainability", "readability", "code smell"]
        if any(keyword in description_lower for keyword in quality_keywords):
            return "quality"

        # Default to the provided type or "general"
        return issue.issue_type or "general"

    def dispatch_fix_agent(self, issue: IssueContext, callback: Optional[Callable] = None) -> Optional[str]:
        """
        Dispatch a fix agent for the given issue.

        Overrides parent's dispatch_fix_agent to add issue-specific context.

        Args:
            issue: The issue context to fix
            callback: Optional callback function

        Returns:
            Message ID if dispatched, None otherwise
        """
        if not self.should_dispatch_fix(issue):
            return None

        # Validate the issue
        validation_result = self.validate_fix_request(issue)
        if not validation_result["valid"]:
            self.logger.error(f"Invalid fix request: {validation_result['reason']}")
            return None

        # Classify the issue
        issue_type = self.classify_issue_type(issue)
        self.logger.info(f"Classified issue {issue.issue_id} as type: {issue_type}")

        with self.fix_lock:
            # Add to active fixes
            self.active_fixes[issue.issue_id] = issue

            # Create or update fix attempt
            if issue.issue_id not in self.fix_history:
                self.fix_history[issue.issue_id] = []

            attempt = FixAttempt(
                attempt_number=len(self.fix_history[issue.issue_id]) + 1,
                status=FixAttemptStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
            self.fix_history[issue.issue_id].append(attempt)

        # Get preserved context
        preserved_context = self.get_fix_context(issue.issue_id)

        # Build context for the fix agent
        context = {
            "task_type": "fix_issue",
            "issue_id": issue.issue_id,
            "severity": issue.severity,
            "issue_type": issue_type,
            "description": issue.description,
            "file_path": issue.file_path,
            "line_numbers": issue.line_numbers,
            "code_review_findings": issue.code_review_findings,
            "suggested_fix": issue.suggested_fix,
            "attempt_number": attempt.attempt_number,
            "preserved_context": preserved_context
        }

        # Define success criteria
        success_criteria = {
            "type": "fix_complete",
            "issue_resolved": True,
            "no_new_issues": True,
            "tests_pass": True
        }

        # Create wrapped callback
        def fix_callback(response: AgentMessage):
            self._handle_fix_response(issue.issue_id, response, callback, attempt)

        # Use parent's dispatch_fix_agent with our context
        message_id = super().dispatch_fix_agent(
            task_id=issue.issue_id,
            context=context,
            success_criteria=success_criteria,
            callback_handler=fix_callback,
            timeout_ms=self.timeout_ms,
            priority=self._get_priority_for_severity(issue.severity)
        )

        if not message_id:
            # Fix was not dispatched (likely due to iteration limits)
            with self.fix_lock:
                self.active_fixes.pop(issue.issue_id, None)
                attempt.status = FixAttemptStatus.EXCEEDED_RETRIES
                attempt.end_time = datetime.now()

        return message_id

    def _get_priority_for_severity(self, severity: str) -> Priority:
        """Convert severity to priority"""
        severity_to_priority = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW
        }
        return severity_to_priority.get(severity.lower(), Priority.MEDIUM)

    def _handle_fix_response(
        self,
        issue_id: str,
        response: AgentMessage,
        callback: Optional[Callable],
        attempt: FixAttempt
    ):
        """Handle response from fix agent"""
        with self.fix_lock:
            if issue_id not in self.active_fixes:
                self.logger.warning(f"Received response for inactive fix: {issue_id}")
                return

            # Update attempt status
            attempt.end_time = datetime.now()

            if response.metadata.get("success", False):
                attempt.status = FixAttemptStatus.SUCCESS
                self.active_fixes.pop(issue_id, None)

                # Run verification if enabled
                if self.enable_verification and self.fix_verifier:
                    self._run_verification(issue_id, attempt)

                if self.on_fix_complete:
                    self.on_fix_complete(issue_id, attempt)
            else:
                attempt.status = FixAttemptStatus.FAILED
                attempt.error_message = response.metadata.get("error", "Unknown error")

                # Check for circular dependencies
                if self._has_circular_dependency(issue_id):
                    self.logger.error(f"Circular dependency detected for issue {issue_id}")
                    self.active_fixes.pop(issue_id, None)
                    if self.on_max_retries:
                        self.on_max_retries(issue_id, "Circular dependency detected")
                else:
                    # Schedule retry if attempts remain
                    if self.fix_manager.can_attempt_fix(issue_id):
                        self.schedule_retry(self.active_fixes[issue_id])
                    else:
                        self.active_fixes.pop(issue_id, None)
                        if self.on_max_retries:
                            self.on_max_retries(issue_id, "Max attempts exceeded")

        # Call original callback if provided
        if callback:
            callback(response)

    # Additional methods for unique FixAgentDispatcher features...

    def get_fix_context(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get preserved context for an issue across fix attempts"""
        return self._issue_contexts.get(issue_id)

    def preserve_fix_context(self, issue_id: str, context: Dict[str, Any]):
        """Preserve context for future fix attempts"""
        self._issue_contexts[issue_id] = context

    def validate_fix_request(self, issue: IssueContext) -> Dict[str, Any]:
        """Validate a fix request before dispatching"""
        if not issue.issue_id:
            return {"valid": False, "reason": "Missing issue_id"}
        if not issue.file_path:
            return {"valid": False, "reason": "Missing file_path"}
        if not issue.description:
            return {"valid": False, "reason": "Missing description"}
        return {"valid": True}

    def _has_circular_dependency(self, issue_id: str) -> bool:
        """Check if fix attempts show circular dependency pattern"""
        if issue_id not in self.fix_history:
            return False

        attempts = self.fix_history[issue_id]
        if len(attempts) < self._min_attempts_before_check:
            return False

        # Track file modification frequency
        file_counts: Dict[str, int] = {}
        for attempt in attempts[-self._min_attempts_before_check:]:
            for file_path in attempt.files_modified:
                file_counts[file_path] = file_counts.get(file_path, 0) + 1

        # Check if any file exceeds modification threshold
        for count in file_counts.values():
            if count >= self._max_file_modifications:
                return True

        return False

    def schedule_retry(self, issue: IssueContext, delay_seconds: Optional[int] = None):
        """Schedule a retry for a failed fix attempt"""
        if delay_seconds is None:
            # Exponential backoff based on attempt number
            attempts = len(self.fix_history.get(issue.issue_id, []))
            delay_seconds = min(300, 30 * (2 ** (attempts - 1)))  # Max 5 minutes

        retry_time = datetime.now() + timedelta(seconds=delay_seconds)

        with self.fix_lock:
            # Add to retry queue
            heapq.heappush(self.retry_queue, (retry_time, issue))

            # Schedule timer if not already scheduled
            if not self.retry_timer or not self.retry_timer.is_alive():
                self._schedule_next_retry()

    def _schedule_next_retry(self):
        """Schedule the next retry from the queue"""
        with self.fix_lock:
            if not self.retry_queue:
                return

            next_time, _ = self.retry_queue[0]
            delay = (next_time - datetime.now()).total_seconds()

            if delay > 0:
                self.retry_timer = threading.Timer(delay, self._process_retries)
                self.retry_timer.start()
            else:
                self._process_retries()

    def _process_retries(self):
        """Process any pending retries"""
        now = datetime.now()

        with self.fix_lock:
            while self.retry_queue and self.retry_queue[0][0] <= now:
                _, issue = heapq.heappop(self.retry_queue)
                self.active_fixes.pop(issue.issue_id, None)  # Remove from active

                # Dispatch the retry
                self.dispatch_fix_agent(issue)

            # Schedule next retry if any remain
            self._schedule_next_retry()

    def _run_verification(self, issue_id: str, attempt: FixAttempt):
        """Run verification on a completed fix"""
        if not self.fix_verifier:
            return

        try:
            verification_result = self.fix_verifier(issue_id, attempt)
            attempt.verification_results = verification_result

            if not verification_result.get("passed", False):
                attempt.status = FixAttemptStatus.VERIFICATION_FAILED
                if self.on_verification_failed:
                    self.on_verification_failed(issue_id, verification_result)
        except Exception as e:
            self.logger.error(f"Verification failed for {issue_id}: {e}")
            attempt.verification_results = {"error": str(e), "passed": False}

    def get_iteration_stats(self, issue_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed iteration statistics"""
        stats = {
            "total_issues": len(self.fix_history),
            "active_fixes": len(self.active_fixes),
            "pending_retries": len(self.retry_queue),
            "by_status": {},
            "by_issue_type": {}
        }

        # Aggregate statistics
        for issue_id_key, attempts in self.fix_history.items():
            if issue_id and issue_id_key != issue_id:
                continue

            for attempt in attempts:
                status = attempt.status.value
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        # Include parent's statistics
        parent_stats = self.fix_manager.get_statistics()
        stats["parent_tracking"] = parent_stats

        return stats
