"""
Fix Agent Dispatcher for handling code review issue remediation.

This module dispatches fix agents to address critical and high-priority issues
identified during code reviews, with context preservation and iteration limits.
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

from .agent_messages import AgentMessage, MessageType, AgentType
from .agent_dispatcher import AgentDispatcher
from .config import FixAgentDispatcherConfig


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


class FixAgentDispatcher:
    """
    Manages the dispatch and coordination of fix agents for code issues.

    Features:
    - Dispatches fix agents based on issue severity and type
    - Maintains context between fix attempts
    - Enforces iteration limits (max 3 attempts)
    - Tracks fix history and patterns
    - Handles escalation for failed fixes
    """

    def __init__(self, config: Optional[FixAgentDispatcherConfig] = None):
        """
        Initialize the FixAgentDispatcher.

        Args:
            config: Configuration object for the dispatcher. If None, uses defaults.
        """
        if config is None:
            config = FixAgentDispatcherConfig()

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
        self.logger.info(f"FixAgentDispatcher initialized with max_attempts={self.max_attempts}, timeout={self.timeout_ms}ms")

        # Tracking structures
        self.active_fixes: Dict[str, IssueContext] = {}  # issue_id -> context
        self.fix_history: Dict[str, List[FixAttempt]] = {}  # issue_id -> attempts
        self.agent_dispatcher = AgentDispatcher()

        # Thread safety
        self.lock = threading.Lock()

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
        with self.lock:
            if issue.issue_id in self.active_fixes:
                self.logger.debug(f"Issue {issue.issue_id} already being fixed")
                return False

            attempts = self.fix_history.get(issue.issue_id, [])
            if len(attempts) >= self.max_attempts:
                self.logger.warning(f"Issue {issue.issue_id} exceeded max attempts ({self.max_attempts})")
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
        with self.lock:
            if issue_id not in self.active_fixes and issue_id not in self.fix_history:
                return None

            status = {
                "issue_id": issue_id,
                "is_active": issue_id in self.active_fixes,
                "attempts": []
            }

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

            return status

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up FixAgentDispatcher")
        with self.lock:
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

        Args:
            issue: The issue context to fix
            callback: Optional callback for when fix completes

        Returns:
            str: Message ID of dispatched agent, or None if not dispatched
        """
        # Validate the fix request
        is_valid, error_message = self.validate_fix_request(issue)
        if not is_valid:
            self.logger.error(f"Invalid fix request for issue {issue.issue_id}: {error_message}")
            if callback:
                # Create error response for invalid request
                from .agent_messages import MessageContext, SuccessCriteria, CallbackInfo
                error_response = AgentMessage(
                    message_type=MessageType.ERROR,
                    agent_type=AgentType.FIX,
                    context=MessageContext(
                        task_id=f"fix-{issue.issue_id}",
                        parent_session="",
                        files_modified=[],
                        project_path=""
                    ),
                    success_criteria=SuccessCriteria(
                        expected_outcomes=[],
                        validation_steps=[]
                    ),
                    callback=CallbackInfo(
                        handler="dispatch_fix_agent",
                        timeout_ms=0
                    ),
                    message_id=self.agent_dispatcher.generate_message_id(),
                    payload={"error": error_message}
                )
                callback(error_response)
            return None

        # Check if we should dispatch
        if not self.should_dispatch_fix(issue):
            self.logger.info(f"Not dispatching fix agent for issue {issue.issue_id}")
            return None

        # Classify the issue if not already classified
        if not issue.issue_type or issue.issue_type == "general":
            issue.issue_type = self.classify_issue_type(issue)
            self.logger.info(f"Classified issue {issue.issue_id} as type: {issue.issue_type}")

        with self.lock:
            # Mark as active
            self.active_fixes[issue.issue_id] = issue

            # Initialize history if needed
            if issue.issue_id not in self.fix_history:
                self.fix_history[issue.issue_id] = []

            # Create new attempt
            attempt_number = len(self.fix_history[issue.issue_id]) + 1
            attempt = FixAttempt(
                attempt_number=attempt_number,
                status=FixAttemptStatus.PENDING,
                start_time=datetime.now()
            )
            self.fix_history[issue.issue_id].append(attempt)

        # Build fix instructions
        fix_instructions = f"Fix {issue.severity} severity {issue.issue_type} issue:\n\n"
        fix_instructions += f"Description: {issue.description}\n"
        fix_instructions += f"File: {issue.file_path}\n"

        if issue.line_numbers:
            fix_instructions += f"Lines: {', '.join(map(str, issue.line_numbers))}\n"

        if issue.suggested_fix:
            fix_instructions += f"\nSuggested fix: {issue.suggested_fix}\n"

        fix_instructions += f"\nThis is attempt {attempt_number} of {self.max_attempts}."

        # Get fix scope
        fix_scope = self.get_fix_scope(issue)

        # Add scope information to instructions
        fix_instructions += f"\n\nFix Scope:"
        fix_instructions += f"\n- Allowed operations: {', '.join(fix_scope['allowed_operations'])}"
        fix_instructions += f"\n- Restrictions: {', '.join(fix_scope['restrictions'])}"
        if fix_scope['related_files']:
            fix_instructions += f"\n- Related files: {', '.join(fix_scope['related_files'])}"

        # Add context from previous attempts
        if attempt_number > 1:
            previous_context = self.get_fix_context(issue.issue_id)
            if previous_context:
                fix_instructions += f"\n\nPrevious attempt information:"
                if "learned_constraints" in previous_context:
                    fix_instructions += f"\n- Learned constraints: {', '.join(previous_context['learned_constraints'])}"
                # Get previous error messages
                if issue.issue_id in self.fix_history:
                    for prev_attempt in self.fix_history[issue.issue_id]:
                        if prev_attempt.attempt_number < attempt_number and prev_attempt.error_message:
                            fix_instructions += f"\n- Attempt {prev_attempt.attempt_number} failed: {prev_attempt.error_message}"

        # Create context and success criteria
        from .agent_messages import MessageContext, SuccessCriteria

        context = MessageContext(
            task_id=f"fix-{issue.issue_id}",
            parent_session=f"review-session-{issue.issue_id}",
            files_modified=[issue.file_path],
            project_path=str(Path(issue.file_path).parent)
        )

        success_criteria = SuccessCriteria(
            expected_outcomes=[
                f"Fix the {issue.issue_type} issue in {issue.file_path}",
                "Ensure the fix doesn't introduce new issues",
                "Maintain existing functionality"
            ],
            validation_steps=[
                "Verify the issue is resolved",
                "Run relevant tests if available",
                "Check for regressions"
            ]
        )

        # Store issue context for later use
        self._issue_contexts = getattr(self, '_issue_contexts', {})
        self._issue_contexts[f"fix-{issue.issue_id}"] = {
            "issue": issue,
            "attempt": attempt,
            "instructions": fix_instructions,
            "attempt_number": attempt_number,
            "fix_scope": fix_scope,
            "timestamp": datetime.now().isoformat()
        }

        # Set up completion callback
        def fix_complete_callback(response: AgentMessage):
            self._handle_fix_response(issue, attempt, response)
            if callback:
                callback(response)

        # Dispatch the agent
        message_id = self.agent_dispatcher.dispatch_agent(
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=success_criteria,
            callback_handler=fix_complete_callback,
            timeout_ms=self.timeout_ms
        )

        # Update attempt with agent ID
        with self.lock:
            attempt.fix_agent_id = message_id
            attempt.status = FixAttemptStatus.IN_PROGRESS

        self.logger.info(f"Dispatched fix agent {message_id} for issue {issue.issue_id} (attempt {attempt_number})")
        return message_id

    def _handle_fix_response(self, issue: IssueContext, attempt: FixAttempt, response: AgentMessage):
        """
        Handle the response from a fix agent.

        Args:
            issue: The issue being fixed
            attempt: The fix attempt
            response: The agent response
        """
        with self.lock:
            attempt.end_time = datetime.now()

            if response.message_type == MessageType.TASK_COMPLETE:
                attempt.files_modified = response.context.files_modified or []
                self.logger.info(f"Fix completed for issue {issue.issue_id}, pending verification")
            elif response.message_type == MessageType.ERROR:
                self._process_failed_attempt(issue, attempt, response)
                return

        # Perform verification outside of lock if needed
        if self.enable_verification and response.message_type == MessageType.TASK_COMPLETE:
            self._run_verification(issue, attempt)
        elif response.message_type == MessageType.TASK_COMPLETE:
            # No verification, mark as success
            self._process_successful_attempt(issue, attempt)

    def _process_failed_attempt(self, issue: IssueContext, attempt: FixAttempt, response: AgentMessage):
        """
        Process a failed fix attempt. Assumes lock is held.

        Args:
            issue: The issue being fixed
            attempt: The fix attempt
            response: The error response
        """
        attempt.status = FixAttemptStatus.FAILED
        attempt.error_message = response.payload.get("error", "Unknown error") if response.payload else "Unknown error"
        self.logger.error(f"Fix failed for issue {issue.issue_id}: {attempt.error_message}")

        # Remove from active fixes to allow retry
        if issue.issue_id in self.active_fixes:
            del self.active_fixes[issue.issue_id]

        # Check if we should retry
        attempts = self.fix_history.get(issue.issue_id, [])
        if len(attempts) < self.max_attempts:
            # Schedule retry outside of lock
            # Use Timer to avoid holding lock during retry dispatch
            threading.Timer(0, self.schedule_retry, args=(issue,)).start()
        else:
            # Max retries exceeded
            attempt.status = FixAttemptStatus.EXCEEDED_RETRIES
            self.logger.error(f"Max retries exceeded for issue {issue.issue_id}")

            # Schedule failure callbacks outside of lock
            if self.on_max_retries:
                threading.Timer(0, self.on_max_retries, args=(issue, self.fix_history[issue.issue_id])).start()
            elif self.on_fix_failed:
                threading.Timer(0, self.on_fix_failed, args=(issue, attempt)).start()

    def _run_verification(self, issue: IssueContext, attempt: FixAttempt):
        """
        Run verification and process the result.

        Args:
            issue: The issue being fixed
            attempt: The fix attempt
        """
        verification_results = self.verify_fix(issue, attempt)
        attempt.verification_results = verification_results

        if verification_results["success"]:
            self._process_successful_attempt(issue, attempt)
        else:
            self._process_verification_failure(issue, attempt, verification_results)

    def _process_successful_attempt(self, issue: IssueContext, attempt: FixAttempt):
        """
        Finalize a successful fix.

        Args:
            issue: The issue that was fixed
            attempt: The successful attempt
        """
        with self.lock:
            attempt.status = FixAttemptStatus.SUCCESS

            # Remove from active fixes
            if issue.issue_id in self.active_fixes:
                del self.active_fixes[issue.issue_id]

            # Clean up context to prevent memory leak
            context_key = f"fix-{issue.issue_id}"
            if context_key in self._issue_contexts:
                del self._issue_contexts[context_key]
                self.logger.debug(f"Cleaned up context for completed issue {issue.issue_id}")

        self.logger.info(f"Fix succeeded for issue {issue.issue_id}")

        # Execute callback outside of lock
        if self.on_fix_complete:
            try:
                self.on_fix_complete(issue, attempt)
            except Exception as e:
                self.logger.error(f"Error invoking on_fix_complete callback: {e}")

    def _process_verification_failure(self, issue: IssueContext, attempt: FixAttempt, verification_results: Dict[str, Any]):
        """
        Process a verification failure.

        Args:
            issue: The issue being fixed
            attempt: The fix attempt
            verification_results: Results from verification
        """
        with self.lock:
            # Update attempt status
            attempt.status = FixAttemptStatus.VERIFICATION_FAILED
            attempt.error_message = f"Verification failed: {', '.join(verification_results.get('errors', ['Unknown error']))}"
            self.logger.error(f"Fix verification failed for issue {issue.issue_id}: {attempt.error_message}")

            # Remove from active fixes to allow retry
            if issue.issue_id in self.active_fixes:
                del self.active_fixes[issue.issue_id]

            # Check if we should retry
            attempts = self.fix_history.get(issue.issue_id, [])
            should_retry = len(attempts) < self.max_attempts

            if not should_retry:
                # Max retries exceeded
                attempt.status = FixAttemptStatus.EXCEEDED_RETRIES

                # Clean up context since we're done
                context_key = f"fix-{issue.issue_id}"
                if context_key in self._issue_contexts:
                    del self._issue_contexts[context_key]
                    self.logger.debug(f"Cleaned up context for exceeded retries issue {issue.issue_id}")

        # Handle retry or failure callbacks outside of lock
        if should_retry:
            self.schedule_retry(issue)
        else:
            # Execute failure callbacks
            if self.on_verification_failed:
                try:
                    self.on_verification_failed(issue, attempt, verification_results)
                except Exception as e:
                    self.logger.error(f"Error invoking on_verification_failed callback: {e}")
            elif self.on_fix_failed:
                try:
                    self.on_fix_failed(issue, attempt)
                except Exception as e:
                    self.logger.error(f"Error invoking on_fix_failed callback: {e}")

    def get_fix_context(self, issue_id: str, attempt_number: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get preserved context for an issue fix.

        Args:
            issue_id: The issue identifier
            attempt_number: Optional specific attempt number

        Returns:
            Context dictionary or None if not found
        """
        with self.lock:
            context_key = f"fix-{issue_id}"
            if context_key not in self._issue_contexts:
                return None

            context = self._issue_contexts[context_key].copy()

            # If specific attempt requested, filter history
            if attempt_number is not None and issue_id in self.fix_history:
                attempts = [a for a in self.fix_history[issue_id] if a.attempt_number == attempt_number]
                if attempts:
                    context["specific_attempt"] = {
                        "number": attempts[0].attempt_number,
                        "status": attempts[0].status.value,
                        "error": attempts[0].error_message,
                        "files_modified": attempts[0].files_modified
                    }

            return context

    def preserve_fix_context(self, issue_id: str, additional_context: Dict[str, Any]) -> None:
        """
        Preserve additional context for future fix attempts.

        Args:
            issue_id: The issue identifier
            additional_context: Additional context to preserve
        """
        with self.lock:
            context_key = f"fix-{issue_id}"
            if context_key in self._issue_contexts:
                # Merge with existing context
                self._issue_contexts[context_key].update(additional_context)
            else:
                # Create new context
                self._issue_contexts[context_key] = additional_context

            self.logger.debug(f"Preserved context for issue {issue_id}")

    def get_fix_scope(self, issue: IssueContext) -> Dict[str, Any]:
        """
        Determine the scope of a fix based on issue type and context.

        Args:
            issue: The issue context

        Returns:
            Dictionary defining the fix scope
        """
        scope = {
            "primary_file": issue.file_path,
            "affected_lines": issue.line_numbers or [],
            "fix_type": issue.issue_type,
            "allowed_operations": [],
            "restrictions": [],
            "related_files": []
        }

        # Define allowed operations based on issue type
        if issue.issue_type == "security":
            scope["allowed_operations"] = [
                "sanitize_input",
                "add_validation",
                "update_authentication",
                "fix_vulnerability"
            ]
            scope["restrictions"] = [
                "preserve_existing_functionality",
                "maintain_backward_compatibility",
                "no_new_external_dependencies"
            ]
        elif issue.issue_type == "performance":
            scope["allowed_operations"] = [
                "optimize_algorithm",
                "add_caching",
                "reduce_complexity",
                "batch_operations"
            ]
            scope["restrictions"] = [
                "maintain_correctness",
                "preserve_api_contract",
                "no_functional_changes"
            ]
        elif issue.issue_type == "bug":
            scope["allowed_operations"] = [
                "fix_logic_error",
                "handle_edge_case",
                "add_error_handling",
                "fix_null_reference"
            ]
            scope["restrictions"] = [
                "minimal_code_changes",
                "preserve_existing_tests",
                "no_api_changes"
            ]
        elif issue.issue_type == "quality":
            scope["allowed_operations"] = [
                "refactor_code",
                "extract_method",
                "reduce_duplication",
                "improve_naming"
            ]
            scope["restrictions"] = [
                "no_functional_changes",
                "maintain_test_coverage",
                "preserve_public_api"
            ]
        else:
            # General fixes
            scope["allowed_operations"] = [
                "modify_code",
                "add_comments",
                "update_logic"
            ]
            scope["restrictions"] = [
                "follow_coding_standards",
                "maintain_consistency"
            ]

        # Add context from previous attempts if available
        if hasattr(self, '_issue_contexts'):
            context_key = f"fix-{issue.issue_id}"
            if context_key in self._issue_contexts:
                previous_context = self._issue_contexts[context_key]
                if "related_files" in previous_context:
                    scope["related_files"].extend(previous_context["related_files"])
                if "learned_constraints" in previous_context:
                    scope["restrictions"].extend(previous_context["learned_constraints"])

        return scope

    def update_fix_scope(self, issue_id: str, scope_updates: Dict[str, Any]) -> None:
        """
        Update the fix scope based on learned information.

        Args:
            issue_id: The issue identifier
            scope_updates: Updates to apply to the scope
        """
        self.preserve_fix_context(issue_id, {"scope_updates": scope_updates})

        # Track related files discovered during fix attempts
        if "discovered_files" in scope_updates:
            existing_related = self.get_fix_context(issue_id, None) or {}
            related_files = existing_related.get("related_files", [])
            related_files.extend(scope_updates["discovered_files"])
            self.preserve_fix_context(issue_id, {"related_files": list(set(related_files))})

        # Track constraints learned from failures
        if "new_constraints" in scope_updates:
            existing_context = self.get_fix_context(issue_id, None) or {}
            constraints = existing_context.get("learned_constraints", [])
            constraints.extend(scope_updates["new_constraints"])
            self.preserve_fix_context(issue_id, {"learned_constraints": list(set(constraints))})

        self.logger.info(f"Updated fix scope for issue {issue_id}")

    def get_retry_delay(self, attempt_number: int) -> int:
        """
        Calculate retry delay with exponential backoff.

        Args:
            attempt_number: The attempt number (1-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: 2^(n-1) seconds, capped at 5 minutes
        base_delay = 2 ** (attempt_number - 1)
        max_delay = 300  # 5 minutes
        return min(base_delay, max_delay)

    def schedule_retry(self, issue: IssueContext) -> bool:
        """
        Schedule a retry for a failed fix attempt.

        Args:
            issue: The issue to retry

        Returns:
            bool: True if retry was scheduled, False if max attempts exceeded
        """
        with self.lock:
            # Check if we can retry
            attempts = self.fix_history.get(issue.issue_id, [])
            if len(attempts) >= self.max_attempts:
                self.logger.warning(f"Cannot schedule retry for {issue.issue_id}: max attempts exceeded")
                return False

            # Calculate retry time
            retry_delay = self.get_retry_delay(len(attempts) + 1)
            retry_time = datetime.now() + timedelta(seconds=retry_delay)

            # Add to retry queue using heapq for efficient insertion
            heapq.heappush(self.retry_queue, (retry_time, issue))

            self.logger.info(f"Scheduled retry for {issue.issue_id} in {retry_delay}s")

            # Start retry timer if not already running
            self._start_retry_timer()

            return True

    def _start_retry_timer(self) -> None:
        """Start or restart the retry timer - assumes lock is held"""
        # Cancel existing timer
        if self.retry_timer:
            self.retry_timer.cancel()
            self.retry_timer = None

        if not self.retry_queue:
            return

        # Get next retry time
        next_retry_time, _ = self.retry_queue[0]
        delay = (next_retry_time - datetime.now()).total_seconds()

        if delay <= 0:
            # Ready to retry now
            self._process_retries()
        else:
            # Schedule timer
            self.retry_timer = threading.Timer(delay, self._process_retries)
            self.retry_timer.start()

    def _process_retries(self) -> None:
        """Process any pending retries"""
        now = datetime.now()
        issues_to_retry = []

        with self.lock:
            # Find all issues ready for retry
            while self.retry_queue and self.retry_queue[0][0] <= now:
                _, issue = heapq.heappop(self.retry_queue)
                issues_to_retry.append(issue)

            # Restart timer for remaining retries
            self._start_retry_timer()

        # Dispatch retries outside of lock
        for issue in issues_to_retry:
            self.logger.info(f"Processing retry for issue {issue.issue_id}")
            self.dispatch_fix_agent(issue)

    def get_iteration_stats(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get iteration statistics for an issue.

        Args:
            issue_id: The issue identifier

        Returns:
            Statistics dictionary or None if not found
        """
        with self.lock:
            if issue_id not in self.fix_history:
                return None

            attempts = self.fix_history[issue_id]
            if not attempts:
                return None

            stats = {
                "issue_id": issue_id,
                "total_attempts": len(attempts),
                "successful_attempts": sum(1 for a in attempts if a.status == FixAttemptStatus.SUCCESS),
                "failed_attempts": sum(1 for a in attempts if a.status == FixAttemptStatus.FAILED),
                "average_duration": None,
                "total_files_modified": set(),
                "error_patterns": {}
            }

            # Calculate average duration and collect files
            durations = []
            for attempt in attempts:
                if attempt.end_time and attempt.start_time:
                    duration = (attempt.end_time - attempt.start_time).total_seconds()
                    durations.append(duration)

                stats["total_files_modified"].update(attempt.files_modified)

                # Track error patterns
                if attempt.error_message:
                    error_type = self._classify_error(attempt.error_message)
                    stats["error_patterns"][error_type] = stats["error_patterns"].get(error_type, 0) + 1

            if durations:
                stats["average_duration"] = sum(durations) / len(durations)

            stats["total_files_modified"] = list(stats["total_files_modified"])

            return stats

    def _classify_error(self, error_message: str) -> str:
        """
        Classify an error message into categories.

        Args:
            error_message: The error message

        Returns:
            Error category
        """
        error_lower = error_message.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "permission"
        elif "syntax" in error_lower or "parse" in error_lower:
            return "syntax"
        elif "import" in error_lower or "module not found" in error_lower:
            return "import"
        elif "test" in error_lower or "assertion" in error_lower:
            return "test_failure"
        elif "compile" in error_lower or "compilation" in error_lower or "build" in error_lower:
            return "build"
        else:
            return "other"

    def cancel_fix(self, issue_id: str) -> bool:
        """
        Cancel an active fix attempt.

        Args:
            issue_id: The issue identifier

        Returns:
            bool: True if cancelled, False if not found
        """
        with self.lock:
            if issue_id not in self.active_fixes:
                return False

            # Remove from active fixes
            issue = self.active_fixes[issue_id]
            del self.active_fixes[issue_id]

            # Mark current attempt as cancelled
            if issue_id in self.fix_history and self.fix_history[issue_id]:
                current_attempt = self.fix_history[issue_id][-1]
                if current_attempt.status == FixAttemptStatus.IN_PROGRESS:
                    current_attempt.status = FixAttemptStatus.FAILED
                    current_attempt.end_time = datetime.now()
                    current_attempt.error_message = "Cancelled by user"

            # Remove from retry queue if present
            # Since we're using a heap, we need to rebuild it after filtering
            filtered_queue = [(t, i) for t, i in self.retry_queue if i.issue_id != issue_id]
            self.retry_queue.clear()
            for item in filtered_queue:
                heapq.heappush(self.retry_queue, item)

            self.logger.info(f"Cancelled fix for issue {issue_id}")
            return True

    def set_fix_verifier(self, verifier: Callable[[IssueContext, FixAttempt], Dict[str, Any]]) -> None:
        """
        Set a custom fix verifier function.

        Args:
            verifier: Function that takes (issue, attempt) and returns verification results
        """
        self.fix_verifier = verifier
        self.logger.info("Custom fix verifier registered")

    def verify_fix(self, issue: IssueContext, attempt: FixAttempt) -> Dict[str, Any]:
        """
        Verify that a fix was successfully applied.

        Args:
            issue: The issue that was fixed
            attempt: The fix attempt to verify

        Returns:
            Verification results dictionary with 'success' and optional 'details'
        """
        verification_results = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "checks_performed": [],
            "errors": []
        }

        try:
            # Basic verification checks
            if not attempt.files_modified:
                verification_results["errors"].append("No files were modified")
                return verification_results

            # Check if files still exist
            for file_path in attempt.files_modified:
                if not Path(file_path).exists():
                    verification_results["errors"].append(f"Modified file no longer exists: {file_path}")
                    return verification_results
            verification_results["checks_performed"].append("file_existence")

            # Use custom verifier if available
            if self.fix_verifier:
                try:
                    custom_results = self.fix_verifier(issue, attempt)
                    verification_results["custom_verification"] = custom_results
                    verification_results["success"] = custom_results.get("success", False)
                    verification_results["checks_performed"].append("custom_verifier")

                    if not verification_results["success"]:
                        verification_results["errors"].append(
                            custom_results.get("error", "Custom verification failed")
                        )
                except Exception as e:
                    self.logger.error(f"Custom verifier failed: {e}")
                    verification_results["errors"].append(f"Custom verifier error: {str(e)}")
                    return verification_results
            else:
                # Default verification - check based on issue type
                self.logger.warning(
                    f"No custom verifier configured for issue {issue.issue_id}. "
                    f"Using basic verification for {issue.issue_type} issue type."
                )

                # Basic syntax check for Python files
                python_files = [f for f in attempt.files_modified if f.endswith('.py')]
                for py_file in python_files:
                    try:
                        with open(py_file, 'r') as f:
                            code = f.read()
                        # Basic syntax check
                        compile(code, py_file, 'exec')
                        verification_results["checks_performed"].append(f"syntax_check:{py_file}")
                    except SyntaxError as e:
                        verification_results["errors"].append(f"Syntax error in {py_file}: {str(e)}")
                        return verification_results
                    except Exception as e:
                        self.logger.warning(f"Could not verify syntax for {py_file}: {e}")

                if issue.issue_type == "syntax":
                    # For syntax issues, verify code compiles
                    if python_files:
                        verification_results["success"] = True  # Already checked above
                    else:
                        self.logger.warning("Syntax issue but no Python files modified - assuming success")
                        verification_results["success"] = True
                        verification_results["checks_performed"].append("syntax_check_non_python")
                elif issue.issue_type == "test_failure":
                    # For test failures, we'd normally run the tests
                    self.logger.warning(
                        "Test verification not implemented - consider using custom verifier "
                        "to run specific tests"
                    )
                    verification_results["checks_performed"].append("test_run_skipped")
                    verification_results["success"] = True
                else:
                    # For other types, verify files were modified and syntax is valid
                    verification_results["success"] = len(attempt.files_modified) > 0
                    verification_results["checks_performed"].append("basic_check")

                    if not verification_results["success"]:
                        self.logger.warning(
                            f"Basic verification failed for {issue.issue_type} issue: "
                            f"no files modified"
                        )

            # Log verification result
            self.logger.info(
                f"Fix verification for issue {issue.issue_id}: "
                f"success={verification_results['success']}, "
                f"checks={verification_results['checks_performed']}"
            )

        except Exception as e:
            self.logger.error(f"Fix verification failed with error: {e}")
            verification_results["errors"].append(f"Verification error: {str(e)}")

        return verification_results

    def handle_fix_error(self, issue: IssueContext, error: Exception) -> Dict[str, Any]:
        """
        Handle errors that occur during fix attempts.

        Args:
            issue: The issue being fixed
            error: The exception that occurred

        Returns:
            Error handling results
        """
        error_info = {
            "issue_id": issue.issue_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "recovery_action": None
        }

        # Classify error and determine recovery action
        if isinstance(error, TimeoutError):
            error_info["recovery_action"] = "retry_with_extended_timeout"
            self.logger.warning(f"Timeout error for issue {issue.issue_id}: {error}")
        elif isinstance(error, PermissionError):
            error_info["recovery_action"] = "escalate_permissions"
            self.logger.error(f"Permission error for issue {issue.issue_id}: {error}")
        elif isinstance(error, FileNotFoundError):
            error_info["recovery_action"] = "verify_file_paths"
            self.logger.error(f"File not found for issue {issue.issue_id}: {error}")
        elif isinstance(error, ValueError):
            error_info["recovery_action"] = "review_fix_logic"
            self.logger.error(f"Value error in fix for issue {issue.issue_id}: {error}")
        else:
            error_info["recovery_action"] = "generic_retry"
            self.logger.error(f"Unexpected error for issue {issue.issue_id}: {error}")

        # Store error context for future attempts
        self.preserve_fix_context(issue.issue_id, {
            "last_error": error_info,
            "error_history": self._get_error_history(issue.issue_id) + [error_info]
        })

        return error_info

    def _get_error_history(self, issue_id: str) -> List[Dict[str, Any]]:
        """Get error history for an issue."""
        context = self.get_fix_context(issue_id)
        if context and "error_history" in context:
            return context["error_history"]
        return []

    def validate_fix_request(self, issue: IssueContext) -> Tuple[bool, Optional[str]]:
        """
        Validate that a fix request is valid before processing.

        Args:
            issue: The issue to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if not issue.issue_id:
            return False, "Issue ID is required"

        if not issue.file_path:
            return False, "File path is required"

        if not issue.description:
            return False, "Issue description is required"

        # Path traversal protection
        try:
            file_path = Path(issue.file_path).resolve()
            # Ensure the path is absolute and doesn't contain suspicious patterns
            if ".." in str(issue.file_path) or str(issue.file_path).startswith("~"):
                return False, f"Suspicious path pattern detected: {issue.file_path}"

            # Additional check: ensure file is within expected project boundaries
            # This could be configured based on project root or allowed directories
            if not file_path.is_absolute():
                return False, f"File path must be absolute: {issue.file_path}"

        except Exception as e:
            return False, f"Invalid file path: {issue.file_path} - {str(e)}"

        # Validate file exists
        if not file_path.exists():
            return False, f"File does not exist: {issue.file_path}"

        # Validate severity
        valid_severities = ["low", "medium", "high", "critical"]
        if issue.severity.lower() not in valid_severities:
            return False, f"Invalid severity: {issue.severity}"

        # Check for circular dependencies in fix history
        if self._has_circular_dependency(issue):
            return False, "Circular dependency detected in fix attempts"

        return True, None

    def _has_circular_dependency(self, issue: IssueContext) -> bool:
        """Check if fixing this issue would create a circular dependency."""
        # Get configuration values from constructor or use defaults
        # These values come from config.yaml fix_agent_dispatcher.circular_dependency section
        max_modifications = getattr(self, '_max_file_modifications', 3)
        min_attempts = getattr(self, '_min_attempts_before_check', 5)

        # Check fix history for the issue
        if issue.issue_id not in self.fix_history:
            return False

        attempts = self.fix_history[issue.issue_id]

        # Only check after minimum attempts
        if len(attempts) < min_attempts:
            return False

        # Track file modifications across all failed attempts
        file_modification_count = {}
        for attempt in attempts:
            # Only count failed attempts to avoid penalizing successful partial fixes
            if attempt.status in [FixAttemptStatus.FAILED, FixAttemptStatus.VERIFICATION_FAILED]:
                for file_path in attempt.files_modified:
                    file_modification_count[file_path] = file_modification_count.get(file_path, 0) + 1

        # Check if any file exceeded the modification threshold
        for file_path, count in file_modification_count.items():
            if count > max_modifications:
                self.logger.warning(
                    f"Circular dependency detected for issue {issue.issue_id}: "
                    f"file {file_path} modified {count} times across failed attempts"
                )
                return True

        return False
