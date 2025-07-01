"""
Fix Iteration Tracker and Escalation Handler System

This module implements the fix iteration tracking, limit enforcement, and escalation
workflow for the Claude Cadence agent dispatch system.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict

from .agent_messages import AgentMessage

logger = logging.getLogger(__name__)


class EscalationStrategy(str, Enum):
    """Escalation strategies when fix limits are exceeded"""
    LOG_ONLY = "log_only"
    NOTIFY_SUPERVISOR = "notify_supervisor"
    PAUSE_AUTOMATION = "pause_automation"
    MARK_FOR_MANUAL_REVIEW = "mark_for_manual_review"


class PersistenceType(str, Enum):
    """Persistence storage types for attempt counts"""
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"


@dataclass
class FixAttempt:
    """Records information about a single fix attempt"""
    attempt_number: int
    timestamp: str
    agent_id: Optional[str] = None
    error_message: Optional[str] = None
    success: bool = False
    duration_seconds: Optional[float] = None
    files_modified: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FixAttempt':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TaskFixHistory:
    """Complete fix history for a task"""
    task_id: str
    current_attempt_count: int = 0
    is_escalated: bool = False
    escalation_timestamp: Optional[str] = None
    escalation_reason: Optional[str] = None
    attempts: List[FixAttempt] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_attempt(self, attempt: FixAttempt) -> None:
        """Add a new fix attempt"""
        self.attempts.append(attempt)
        self.current_attempt_count = len(self.attempts)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_escalated(self, reason: str) -> None:
        """Mark task as escalated"""
        self.is_escalated = True
        self.escalation_timestamp = datetime.now(timezone.utc).isoformat()
        self.escalation_reason = reason
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def reset_attempts(self) -> None:
        """Reset attempts after successful fix"""
        self.current_attempt_count = 0
        self.is_escalated = False
        self.escalation_timestamp = None
        self.escalation_reason = None
        self.attempts.clear()
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['attempts'] = [attempt.to_dict() for attempt in self.attempts]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskFixHistory':
        """Create from dictionary"""
        attempts_data = data.pop('attempts', [])
        history = cls(**data)
        history.attempts = [FixAttempt.from_dict(attempt_data) for attempt_data in attempts_data]
        return history


class FixIterationTracker:
    """
    Tracks fix attempts per task with configurable persistence.

    Maintains state for each fix attempt across the workflow, supporting
    multiple persistence backends and thread-safe operations.
    """

    def __init__(
        self,
        persistence_type: PersistenceType = PersistenceType.MEMORY,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the fix iteration tracker.

        Args:
            persistence_type: Type of persistence to use
            storage_path: Path for file-based persistence
        """
        self.persistence_type = persistence_type
        self.storage_path = storage_path
        self._task_histories: Dict[str, TaskFixHistory] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations

        # Initialize persistence
        if persistence_type == PersistenceType.FILE:
            self._init_file_storage()
        elif persistence_type == PersistenceType.DATABASE:
            self._init_database_storage()

        logger.info(f"FixIterationTracker initialized with {persistence_type.value} persistence")

    def _init_file_storage(self) -> None:
        """Initialize file-based storage"""
        if not self.storage_path:
            self.storage_path = ".cadence/fix_attempts.json"

        storage_file = Path(self.storage_path)
        storage_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if file exists
        if storage_file.exists():
            try:
                with open(storage_file, 'r') as f:
                    data = json.load(f)
                    self._task_histories = {
                        task_id: TaskFixHistory.from_dict(history_data)
                        for task_id, history_data in data.items()
                    }
                logger.info(f"Loaded {len(self._task_histories)} task histories from {storage_file}")
            except Exception as e:
                logger.error(f"Error loading fix attempt data: {e}")
                self._task_histories = {}

    def _init_database_storage(self) -> None:
        """Initialize database storage (placeholder for future implementation)"""
        # Placeholder for database implementation
        logger.warning("Database persistence not yet implemented, falling back to memory")
        self.persistence_type = PersistenceType.MEMORY

    def _save_to_storage(self) -> None:
        """Save current state to persistent storage"""
        if self.persistence_type == PersistenceType.FILE and self.storage_path:
            try:
                data = {
                    task_id: history.to_dict()
                    for task_id, history in self._task_histories.items()
                }

                storage_file = Path(self.storage_path)
                storage_file.parent.mkdir(parents=True, exist_ok=True)

                # Atomic write using temporary file
                temp_file = storage_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)

                temp_file.replace(storage_file)
                logger.debug(f"Saved fix attempt data to {storage_file}")

            except Exception as e:
                logger.error(f"Error saving fix attempt data: {e}")

    def get_task_history(self, task_id: str) -> TaskFixHistory:
        """
        Get or create task fix history.

        Args:
            task_id: Task identifier

        Returns:
            TaskFixHistory for the specified task
        """
        with self._lock:
            if task_id not in self._task_histories:
                self._task_histories[task_id] = TaskFixHistory(task_id=task_id)
                if self.persistence_type != PersistenceType.MEMORY:
                    self._save_to_storage()

            return self._task_histories[task_id]

    def start_fix_attempt(
        self,
        task_id: str,
        agent_id: Optional[str] = None
    ) -> int:
        """
        Start a new fix attempt for a task.

        Args:
            task_id: Task identifier
            agent_id: Optional agent identifier

        Returns:
            Current attempt number
        """
        with self._lock:
            history = self.get_task_history(task_id)

            attempt = FixAttempt(
                attempt_number=history.current_attempt_count + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_id=agent_id
            )

            history.add_attempt(attempt)

            if self.persistence_type != PersistenceType.MEMORY:
                self._save_to_storage()

            logger.info(f"Started fix attempt {attempt.attempt_number} for task {task_id}")
            return attempt.attempt_number

    def complete_fix_attempt(
        self,
        task_id: str,
        success: bool,
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        files_modified: Optional[List[str]] = None
    ) -> None:
        """
        Complete the current fix attempt for a task.

        Args:
            task_id: Task identifier
            success: Whether the fix was successful
            error_message: Error message if fix failed
            duration_seconds: Duration of the fix attempt
            files_modified: List of files modified during fix
        """
        with self._lock:
            history = self.get_task_history(task_id)

            if not history.attempts:
                logger.warning(f"No active fix attempt found for task {task_id}")
                return

            # Update the latest attempt
            latest_attempt = history.attempts[-1]
            latest_attempt.success = success
            latest_attempt.error_message = error_message
            latest_attempt.duration_seconds = duration_seconds
            latest_attempt.files_modified = files_modified or []

            history.updated_at = datetime.now(timezone.utc).isoformat()

            if success:
                logger.info(f"Fix attempt {latest_attempt.attempt_number} succeeded for task {task_id}")
            else:
                logger.warning(f"Fix attempt {latest_attempt.attempt_number} failed for task {task_id}: {error_message}")

            if self.persistence_type != PersistenceType.MEMORY:
                self._save_to_storage()

    def reset_task_attempts(self, task_id: str) -> None:
        """
        Reset fix attempts for a task (e.g., after successful fix).

        Args:
            task_id: Task identifier
        """
        with self._lock:
            history = self.get_task_history(task_id)
            history.reset_attempts()

            if self.persistence_type != PersistenceType.MEMORY:
                self._save_to_storage()

            logger.info(f"Reset fix attempts for task {task_id}")

    def get_attempt_count(self, task_id: str) -> int:
        """
        Get current attempt count for a task.

        Args:
            task_id: Task identifier

        Returns:
            Current attempt count
        """
        with self._lock:
            return self.get_task_history(task_id).current_attempt_count

    def is_task_escalated(self, task_id: str) -> bool:
        """
        Check if a task has been escalated.

        Args:
            task_id: Task identifier

        Returns:
            True if task is escalated
        """
        with self._lock:
            return self.get_task_history(task_id).is_escalated

    def mark_task_escalated(self, task_id: str, reason: str) -> None:
        """
        Mark a task as escalated.

        Args:
            task_id: Task identifier
            reason: Reason for escalation
        """
        with self._lock:
            history = self.get_task_history(task_id)
            history.mark_escalated(reason)

            if self.persistence_type != PersistenceType.MEMORY:
                self._save_to_storage()

            logger.warning(f"Task {task_id} escalated: {reason}")

    def get_all_escalated_tasks(self) -> List[str]:
        """
        Get list of all escalated task IDs.

        Returns:
            List of escalated task IDs
        """
        with self._lock:
            return [
                task_id for task_id, history in self._task_histories.items()
                if history.is_escalated
            ]

    def cleanup(self) -> None:
        """Clean up resources and save final state"""
        with self._lock:
            if self.persistence_type != PersistenceType.MEMORY:
                self._save_to_storage()
            logger.info("FixIterationTracker cleanup completed")


class FixAttemptLimitEnforcer:
    """
    Enforces maximum fix attempt limits and triggers escalation.

    Checks current attempt count against configurable limits before allowing
    new fix attempts and triggers escalation when limits are exceeded.
    """

    def __init__(
        self,
        iteration_tracker: FixIterationTracker,
        max_fix_iterations: int = 3,
        escalation_handler: Optional['EscalationHandler'] = None
    ):
        """
        Initialize the fix attempt limit enforcer.

        Args:
            iteration_tracker: Fix iteration tracker instance
            max_fix_iterations: Maximum allowed fix attempts
            escalation_handler: Optional escalation handler
        """
        self.iteration_tracker = iteration_tracker
        self.max_fix_iterations = max_fix_iterations
        self.escalation_handler = escalation_handler
        self._lock = threading.Lock()

        logger.info(f"FixAttemptLimitEnforcer initialized with max_iterations={max_fix_iterations}")

    def can_attempt_fix(self, task_id: str) -> bool:
        """
        Check if a fix attempt is allowed for a task.

        Args:
            task_id: Task identifier

        Returns:
            True if fix attempt is allowed
        """
        with self._lock:
            # Check if task is already escalated
            if self.iteration_tracker.is_task_escalated(task_id):
                logger.warning(f"Fix attempt blocked for escalated task {task_id}")
                return False

            # Check attempt count
            current_count = self.iteration_tracker.get_attempt_count(task_id)
            allowed = current_count < self.max_fix_iterations

            if not allowed:
                logger.warning(f"Fix attempt blocked for task {task_id}: {current_count}/{self.max_fix_iterations} attempts used")

            return allowed

    def validate_fix_attempt(self, task_id: str) -> bool:
        """
        Validate and potentially escalate a fix attempt.

        Args:
            task_id: Task identifier

        Returns:
            True if attempt is allowed, False if escalated
        """
        with self._lock:
            if not self.can_attempt_fix(task_id):
                # Trigger escalation
                current_count = self.iteration_tracker.get_attempt_count(task_id)
                reason = f"Maximum fix attempts exceeded: {current_count}/{self.max_fix_iterations}"

                self.iteration_tracker.mark_task_escalated(task_id, reason)

                if self.escalation_handler:
                    try:
                        self.escalation_handler.handle_escalation(task_id, reason, current_count)
                    except Exception as e:
                        logger.error(f"Error in escalation handler: {e}")

                return False

            return True

    def check_and_escalate_if_needed(self, task_id: str) -> bool:
        """
        Check if escalation is needed after a failed fix attempt.

        Args:
            task_id: Task identifier

        Returns:
            True if escalated, False otherwise
        """
        with self._lock:
            current_count = self.iteration_tracker.get_attempt_count(task_id)

            if current_count >= self.max_fix_iterations:
                reason = f"Fix attempts exhausted: {current_count}/{self.max_fix_iterations}"
                self.iteration_tracker.mark_task_escalated(task_id, reason)

                if self.escalation_handler:
                    try:
                        self.escalation_handler.handle_escalation(task_id, reason, current_count)
                    except Exception as e:
                        logger.error(f"Error in escalation handler: {e}")

                return True

            return False


class EscalationHandler:
    """
    Handles escalation when fix attempt limits are exceeded.

    Provides configurable escalation strategies including logging, supervisor
    notification, automation pausing, and manual review marking.
    """

    def __init__(
        self,
        escalation_strategy: EscalationStrategy = EscalationStrategy.LOG_ONLY,
        supervisor_callback: Optional[Callable[[str, str, int], None]] = None,
        notification_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the escalation handler.

        Args:
            escalation_strategy: Strategy to use for escalation
            supervisor_callback: Callback for supervisor notifications
            notification_callback: Callback for general notifications
        """
        self.escalation_strategy = escalation_strategy
        self.supervisor_callback = supervisor_callback
        self.notification_callback = notification_callback
        self._escalated_tasks: Dict[str, Dict[str, Any]] = {}
        self._automation_paused = False
        self._lock = threading.Lock()

        logger.info(f"EscalationHandler initialized with strategy: {escalation_strategy.value}")

    def handle_escalation(self, task_id: str, reason: str, attempt_count: int) -> None:
        """
        Handle escalation for a task.

        Args:
            task_id: Task identifier
            reason: Reason for escalation
            attempt_count: Number of attempts made
        """
        with self._lock:
            escalation_info = {
                'task_id': task_id,
                'reason': reason,
                'attempt_count': attempt_count,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'strategy': self.escalation_strategy.value
            }

            self._escalated_tasks[task_id] = escalation_info

            # Execute escalation strategy
            if self.escalation_strategy == EscalationStrategy.LOG_ONLY:
                self._log_escalation(escalation_info)

            elif self.escalation_strategy == EscalationStrategy.NOTIFY_SUPERVISOR:
                self._notify_supervisor(escalation_info)

            elif self.escalation_strategy == EscalationStrategy.PAUSE_AUTOMATION:
                self._pause_automation(escalation_info)

            elif self.escalation_strategy == EscalationStrategy.MARK_FOR_MANUAL_REVIEW:
                self._mark_for_manual_review(escalation_info)

            # Always log the escalation
            logger.error(f"ESCALATION: Task {task_id} escalated after {attempt_count} attempts - {reason}")

    def _log_escalation(self, escalation_info: Dict[str, Any]) -> None:
        """Log escalation event"""
        logger.warning(f"Task {escalation_info['task_id']} escalated: {escalation_info['reason']}")

    def _notify_supervisor(self, escalation_info: Dict[str, Any]) -> None:
        """Notify supervisor of escalation"""
        self._log_escalation(escalation_info)

        if self.supervisor_callback:
            try:
                self.supervisor_callback(
                    escalation_info['task_id'],
                    escalation_info['reason'],
                    escalation_info['attempt_count']
                )
                logger.info(f"Supervisor notified of escalation for task {escalation_info['task_id']}")
            except Exception as e:
                logger.error(f"Error notifying supervisor: {e}")
        else:
            logger.warning("No supervisor callback configured for notification")

    def _pause_automation(self, escalation_info: Dict[str, Any]) -> None:
        """Pause automation"""
        self._log_escalation(escalation_info)
        self._automation_paused = True

        logger.critical(f"AUTOMATION PAUSED due to escalation of task {escalation_info['task_id']}")

        if self.notification_callback:
            try:
                self.notification_callback(
                    "AUTOMATION_PAUSED",
                    f"Automation paused due to task {escalation_info['task_id']} escalation"
                )
            except Exception as e:
                logger.error(f"Error sending pause notification: {e}")

    def _mark_for_manual_review(self, escalation_info: Dict[str, Any]) -> None:
        """Mark task for manual review"""
        self._log_escalation(escalation_info)

        # Create escalation message for agent dispatcher
        escalation_message = {
            'message_type': 'ESCALATION_REQUIRED',
            'task_id': escalation_info['task_id'],
            'reason': escalation_info['reason'],
            'attempt_count': escalation_info['attempt_count'],
            'timestamp': escalation_info['timestamp'],
            'requires_manual_review': True
        }

        logger.info(f"Task {escalation_info['task_id']} marked for manual review")

        if self.notification_callback:
            try:
                self.notification_callback(
                    "MANUAL_REVIEW_REQUIRED",
                    json.dumps(escalation_message)
                )
            except Exception as e:
                logger.error(f"Error sending manual review notification: {e}")

    def is_automation_paused(self) -> bool:
        """Check if automation is paused"""
        with self._lock:
            return self._automation_paused

    def resume_automation(self) -> None:
        """Resume automation (manual action)"""
        with self._lock:
            self._automation_paused = False
            logger.info("Automation resumed manually")

    def get_escalated_tasks(self) -> List[Dict[str, Any]]:
        """Get list of escalated tasks"""
        with self._lock:
            return list(self._escalated_tasks.values())

    def clear_escalation(self, task_id: str) -> bool:
        """
        Clear escalation for a task (e.g., after manual resolution).

        Args:
            task_id: Task identifier

        Returns:
            True if escalation was cleared
        """
        with self._lock:
            if task_id in self._escalated_tasks:
                del self._escalated_tasks[task_id]
                logger.info(f"Escalation cleared for task {task_id}")
                return True
            return False


class FixIterationManager:
    """
    Combined manager for fix iteration tracking, limit enforcement, and escalation.

    Provides a unified interface for all fix iteration management functionality
    and integrates with the AgentDispatcher messaging protocol.
    """

    def __init__(
        self,
        max_fix_iterations: int = 3,
        escalation_strategy: EscalationStrategy = EscalationStrategy.LOG_ONLY,
        persistence_type: PersistenceType = PersistenceType.MEMORY,
        storage_path: Optional[str] = None,
        supervisor_callback: Optional[Callable[[str, str, int], None]] = None,
        notification_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the fix iteration manager.

        Args:
            max_fix_iterations: Maximum allowed fix attempts
            escalation_strategy: Strategy for handling escalations
            persistence_type: Type of persistence for attempt counts
            storage_path: Path for file-based persistence
            supervisor_callback: Callback for supervisor notifications
            notification_callback: Callback for general notifications
        """
        # Initialize components
        self.iteration_tracker = FixIterationTracker(
            persistence_type=persistence_type,
            storage_path=storage_path
        )

        self.escalation_handler = EscalationHandler(
            escalation_strategy=escalation_strategy,
            supervisor_callback=supervisor_callback,
            notification_callback=notification_callback
        )

        self.limit_enforcer = FixAttemptLimitEnforcer(
            iteration_tracker=self.iteration_tracker,
            max_fix_iterations=max_fix_iterations,
            escalation_handler=self.escalation_handler
        )

        logger.info("FixIterationManager initialized successfully")

    def can_attempt_fix(self, task_id: str) -> bool:
        """Check if a fix attempt is allowed"""
        return self.limit_enforcer.can_attempt_fix(task_id)

    def start_fix_attempt(self, task_id: str, agent_id: Optional[str] = None) -> Optional[int]:
        """
        Start a new fix attempt if allowed.

        Args:
            task_id: Task identifier
            agent_id: Optional agent identifier

        Returns:
            Attempt number if allowed, None if escalated
        """
        if not self.limit_enforcer.validate_fix_attempt(task_id):
            return None

        return self.iteration_tracker.start_fix_attempt(task_id, agent_id)

    def complete_fix_attempt(
        self,
        task_id: str,
        success: bool,
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        files_modified: Optional[List[str]] = None
    ) -> bool:
        """
        Complete a fix attempt and check for escalation.

        Args:
            task_id: Task identifier
            success: Whether the fix was successful
            error_message: Error message if fix failed
            duration_seconds: Duration of the fix attempt
            files_modified: List of files modified during fix

        Returns:
            True if no escalation occurred, False if escalated
        """
        self.iteration_tracker.complete_fix_attempt(
            task_id, success, error_message, duration_seconds, files_modified
        )

        if success:
            # Reset attempts on success
            self.iteration_tracker.reset_task_attempts(task_id)
            return True
        else:
            # Check for escalation on failure
            return not self.limit_enforcer.check_and_escalate_if_needed(task_id)

    def enhance_dispatch_message(self, message: AgentMessage, task_id: str) -> AgentMessage:
        """
        Enhance a dispatch message with attempt count metadata.

        Args:
            message: Original agent message
            task_id: Task identifier

        Returns:
            Enhanced message with attempt metadata
        """
        attempt_count = self.iteration_tracker.get_attempt_count(task_id)
        is_escalated = self.iteration_tracker.is_task_escalated(task_id)

        # Add metadata to payload
        if message.payload is None:
            message.payload = {}

        message.payload.update({
            'attempt_count': attempt_count,
            'is_escalated': is_escalated,
            'max_attempts': self.limit_enforcer.max_fix_iterations
        })

        return message

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a task.

        Args:
            task_id: Task identifier

        Returns:
            Task status dictionary
        """
        history = self.iteration_tracker.get_task_history(task_id)

        return {
            'task_id': task_id,
            'attempt_count': history.current_attempt_count,
            'max_attempts': self.limit_enforcer.max_fix_iterations,
            'is_escalated': history.is_escalated,
            'escalation_reason': history.escalation_reason,
            'escalation_timestamp': history.escalation_timestamp,
            'can_attempt_fix': self.can_attempt_fix(task_id),
            'automation_paused': self.escalation_handler.is_automation_paused(),
            'created_at': history.created_at,
            'updated_at': history.updated_at,
            'recent_attempts': [attempt.to_dict() for attempt in history.attempts[-3:]]  # Last 3 attempts
        }

    def cleanup(self) -> None:
        """Clean up all components"""
        self.iteration_tracker.cleanup()
        logger.info("FixIterationManager cleanup completed")
