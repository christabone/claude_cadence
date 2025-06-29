"""
Workflow State Machine for tracking code review dispatch workflow states.

This module provides a state machine to track workflow transitions through:
working → review → fixing → complete states with proper event logging and persistence.
"""

import json
import logging
import threading
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import deque

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Workflow states for the code review dispatch system"""
    WORKING = "working"
    REVIEW_TRIGGERED = "review_triggered"
    REVIEWING = "reviewing"
    FIX_REQUIRED = "fix_required"
    FIXING = "fixing"
    VERIFICATION = "verification"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StateTransition:
    """Represents a state transition event"""
    from_state: WorkflowState
    to_state: WorkflowState
    timestamp: datetime
    trigger: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class WorkflowContext:
    """Context information for the workflow"""
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    project_path: Optional[str] = None
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""
    pass


class WorkflowStateMachine:
    """
    State machine for tracking workflow states through the code review dispatch process.

    Manages transitions through: WORKING → REVIEW_TRIGGERED → REVIEWING →
    FIX_REQUIRED → FIXING → VERIFICATION → COMPLETE (or ERROR at any point).

    Features:
    - Thread-safe state management
    - Event logging for all transitions
    - State persistence for recovery
    - Transition guards to prevent invalid changes
    - Integration hooks for external systems
    """

    # Define valid state transitions
    VALID_TRANSITIONS: Dict[WorkflowState, Set[WorkflowState]] = {
        WorkflowState.WORKING: {
            WorkflowState.REVIEW_TRIGGERED,
            WorkflowState.COMPLETE,
            WorkflowState.ERROR
        },
        WorkflowState.REVIEW_TRIGGERED: {
            WorkflowState.REVIEWING,
            WorkflowState.ERROR
        },
        WorkflowState.REVIEWING: {
            WorkflowState.FIX_REQUIRED,
            WorkflowState.COMPLETE,
            WorkflowState.ERROR
        },
        WorkflowState.FIX_REQUIRED: {
            WorkflowState.FIXING,
            WorkflowState.ERROR
        },
        WorkflowState.FIXING: {
            WorkflowState.VERIFICATION,
            WorkflowState.FIX_REQUIRED,  # Allow retry if fix fails
            WorkflowState.ERROR
        },
        WorkflowState.VERIFICATION: {
            WorkflowState.COMPLETE,
            WorkflowState.FIX_REQUIRED,  # Allow retry if verification fails
            WorkflowState.ERROR
        },
        WorkflowState.COMPLETE: set(),  # Terminal state
        WorkflowState.ERROR: {
            # Allow recovery from error state
            WorkflowState.WORKING,
            WorkflowState.REVIEW_TRIGGERED,
            WorkflowState.REVIEWING,
            WorkflowState.FIXING
        }
    }

    def __init__(
        self,
        initial_state: WorkflowState = WorkflowState.WORKING,
        context: Optional[WorkflowContext] = None,
        persistence_path: Optional[Path] = None,
        max_history_size: Optional[int] = None,
        enable_persistence: bool = True
    ):
        """
        Initialize the workflow state machine.

        Args:
            initial_state: Starting state for the workflow
            context: Context information for the workflow
            persistence_path: Path to save state for recovery
            max_history_size: Maximum number of transitions to keep in history (default: 1000)
            enable_persistence: Whether to enable state persistence
        """
        self._current_state = initial_state
        self._context = context or WorkflowContext()
        self._persistence_path = persistence_path
        self._enable_persistence = enable_persistence
        self._lock = threading.Lock()

        # Use bounded deque for transition history to prevent memory leaks
        # Default to 1000 if not specified
        history_size = max_history_size if max_history_size is not None else 1000
        self._transition_history: deque = deque(maxlen=history_size)
        self._event_handlers: Dict[WorkflowState, List[Callable]] = {}
        self._transition_handlers: Dict[tuple, List[Callable]] = {}

        logger.info(f"WorkflowStateMachine initialized in state: {initial_state.value}")

        # Load persisted state if available
        if self._enable_persistence and self._persistence_path and self._persistence_path.exists():
            self._load_state()

    @property
    def current_state(self) -> WorkflowState:
        """Get the current workflow state"""
        with self._lock:
            return self._current_state

    @property
    def context(self) -> WorkflowContext:
        """Get the workflow context"""
        import copy
        with self._lock:
            return copy.deepcopy(self._context)

    @property
    def transition_history(self) -> List[StateTransition]:
        """Get the complete transition history"""
        with self._lock:
            return list(self._transition_history)

    def can_transition_to(self, target_state: WorkflowState) -> bool:
        """
        Check if transition to target state is valid.

        Args:
            target_state: State to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        with self._lock:
            return target_state in self.VALID_TRANSITIONS.get(self._current_state, set())

    def transition_to(
        self,
        target_state: WorkflowState,
        trigger: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to transition to the target state.

        Args:
            target_state: State to transition to
            trigger: What triggered this transition
            metadata: Additional metadata for the transition

        Returns:
            True if transition succeeded, False otherwise

        Raises:
            StateTransitionError: If transition is invalid
        """
        # Validate metadata input
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        with self._lock:
            if not self.can_transition_to(target_state):
                error_msg = f"Invalid transition from {self._current_state.value} to {target_state.value}"
                logger.error(error_msg)
                raise StateTransitionError(error_msg)

            # Record the transition
            transition = StateTransition(
                from_state=self._current_state,
                to_state=target_state,
                timestamp=datetime.now(),
                trigger=trigger,
                metadata=metadata or {},
                session_id=self._context.session_id
            )

            # Update state
            old_state = self._current_state
            self._current_state = target_state
            self._transition_history.append(transition)

            logger.info(
                f"State transition: {old_state.value} → {target_state.value} "
                f"(trigger: {trigger})"
            )

            # Persist state if configured
            if self._enable_persistence:
                self._persist_state()

            # Trigger event handlers
            self._trigger_event_handlers(old_state, target_state, transition)

            return True

    def update_context(self, **kwargs) -> None:
        """
        Update the workflow context with new information.

        Args:
            **kwargs: Context fields to update
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._context, key):
                    setattr(self._context, key, value)
                else:
                    # Store in metadata if not a direct field
                    self._context.metadata[key] = value

            # Persist updated context
            if self._enable_persistence:
                self._persist_state()

    def add_issue(self, issue: Dict[str, Any]) -> None:
        """Add an issue to the context"""
        # Validate issue data
        if not isinstance(issue, dict):
            raise ValueError("Issue must be a dictionary")

        with self._lock:
            self._context.issues_found.append(issue)
            if self._enable_persistence:
                self._persist_state()

    def add_fix(self, fix: Dict[str, Any]) -> None:
        """Add a fix to the context"""
        # Validate fix data
        if not isinstance(fix, dict):
            raise ValueError("Fix must be a dictionary")

        with self._lock:
            self._context.fixes_applied.append(fix)
            if self._enable_persistence:
                self._persist_state()

    def register_state_handler(
        self,
        state: WorkflowState,
        handler: Callable[[WorkflowState, StateTransition], None]
    ) -> None:
        """
        Register a handler to be called when entering a specific state.

        Args:
            state: State to watch for
            handler: Function to call when entering the state
        """
        if state not in self._event_handlers:
            self._event_handlers[state] = []
        self._event_handlers[state].append(handler)

    def register_transition_handler(
        self,
        from_state: WorkflowState,
        to_state: WorkflowState,
        handler: Callable[[StateTransition], None]
    ) -> None:
        """
        Register a handler for specific state transitions.

        Args:
            from_state: Source state
            to_state: Target state
            handler: Function to call on this transition
        """
        key = (from_state, to_state)
        if key not in self._transition_handlers:
            self._transition_handlers[key] = []
        self._transition_handlers[key].append(handler)

    def reset(self, initial_state: WorkflowState = WorkflowState.WORKING) -> None:
        """
        Reset the state machine to initial state.

        Args:
            initial_state: State to reset to
        """
        with self._lock:
            logger.info(f"Resetting workflow state machine to {initial_state.value}")
            self._current_state = initial_state
            self._transition_history.clear()
            self._context = WorkflowContext()

            # Force immediate persistence for reset
            if self._enable_persistence:
                self._persist_state()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state and context.

        Returns:
            Dictionary with state information
        """
        with self._lock:
            return {
                "current_state": self._current_state.value,
                "context": asdict(self._context),
                "transition_count": len(self._transition_history),
                "last_transition": (
                    asdict(self._transition_history[-1])
                    if self._transition_history else None
                ),
                "valid_next_states": [
                    state.value for state in
                    self.VALID_TRANSITIONS.get(self._current_state, set())
                ]
            }

    def _trigger_event_handlers(
        self,
        old_state: WorkflowState,
        new_state: WorkflowState,
        transition: StateTransition
    ) -> None:
        """Trigger registered event handlers"""
        # Collect handlers while holding lock
        with self._lock:
            state_handlers = list(self._event_handlers.get(new_state, []))
            transition_handlers = list(self._transition_handlers.get((old_state, new_state), []))

        # Execute handlers outside lock to prevent deadlock
        try:
            # Trigger state entry handlers
            for handler in state_handlers:
                try:
                    handler(new_state, transition)
                except Exception as e:
                    logger.error(f"Error in state handler for {new_state.value}: {e}")

            # Trigger transition handlers
            for handler in transition_handlers:
                try:
                    handler(transition)
                except Exception as e:
                    logger.error(f"Error in transition handler {old_state.value}→{new_state.value}: {e}")
        except Exception as e:
            logger.error(f"Error triggering event handlers: {e}")

    def _persist_state(self) -> None:
        """Persist current state to disk for recovery"""
        if not self._persistence_path:
            return

        try:
            state_data = {
                "current_state": self._current_state.value,
                "context": asdict(self._context),
                "transition_history": [
                    {
                        "from_state": t.from_state.value,
                        "to_state": t.to_state.value,
                        "timestamp": t.timestamp.isoformat(),
                        "trigger": t.trigger,
                        "metadata": t.metadata,
                        "session_id": t.session_id
                    }
                    for t in self._transition_history
                ],
                "metadata": {
                    "version": "1.0",
                    "persisted_at": datetime.now().isoformat(),
                    "history_size": len(self._transition_history)
                }
            }

            # Ensure directory exists
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Write state atomically using temp file
            temp_path = self._persistence_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            # Atomic rename
            temp_path.replace(self._persistence_path)

        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def _load_state(self) -> None:
        """Load persisted state from disk"""
        try:
            with open(self._persistence_path, 'r') as f:
                state_data = json.load(f)

            # Validate loaded data
            if not isinstance(state_data, dict):
                raise ValueError("Invalid state data format")

            if "current_state" not in state_data:
                raise ValueError("Missing current_state in persisted data")

            # Restore current state
            self._current_state = WorkflowState(state_data["current_state"])

            # Restore context
            context_data = state_data.get("context", {})
            if not isinstance(context_data, dict):
                raise ValueError("Invalid context data format")

            self._context = WorkflowContext(**context_data)

            # Restore transition history
            self._transition_history.clear()
            for t_data in state_data.get("transition_history", []):
                if not isinstance(t_data, dict):
                    logger.warning("Skipping invalid transition data")
                    continue

                try:
                    transition = StateTransition(
                        from_state=WorkflowState(t_data["from_state"]),
                        to_state=WorkflowState(t_data["to_state"]),
                        timestamp=datetime.fromisoformat(t_data["timestamp"]),
                        trigger=t_data["trigger"],
                        metadata=t_data.get("metadata", {}),
                        session_id=t_data.get("session_id")
                    )
                    self._transition_history.append(transition)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid transition: {e}")

            logger.info(f"Restored workflow state: {self._current_state.value}")

        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")
            # Continue with current state if loading fails


class WorkflowStateManager:
    """
    Manager for multiple workflow state machines.

    Allows managing multiple concurrent workflows by session or task ID.
    """

    def __init__(self, persistence_dir: Optional[Path] = None, max_history_size: Optional[int] = None):
        """
        Initialize the workflow state manager.

        Args:
            persistence_dir: Directory to store state files
            max_history_size: Maximum number of transitions to keep in history for new workflows
        """
        self._workflows: Dict[str, WorkflowStateMachine] = {}
        self._persistence_dir = persistence_dir or Path(".cadence/workflow_states")
        self._max_history_size = max_history_size
        self._lock = threading.Lock()

    def get_or_create_workflow(
        self,
        workflow_id: str,
        initial_state: WorkflowState = WorkflowState.WORKING,
        context: Optional[WorkflowContext] = None
    ) -> WorkflowStateMachine:
        """
        Get existing workflow or create a new one.

        Args:
            workflow_id: Unique identifier for the workflow
            initial_state: Initial state if creating new workflow
            context: Context for new workflow

        Returns:
            WorkflowStateMachine instance
        """
        # Basic path validation to prevent traversal
        if ".." in workflow_id or "/" in workflow_id or "\\" in workflow_id:
            raise ValueError("workflow_id cannot contain path separators or '..'")

        with self._lock:
            if workflow_id not in self._workflows:
                persistence_path = self._persistence_dir / f"{workflow_id}.json"
                self._workflows[workflow_id] = WorkflowStateMachine(
                    initial_state=initial_state,
                    context=context,
                    persistence_path=persistence_path,
                    max_history_size=self._max_history_size
                )

            return self._workflows[workflow_id]

    def remove_workflow(self, workflow_id: str) -> bool:
        """
        Remove a workflow from management.

        Args:
            workflow_id: Workflow to remove

        Returns:
            True if workflow was removed, False if not found
        """
        with self._lock:
            if workflow_id in self._workflows:
                # Clean up persistence file
                persistence_path = self._persistence_dir / f"{workflow_id}.json"
                if persistence_path.exists():
                    try:
                        persistence_path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to remove persistence file: {e}")

                del self._workflows[workflow_id]
                return True
            return False

    def list_workflows(self) -> List[str]:
        """Get list of active workflow IDs"""
        with self._lock:
            return list(self._workflows.keys())

    def get_workflow_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries of all active workflows"""
        with self._lock:
            return {
                workflow_id: workflow.get_state_summary()
                for workflow_id, workflow in self._workflows.items()
            }
