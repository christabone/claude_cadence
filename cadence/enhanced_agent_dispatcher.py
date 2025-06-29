"""
Enhanced Agent Dispatcher with Fix Iteration Tracking and Escalation Support

This module extends the basic AgentDispatcher with fix iteration tracking,
limit enforcement, and escalation handling capabilities.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass

from .agent_dispatcher import AgentDispatcher
from .agent_messages import (
    AgentMessage, MessageType, AgentType, Priority,
    MessageContext, SuccessCriteria, CallbackInfo
)
from .fix_iteration_tracker import (
    FixIterationManager, EscalationStrategy, PersistenceType
)
from .config import DEFAULT_AGENT_TIMEOUT_MS

logger = logging.getLogger(__name__)


@dataclass
class DispatchConfig:
    """Configuration for Enhanced Agent Dispatcher"""
    max_concurrent_agents: int = 2
    default_timeout_ms: int = DEFAULT_AGENT_TIMEOUT_MS
    enable_fix_tracking: bool = True
    enable_escalation: bool = True
    max_fix_iterations: int = 3
    escalation_strategy: str = "log_only"
    persistence_type: str = "memory"
    storage_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for enhanced dispatcher"""
        return {
            "max_concurrent_agents": self.max_concurrent_agents,
            "default_timeout_ms": self.default_timeout_ms,
            "enable_fix_tracking": self.enable_fix_tracking,
            "enable_escalation": self.enable_escalation,
            "max_fix_iterations": self.max_fix_iterations,
            "escalation_strategy": self.escalation_strategy,
            "persistence_type": self.persistence_type,
            "storage_path": self.storage_path
        }


class EnhancedAgentDispatcher(AgentDispatcher):
    """
    Enhanced agent dispatcher with fix iteration tracking and escalation support.

    Extends the basic AgentDispatcher to include:
    - Fix attempt tracking and limits
    - Automatic escalation handling
    - Enhanced message metadata
    - Configuration-driven behavior
    """

    def __init__(
        self,
        max_fix_iterations: int = 3,
        escalation_strategy: EscalationStrategy = EscalationStrategy.LOG_ONLY,
        persistence_type: PersistenceType = PersistenceType.MEMORY,
        storage_path: Optional[str] = None,
        supervisor_callback: Optional[Callable[[str, str, int], None]] = None,
        notification_callback: Optional[Callable[[str, str], None]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the enhanced agent dispatcher.

        Args:
            max_fix_iterations: Maximum allowed fix attempts per task
            escalation_strategy: Strategy for handling escalations
            persistence_type: Type of persistence for attempt tracking
            storage_path: Path for file-based persistence
            supervisor_callback: Callback for supervisor notifications
            notification_callback: Callback for general notifications
            config: Optional configuration override
        """
        super().__init__()

        # Load configuration
        if config:
            max_fix_iterations = config.get('max_fix_iterations', max_fix_iterations)
            escalation_strategy_str = config.get('escalation_strategy', escalation_strategy.value)
            if isinstance(escalation_strategy_str, str):
                escalation_strategy = EscalationStrategy(escalation_strategy_str)
            persistence_type_str = config.get('persistence_type', persistence_type.value)
            if isinstance(persistence_type_str, str):
                persistence_type = PersistenceType(persistence_type_str)
            storage_path = config.get('storage_path', storage_path)

        # Initialize fix iteration manager
        self.fix_manager = FixIterationManager(
            max_fix_iterations=max_fix_iterations,
            escalation_strategy=escalation_strategy,
            persistence_type=persistence_type,
            storage_path=storage_path,
            supervisor_callback=supervisor_callback,
            notification_callback=notification_callback
        )

        self.max_fix_iterations = max_fix_iterations
        logger.info(f"EnhancedAgentDispatcher initialized with max_iterations={max_fix_iterations}")

    def dispatch_fix_agent(
        self,
        task_id: str,
        context: MessageContext,
        success_criteria: SuccessCriteria,
        callback_handler: Callable,
        timeout_ms: int = 30000,
        priority: Priority = Priority.MEDIUM,
        use_queue: bool = False
    ) -> Optional[str]:
        """
        Dispatch a fix agent with iteration tracking and limit enforcement.

        Args:
            task_id: Task identifier for iteration tracking
            context: Message context
            success_criteria: Success criteria
            callback_handler: Callback function
            timeout_ms: Timeout in milliseconds
            priority: Message priority
            use_queue: Whether to queue the message

        Returns:
            Message ID if dispatch allowed, None if escalated
        """
        # Check if fix attempt is allowed
        if not self.fix_manager.can_attempt_fix(task_id):
            logger.warning(f"Fix dispatch blocked for task {task_id}: max attempts exceeded or escalated")
            return None

        # Start fix attempt tracking
        attempt_number = self.fix_manager.start_fix_attempt(task_id)
        if attempt_number is None:
            logger.warning(f"Fix dispatch blocked for task {task_id}: escalated during validation")
            return None

        logger.info(f"Starting fix attempt {attempt_number} for task {task_id}")

        # Create enhanced callback that tracks completion
        original_callback = callback_handler
        def enhanced_callback(response: AgentMessage):
            self._handle_fix_response(task_id, response, original_callback, attempt_number)

        # Dispatch the agent
        message_id = self.dispatch_agent(
            agent_type=AgentType.FIX,
            context=context,
            success_criteria=success_criteria,
            callback_handler=enhanced_callback,
            timeout_ms=timeout_ms,
            use_queue=use_queue
        )

        if message_id:
            logger.info(f"Fix agent dispatched for task {task_id}, attempt {attempt_number}, message_id: {message_id}")

        return message_id

    def dispatch_agent_with_tracking(
        self,
        agent_type: AgentType,
        task_id: str,
        context: MessageContext,
        success_criteria: SuccessCriteria,
        callback_handler: Callable,
        timeout_ms: int = 30000,
        priority: Priority = Priority.MEDIUM,
        use_queue: bool = False
    ) -> Optional[str]:
        """
        Dispatch any agent type with enhanced tracking and metadata.

        Args:
            agent_type: Type of agent to dispatch
            task_id: Task identifier
            context: Message context
            success_criteria: Success criteria
            callback_handler: Callback function
            timeout_ms: Timeout in milliseconds
            priority: Message priority
            use_queue: Whether to queue the message

        Returns:
            Message ID if successful
        """
        # For fix agents, use special tracking
        if agent_type == AgentType.FIX:
            return self.dispatch_fix_agent(
                task_id=task_id,
                context=context,
                success_criteria=success_criteria,
                callback_handler=callback_handler,
                timeout_ms=timeout_ms,
                priority=priority,
                use_queue=use_queue
            )

        # For other agent types, use standard dispatch with enhanced message
        message_id = self.generate_message_id()

        # Create enhanced message
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=agent_type,
            priority=priority,
            context=context,
            success_criteria=success_criteria,
            callback=CallbackInfo(
                handler=callback_handler.__name__,
                timeout_ms=timeout_ms
            ),
            message_id=message_id
        )

        # Add tracking metadata
        enhanced_message = self.fix_manager.enhance_dispatch_message(message, task_id)

        # Store message and callback with thread safety
        with self.lock:
            self.pending_messages[message_id] = enhanced_message
            self.callbacks[message_id] = callback_handler
            self._start_timeout_timer(message_id, timeout_ms)

        if use_queue:
            self.queue_message(enhanced_message)
        else:
            logger.info(f"Dispatched {agent_type.value} agent for task {task_id}, message_id: {message_id}")

        return message_id

    def _handle_fix_response(
        self,
        task_id: str,
        response: AgentMessage,
        original_callback: Callable,
        attempt_number: int
    ) -> None:
        """
        Handle fix agent response with completion tracking.

        Args:
            task_id: Task identifier
            response: Agent response message
            original_callback: Original callback function
            attempt_number: Current attempt number
        """
        start_time = datetime.now()

        try:
            # Determine if fix was successful
            success = response.message_type == MessageType.TASK_COMPLETE
            error_message = None
            files_modified = []

            if not success:
                if response.message_type == MessageType.ERROR:
                    error_message = response.payload.get('error', 'Unknown error') if response.payload else 'Unknown error'
                else:
                    error_message = f"Fix incomplete: {response.message_type.value}"

            # Extract files modified from context or payload
            if response.context and response.context.files_modified:
                files_modified = response.context.files_modified
            elif response.payload and 'files_modified' in response.payload:
                files_modified = response.payload['files_modified']

            # Calculate duration (rough estimate)
            duration_seconds = (datetime.now() - start_time).total_seconds()

            # Complete the fix attempt
            not_escalated = self.fix_manager.complete_fix_attempt(
                task_id=task_id,
                success=success,
                error_message=error_message,
                duration_seconds=duration_seconds,
                files_modified=files_modified
            )

            if success:
                logger.info(f"Fix attempt {attempt_number} succeeded for task {task_id}")
            elif not_escalated:
                logger.warning(f"Fix attempt {attempt_number} failed for task {task_id}: {error_message}")
            else:
                logger.error(f"Fix attempt {attempt_number} failed and task {task_id} escalated: {error_message}")

            # Call original callback
            original_callback(response)

        except Exception as e:
            logger.error(f"Error handling fix response for task {task_id}: {e}")

            # Mark attempt as failed due to processing error
            self.fix_manager.complete_fix_attempt(
                task_id=task_id,
                success=False,
                error_message=f"Response processing error: {str(e)}"
            )

            # Still call original callback
            try:
                original_callback(response)
            except Exception as callback_error:
                logger.error(f"Error in original callback: {callback_error}")

    def get_task_fix_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get comprehensive fix status for a task.

        Args:
            task_id: Task identifier

        Returns:
            Task status dictionary
        """
        return self.fix_manager.get_task_status(task_id)

    def reset_task_fix_attempts(self, task_id: str) -> None:
        """
        Reset fix attempts for a task (e.g., after manual resolution).

        Args:
            task_id: Task identifier
        """
        self.fix_manager.iteration_tracker.reset_task_attempts(task_id)
        logger.info(f"Reset fix attempts for task {task_id}")

    def get_all_escalated_tasks(self) -> List[str]:
        """
        Get list of all escalated task IDs.

        Returns:
            List of escalated task IDs
        """
        return self.fix_manager.iteration_tracker.get_all_escalated_tasks()

    def is_automation_paused(self) -> bool:
        """
        Check if automation is paused due to escalations.

        Returns:
            True if automation is paused
        """
        return self.fix_manager.escalation_handler.is_automation_paused()

    def resume_automation(self) -> None:
        """Resume automation after manual intervention"""
        self.fix_manager.escalation_handler.resume_automation()
        logger.info("Automation resumed via dispatcher")

    def create_escalation_message(self, task_id: str, reason: str, attempt_count: int) -> AgentMessage:
        """
        Create an escalation message for supervisor notification.

        Args:
            task_id: Task identifier
            reason: Escalation reason
            attempt_count: Number of attempts made

        Returns:
            Escalation message
        """
        return AgentMessage(
            message_type=MessageType.ESCALATION_REQUIRED,
            agent_type=AgentType.FIX,  # Escalation related to fix agent
            priority=Priority.HIGH,
            context=MessageContext(
                task_id=task_id,
                parent_session="escalation_handler",
                files_modified=[],
                project_path=""
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["manual_intervention"],
                validation_steps=["supervisor_review"]
            ),
            callback=CallbackInfo(handler="escalation_callback"),
            payload={
                'escalation_reason': reason,
                'attempt_count': attempt_count,
                'requires_manual_review': True,
                'task_status': self.get_task_fix_status(task_id)
            }
        )

    def handle_escalation_message(self, escalation_message: AgentMessage) -> None:
        """
        Handle an escalation message (e.g., from external sources).

        Args:
            escalation_message: Escalation message to process
        """
        if escalation_message.message_type != MessageType.ESCALATION_REQUIRED:
            logger.warning(f"Invalid escalation message type: {escalation_message.message_type}")
            return

        task_id = escalation_message.context.task_id
        payload = escalation_message.payload or {}
        reason = payload.get('escalation_reason', 'External escalation')

        # Mark task as escalated if not already
        if not self.fix_manager.iteration_tracker.is_task_escalated(task_id):
            self.fix_manager.iteration_tracker.mark_task_escalated(task_id, reason)

        logger.warning(f"Processed escalation message for task {task_id}: {reason}")

    def cleanup(self) -> None:
        """Clean up all resources"""
        super().cleanup()
        self.fix_manager.cleanup()
        logger.info("EnhancedAgentDispatcher cleanup completed")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EnhancedAgentDispatcher':
        """
        Create dispatcher from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Configured dispatcher instance
        """
        return cls(config=config)
