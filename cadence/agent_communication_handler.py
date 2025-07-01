"""
Agent Communication Handler for managing asynchronous agent lifecycle events.

This module provides callback mechanisms and timeout watchdog functionality
for agent operations, integrating with the existing AgentDispatcher.
"""

import asyncio
import logging
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from .agent_dispatcher import AgentDispatcher
from .agent_messages import AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria

logger = logging.getLogger(__name__)


class CallbackType(str, Enum):
    """Types of callbacks supported by the communication handler"""
    ON_COMPLETE = "on_complete"
    ON_ERROR = "on_error"
    ON_TIMEOUT = "on_timeout"


@dataclass
class AgentOperation:
    """Represents an ongoing agent operation"""
    agent_id: str
    agent_type: AgentType
    context: MessageContext
    success_criteria: SuccessCriteria
    start_time: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 300.0  # Default 5 minutes
    callbacks: Dict[CallbackType, List[Callable]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackEvent:
    """Event to be processed by the callback system"""
    callback_type: CallbackType
    agent_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[AgentOperation] = None


class AgentCommunicationHandler:
    """
    High-level communication handler for managing asynchronous agent operations.

    Provides callback mechanisms (on_complete, on_error, on_timeout) and timeout
    watchdog functionality. Integrates with AgentDispatcher for message handling.

    Features:
    - Callback registration/deregistration per operation
    - Timeout watchdog using dedicated asyncio task per operation
    - Event queuing for high-throughput scenarios
    - Thread-safe concurrent operations
    - Graceful shutdown handling
    - Callback chaining support
    """

    def __init__(self,
                 dispatcher: Optional[AgentDispatcher] = None,
                 default_timeout: float = 300.0,
                 enable_event_queue: bool = True):
        """
        Initialize the communication handler.

        Args:
            dispatcher: AgentDispatcher instance (creates new if None)
            default_timeout: Default timeout in seconds
            enable_event_queue: Whether to enable event queuing
        """
        self.dispatcher = dispatcher or AgentDispatcher()
        self.default_timeout = default_timeout
        self.enable_event_queue = enable_event_queue

        # Active operations tracking
        self.active_operations: Dict[str, AgentOperation] = {}
        self.operation_lock = threading.Lock()

        # Callback management
        self.global_callbacks: Dict[CallbackType, List[Callable]] = {
            CallbackType.ON_COMPLETE: [],
            CallbackType.ON_ERROR: [],
            CallbackType.ON_TIMEOUT: []
        }
        self.callback_lock = threading.Lock()

        # Event queuing
        self.event_queue: Optional[asyncio.Queue] = None
        self.event_processor_task: Optional[asyncio.Task] = None

        # Shutdown management
        self.shutdown_event = threading.Event()
        self.pending_timeouts: Dict[str, asyncio.Task] = {}

        logger.info(f"AgentCommunicationHandler initialized with default timeout: {default_timeout}s")

    def generate_agent_id(self) -> str:
        """Generate unique agent operation ID"""
        return f"agent-{uuid.uuid4().hex[:8]}-{int(datetime.now().timestamp())}"

    def register_global_callback(self, callback_type: CallbackType, callback: Callable) -> None:
        """
        Register a global callback for all operations.

        Args:
            callback_type: Type of callback (on_complete, on_error, on_timeout)
            callback: Callback function
        """
        with self.callback_lock:
            self.global_callbacks[callback_type].append(callback)
        logger.debug(f"Registered global {callback_type.value} callback")

    def deregister_global_callback(self, callback_type: CallbackType, callback: Callable) -> bool:
        """
        Deregister a global callback.

        Args:
            callback_type: Type of callback
            callback: Callback function to remove

        Returns:
            True if callback was found and removed
        """
        with self.callback_lock:
            try:
                self.global_callbacks[callback_type].remove(callback)
                logger.debug(f"Deregistered global {callback_type.value} callback")
                return True
            except ValueError:
                return False

    def register_operation_callback(self, agent_id: str, callback_type: CallbackType, callback: Callable) -> bool:
        """
        Register a callback for a specific operation.

        Args:
            agent_id: Agent operation ID
            callback_type: Type of callback
            callback: Callback function

        Returns:
            True if operation exists and callback was registered
        """
        with self.operation_lock:
            operation = self.active_operations.get(agent_id)
            if not operation:
                return False

            if callback_type not in operation.callbacks:
                operation.callbacks[callback_type] = []
            operation.callbacks[callback_type].append(callback)

        logger.debug(f"Registered {callback_type.value} callback for agent {agent_id}")
        return True

    async def start_agent_operation(self,
                                   agent_type: AgentType,
                                   context: MessageContext,
                                   success_criteria: SuccessCriteria,
                                   timeout_seconds: Optional[float] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new agent operation with timeout watchdog.

        Args:
            agent_type: Type of agent to dispatch
            context: Message context
            success_criteria: Success criteria
            timeout_seconds: Timeout in seconds (uses default if None)
            metadata: Additional metadata for the operation

        Returns:
            agent_id: Unique identifier for this operation
        """
        agent_id = self.generate_agent_id()
        timeout = timeout_seconds or self.default_timeout

        # Create operation record
        operation = AgentOperation(
            agent_id=agent_id,
            agent_type=agent_type,
            context=context,
            success_criteria=success_criteria,
            timeout_seconds=timeout,
            metadata=metadata or {}
        )

        with self.operation_lock:
            self.active_operations[agent_id] = operation

        # Start timeout watchdog
        timeout_task = asyncio.create_task(self._timeout_watchdog(agent_id, timeout))
        self.pending_timeouts[agent_id] = timeout_task

        # Dispatch agent through existing dispatcher
        try:
            message_id = self.dispatcher.dispatch_agent(
                agent_type=agent_type,
                context=context,
                success_criteria=success_criteria,
                callback_handler=lambda msg: asyncio.create_task(self._handle_agent_response(agent_id, msg)),
                timeout_ms=int(timeout * 1000)
            )

            operation.metadata["message_id"] = message_id
            logger.info(f"Started agent operation {agent_id} with message {message_id}")

        except Exception as e:
            # Clean up on dispatch failure
            with self.operation_lock:
                self.active_operations.pop(agent_id, None)
            timeout_task.cancel()
            self.pending_timeouts.pop(agent_id, None)

            await self._trigger_callbacks(CallbackType.ON_ERROR, agent_id, {
                "error": f"Failed to dispatch agent: {e}",
                "error_type": "dispatch_error"
            })
            raise

        return agent_id

    async def _timeout_watchdog(self, agent_id: str, timeout_seconds: float) -> None:
        """
        Timeout watchdog for agent operations.

        Args:
            agent_id: Agent operation ID
            timeout_seconds: Timeout duration
        """
        try:
            await asyncio.sleep(timeout_seconds)

            # Check if operation is still active
            with self.operation_lock:
                operation = self.active_operations.get(agent_id)
                if operation:
                    logger.warning(f"Agent operation {agent_id} timed out after {timeout_seconds}s")
                    # Remove from active operations
                    del self.active_operations[agent_id]

            # Clean up timeout tracking
            self.pending_timeouts.pop(agent_id, None)

            # Trigger timeout callbacks
            await self._trigger_callbacks(CallbackType.ON_TIMEOUT, agent_id, {
                "timeout_seconds": timeout_seconds,
                "duration": timeout_seconds
            })

        except asyncio.CancelledError:
            # Operation completed before timeout
            logger.debug(f"Timeout watchdog cancelled for agent {agent_id}")

    async def _handle_agent_response(self, agent_id: str, message: AgentMessage) -> None:
        """
        Handle response from agent dispatcher.

        Args:
            agent_id: Agent operation ID
            message: Response message
        """
        with self.operation_lock:
            operation = self.active_operations.get(agent_id)
            if not operation:
                logger.warning(f"Received response for unknown agent {agent_id}")
                return

        try:
            if message.message_type == MessageType.TASK_COMPLETE:
                # Operation completed successfully
                await self._complete_operation(agent_id, message)

            elif message.message_type == MessageType.ERROR:
                # Operation failed
                error_info = message.payload or {}
                await self._error_operation(agent_id, message, error_info)

            elif message.message_type == MessageType.AGENT_RESPONSE:
                # Intermediate response - could be progress update
                logger.debug(f"Received intermediate response for agent {agent_id}")
                # Could trigger progress callbacks here in the future

        except Exception as e:
            logger.error(f"Error handling response for agent {agent_id}: {e}")
            await self._error_operation(agent_id, message, {"error": str(e), "error_type": "handler_error"})

    async def _complete_operation(self, agent_id: str, message: AgentMessage) -> None:
        """Complete an agent operation successfully"""
        # Cancel timeout watchdog
        timeout_task = self.pending_timeouts.pop(agent_id, None)
        if timeout_task:
            timeout_task.cancel()

        # Get operation and remove from active operations
        with self.operation_lock:
            operation = self.active_operations.pop(agent_id, None)

        if operation:
            result_data = {
                "message": message,
                "operation": operation,
                "result": message.payload or {},
                "duration": (datetime.now() - operation.start_time).total_seconds()
            }

            await self._trigger_callbacks(CallbackType.ON_COMPLETE, agent_id, result_data, operation)
            logger.info(f"Agent operation {agent_id} completed successfully")

    async def _error_operation(self, agent_id: str, message: AgentMessage, error_info: Dict[str, Any]) -> None:
        """Handle agent operation error"""
        # Cancel timeout watchdog
        timeout_task = self.pending_timeouts.pop(agent_id, None)
        if timeout_task:
            timeout_task.cancel()

        # Get operation and remove from active operations
        with self.operation_lock:
            operation = self.active_operations.pop(agent_id, None)

        if operation:
            error_data = {
                "message": message,
                "operation": operation,
                "error": error_info.get("error", "Unknown error"),
                "error_type": error_info.get("error_type", "agent_error"),
                "context": operation.context,
                "duration": (datetime.now() - operation.start_time).total_seconds()
            }

            await self._trigger_callbacks(CallbackType.ON_ERROR, agent_id, error_data, operation)
            logger.error(f"Agent operation {agent_id} failed: {error_info.get('error', 'Unknown error')}")

    async def _trigger_callbacks(self, callback_type: CallbackType, agent_id: str, data: Dict[str, Any], operation: Optional[AgentOperation] = None) -> None:
        """
        Trigger callbacks for an event.

        Args:
            callback_type: Type of callback
            agent_id: Agent operation ID
            data: Event data
            operation: Optional operation object (for when operation is already removed from active_operations)
        """
        if self.enable_event_queue and self.event_queue:
            # Queue event for processing
            event = CallbackEvent(callback_type, agent_id, data, operation=operation)
            await self.event_queue.put(event)
        else:
            # Process callbacks immediately
            await self._process_callbacks(callback_type, agent_id, data, operation)

    async def _process_callbacks(self, callback_type: CallbackType, agent_id: str, data: Dict[str, Any], operation: Optional[AgentOperation] = None) -> None:
        """Process callbacks for an event"""
        callbacks_to_execute = []

        # Collect global callbacks
        with self.callback_lock:
            callbacks_to_execute.extend(self.global_callbacks[callback_type])

        # Collect operation-specific callbacks
        if operation:
            # Use the provided operation (for completed/errored operations)
            if callback_type in operation.callbacks:
                callbacks_to_execute.extend(operation.callbacks[callback_type])
        else:
            # Look up operation in active_operations (for timeouts or other cases)
            with self.operation_lock:
                active_operation = self.active_operations.get(agent_id)
                if active_operation and callback_type in active_operation.callbacks:
                    callbacks_to_execute.extend(active_operation.callbacks[callback_type])

        # Execute callbacks
        for callback in callbacks_to_execute:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, data)
                else:
                    callback(agent_id, data)
            except Exception as e:
                logger.error(f"Error in {callback_type.value} callback for agent {agent_id}: {e}")

    async def start_event_processor(self) -> None:
        """Start the event processing queue"""
        if not self.enable_event_queue:
            return

        self.event_queue = asyncio.Queue()
        self.event_processor_task = asyncio.create_task(self._event_processor_loop())
        logger.info("Event processor started")

    async def _event_processor_loop(self) -> None:
        """Main event processing loop"""
        while not self.shutdown_event.is_set():
            try:
                # Wait for event or shutdown
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_callbacks(event.callback_type, event.agent_id, event.data, event.operation)
            except asyncio.TimeoutError:
                # Check for shutdown periodically
                continue
            except Exception as e:
                logger.error(f"Error in event processor: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown the communication handler"""
        logger.info("Shutting down AgentCommunicationHandler")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel all pending timeouts
        for agent_id, timeout_task in list(self.pending_timeouts.items()):
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass

        self.pending_timeouts.clear()

        # Stop event processor
        if self.event_processor_task:
            self.event_processor_task.cancel()
            try:
                await self.event_processor_task
            except asyncio.CancelledError:
                pass

        # Cleanup dispatcher
        self.dispatcher.cleanup()

        # Clear active operations
        with self.operation_lock:
            remaining_count = len(self.active_operations)
            self.active_operations.clear()

        if remaining_count > 0:
            logger.warning(f"Shutdown with {remaining_count} active operations")

        logger.info("AgentCommunicationHandler shutdown complete")

    def get_operation_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for an operation"""
        with self.operation_lock:
            operation = self.active_operations.get(agent_id)
            if not operation:
                return None

            return {
                "agent_id": agent_id,
                "agent_type": operation.agent_type.value,
                "start_time": operation.start_time.isoformat(),
                "timeout_seconds": operation.timeout_seconds,
                "elapsed_seconds": (datetime.now() - operation.start_time).total_seconds(),
                "context": {
                    "task_id": operation.context.task_id,
                    "project_path": operation.context.project_path
                },
                "metadata": operation.metadata
            }

    def list_active_operations(self) -> List[str]:
        """Get list of active operation IDs"""
        with self.operation_lock:
            return list(self.active_operations.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics"""
        with self.operation_lock:
            active_count = len(self.active_operations)

        with self.callback_lock:
            global_callbacks = {
                callback_type.value: len(callbacks)
                for callback_type, callbacks in self.global_callbacks.items()
            }

        return {
            "active_operations": active_count,
            "pending_timeouts": len(self.pending_timeouts),
            "global_callbacks": global_callbacks,
            "event_queue_enabled": self.enable_event_queue,
            "default_timeout": self.default_timeout,
            "is_shutdown": self.shutdown_event.is_set()
        }
