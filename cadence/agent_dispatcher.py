"""
Simple Agent Dispatcher for sending and receiving agent messages
"""
import json
import uuid
import logging
import threading
from queue import Queue
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from .agent_messages import (
    AgentMessage, MessageType, AgentType,
    MessageContext, SuccessCriteria, CallbackInfo
)

logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Simple dispatcher for agent messages"""

    def __init__(self):
        self.pending_messages: Dict[str, AgentMessage] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.message_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.timers: Dict[str, threading.Timer] = {}
        self.lock = threading.Lock()  # Thread synchronization

    def generate_message_id(self) -> str:
        """Generate unique message ID"""
        return str(uuid.uuid4())

    def dispatch_agent(
        self,
        agent_type: AgentType,
        context: MessageContext,
        success_criteria: SuccessCriteria,
        callback_handler: Callable,
        timeout_ms: int = 30000,
        use_queue: bool = False
    ) -> str:
        """
        Dispatch an agent with given parameters.

        Args:
            agent_type: Type of agent to dispatch
            context: Message context
            success_criteria: Success criteria
            callback_handler: Callback function
            timeout_ms: Timeout in milliseconds
            use_queue: Whether to queue the message instead of direct dispatch

        Returns:
            message_id: The ID of the dispatched message
        """
        message_id = self.generate_message_id()

        # Create dispatch message
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=agent_type,
            context=context,
            success_criteria=success_criteria,
            callback=CallbackInfo(
                handler=callback_handler.__name__,
                timeout_ms=timeout_ms
            ),
            message_id=message_id
        )

        # Store message and callback with thread safety
        with self.lock:
            self.pending_messages[message_id] = message
            self.callbacks[message_id] = callback_handler

            # Start timeout timer within lock
            self._start_timeout_timer(message_id, timeout_ms)

        if use_queue:
            # Queue for later processing
            self.queue_message(message)
        else:
            # Direct dispatch (in real implementation, would send to agent)
            logger.info(f"Dispatched {agent_type.value} agent with message_id: {message_id}")

        return message_id

    def receive_response(self, response_data: Dict[str, Any]) -> bool:
        """
        Process an agent response.

        Args:
            response_data: Dictionary containing response data

        Returns:
            bool: True if response was processed successfully
        """
        try:
            # Parse response
            response = AgentMessage.from_dict(response_data)

            # Validate response type
            if response.message_type not in [MessageType.AGENT_RESPONSE, MessageType.TASK_COMPLETE, MessageType.ERROR]:
                logger.error(f"Invalid response type: {response.message_type}")
                return False

            # Find original message
            original_id = response.payload.get("original_message_id") if response.payload else None

            # Thread-safe access to shared state
            with self.lock:
                if not original_id or original_id not in self.pending_messages:
                    logger.error(f"No pending message found for response: {original_id}")
                    return False

                # Get callback
                callback = self.callbacks.get(original_id)

                # Clean up before executing callback
                del self.pending_messages[original_id]
                del self.callbacks[original_id]
                self._cancel_timeout_timer(original_id)

            # Execute callback outside of lock to avoid deadlock
            if callback:
                callback(response)

            logger.info(f"Processed {response.message_type.value} for message: {original_id}")
            return True

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return False

    def create_error_response(self, original_message_id: str, error_message: str) -> AgentMessage:
        """Create an error response for a message - assumes lock is already held"""
        original = self.pending_messages.get(original_message_id)
        if not original:
            raise ValueError(f"No pending message found: {original_message_id}")

        # Create a new callback with valid timeout_ms (minimum 1000ms)
        error_callback = CallbackInfo(
            handler=original.callback.handler,
            timeout_ms=max(1000, original.callback.timeout_ms)
        )

        return AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=original.agent_type,
            context=original.context,
            success_criteria=original.success_criteria,
            callback=error_callback,
            message_id=self.generate_message_id(),
            payload={
                "original_message_id": original_message_id,
                "error": error_message
            }
        )

    def get_pending_count(self) -> int:
        """Get count of pending messages"""
        with self.lock:
            return len(self.pending_messages)

    def queue_message(self, message: AgentMessage) -> None:
        """Add message to outgoing queue"""
        self.message_queue.put(message)
        logger.debug(f"Queued message: {message.message_id}")

    def get_next_message(self) -> Optional[AgentMessage]:
        """Get next message from queue (non-blocking)"""
        if not self.message_queue.empty():
            return self.message_queue.get_nowait()
        return None

    def queue_response(self, response: AgentMessage) -> None:
        """Add response to response queue"""
        self.response_queue.put(response)
        logger.debug(f"Queued response: {response.message_id}")

    def get_next_response(self) -> Optional[AgentMessage]:
        """Get next response from queue (non-blocking)"""
        if not self.response_queue.empty():
            return self.response_queue.get_nowait()
        return None

    def process_queue(self) -> List[str]:
        """
        Process all messages in the queue.

        Returns:
            List of message IDs that were processed
        """
        processed = []

        # Process outgoing messages
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                # In real implementation, this would send to agent
                logger.info(f"Processing message from queue: {message.message_id}")
                processed.append(message.message_id)
            except Exception as e:
                logger.error(f"Error processing message: {e}")

        # Process responses
        while not self.response_queue.empty():
            try:
                response = self.response_queue.get_nowait()
                self.receive_response(response.to_dict())
                processed.append(response.message_id)
            except Exception as e:
                logger.error(f"Error processing response: {e}")

        return processed

    def _start_timeout_timer(self, message_id: str, timeout_ms: int) -> None:
        """Start a timeout timer for a message - assumes lock is already held"""
        timeout_seconds = timeout_ms / 1000.0

        def timeout_handler():
            logger.warning(f"Timeout for message {message_id} after {timeout_ms}ms")

            # Thread-safe timeout handling
            with self.lock:
                # Check if message still exists (it might have been processed)
                if message_id in self.pending_messages:
                    try:
                        error_response = self.create_error_response(
                            message_id,
                            f"Agent timeout after {timeout_ms}ms"
                        )

                        # Queue the error response
                        self.queue_response(error_response)
                    except ValueError:
                        # Message was already processed
                        pass

                # Clean up timer reference
                if message_id in self.timers:
                    del self.timers[message_id]

        # Create timer and store reference while holding lock
        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.daemon = True  # Set as daemon thread to prevent blocking shutdown
        self.timers[message_id] = timer
        # Start timer after storing reference but still within lock
        timer.start()

        logger.debug(f"Started timeout timer for {message_id}: {timeout_ms}ms")

    def _cancel_timeout_timer(self, message_id: str) -> None:
        """Cancel a timeout timer - assumes lock is already held"""
        if message_id in self.timers:
            timer = self.timers[message_id]
            timer.cancel()
            del self.timers[message_id]
            logger.debug(f"Cancelled timeout timer for {message_id}")

    def cleanup(self) -> None:
        """Clean up all pending timers"""
        with self.lock:
            for message_id, timer in list(self.timers.items()):
                timer.cancel()
            self.timers.clear()
        logger.info("Cleaned up all timeout timers")
