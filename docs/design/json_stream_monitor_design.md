# JSONStreamMonitor Design Document - Task 1.2

## Overview
This document defines the design for the JSONStreamMonitor class, which will provide robust JSON stream parsing with buffering, error recovery, and event-driven architecture for the Claude Cadence supervisor/agent communication system.

## Design Goals
1. **Robust Parsing**: Handle partial JSON, malformed data, and mixed text/JSON streams
2. **Event-Driven**: Decouple parsing from consumption through events
3. **State Management**: Maintain parsing state across stream chunks
4. **Error Recovery**: Continue processing after errors with detailed logging
5. **Performance**: Efficient processing with minimal overhead
6. **Integration**: Seamless integration with existing orchestrator

## Class Architecture

### Core Class: JSONStreamMonitor

```python
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import logging
from abc import ABC, abstractmethod


class MessageType(Enum):
    """Claude message types"""
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    RESULT = "result"
    UNKNOWN = "unknown"


class ParseState(Enum):
    """JSON parsing state"""
    SEEKING_JSON = "seeking_json"     # Looking for start of JSON
    IN_JSON = "in_json"               # Inside JSON object
    IN_STRING = "in_string"           # Inside JSON string
    ESCAPE_CHAR = "escape_char"       # After backslash in string


@dataclass
class JSONEvent:
    """Event emitted when complete JSON object is parsed"""
    json_data: Dict[str, Any]
    message_type: MessageType
    raw_json: str
    line_number: int
    timestamp: float
    source: str  # "agent" or "supervisor"


@dataclass
class ParseError:
    """Error information for failed parsing"""
    error: Exception
    partial_data: str
    line_number: int
    timestamp: float
    recoverable: bool


class JSONStreamMonitor:
    """
    Monitors streaming output for JSON objects with robust parsing and event emission.

    Features:
    - Buffers partial JSON across line boundaries
    - Maintains parsing state between chunks
    - Emits events for complete JSON objects
    - Recovers from parsing errors
    - Handles nested objects and arrays
    - Configurable buffer size limits
    """

    def __init__(
        self,
        source: str = "unknown",
        max_buffer_size: int = 1024 * 1024,  # 1MB default
        max_nesting_depth: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize JSONStreamMonitor.

        Args:
            source: Identifier for the stream source ("agent" or "supervisor")
            max_buffer_size: Maximum size for JSON buffer in bytes
            max_nesting_depth: Maximum nesting level for JSON objects
            logger: Optional logger instance
        """
        self.source = source
        self.max_buffer_size = max_buffer_size
        self.max_nesting_depth = max_nesting_depth
        self.logger = logger or logging.getLogger(__name__)

        # Parsing state
        self._buffer = ""
        self._state = ParseState.SEEKING_JSON
        self._brace_count = 0
        self._bracket_count = 0
        self._in_string = False
        self._escape_next = False
        self._line_number = 0

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            'json_complete': [],
            'parse_error': [],
            'buffer_overflow': [],
        }

        # Statistics
        self._stats = {
            'objects_parsed': 0,
            'errors_recovered': 0,
            'bytes_processed': 0,
            'buffer_overflows': 0
        }

    # Stream Processing Methods

    async def process_stream(self, stream: asyncio.StreamReader) -> None:
        """
        Process an async stream, emitting events for complete JSON objects.

        Args:
            stream: Async stream reader to process
        """
        while True:
            try:
                # Read chunk from stream
                chunk = await stream.read(8192)  # 8KB chunks
                if not chunk:
                    break

                # Process the chunk
                await self.process_chunk(chunk.decode('utf-8', errors='replace'))

            except Exception as e:
                self.logger.error(f"Stream processing error: {e}")
                self._emit_error(ParseError(
                    error=e,
                    partial_data=self._buffer,
                    line_number=self._line_number,
                    timestamp=asyncio.get_event_loop().time(),
                    recoverable=True
                ))

    async def process_chunk(self, chunk: str) -> None:
        """
        Process a chunk of text that may contain JSON objects.

        Args:
            chunk: Text chunk to process
        """
        for char in chunk:
            self._stats['bytes_processed'] += 1

            # Update line number
            if char == '\n':
                self._line_number += 1

            # State machine processing
            await self._process_character(char)

            # Check buffer size
            if len(self._buffer) > self.max_buffer_size:
                await self._handle_buffer_overflow()

    async def process_line(self, line: str) -> None:
        """
        Process a complete line (convenience method for line-based processing).

        Args:
            line: Complete line to process
        """
        await self.process_chunk(line + '\n')

    # State Machine Implementation

    async def _process_character(self, char: str) -> None:
        """Process a single character through the state machine."""
        if self._state == ParseState.SEEKING_JSON:
            if char == '{':
                self._state = ParseState.IN_JSON
                self._buffer = char
                self._brace_count = 1
                self._bracket_count = 0

        elif self._state == ParseState.IN_JSON:
            self._buffer += char

            if self._escape_next:
                self._escape_next = False
            elif char == '\\' and self._in_string:
                self._escape_next = True
            elif char == '"' and not self._escape_next:
                self._in_string = not self._in_string
            elif not self._in_string:
                if char == '{':
                    self._brace_count += 1
                elif char == '}':
                    self._brace_count -= 1
                elif char == '[':
                    self._bracket_count += 1
                elif char == ']':
                    self._bracket_count -= 1

                # Check if JSON is complete
                if self._brace_count == 0 and self._bracket_count == 0:
                    await self._complete_json_object()

    async def _complete_json_object(self) -> None:
        """Handle a complete JSON object."""
        try:
            # Parse the JSON
            json_data = json.loads(self._buffer)

            # Determine message type
            message_type = self._determine_message_type(json_data)

            # Create and emit event
            event = JSONEvent(
                json_data=json_data,
                message_type=message_type,
                raw_json=self._buffer,
                line_number=self._line_number,
                timestamp=asyncio.get_event_loop().time(),
                source=self.source
            )

            await self._emit_json_complete(event)

            # Update statistics
            self._stats['objects_parsed'] += 1

        except json.JSONDecodeError as e:
            # Emit error but try to recover
            await self._handle_parse_error(e)

        finally:
            # Reset state
            self._reset_state()

    def _determine_message_type(self, json_data: Dict[str, Any]) -> MessageType:
        """Determine the Claude message type from JSON data."""
        msg_type = json_data.get('type', '').lower()

        if msg_type in [t.value for t in MessageType]:
            return MessageType(msg_type)
        return MessageType.UNKNOWN

    async def _handle_parse_error(self, error: json.JSONDecodeError) -> None:
        """Handle JSON parsing error with recovery attempt."""
        self.logger.warning(f"JSON parse error: {error}")

        # Try to recover by finding next potential JSON start
        recovery_index = self._buffer.find('{', 1)

        if recovery_index > 0:
            # Emit error for the bad JSON
            await self._emit_error(ParseError(
                error=error,
                partial_data=self._buffer[:recovery_index],
                line_number=self._line_number,
                timestamp=asyncio.get_event_loop().time(),
                recoverable=True
            ))

            # Continue with remaining buffer
            self._buffer = self._buffer[recovery_index:]
            self._brace_count = 1
            self._stats['errors_recovered'] += 1
        else:
            # Cannot recover, reset completely
            await self._emit_error(ParseError(
                error=error,
                partial_data=self._buffer,
                line_number=self._line_number,
                timestamp=asyncio.get_event_loop().time(),
                recoverable=False
            ))
            self._reset_state()

    async def _handle_buffer_overflow(self) -> None:
        """Handle buffer overflow condition."""
        self.logger.error(f"Buffer overflow: {len(self._buffer)} bytes")

        await self._emit_event('buffer_overflow', {
            'buffer_size': len(self._buffer),
            'partial_data': self._buffer[:1000] + '...',
            'line_number': self._line_number
        })

        self._stats['buffer_overflows'] += 1
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset parsing state."""
        self._buffer = ""
        self._state = ParseState.SEEKING_JSON
        self._brace_count = 0
        self._bracket_count = 0
        self._in_string = False
        self._escape_next = False

    # Event System

    def on(self, event: str, handler: Callable) -> None:
        """
        Register an event handler.

        Args:
            event: Event name ('json_complete', 'parse_error', 'buffer_overflow')
            handler: Async callback function
        """
        if event not in self._event_handlers:
            raise ValueError(f"Unknown event: {event}")
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """
        Unregister an event handler.

        Args:
            event: Event name
            handler: Handler to remove
        """
        if event in self._event_handlers and handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)

    async def _emit_json_complete(self, event: JSONEvent) -> None:
        """Emit json_complete event."""
        await self._emit_event('json_complete', event)

    async def _emit_error(self, error: ParseError) -> None:
        """Emit parse_error event."""
        await self._emit_event('parse_error', error)

    async def _emit_event(self, event_name: str, data: Any) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers.get(event_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_name}: {e}")

    # Utility Methods

    def get_stats(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset parsing statistics."""
        self._stats = {
            'objects_parsed': 0,
            'errors_recovered': 0,
            'bytes_processed': 0,
            'buffer_overflows': 0
        }


## Integration with Orchestrator

### 1. Replacing Current JSON Parsing in `run_claude_with_realtime_output`

```python
async def run_claude_with_realtime_output(self, cmd: List[str], cwd: str, process_name: str) -> tuple[int, List[str]]:
    """Run claude command with real-time output display using JSONStreamMonitor"""

    # Create monitor for this process
    monitor = JSONStreamMonitor(source=process_name.lower())

    # Register event handlers
    async def on_json_complete(event: JSONEvent):
        # Process based on message type
        if event.message_type == MessageType.SYSTEM:
            await self._handle_system_message(event.json_data, process_name)
        elif event.message_type == MessageType.ASSISTANT:
            await self._handle_assistant_message(event.json_data, process_name)
        # ... etc

    monitor.on('json_complete', on_json_complete)

    # Start subprocess and process stream
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=env
    )

    # Process the stream
    await monitor.process_stream(process.stdout)
```

### 2. Supervisor Decision Extraction

```python
class SupervisorDecisionExtractor:
    """Extract supervisor decision from JSON stream"""

    def __init__(self):
        self.monitor = JSONStreamMonitor(source="supervisor")
        self.decision_candidates = []

        # Register handler to collect decision JSONs
        self.monitor.on('json_complete', self._on_json)

    async def _on_json(self, event: JSONEvent):
        """Collect potential decision JSONs"""
        if event.message_type == MessageType.ASSISTANT:
            content = event.json_data.get('message', {}).get('content', [])
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '')
                    # Look for decision JSON
                    if '"action"' in text:
                        self._extract_decision_json(text)

    def get_final_decision(self) -> Optional[Dict[str, Any]]:
        """Get the last valid decision JSON"""
        return self.decision_candidates[-1] if self.decision_candidates else None
```

### 3. Code Review Trigger Detection (Future)

```python
class CodeReviewTriggerMonitor:
    """Monitor for code review triggers in agent output"""

    def __init__(self, on_trigger: Callable):
        self.monitor = JSONStreamMonitor(source="agent")
        self.on_trigger = on_trigger

        self.monitor.on('json_complete', self._check_for_trigger)

    async def _check_for_trigger(self, event: JSONEvent):
        """Check if JSON contains code review trigger"""
        if event.message_type == MessageType.ASSISTANT:
            # Check for completion signals or specific patterns
            if self._is_code_review_trigger(event.json_data):
                await self.on_trigger(event)
```

## Event Flow Diagram

```
Stream Input → JSONStreamMonitor → Parse Characters → State Machine
                                                          ↓
                                                    Complete JSON?
                                                          ↓
                                               Yes ←------+-----→ No
                                                ↓                   ↓
                                          Parse JSON          Continue
                                                ↓                   ↓
                                          Emit Event          Buffer
                                                ↓
                                          Event Handlers
                                                ↓
                                    Application Logic (Display/Store/Act)
```

## Error Handling Strategy

1. **Malformed JSON**: Try to recover by finding next JSON start
2. **Buffer Overflow**: Emit event and reset, log partial data
3. **Nested Depth Exceeded**: Treat as complete and attempt parse
4. **Stream Errors**: Log and emit error event, continue processing
5. **Handler Errors**: Catch and log, don't break parsing

## Performance Considerations

1. **Efficient Character Processing**: Single pass state machine
2. **Minimal Regex Use**: Only for specific pattern matching
3. **Bounded Buffers**: Prevent memory exhaustion
4. **Async Processing**: Non-blocking stream handling
5. **Event Batching**: Option to batch multiple events

## Testing Strategy

1. **Unit Tests**:
   - State machine transitions
   - Complete JSON detection
   - Error recovery scenarios
   - Event emission

2. **Integration Tests**:
   - Real Claude output streams
   - Mixed text/JSON content
   - Concurrent stream processing
   - Large JSON objects

3. **Performance Tests**:
   - High-volume streams
   - Memory usage under load
   - Event handler performance

## Future Enhancements

1. **JSON Schema Validation**: Validate against expected schemas
2. **Stream Filtering**: Filter events by criteria before emission
3. **Compression Support**: Handle compressed streams
4. **Replay Capability**: Record and replay streams for debugging
5. **Metrics Collection**: Detailed performance metrics

## Conclusion

The JSONStreamMonitor design provides a robust, event-driven solution for parsing JSON from mixed text streams. It addresses all gaps identified in the current implementation while maintaining compatibility with existing code through careful integration points.
