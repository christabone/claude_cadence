# JSONStreamMonitor Design Summary

## Key Design Decisions

### 1. State Machine Approach
- **ParseState enum**: SEEKING_JSON → IN_JSON → (IN_STRING/ESCAPE_CHAR)
- Single-pass character processing for efficiency
- Maintains state between stream chunks

### 2. Event-Driven Architecture
- **Event Types**:
  - `json_complete`: Emitted when complete JSON object is parsed
  - `parse_error`: Emitted on parsing failures with recovery info
  - `buffer_overflow`: Emitted when buffer exceeds limits
- Async event handlers support
- Decouples parsing from business logic

### 3. Robust Error Recovery
- Attempts to find next JSON start after parse errors
- Preserves partial data for debugging
- Continues processing after errors
- Configurable recovery strategies

### 4. Buffer Management
- Configurable max buffer size (default 1MB)
- Handles partial JSON across line boundaries
- Bracket/brace counting for object completion detection
- String state tracking to avoid false positives

### 5. Integration Strategy
- Drop-in replacement for current line-by-line parsing
- Separate extractors for specific use cases:
  - `SupervisorDecisionExtractor`: Finds decision JSON in output
  - `CodeReviewTriggerDetector`: Monitors for review triggers
- Backward compatible with existing code

## Implementation Priorities

### Phase 1: Core Parser (Subtasks 1.3-1.5)
1. Implement state machine and buffer management
2. Add JSON completion detection
3. Create event system

### Phase 2: Integration (Subtask 1.6)
1. Replace parsing in `run_claude_with_realtime_output`
2. Implement `SupervisorDecisionExtractor`
3. Update supervisor analysis to use new extractor

### Phase 3: Advanced Features
1. Add `CodeReviewTriggerDetector` for future use
2. Implement statistics collection
3. Add performance optimizations

## Key Benefits

1. **Reliability**: Handles edge cases current implementation misses
2. **Maintainability**: Clean separation of concerns
3. **Extensibility**: Easy to add new event handlers
4. **Performance**: Efficient single-pass processing
5. **Debugging**: Detailed error information and statistics

## Testing Approach

1. **Unit Tests**:
   - State transitions
   - Buffer management
   - JSON completion detection
   - Error recovery

2. **Integration Tests**:
   - Real Claude output streams
   - Supervisor decision extraction
   - Mixed content handling

3. **Edge Cases**:
   - Partial JSON across chunks
   - Deeply nested objects
   - Malformed JSON recovery
   - Buffer overflow handling

## Migration Path

1. Implement JSONStreamMonitor as standalone module
2. Add integration wrapper methods
3. Test with real Claude output
4. Gradually replace existing parsing
5. Remove old parsing code

This design provides a solid foundation for reliable JSON stream processing while maintaining compatibility with the existing system.
