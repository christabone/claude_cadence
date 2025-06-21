# Claude Cadence TODO List

## High Priority

### 1. Replace Mock Implementations in Zen Integration
**File**: `cadence/zen_integration.py`
**Lines**: 383-387 and similar sections in other `_zen_*` methods

Currently, the zen integration methods return mock responses:
```python
# For now, return mock response - real implementation would call zen
return {
    "tool": "debug",
    "success": True,
    "model": model,
    "thinking_mode": thinking_mode,
    "guidance": f"[Zen debug analysis would appear here for: {reason}]",
    ...
}
```

**TODO**: Replace with actual MCP tool calls using subprocess to invoke the zen tools properly.

## Medium Priority

### 2. Extract Hardcoded Values to Configuration
- Magic number '200' for output lines (zen_integration.py:215) → config.yaml
- Completion patterns ("ALL TASKS COMPLETE", "HELP NEEDED") → config.yaml
- Session ID format patterns → config.yaml
- Error categorization patterns → config.yaml

### 3. Add Input Validation
- Validate session_id format in all methods that accept it
- Add regex pattern for valid session IDs
- Validate file paths and ensure they're within project bounds

## Low Priority

### 4. Remove Unused Constants
- `SECONDS_PER_TURN_ESTIMATE` in prompts.py appears to be defined but unused
- Review all constants for actual usage

### 5. Make Completion Patterns Configurable
- Move hardcoded strings like "ALL TASKS COMPLETE" and "HELP NEEDED" to configuration
- Allow users to customize these patterns for different use cases

### 6. Improve Error Messages
- Add more context to error messages
- Include suggestions for common issues
- Better formatting for multi-line errors

## Future Enhancements

### 7. Add Metrics and Monitoring
- Track execution times
- Monitor error rates
- Collect usage statistics (opt-in)

### 8. Enhance Test Coverage
- Add integration tests for actual Claude CLI interaction
- Test edge cases for session cleanup
- Add performance benchmarks

### 9. Documentation Improvements
- Add architecture diagrams
- Create developer guide for extending the system
- Add more examples for different use cases

## Technical Debt

### 10. Refactor Backward Compatibility Code
- The `_detect_cutoff` method has backward compatibility that adds complexity
- Plan migration path to remove old detection method

### 11. Standardize Logging
- Consistent log levels across all modules
- Structured logging for better parsing
- Add log rotation configuration