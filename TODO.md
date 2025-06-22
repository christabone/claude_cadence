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

### 2. ✅ COMPLETED: Fix Code Review to Use Supervisor Flow
**Files**: `cadence/orchestrator.py`

**Implemented Solution**:
1. ✅ Removed `run_zen_code_review()` and `should_run_code_review()` methods from orchestrator
2. ✅ Removed `save_zen_documentation()` method as it was only used by direct code review
3. ✅ Modified supervisor prompt construction to include code review instructions:
   - If `code_review_frequency` = "task": Supervisor is instructed to request code review after agent completes all TODOs
   - If `code_review_frequency` = "project": Supervisor is instructed to request code review when all project tasks are complete
   - If `code_review_frequency` = "none": No code review instructions are added
4. ✅ The supervisor will now trigger code review via the existing zen_assistance flow

**How it works now**:
- Orchestrator reads config and adds appropriate code review instructions to supervisor prompt
- Supervisor analyzes agent work and decides if code review is needed based on instructions
- If needed, supervisor returns `action: "zen_assistance"` with `tool: "review"`
- Orchestrator's existing zen_assistance handler processes the request

### 3. Implement Consensus and Review Zen Tools
**Files**: `cadence/zen_integration.py`, `cadence/orchestrator.py`

Currently, consensus and review tools are designed but not implemented:
- **Consensus**: For helping agents choose between multiple implementation approaches
- **Review**: For proactive code review before implementation (different from post-task review)

Both tools:
- Have specialized prompts in `zen_prompts.py` ✅
- Can be triggered via supervisor `zen_assistance` action ✅
- Return mock responses (need actual implementation) ❌
- Need integration with the orchestrator feedback loop ❌

**TODO**: 
1. Implement actual MCP calls for `_zen_consensus()` and `_zen_review()`
2. Update orchestrator to handle `zen_assistance` action properly (currently just logs warning)
3. Create feedback mechanism to pass Zen guidance to agent in next iteration

## Medium Priority

### 4. Extract Hardcoded Values to Configuration
- Magic number '200' for output lines (zen_integration.py:215) → config.yaml
- Completion patterns ("ALL TASKS COMPLETE", "HELP NEEDED") → config.yaml
- Session ID format patterns → config.yaml
- Error categorization patterns → config.yaml

### 5. Add Input Validation
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

## Code Quality Improvements (from Code Review)

### 12. Replace os.chdir Usage (Medium Priority)
**Files**: `cadence/orchestrator.py` (lines 208, 276)
- Replace `os.chdir()` with the `cwd` argument in `subprocess.run()` calls
- This prevents global state modification and potential threading issues
- More robust pattern that isolates directory changes to subprocesses

### 13. Fix Inconsistent Signal Checking (Medium Priority)
**File**: `cadence/task_supervisor.py` (line 155)
- The `_check_scratchpad_status` method uses hardcoded strings instead of constants
- Make signal checking case-insensitive like elsewhere in the codebase
- Use `AgentPromptDefaults.HELP_SIGNAL` and `AgentPromptDefaults.COMPLETION_SIGNAL`

### 14. Simplify Per-Instance Logging (Medium Priority)
**File**: `cadence/task_supervisor.py` (line 107)
- Each TaskSupervisor creates its own logger with unique ID
- Consider using module-level logger: `logger = logging.getLogger(__name__)`
- Extract Markdown logging to a separate `MarkdownLogger` class for better separation of concerns

### 15. Centralize Logging Configuration (Low Priority)
**Files**: `orchestrate.py` (lines 22-29), `cadence/supervisor_cli.py` (lines 18-25)
- The `setup_logging` function is duplicated in both files
- Move to `cadence/utils.py` or create new `cadence/logging_utils.py`
- Follows DRY principle and simplifies future logging changes