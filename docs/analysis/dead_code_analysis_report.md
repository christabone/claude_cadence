# Dead Code Analysis Report - Claude Cadence

## Summary
This report analyzes dead code in the Claude Cadence Python codebase, focusing on the requested files. Dead code includes unused functions, classes, imports, and any code that is defined but never called.

## 1. __init__.py and __main__.py

### cadence/__init__.py
- **Status**: Minimal dead code
- **Exports**: ConfigLoader, CadenceConfig, TaskManager, Task, ZenIntegration, ZenRequest
- **Issues**:
  - `TaskManager` and `Task` are exported but not defined in the codebase
  - Examples in the codebase import `TaskSupervisor` which doesn't exist in __init__.py

### cadence/__main__.py
- **Status**: Clean, no dead code
- Simply imports and calls `main` from the parent orchestrate module

## 2. code_review_agent.py

### Unused Classes/Methods
- **`quick_review()` function** (lines 445-469): Convenience function never called anywhere
- **`get_review_history()` method** (lines 412-414): Never called
- **`clear_history()` method** (lines 416-419): Never called
- **`get_health_status()` method** (lines 421-441): Never called

### Unused Imports
- None found

### Unused Enums
- All enums (ReviewSeverity, ReviewType, ModelProvider) are used

## 3. agent_dispatcher.py

### Status
- **COMPLETELY UNUSED MODULE** - No imports of this module found anywhere
- The enhanced_agent_dispatcher.py extends it, but the base AgentDispatcher is never used directly

### All Dead Code
- Entire `AgentDispatcher` class and all its methods
- Helper functions are all unused

## 4. json_stream_monitor.py

### Used By
- unified_agent.py (lines 139, 404)
- orchestrator.py (line 216)

### Unused Methods
- **`reset()` method** (lines 253-257): Never called
- **`get_last_json_object()` method** (lines 259-261): Set but never retrieved

### Unused Patterns
- `ESCAPED_JSON_PATTERN` (lines 22-25): Defined but never matches anything in practice

## 5. prompts.py

### Unused Classes
- **`TodoPromptManager` class** (lines 353-415): Never instantiated or used

### Unused Methods in PromptGenerator
- **`get_initial_prompt()` alias** (lines 58-60): Alias never used
- **`get_continuation_prompt()` alias** (lines 62-64): Alias never used
- **`generate_final_summary()` method** (lines 289-350): Never called

### Used By
- orchestrator.py imports PromptGenerator and ExecutionContext
- examples/custom_prompts.py uses PromptGenerator

## 6. log_utils.py

### Unused Functions
- **`get_colored_logger()` function** (lines 132-152): Never called
- **`setup_colored_logging()` function** (lines 107-129): Never called (orchestrate.py sets up its own)

### Unused Color Codes
Many color constants in the Colors class are unused:
- BLACK, WHITE (regular, not bold)
- All background colors (BG_*)
- MAGENTA (regular, not bold)

### Used By
- unified_agent.py uses Colors and setup_file_logging
- orchestrator.py uses Colors and setup_file_logging

## 7. config.py

### Unused Methods
- **`save()` method** (lines 435-459): Never called
- **`override_from_args()` method** (lines 461-540): Never called

### Unused Constants
Many constants defined at module level are redundant with dataclass defaults:
- SESSION_ID_FORMAT, SESSION_FILE_PREFIX (duplicated in SessionConfig)
- Various timeout constants (duplicated in configs)

### Dead Configuration Fields
Several configuration fields appear to be unused:
- ExecutionConfig: clean_logs_on_startup, max_log_size_mb
- SupervisorConfig: intervention_threshold, max_output_lines, stream_buffer_size
- OrchestrationConfig: quick_quit_seconds, workflow_max_history_size
- Many fields in various configs that have no corresponding usage

## 8. unified_agent.py

### Unused Methods
- **`_run_async_safely()` method** (lines 203-236): Defined but only used internally

### Unused Security Constants
- Many tools in ALLOWED_TOOLS are never used (e.g., NotebookRead, NotebookEdit)

### Used By
- orchestrator.py imports UnifiedAgent and AgentResult

## 9. retry_utils.py

### Unused Functions
- **`save_json_with_retry()` function** (lines 199-241): Never called
- **`parse_json_with_retry()` function** (lines 29-85): Never called

### Unused Exception Fields
- `RetryError.last_output` and `last_error` are set but never accessed

### Used By
- orchestrator.py uses run_claude_with_realtime_retry and RetryError

## 10. orchestrator.py

### Unused Methods
- **`cleanup_completion_marker()` method** (lines 101-110): Defined but never called
- **`load_state()` and `save_state()` methods**: Referenced but not shown in excerpt

### Unused Fields in SupervisorDecision
- `todos` field: Marked as "for backward compatibility" but never used
- Several code review fields that may not be fully utilized

## Key Findings

1. **Entire unused module**: agent_dispatcher.py is completely dead code
2. **Major unused class**: TodoPromptManager in prompts.py
3. **Many utility methods never called**: History/health methods in various classes
4. **Configuration bloat**: Many config fields defined but never used
5. **Convenience functions unused**: quick_review, save_json_with_retry, etc.

## Recommendations

1. **Remove agent_dispatcher.py** entirely - it's superseded by enhanced version
2. **Remove TodoPromptManager** class from prompts.py
3. **Clean up config.py** - remove unused fields and consolidate constants
4. **Remove unused utility methods** from code_review_agent.py
5. **Audit ALLOWED_TOOLS** in unified_agent.py - remove unused entries
6. **Consider removing unused color codes** from Colors class
7. **Fix __init__.py exports** - either implement TaskManager/Task or remove them
