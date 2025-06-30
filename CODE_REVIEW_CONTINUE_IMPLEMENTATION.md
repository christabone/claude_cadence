# Code Review Continue Logic Implementation

## Summary of Changes

This document summarizes the implementation of the special continue logic for code review fixes in Claude Cadence.

## Requirement

The supervisor should ALWAYS use `--continue` when returning from an agent that was dispatched specifically to fix code review issues. This is not configurable and should be hardcoded behavior.

## Implementation Details

### 1. Added Tracking Flag

In `orchestrator.py` (line 549):
```python
agent_was_fixing_code_review = False  # Track if the previous agent was fixing code review issues
```

### 2. Detection Logic

When the orchestrator detects that:
- `code_review_pending` is True (code review just completed)
- The supervisor's decision is "execute" with guidance containing fix-related keywords
- Keywords checked: "fix", "critical", "vulnerability", "security", "bug"

Then it sets `agent_was_fixing_code_review = True` (lines 751-753)

### 3. Updated Continue Logic

Modified `_should_use_continue_for_supervisor` method (lines 1137-1155):
- Changed parameter from `code_review_pending` to `agent_was_fixing_code_review`
- Always returns True when `agent_was_fixing_code_review` is True
- This overrides any configuration settings

### 4. Flag Reset

The `agent_was_fixing_code_review` flag is reset to False when:
- Supervisor returns "complete" action
- Supervisor returns "skip" action
- Code review passes without critical issues

### 5. Code Review Pending Management

The `code_review_pending` flag is also properly managed:
- Set to True when agent completes task successfully and code_review_frequency is "task"
- Reset to False when code review passes without critical issues

## Testing

The implementation is ready for testing. The orchestrator will now:
1. Run agent to complete a task
2. Supervisor performs code review
3. If critical issues found, supervisor dispatches agent with fix guidance
4. When that agent completes, supervisor ALWAYS uses --continue
5. Normal supervisor continue configuration applies in all other cases
