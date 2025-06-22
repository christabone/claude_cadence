# Zen MCP Integration Summary

## Overview
Implemented specialized prompts and safeguards for Zen MCP usage in Claude Cadence to prevent scope creep and maintain focus on task completion.

## Key Changes

### 1. Specialized Zen Prompts (`cadence/zen_prompts.py`)
Created focused prompts for each Zen tool scenario:

- **Debug**: Minimal fixes to unblock stuck agents
- **Review**: Critical functionality checks only (no style nitpicks)
- **Consensus**: Quick, practical decisions to keep moving forward
- **Precommit**: Safety and correctness validation
- **Analyze**: Retrospective analysis with minimal recovery steps

Each prompt includes:
- Critical constraints to prevent scope expansion
- Focus on task completion over perfection
- Clear output requirements
- Reminders to document broader suggestions separately

### 2. Code Review Integration
- Added `code_review_frequency` config option in `config.yaml`:
  - `"none"`: Disabled
  - `"task"`: Run after each task completion (default)
  - `"project"`: Run at project end
  
- Supervisor automatically triggers code review when:
  - Agent completes all TODOs successfully
  - Code review is enabled for tasks

### 3. Zen Usage Safeguards

#### Supervisor Prompt Updates:
- Clear guidance on when to request Zen assistance
- Explicit reminders to stay focused on current task
- Instructions to document broader suggestions
- Specification of which Zen tool to use

#### Agent Prompt Updates:
- Special handling when guidance includes Zen assistance
- Reminders to not expand scope
- Instructions to note but not implement out-of-scope suggestions

### 4. Zen Documentation Feature
- Automatically saves all Zen interactions to markdown file
- Located in `.cadence/zen_suggestions/zen_suggestions_{session_id}.md`
- Includes:
  - All Zen requests and responses
  - Suggestions for future consideration
  - Priority areas for improvement
  - Reminder that task completion takes precedence

## Usage Scenarios

### 1. Debug Assistance
**Trigger**: Agent stuck or repeated errors
**Focus**: Unblock with minimal intervention
```json
{
  "zen_needed": {
    "required": true,
    "tool": "debug",
    "reason": "Agent stuck on import error",
    "focus_area": "dependency resolution"
  }
}
```

### 2. Code Review
**Trigger**: Task completion (all TODOs done)
**Focus**: Critical functionality validation
- Runs automatically based on config
- Checks for security, data integrity, runtime errors
- Documents future improvements separately

### 3. Architecture Decisions
**Trigger**: Complex choices blocking progress
**Focus**: Quick, practical decisions
```json
{
  "zen_needed": {
    "required": true,
    "tool": "consensus",
    "reason": "Need to choose between 3 auth approaches",
    "focus_area": "authentication implementation"
  }
}
```

### 4. Critical Task Validation
**Trigger**: Security/payment/database tasks
**Focus**: Safety and correctness only
- Pattern matching on task descriptions
- Precommit validation before marking complete

### 5. Retrospective Analysis
**Trigger**: Task cut off at turn limit
**Focus**: Minimal steps to complete
- Analyzes what was done vs. remaining
- Suggests focused next steps
- Documents lessons learned

## Configuration

### config.yaml
```yaml
zen_integration:
  code_review_frequency: "task"  # When to run code review
  auto_debug_error_threshold: 3   # Errors before debug help
  validate_on_complete:           # Patterns requiring validation
    - "*security*"
    - "*payment*"
    - "*auth*"
    - "*database*"
```

### Best Practices
1. **Stay Focused**: All Zen interactions emphasize task completion
2. **Document Later**: Broader suggestions saved for future review
3. **Minimal Fixes**: Just enough to unblock progress
4. **Clear Decisions**: No "it depends" - make practical choices
5. **Safety First**: Critical tasks get validation, others move forward

## Benefits
- Prevents scope creep from external model suggestions
- Maintains focus on task completion
- Documents improvement opportunities without implementing them
- Provides expert help only when truly needed
- Balances quality with pragmatic progress