# Zen MCP Trigger Paths Documentation

## Overview
The Claude Cadence system has two distinct paths for triggering Zen MCP assistance:

### Path 1: Automatic Detection (Post-Execution)
Location: `TaskSupervisor.execute_with_todos()` → `ZenIntegration.should_call_zen()`

**Triggers:**
1. **Debug** - Automatically detected when:
   - Agent writes "Status: STUCK" or "HELP NEEDED" in scratchpad
   - Output contains help patterns like "HELP NEEDED - STUCK", "ARCHITECTURE_REVIEW_NEEDED", etc.
   - Same error occurs 3+ times (configurable via `auto_debug_error_threshold`)

2. **Precommit** - Automatically detected when:
   - Task description matches critical patterns in `validate_on_complete` config
   - Default patterns: "*security*", "*database*", "*critical*", "*auth*", "*payment*"

3. **Analyze** - Automatically detected when:
   - Task appears cut off at turn limit
   - Multiple indicators suggest incomplete work (3+ indicators)
   - No "ALL TASKS COMPLETE" and has remaining todos

**Note:** This detection happens AFTER agent execution, stored in metadata but not currently acted upon.

### Path 2: Supervisor Decision (Pre-Execution)
Location: `SupervisorOrchestrator` → Supervisor prompt → JSON decision

**Triggers:**
The supervisor can request ANY Zen tool by setting `action: "zen_assistance"` with:
```json
{
    "action": "zen_assistance",
    "zen_needed": {
        "required": true,
        "tool": "debug|review|consensus|precommit|analyze",
        "reason": "Description of why help is needed",
        "focus_area": "Specific area to focus on"
    }
}
```

**When Supervisor Should Request:**
- **Debug**: Task involves complex debugging requiring external expertise
- **Consensus**: Architecture decisions need validation before implementation  
- **Review**: Code quality concerns (Note: different from automatic post-task review)
- **Precommit**: Security-critical features need review before coding
- **Analyze**: Performance optimization requires analysis

### Special Case: Code Review
Location: `SupervisorOrchestrator.run_orchestration_loop()`

**Trigger:**
- Runs automatically AFTER agent completes all TODOs successfully
- Controlled by `zen_integration.code_review_frequency` config
- Options: "none", "task" (default), "project"
- Uses `ZenPrompts.review_prompt()` for focused review

## Current Implementation Status

### ✅ Working:
1. Automatic detection logic (Path 1) - detects and logs but doesn't act
2. Code review after task completion
3. Specialized prompts for all Zen tools
4. Zen documentation saving

### ⚠️ Partially Implemented:
1. Supervisor zen_assistance action - recognized but Zen call not implemented
2. Path 1 detection results stored in metadata but not used

### ❌ Not Implemented:
1. Actual Zen MCP tool calls (currently returns mock responses)
2. Feeding Zen guidance back to agent for next iteration
3. Consensus and Review tools have no automatic triggers

## Recommended Architecture

### Unified Approach:
1. **Pre-execution Check**: Supervisor analyzes task and can request Zen help
2. **Execution**: Agent works with any Zen guidance provided
3. **Post-execution Check**: Automatic detection for issues
4. **Feedback Loop**: If issues detected, supervisor gets Zen guidance for next iteration

### Implementation Priority:
1. Complete zen_assistance action handling in orchestrator
2. Implement actual Zen MCP calls (replace mock responses)
3. Create feedback mechanism to use Path 1 detection results
4. Add automatic triggers for consensus/review scenarios

## Configuration

### config.yaml
```yaml
zen_integration:
  enabled: true
  stuck_detection: true
  auto_debug_threshold: 3
  code_review_frequency: "task"
  validate_on_complete:
    - "*security*"
    - "*database*"
    - "*critical*"
    - "*auth*"
    - "*payment*"
```

### Usage Examples

#### Supervisor Requesting Debug Help:
```json
{
    "action": "zen_assistance",
    "zen_needed": {
        "required": true,
        "tool": "debug",
        "reason": "Agent failed 3 times with import errors",
        "focus_area": "dependency resolution"
    }
}
```

#### Automatic Precommit Detection:
Task: "Implement user authentication system"
- Matches "*auth*" pattern
- `should_call_zen()` returns ("precommit", "Critical task validation: Matches pattern: *auth*")

#### Code Review After Completion:
- Agent completes all TODOs
- `agent_result.completed_normally == True`
- `code_review_frequency == "task"`
- Triggers `run_zen_code_review()`