# Code Review Agent Dispatch System for Claude Cadence

## Executive Summary

Claude Cadence currently executes code reviews but fails to act on identified issues. This PRD proposes a dedicated agent dispatch system that ensures critical code review findings are addressed before task completion.

## Problem Statement

The current workflow has a critical gap:
1. Supervisor runs code reviews with zen MCP (o3/Gemini 2.5 Pro)
2. Critical and high-priority issues are identified
3. **Gap**: No automatic action is taken to fix these issues
4. Tasks are marked complete despite unresolved problems

## Proposed Solution

### System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Orchestrator  │────>│    Supervisor    │────>│  Primary Agent  │
│  (Python Core)  │     │  (Dispatcher)    │     │   (Worker)      │
└────────┬────────┘     └────────┬─────────┘     └─────────────────┘
         │                       │
         │                       ├──────────────┐
         │                       ▼              ▼
         │              ┌──────────────┐  ┌──────────────┐
         │              │ Code Review  │  │  Fix Agent   │
         │              │    Agent     │  │ (Conditional)│
         │              └──────────────┘  └──────────────┘
         │
         └─────> State Machine & JSON Stream Monitor
```

### Core Components

#### 1. Enhanced Orchestrator
- **JSON Stream Monitor**: Real-time parsing of agent/supervisor output
- **State Machine**: Track workflow states (working → review → fixing → complete)
- **Dispatch Controller**: Manage agent lifecycle and communication
- **Fix Verification Loop**: Ensure fixes are applied and validated

#### 2. Code Review Trigger Detection
```python
class ReviewTriggerDetector:
    def __init__(self):
        self.triggers = [
            'zen_mcp_codereview',  # Explicit review call
            'TASK_COMPLETE',       # Task completion marker
            'CODE_CHANGES_DONE'    # Code modification complete
        ]

    def should_trigger_review(self, json_output):
        # Logic to detect review triggers
```

#### 3. Agent Communication Protocol
```json
{
  "message_type": "DISPATCH_AGENT",
  "agent_type": "code_review",
  "context": {
    "task_id": "8.1",
    "parent_session": "abc123",
    "files_modified": ["path/to/file1.py", "path/to/file2.py"],
    "review_config": {
      "models": ["o3", "gemini-2.5-pro"],
      "severity_threshold": "medium",
      "focus_areas": ["security", "performance", "quality"]
    }
  },
  "success_criteria": {
    "review_complete": true,
    "issues_documented": true,
    "structured_output": true
  },
  "callback": {
    "on_complete": "PARSE_REVIEW_RESULTS",
    "on_error": "RETRY_OR_SKIP"
  }
}
```

#### 4. Review Result Processing
```python
class ReviewResultProcessor:
    def parse_review_output(self, review_output):
        # Extract structured issues
        issues = self._extract_issues(review_output)

        # Categorize by severity
        critical = [i for i in issues if i['severity'] == 'critical']
        high = [i for i in issues if i['severity'] == 'high']

        # Determine action
        if critical or high:
            return {
                'action': 'DISPATCH_FIX_AGENT',
                'issues': critical + high,
                'scope_check': self._check_scope_creep(issues)
            }
        return {'action': 'PROCEED'}
```

#### 5. Fix Agent Dispatch Logic
- **Scope Validation**: Ensure fixes don't exceed original task boundaries
- **Context Preservation**: Pass review findings and file context
- **Iteration Limits**: Max 3 fix attempts before escalation
- **Verification**: Re-run targeted review after fixes

### Implementation Phases

#### Phase 1: Foundation
1. Extend orchestrator with JSON stream parsing
2. Implement state machine for workflow tracking
3. Create ReviewTriggerDetector class
4. Add basic agent dispatch messaging

#### Phase 2: Code Review Integration
1. Build code review agent wrapper
2. Implement review result parser
3. Create issue severity classifier
4. Add structured output formatting

#### Phase 3: Fix Automation
1. Develop fix agent dispatch logic
2. Implement scope creep detection
3. Add fix verification workflow
4. Create iteration limits and escalation

#### Phase 4: Monitoring & Optimization
1. Robust logging for tracking errors and debugging

### Success Criteria

1. **Coverage**: 100% of completed tasks undergo code review (when task is flagged for code review in config.yaml)

### Configuration Options

```yaml
code_review_dispatch:
  enabled: true
  triggers:
    - zen_mcp_codereview
    - task_complete
  severity_threshold: medium  # low, medium, high, critical
  max_fix_iterations: 3
  timeout_seconds: 300
  models:
    review: ["o3", "gemini-2.5-pro"]
    fix: ["claude-3-opus"]
  scope_check:
    enabled: true
    max_file_changes: 10
    max_line_changes: 500
```

### Monitoring & Metrics

1. Just logging for now.

### Testing Strategy

1. Minimal unit tests for now

---

This PRD incorporates feedback from both o3 and Gemini 2.5 Pro models to create a comprehensive solution for the code review dispatch problem. The system is designed to be modular, testable, and configurable while maintaining clear boundaries and preventing scope creep.
