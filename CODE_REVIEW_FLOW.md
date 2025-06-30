# Code Review Flow Design

## Overview
Implement a code review flow where the supervisor orchestrates code review after task completion based on configuration settings. The supervisor performs its own code review first, then coordinates with a review agent.

## Configuration
```yaml
zen_integration:
  code_review_frequency: "task"  # Options: "none", "task", "project"
```

## Flow Sequence

### 1. Normal Task Execution Flow
```
Orchestrator → Supervisor (1st run) → Agent (executes task) → Orchestrator checks config
```

### 2. Orchestrator Code Review Decision
After agent completes, orchestrator checks if code review is needed:
- If `code_review_frequency` is "task" → Flag that code review is needed
- Dispatch supervisor (2nd run) with special instructions

### 3. Supervisor 2nd Run (Code Review Check)
When supervisor runs the second time with code review flag:

1. **First Priority - Check Task Completion**:
   - Verify if all subtasks are actually done
   - If subtasks remain → Return "execute" action to continue work
   - If all subtasks done → Proceed to code review

2. **Perform Supervisor Code Review** (NO zen tools):
   - Analyze the code changes directly
   - Look for obvious issues, patterns, security concerns
   - Document findings internally
   - Output special JSON decision to orchestrator

### 4. Code Review JSON Decision Formats

**If subtasks not complete**:
```json
{
    "action": "execute",
    "task_id": "1",
    "subtasks": [/* remaining subtasks */],
    "reason": "Task has remaining subtasks to complete before code review"
}
```

**If subtasks complete, trigger code review**:
```json
{
    "action": "code_review",
    "task_id": "1.2",
    "task_title": "Implement user authentication",
    "review_scope": "task",
    "files_to_review": ["src/auth.py", "src/models/user.py"],
    "supervisor_findings": "Identified potential security concerns in password handling",
    "session_id": "20240115_143022_abc123",
    "reason": "All subtasks completed, initiating code review per configuration"
}
```

### 5. Orchestrator Handling Code Review Action
When orchestrator receives `"action": "code_review"`:

1. **Save Supervisor's Findings** for later use
2. **Dispatch Code Review Agent**:
   ```
   Orchestrator → Agent (code review profile) → Performs detailed review
   ```

### 6. Code Review Agent Execution
The code review agent:
- Reviews the specified files independently
- Checks for security issues, performance problems, code quality
- Generates detailed findings
- Outputs structured review results

### 7. Post-Review Supervisor Continuation (with --continue)
After code review agent completes:

```
Orchestrator → Supervisor (resumed with --continue flag)
```

The supervisor:
1. **Reviews Both Analyses**:
   - Its own initial code review findings
   - The code review agent's detailed findings
2. **Makes Decision**:
   - If critical issues found → Dispatch fix agent ("execute" with fix tasks)
   - If minor issues or all good → Continue to next task ("skip" or "execute" next)

### 8. Complete Decision Flow

```
Agent completes task
    └─ Orchestrator (checks code_review_frequency)
        └─ Supervisor (2nd run with review flag)
            ├─ All subtasks done?
            │   ├─ No → Return "execute" (continue subtasks)
            │   │   └─ Agent → Complete remaining work
            │   └─ Yes → Perform own review
            │       └─ Return "code_review" action
            │           └─ Orchestrator → Code Review Agent
            │               └─ Supervisor (continued)
            │                   ├─ Critical issues?
            │                   │   ├─ Yes → "execute" (fix issues)
            │                   │   └─ No → Continue next task
```

## Implementation Details

### Orchestrator Modifications

1. **Track Code Review State**:
```python
# After agent completes
if agent_result.success:
    review_freq = self.config.zen_integration.code_review_frequency
    if review_freq == "task":
        # Flag that code review is needed
        needs_code_review = True
        supervisor_context["code_review_pending"] = True
```

2. **Handle Code Review Action**:
```python
elif decision.action == "code_review":
    logger.info("Supervisor completed initial review, dispatching review agent")

    # Save supervisor's findings
    self.save_supervisor_findings(decision.supervisor_findings, session_id)

    # Run code review agent with special profile
    review_result = self.run_agent(
        todos=["Review code changes for security, performance, and quality issues"],
        guidance="Perform thorough code review of recent changes",
        profile="review",  # Use review agent profile
        session_id=session_id,
        files_to_review=decision.files_to_review
    )

    # Save review results
    self.save_code_review_results(review_result, session_id)

    # Continue supervisor with --continue flag
    supervisor_decision = self.run_supervisor_analysis(
        session_id=session_id,
        use_continue=True,  # IMPORTANT: Use --continue
        iteration=iteration,
        context={
            "supervisor_findings": decision.supervisor_findings,
            "review_agent_results": review_result
        }
    )
```

### Supervisor Modifications

1. **Handle Code Review Mode** (2nd run):
```python
# In supervisor prompt or logic
if context.get("code_review_pending"):
    # First check if all subtasks are done
    remaining_subtasks = self.get_remaining_subtasks(current_task)

    if remaining_subtasks:
        # Not ready for review yet
        return {
            "action": "execute",
            "task_id": current_task.id,
            "subtasks": remaining_subtasks,
            "reason": "Task has remaining subtasks to complete before code review"
        }

    # All subtasks done - perform supervisor's own review
    supervisor_findings = self.perform_code_review(modified_files)

    return {
        "action": "code_review",
        "task_id": current_task.id,
        "supervisor_findings": supervisor_findings,
        "files_to_review": modified_files,
        "reason": "All subtasks completed, initiating code review"
    }
```

2. **Handle Post-Review Continuation** (with --continue):
```python
# When resumed after code review agent
if context.get("review_agent_results"):
    # Analyze both reviews
    supervisor_findings = context.get("supervisor_findings")
    agent_findings = context.get("review_agent_results")

    critical_issues = self.identify_critical_issues(
        supervisor_findings,
        agent_findings
    )

    if critical_issues:
        # Need fixes
        return {
            "action": "execute",
            "task_id": current_task.id,
            "subtasks": self.create_fix_subtasks(critical_issues),
            "guidance": "Fix critical issues identified in code review",
            "reason": "Code review found issues requiring fixes"
        }
    else:
        # All good, continue
        return {
            "action": "skip",
            "reason": "Code review passed, proceeding to next task"
        }
```

### Agent Profile for Code Review

```yaml
agent:
  profiles:
    review:
      description: "Agent specialized in code review"
      temperature: 0.1  # Lower for focused analysis
      custom_prompt_prefix: |
        You are a senior code reviewer. Review the provided changes for:
        - Security vulnerabilities
        - Performance issues
        - Code quality and maintainability
        - Best practices adherence
        - Potential bugs
```

## Key Design Principles

1. **Task Completion First**: Always verify subtasks are done before reviewing
2. **Two-Stage Review**: Supervisor does initial review, then agent does detailed review
3. **No Zen Tools**: Supervisor performs its own code review without zen MCP
4. **Continue Flag**: Supervisor resumes with --continue after review agent
5. **Context Preservation**: Review findings passed between stages

## Benefits

1. **Automated Quality Control**: Code review happens automatically after tasks
2. **Flexible Configuration**: Can be disabled, per-task, or per-project
3. **Two-Perspective Review**: Supervisor + agent provide different viewpoints
4. **Actionable Results**: Issues found trigger fix agents automatically
5. **Efficient Flow**: No unnecessary reviews if subtasks incomplete

## State Management

The orchestrator needs to track:
- `code_review_pending`: Set when agent completes and review is configured
- `supervisor_findings`: Saved after supervisor's initial review
- `review_agent_results`: Saved after review agent completes
- `in_code_review`: Flag to know we're in review flow

## Edge Cases

1. **Review Agent Fails**: Supervisor should continue without blocking
2. **No Files Changed**: Skip code review if no code modifications
3. **Review Timeout**: Set reasonable time limits for review agents
4. **Circular Reviews**: Prevent reviewing the same code multiple times
5. **Subtasks Added During Review**: Handle new subtasks gracefully
