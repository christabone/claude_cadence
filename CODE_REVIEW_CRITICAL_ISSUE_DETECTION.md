# Code Review Critical Issue Detection Strategy

## Problem Statement

The supervisor correctly runs code reviews after task completion but proceeds to the next task even when CRITICAL and HIGH severity issues are identified. This violates the intended workflow where critical issues should be fixed before moving on.

## Current Behavior (From Terminal Output)

1. Supervisor runs two code reviews:
   - `mcp__zen__codereview` with O3 model
   - `mcp__zen__codereview` with Gemini 2.5 Pro model

2. Both identify issues with severity levels:
   - 🔴 CRITICAL - Missing HTTP Status Check
   - 🟠 HIGH - Unsafe HTTP Client Mutation
   - 🟠 HIGH - Missing JSON Error Handling
   - 🟠 HIGH - Overly Broad Exception Handling

3. Despite these findings, supervisor outputs:
   ```json
   {
       "action": "execute",
       "task_id": "2",
       "task_title": "Register Tool and Update Documentation",
       ...
   }
   ```

## Root Cause

The supervisor is not properly parsing the code review results to detect CRITICAL/HIGH issues before making its decision. The code review output is embedded in the MCP tool response but not analyzed.

## Proposed Solution

### 1. Parse Code Review Results

Add a method to extract severity levels from code review output:

```python
def parse_code_review_severity(self, mcp_response: str) -> Dict[str, int]:
    """
    Parse code review response to count issues by severity.

    Returns:
        Dict with counts: {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    """
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

    # Look for severity indicators in the response
    patterns = {
        'critical': [r'🔴\s*CRITICAL', r'CRITICAL\s*-', r'severity.*critical'],
        'high': [r'🟠\s*HIGH', r'HIGH\s*Priority', r'severity.*high'],
        'medium': [r'🟡\s*MEDIUM', r'MEDIUM\s*Priority', r'severity.*medium'],
        'low': [r'🟢\s*LOW', r'LOW\s*Priority', r'severity.*low']
    }

    response_text = str(mcp_response).lower()

    for severity, patterns_list in patterns.items():
        for pattern in patterns_list:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            severity_counts[severity] += len(matches)

    return severity_counts
```

### 2. Analyze Supervisor JSON Output

Check the supervisor's final JSON decision to see if it's re-dispatching the same task:

```python
def is_redispatching_for_fixes(self, decision: SupervisorDecision,
                                previous_task_id: str) -> bool:
    """
    Check if supervisor is re-dispatching agent to fix code review issues.

    Indicators:
    - Same task_id as previous execution
    - Guidance contains fix-related keywords
    - Subtasks are being reset to pending
    """
    if decision.task_id != previous_task_id:
        return False

    fix_keywords = ['fix', 'critical', 'high', 'issue', 'vulnerability',
                    'error handling', 'status check']

    guidance_lower = decision.guidance.lower()
    return any(keyword in guidance_lower for keyword in fix_keywords)
```

### 3. Post-Decision Validation with Retry Loop

After supervisor outputs its decision but before dispatching the agent:

```python
# In orchestrator.py, after receiving supervisor decision
if decision.action == "execute":
    # Check if we just completed code review with issues
    if code_review_pending:
        # Get max retries from config (default to 5)
        max_fix_retries = self._get_config_value(
            'orchestration.code_review_fix_retries', 5
        )
        fix_retry_count = 0

        while fix_retry_count < max_fix_retries:
            # Look for code review results in supervisor output
            supervisor_output = read_supervisor_output(session_id)
            severity_counts = parse_code_review_severity(supervisor_output)

            if severity_counts['critical'] > 0 or severity_counts['high'] > 0:
                # Supervisor should have re-dispatched for fixes
                if not is_redispatching_for_fixes(decision, previous_task_id):
                    fix_retry_count += 1
                    logger.warning(
                        f"Code review found {severity_counts['critical']} CRITICAL "
                        f"and {severity_counts['high']} HIGH issues, but supervisor "
                        f"is moving to next task. Forcing re-evaluation "
                        f"(attempt {fix_retry_count}/{max_fix_retries})."
                    )

                    # Force supervisor to re-evaluate with --continue
                    decision = self.run_supervisor_analysis(
                        session_id,
                        use_continue=True,  # Force continue
                        iteration=iteration,
                        code_review_pending=True,
                        additional_context={
                            'force_review_fixes': True,
                            'severity_counts': severity_counts,
                            'previous_task_id': previous_task_id,
                            'fix_retry_attempt': fix_retry_count
                        }
                    )

                    # Check if supervisor made the right decision this time
                    if is_redispatching_for_fixes(decision, previous_task_id):
                        logger.info(
                            f"Supervisor correctly re-dispatched for fixes on "
                            f"retry {fix_retry_count}"
                        )
                        break
                else:
                    # Supervisor made correct decision, no retry needed
                    break
            else:
                # No critical/high issues found, proceed normally
                break

        # If we exhausted retries, log error but continue
        if fix_retry_count >= max_fix_retries:
            logger.error(
                f"Supervisor failed to handle critical code review issues after "
                f"{max_fix_retries} attempts. Proceeding anyway to avoid infinite loop."
            )
```

### 4. Enhanced Supervisor Prompt

Add explicit instructions in the supervisor prompt:

```markdown
### CRITICAL: Code Review Issue Resolution

When code reviews identify CRITICAL or HIGH severity issues:

1. **MANDATORY**: You MUST return action "execute" with:
   - The SAME task_id that just completed
   - Subtasks reset to "pending" status
   - Guidance specifically addressing each CRITICAL/HIGH issue
   - Clear instructions like: "Fix the following critical issues identified in code review:
     1. Add response.raise_for_status() after HTTP request (line 60)
     2. Add specific exception handling for JSONDecodeError
     3. Replace broad Exception catches with specific exceptions"

2. **NEVER** proceed to the next task when CRITICAL/HIGH issues exist

3. **Detection**: Look for these severity indicators in code review output:
   - 🔴 CRITICAL or "CRITICAL -"
   - 🟠 HIGH or "HIGH Priority"
   - Lines containing "severity": "critical" or "severity": "high"
```

## Implementation Steps

1. **Add severity parsing** to orchestrator.py
2. **Enhance supervisor prompt** with explicit critical issue handling
3. **Add post-decision validation with retry loop** to catch missed critical issues
4. **Track code review state** to know when fixes are expected
5. **Force re-evaluation** if supervisor tries to skip critical fixes
6. **Add config setting** `orchestration.code_review_fix_retries` to config.yaml (default: 5)

## Testing Strategy

1. Run a task that will have known issues
2. Verify code review identifies CRITICAL/HIGH issues
3. Confirm supervisor re-dispatches to fix issues (not next task)
4. Verify agent receives specific fix guidance
5. Confirm supervisor uses --continue after fix completion
