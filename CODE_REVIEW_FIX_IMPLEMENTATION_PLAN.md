# Code Review Fix Implementation Plan

## Overview

This document outlines a robust implementation strategy to ensure the supervisor properly handles CRITICAL and HIGH severity code review findings before proceeding to the next task.

## Key Insight from Gemini 2.5 Pro

The most robust solution combines:
1. **Structured data** from code review tools (not parsing natural language)
2. **Explicit prompts** that reference structured fields
3. **Deterministic guardrails** that validate supervisor decisions
4. **Re-dispatch pattern** for fix attempts

## Implementation Strategy

### Phase 1: Structured Code Review Output

Currently, the zen MCP tools return unstructured text. We need to parse their output into a structured format:

```python
def extract_code_review_severity(self, tool_response: str) -> Dict[str, Any]:
    """
    Extract severity information from zen code review response.

    Returns structured data:
    {
        "status": "issues_found",  # or "passed"
        "highest_severity": "CRITICAL",  # CRITICAL, HIGH, MEDIUM, LOW, or None
        "issues": [
            {
                "severity": "CRITICAL",
                "description": "Missing HTTP status check",
                "location": "line 60"
            }
        ],
        "raw_response": "..."  # Original response for reference
    }
    """
    # Parse the zen tool response
    severity_levels = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    found_severities = []
    issues = []

    # Look for severity patterns in the response
    patterns = {
        'CRITICAL': [r'🔴\s*CRITICAL', r'CRITICAL\s*-', r'"severity":\s*"critical"'],
        'HIGH': [r'🟠\s*HIGH', r'HIGH\s*Priority', r'"severity":\s*"high"'],
        'MEDIUM': [r'🟡\s*MEDIUM', r'MEDIUM\s*Priority', r'"severity":\s*"medium"'],
        'LOW': [r'🟢\s*LOW', r'LOW\s*Priority', r'"severity":\s*"low"']
    }

    for severity, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, tool_response, re.IGNORECASE):
                found_severities.append(severity)
                # Extract issue description (simplified for now)
                issues.append({
                    "severity": severity,
                    "description": f"Found {severity} severity issue",
                    "location": "See full review"
                })

    # Determine highest severity
    if found_severities:
        highest = max(found_severities, key=lambda x: severity_levels.get(x, 0))
        return {
            "status": "issues_found",
            "highest_severity": highest,
            "issues": issues,
            "raw_response": tool_response
        }
    else:
        return {
            "status": "passed",
            "highest_severity": None,
            "issues": [],
            "raw_response": tool_response
        }
```

### Phase 2: Store Code Review Results

Add to `SupervisorDecision` dataclass:

```python
@dataclass
class SupervisorDecision:
    """Decision made by supervisor analysis"""
    # ... existing fields ...

    # Code review results
    code_review_results: Optional[Dict[str, Any]] = None
    has_critical_issues: bool = False
    requires_fixes: bool = False
```

### Phase 3: Post-Decision Validation

Add validation after supervisor makes a decision:

```python
def validate_code_review_decision(self, decision: SupervisorDecision,
                                  code_review_results: Dict[str, Any],
                                  previous_task_id: str) -> bool:
    """
    Validate supervisor's decision against code review results.

    Returns True if valid, False if supervisor is incorrectly skipping fixes.
    """
    if not code_review_results:
        return True  # No review to validate against

    highest_severity = code_review_results.get("highest_severity")
    needs_fixes = highest_severity in ["CRITICAL", "HIGH"]

    if needs_fixes:
        # Check if supervisor is addressing the issues
        if decision.action == "execute":
            # Is it re-dispatching the same task?
            if decision.task_id == previous_task_id:
                # Check if guidance mentions fixes
                fix_keywords = ['fix', 'critical', 'high', 'issue', 'error',
                                'vulnerability', 'review']
                guidance_lower = (decision.guidance or "").lower()
                has_fix_guidance = any(kw in guidance_lower for kw in fix_keywords)

                if has_fix_guidance:
                    logger.info("Supervisor correctly re-dispatching for fixes")
                    return True
                else:
                    logger.warning("Supervisor re-dispatching same task but without fix guidance")
                    return False
            else:
                # Moving to different task despite critical issues
                logger.error(
                    f"VALIDATION FAILED: Supervisor moving to task {decision.task_id} "
                    f"despite {highest_severity} issues in task {previous_task_id}"
                )
                return False

    return True  # No critical issues or properly handled
```

### Phase 4: Force Re-evaluation

In the orchestrator, after supervisor decision:

```python
# In orchestrator.py, after receiving supervisor decision
if decision.action == "execute" and code_review_pending:
    # Extract code review results from supervisor output
    supervisor_output = read_last_supervisor_output(session_id)
    review_results = extract_code_review_severity(supervisor_output)

    # Store in decision for tracking
    decision.code_review_results = review_results

    # Validate the decision
    is_valid = validate_code_review_decision(
        decision, review_results, previous_task_id
    )

    if not is_valid:
        logger.warning("Supervisor decision failed validation - forcing re-evaluation")

        # Create explicit context for re-evaluation
        fix_context = {
            "force_fix_evaluation": True,
            "critical_issues_found": review_results["issues"],
            "previous_task_id": previous_task_id,
            "instruction": (
                f"Code review found {review_results['highest_severity']} severity issues. "
                f"You MUST re-dispatch task {previous_task_id} with specific fix instructions."
            )
        }

        # Re-run supervisor with continue flag and explicit context
        decision = self.run_supervisor_analysis(
            session_id,
            use_continue=True,
            iteration=iteration,
            code_review_pending=True,
            additional_context=fix_context
        )

        # Track that this is a fix dispatch
        agent_was_fixing_code_review = True
```

### Phase 5: Enhanced Supervisor Prompt

Add to the supervisor prompt template:

```markdown
## CRITICAL: Code Review Decision Rules

When code review results are available, you MUST follow these rules:

1. **Check Severity**: Look for these indicators in code review output:
   - 🔴 CRITICAL or "CRITICAL -" = Immediate fixes required
   - 🟠 HIGH or "HIGH Priority" = Must fix before proceeding
   - 🟡 MEDIUM = Should fix but can proceed with caution
   - 🟢 LOW = Optional improvements

2. **Decision Logic**:
   - IF severity is CRITICAL or HIGH:
     - action: "execute"
     - task_id: SAME AS CURRENT (not next task!)
     - guidance: MUST include specific fixes from review
     - Example: "Fix critical issues: 1) Add response.raise_for_status() after HTTP request (line 60)"

   - IF severity is MEDIUM or LOW only:
     - Document issues in recommendations file
     - Proceed to next task

3. **Validation**: Your decision will be validated. If you skip critical fixes, you'll be asked to reconsider.

{% if additional_context.force_fix_evaluation %}
⚠️ **VALIDATION OVERRIDE ACTIVE** ⚠️
{{ additional_context.instruction }}

You previously tried to skip critical fixes. This is your chance to correct that decision.
{% endif %}
```

## Testing Plan

1. **Unit Test**: `test_extract_code_review_severity()`
   - Test parsing of various code review formats
   - Verify correct severity extraction

2. **Integration Test**: `test_validate_code_review_decision()`
   - Test validation logic with various scenarios
   - Verify correct detection of skipped fixes

3. **End-to-End Test**:
   - Run task with known issues
   - Verify supervisor gets code review results
   - Confirm supervisor re-dispatches for fixes
   - Verify agent receives fix guidance
   - Confirm supervisor uses --continue after fixes

## Benefits

1. **Deterministic**: Not relying solely on LLM interpretation
2. **Transparent**: Clear logging of validation failures
3. **Recoverable**: Automatic re-evaluation on validation failure
4. **Traceable**: Structured data throughout the process

## Next Steps

1. Implement `extract_code_review_severity()` function
2. Add validation logic to orchestrator
3. Update supervisor prompt with explicit rules
4. Add comprehensive logging
5. Test with known code review scenarios
