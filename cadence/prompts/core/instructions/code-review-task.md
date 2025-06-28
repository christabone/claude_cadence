# Code Review Instructions - Task Level

{% if has_previous_agent_result %}
**IMPORTANT**: This should be done as part of processing the agent's work BEFORE finding the next task.
{% endif %}

When the agent completes all TODOs for a task:

## 1. Do Your Own Code Review First
* Read the files that were modified
* Check if the implementation matches the task requirements
* Verify basic code quality and correctness

## 2. Update Task Status
If your review passes, update the task status to "done":
`mcp__taskmaster-ai__set_task_status --id=<task_id> --status=done --projectRoot={{ project_path }}`

## 3. Run AI-Powered Code Reviews
Use multiple models for comprehensive validation:

### First Review: O3 Analysis
* Run: `mcp__zen__codereview` with model="o3" for thorough analysis (use full o3, NOT o3-mini)

### Second Review: Gemini Expert Validation
* Run: `mcp__zen__codereview` with model="gemini-2.5-pro" for expert validation (use gemini-2.5-pro specifically, NOT flash or older versions)

### Critical Review Guidelines
**CRITICAL**: When calling these reviews, present ONLY the facts:
- What the agent was asked to do (the exact TODOs)
- What files were created/modified
- What the implementation does functionally
- DO NOT bias the review with your own assessment
- DO NOT mention whether you think it's good or bad
- Let the models form their own unbiased opinions

* Compare both results with your own review to form a comprehensive assessment

**IMPORTANT**: Wait for ALL THREE reviews (yours, o3, and gemini-2.5-pro) to complete before proceeding

## Critical Review Evaluation Guidelines

Reviews are NOT pass/fail gates - they provide guidance for improvement.

### Focus on CRITICAL Issues Only
These would break functionality:
- Bugs that prevent the code from working correctly
- Security vulnerabilities that expose sensitive data
- Major efficiency problems that severely impact performance
- Missing error handling for critical paths

### AVOID SCOPE CREEP - Do NOT Address
- Style preferences or minor refactoring suggestions
- "Nice to have" improvements unrelated to current task
- Architectural changes beyond the task requirements
- Additional features not requested in the task

## Actions Based on Review Results

### If Reviews Identify CRITICAL Issues
* Update task status back to "in-progress"
* Return action: "execute" with SPECIFIC guidance limited to fixing critical issues
* Keep guidance focused: "Fix the SQL injection vulnerability in login function"

### If NO CRITICAL Issues (Only Suggestions/Improvements)
* Task remains "done"
* Save all review suggestions to `.cadence/code_review_recommendations.md`
* Include timestamp, task ID, and categorized suggestions
* Continue to analyze the next task
