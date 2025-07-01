# Code Review Instructions - Project Level

**IMPORTANT**: This project-level review MUST be run when:
1. The agent has completed work on the LAST task's subtasks
2. You've verified the agent's scratchpad shows "ALL TASKS COMPLETE"
3. All Task Master tasks show status "done"
4. You are about to return action: "complete"

## If All Project Tasks Are Complete (BEFORE returning action: "complete")

### 1. Do Your Own Comprehensive Project Review
* Review the overall project structure and organization
* Verify all tasks have been properly implemented
* Check that the code meets the project requirements
* Ensure consistent code quality throughout

### 2. Run Final AI-Powered Code Reviews
Use multiple models for expert validation:

#### First Review: O3 Project Analysis
* Run: `mcp__zen__codereview` with model="o3" for thorough project analysis (use full o3, NOT o3-mini)

#### Second Review: Gemini Expert Project Validation
* Run: `mcp__zen__codereview` with model="gemini-2.5-pro" for expert project validation (use gemini-2.5-pro specifically, NOT flash or older versions)

#### Critical Review Guidelines
**CRITICAL**: When calling these reviews, present ONLY the facts:
- The original project requirements from PRD
- What was implemented across all tasks
- The final state of the codebase
- DO NOT bias the review with your own assessment
- DO NOT mention whether you think it's good or bad
- Let the models form their own unbiased opinions

* Compare both results with your own review to form a comprehensive assessment

### 3. Wait for All Reviews
**IMPORTANT**: Wait for ALL THREE reviews (yours, o3, and gemini-2.5-pro) to complete before proceeding

## Critical Project Review Evaluation

### Focus on SHOWSTOPPER Issues Only
- Missing critical functionality from project requirements
- System-breaking bugs that prevent deployment
- Security vulnerabilities that compromise the entire system
- Major architectural flaws that make the system unmaintainable
- **PROJECT DRIFT**: Reviews indicate the project has gone "off the rails" from original PRD/tasks
  * Check if implementation diverged significantly from `.taskmaster/docs/prd.txt`
  * Verify work aligns with original Task Master task definitions
  * Flag if scope expanded beyond what was requested

### IGNORE Non-Critical Suggestions
- Code style improvements
- Minor optimizations
- Feature enhancements beyond original scope
- Refactoring suggestions that don't affect functionality

## Actions Based on Project Review Results

### CRITICAL: Project Review Issue Resolution

When reviews identify SHOWSTOPPER issues (CRITICAL or HIGH severity):

1. **MANDATORY**: You MUST return action "execute" with:
   - Clear task_id for the task that needs fixes
   - Guidance specifically addressing each CRITICAL/HIGH issue
   - Focus ONLY on showstopper issues, not suggestions

2. **Detection**: Look for these severity indicators:
   - 🔴 CRITICAL or "CRITICAL -"
   - 🟠 HIGH or "HIGH Priority"
   - Lines containing "severity": "critical" or "severity": "high"
   - References to security vulnerabilities, missing functionality, or system-breaking bugs

### If Reviews Identify SHOWSTOPPER Issues
* Return action: "execute" with targeted fixes ONLY for critical issues
* Do NOT attempt to implement all suggestions from reviews
* If project drift detected, guidance should realign with original requirements
* Provide SPECIFIC guidance listing each critical/high issue to fix

### If NO SHOWSTOPPER Issues (Project Works as Specified)
* Return action: "complete"
* Save ALL review feedback to `.cadence/code_review_recommendations.md`:
  - Include timestamp and final project status
  - Categorize suggestions by type (optimization, style, architecture, etc.)
  - Mark which were addressed vs. deferred for future consideration
