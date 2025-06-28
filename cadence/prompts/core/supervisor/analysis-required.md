# Analysis Required

## First: Read the Scratchpad

Read the scratchpad at `{{ project_path }}/.cadence/scratchpad/session_{{ session_id }}.md`

## Then Provide Analysis Covering

1. What TODOs were completed (based on scratchpad)?
2. What TODOs remain incomplete?
3. Were there any issues or blockers noted?
4. If execution stopped early, what caused it?
5. What guidance would help complete remaining work?

## Zen Assistance Evaluation

Evaluate if zen assistance would be beneficial:

### Stuck Detection
- Did the agent explicitly request help ("HELP NEEDED", "Status: STUCK")?
- Is the agent blocked on a specific technical issue?

### Error Patterns
- Are there repeated errors (same error 3+ times)?
- Are the errors preventing progress?

### Task Cutoff
- Did the agent appear to be cut off mid-work?
- Are there signs of incomplete execution (no completion message, work in progress)?

### Critical Validation
- Does this task involve security, database, or payment operations?
- Would expert review improve safety/quality?

## Zen Recommendation Format

If ANY of these conditions are met, recommend zen assistance with:
- Tool to use (debug, review, precommit, analyze)
- Specific reason for the recommendation

## Response Format

Format your response with:
1. Task progress summary
2. Issues/blockers identified
3. Zen recommendation (if applicable): "ZEN_RECOMMENDED: [tool] - [reason]"
4. Continuation guidance
