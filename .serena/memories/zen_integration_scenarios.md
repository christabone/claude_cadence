# Zen Integration - 5 Scenarios

Claude Cadence integrates Zen assistance for these 5 scenarios:

1. **Stuck Detection** (`zen__debug`)
   - Triggered when agent writes "HELP NEEDED" or "Status: STUCK"
   - Also for architecture/security/performance review requests
   - Provides debugging assistance and unblocking guidance

2. **Error Pattern Detection** (`zen__debug`)
   - Triggered when same error occurs 3+ times
   - Categorizes errors (import, syntax, permission, etc.)
   - Helps break error loops with targeted solutions

3. **Task Validation** (`zen__precommit`)
   - Triggered for critical tasks (payment, security, auth, database, etc.)
   - Validates work before marking complete
   - Safety check for high-risk changes

4. **Cutoff Detection** (`zen__analyze`)
   - Triggered when agent stops unexpectedly (hit turn limit)
   - Provides retrospective analysis
   - Helps plan continuation strategy

5. **Review/Consensus** (`zen__review` / `zen__consensus`)
   - For code review needs
   - For architectural decisions requiring multiple perspectives
   - Gets consensus from multiple models when needed

Each scenario generates continuation guidance that's injected into the next agent execution.
