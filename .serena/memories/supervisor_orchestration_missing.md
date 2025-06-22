# Supervisor Orchestration Missing

## Key Issue
The Claude Cadence supervisor is missing its core orchestration logic. Currently it only executes once with main tasks, but it should:

1. Process subtasks (not main tasks) as TODOs
2. Run in a loop with AI supervisor analysis after each execution
3. Make strategic decisions about continuation
4. Use AI model for supervision (never heuristic)

## Original Design Intent
- Supervisor extracts subtasks from current task → converts to TODOs
- Agent executes until done or max turns
- Supervisor (AI) analyzes results and decides next steps
- Zen integration for 5 scenarios (stuck, errors, validation, cutoff, review)
- Loop continues with same task or moves to next

## Current State
- Only processes main tasks (ignores subtasks completely)
- Executes once and exits
- No continuation logic despite tests expecting it
- Defaults to heuristic supervisor instead of AI

See IMPLEMENTATION_FIX_PLAN.md for detailed fix proposal.