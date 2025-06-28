# Zen Guidance

## When to Use Zen Tools

Consider calling Zen tools directly when:
- The agent explicitly requested help ("HELP NEEDED", "Status: STUCK")
- The task involves complex debugging that might require external expertise
- Architecture decisions need validation before implementation
- Security-critical features need review before coding
- Performance optimization requires analysis
- The agent has repeatedly failed with similar errors
{% if has_previous_agent_result and not agent_completed_normally %}
- The agent was cut off at the turn limit (use zen analyze)
- You need help understanding what the agent completed
{% endif %}

{% if has_previous_agent_result and not agent_completed_normally %}
## Special Handling for Incomplete Runs

Since the agent didn't complete normally, consider using zen analysis:

### 1. Call Zen Analyze for Cutoff Analysis
`mcp__zen__analyze` with model="o3" (use full o3, NOT o3-mini)
- Present ONLY factual information:
  * Agent was cut off at {{ max_turns }} turn limit
  * List what TODOs were assigned
  * List what was completed based on scratchpad/logs
  * List what remains incomplete
  * DO NOT interpret or judge the agent's performance
- Get unbiased recommendations on how to proceed

### 2. Based on Zen Analysis, Decide Whether To
- Re-dispatch agent with remaining work
- Call zen debug for specific blockers
- Skip the task temporarily
- Break down the task differently

### 3. If Re-dispatching
Include zen's guidance in your instructions
{% endif %}

## Critical Guidance for Zen Usage

If you determine that Zen assistance is needed:

### Direct Help
- For direct help requests, you can call zen tools directly

### Always Specify Exact Models
* For debugging: `mcp__zen__debug` with model="o3" (full version)
* For analysis: `mcp__zen__analyze` with model="o3" (full version)
* For consensus: `mcp__zen__consensus` with models=["o3", "gemini-2.5-pro"]
* For code review: `mcp__zen__codereview` with model="o3" or "gemini-2.5-pro"

### Present ONLY Factual Information Without Bias
* State what was requested
* State what was done
* State what errors/issues occurred
* DO NOT interpret or judge performance

### Stay Focused
- Stay STRICTLY focused on the specific task at hand
- Do NOT expand scope based on Zen's suggestions
- Focus ONLY on critical fixes needed to complete the current task
- Document any broader suggestions from Zen in a markdown file for later review
- Remember: The goal is task completion, not architectural perfection
