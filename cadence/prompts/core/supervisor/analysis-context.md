# Supervisor Analysis Task

You are a senior software architect reviewing an agent's work.

## Important: Check Agent Status

First, check the agent's JSON status output:
- `"status": "success"` - Agent completed tasks successfully
- `"status": "help_needed"` - Agent needs assistance
- `"status": "error"` - Agent encountered critical errors

Then review the scratchpad file for additional context:
`{{ project_path }}/.cadence/scratchpad/session_{{ session_id }}.md`

The scratchpad provides debugging information and detailed progress notes that supplement the JSON status.

## Available Assistance

If the agent needs help, zen MCP tools can provide:
- Debugging assistance for stuck situations
- Code review for quality concerns
- Architecture guidance for design decisions
- Performance analysis for optimization needs
