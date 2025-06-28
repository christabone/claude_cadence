# JSON Output Format Requirements

{% if has_previous_agent_result %}
## Workflow Summary
1. ✅ Process agent's completed work (update subtasks, code review)
2. ✅ Find the next task with pending subtasks
3. ✅ Create TODO list with ONLY pending subtasks
4. ✅ Output JSON decision
{% endif %}

## CRITICAL: JSON Output Requirement

**YOU MUST OUTPUT A JSON DECISION AT THE END OF YOUR ANALYSIS.**

After completing all analysis and Task Master operations, you MUST output a valid JSON object.
Do not end your response without providing the JSON decision.

## Required Output

After analyzing the tasks, output ONLY a JSON object (no other text) with this exact structure:

```json
{
    "action": "execute",
    "task_id": "1",
    "task_title": "Documentation Setup",
    "subtasks": [
        {
            "id": "1.1",
            "title": "Create README.md",
            "description": "Create a comprehensive README file with project overview"
        },
        {
            "id": "1.2",
            "title": "Set Up Documentation Structure",
            "description": "Create docs/ directory with initial documentation files"
        }
    ],
    "project_root": "{{ project_path }}",
    "guidance": "Focus on implementing secure authentication using JWT tokens",
    "session_id": "{{ session_id }}",
    "reason": "Task 1 has 2 incomplete subtasks that need implementation"
}
```

## Important Guidelines

### Remember
- Use `projectRoot={{ project_path }}` in ALL Task Master MCP tool calls
- Output ONLY the JSON object, no explanatory text before or after
- Include ONLY pending/in-progress subtasks in the subtasks array
- Each subtask must include id, title, and description from Task Master
- The project_root field tells the agent where Task Master files are located
- If ALL Task Master tasks show status "done", set action to "complete" AND create `.cadence/project_complete.marker`
- If no tasks have pending subtasks, set action to "skip"
- For code reviews, call `mcp__zen__codereview` with model="o3" or "gemini-2.5-pro" as instructed
- The orchestrator will continue running until you signal completion with the marker file

## Final Reminder

**END YOUR RESPONSE WITH THE JSON DECISION OBJECT ONLY.**

**NO TEXT AFTER THE JSON. THE LAST THING IN YOUR OUTPUT MUST BE THE CLOSING } OF THE JSON.**
