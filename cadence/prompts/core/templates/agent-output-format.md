# Agent JSON Output Format Requirements

## CRITICAL: JSON Output Requirement

**YOU MUST OUTPUT A JSON RESULT AT THE END OF YOUR WORK.**

After completing all your assigned tasks, you MUST output a valid JSON object to signal your completion status.

## Required Output Format

At the end of your work session, output ONLY a JSON object (no other text after it).

### Available Status Actions

1. **"success"** - When ALL assigned subtasks are finished OR partial work completed successfully:
```json
{
    "status": "success",
    "completed_subtasks": ["1.1", "1.2", "1.3"],
    "session_id": "{{ session_id }}",
    "summary": "All assigned subtasks completed successfully",
    "help_needed": false,
    "execution_notes": "Brief summary of what was accomplished"
}
```

2. **"help_needed"** - When stuck, blocked, or need supervisor assistance:
```json
{
    "status": "help_needed",
    "completed_subtasks": ["1.1"],
    "session_id": "{{ session_id }}",
    "summary": "Need assistance to proceed",
    "help_needed": true,
    "execution_notes": "Detailed explanation of what help is needed and what was tried"
}
```

3. **"error"** - When encountering critical errors that prevent progress:
```json
{
    "status": "error",
    "error_type": "build_failure",
    "error_message": "Tests failed due to missing dependencies",
    "completed_subtasks": ["1.1"],
    "session_id": "{{ session_id }}",
    "summary": "Critical error encountered",
    "help_needed": true,
    "execution_notes": "Error details and attempted solutions"
}
```

## Requesting Help from Supervisor

**When you get stuck, blocked, or need assistance:**

1. Set `"status": "help_needed"` or `"status": "error"`
2. Set `"help_needed": true`
3. Use `"execution_notes"` to be very specific about what help you need and what you've tried

**Examples of good execution_notes for help requests:**
- `"execution_notes": "Need guidance on which testing framework to use for this React component. I've researched Jest vs Vitest but unclear which fits this project's setup."`
- `"execution_notes": "Cannot resolve import errors - tried relative and absolute imports. Module structure seems unclear. Need architectural guidance."`
- `"execution_notes": "Stuck on database schema design - need help deciding between normalization approaches for user relationships."`
- `"execution_notes": "Tests are failing with unclear error messages about async operations. Need debugging help to understand the root cause."`

**The supervisor will review your request and provide specific guidance or escalate to zen tools for deeper analysis.**

## Backward Compatibility

**IMPORTANT**: For compatibility with existing systems, you must ALSO:

1. **Update your scratchpad** with completion status
2. **Include the completion phrase** "ALL TASKS COMPLETE" in your scratchpad when status is "success"
3. **Include "HELP NEEDED"** phrase when help_needed is true

## Field Descriptions

- **status**: Current execution state (success/help_needed/error)
- **completed_subtasks**: Array of subtask IDs that are fully done
- **remaining_subtasks**: Array of subtask IDs still needing work (optional)
- **session_id**: Current session identifier
- **summary**: Brief description of current state
- **todos_remaining**: Array of human-readable remaining tasks
- **help_needed**: Boolean indicating if supervisor assistance is required
- **execution_notes**: Detailed notes about progress, issues, or help requests
- **blocked_on**: Description of blocking issue (only for "blocked" status)
- **error_type**: Category of error (only for "error" status)
- **error_message**: Detailed error description (only for "error" status)

## Final Reminder

**END YOUR RESPONSE WITH THE JSON RESULT OBJECT ONLY.**

**NO TEXT AFTER THE JSON. THE LAST THING IN YOUR OUTPUT MUST BE THE CLOSING } OF THE JSON.**

This JSON output will be parsed by the orchestrator to determine next steps.
