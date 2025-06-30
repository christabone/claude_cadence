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
    "session_id": "{{ session_id }}",
    "summary": "All assigned subtasks completed successfully",
    "execution_notes": "Brief summary of what was accomplished"
}
```

2. **"help_needed"** - When stuck, blocked, or need supervisor assistance:
```json
{
    "status": "help_needed",
    "session_id": "{{ session_id }}",
    "summary": "Need assistance to proceed",
    "execution_notes": "Detailed explanation of what help is needed and what was tried"
}
```

3. **"error"** - When encountering critical errors that prevent progress:
```json
{
    "status": "error",
    "error_type": "build_failure",
    "error_message": "Tests failed due to missing dependencies",
    "session_id": "{{ session_id }}",
    "summary": "Critical error encountered",
    "execution_notes": "Error details and attempted solutions"
}
```

## Requesting Help from Supervisor

**When you get stuck, blocked, or need assistance:**

1. Set `"status": "help_needed"` or `"status": "error"`
2. Use `"execution_notes"` to be very specific about what help you need and what you've tried

**Examples of good execution_notes for help requests:**
- `"execution_notes": "Need guidance on which testing framework to use for this React component. I've researched Jest vs Vitest but unclear which fits this project's setup."`
- `"execution_notes": "Cannot resolve import errors - tried relative and absolute imports. Module structure seems unclear. Need architectural guidance."`
- `"execution_notes": "Stuck on database schema design - need help deciding between normalization approaches for user relationships."`
- `"execution_notes": "Tests are failing with unclear error messages about async operations. Need debugging help to understand the root cause."`

**The supervisor will review your request and provide specific guidance or escalate to zen tools for deeper analysis.**

## Important Notes

**JSON Output is Primary**: The JSON output is the authoritative signal for the orchestrator.

**Scratchpad is for Context**: The scratchpad provides additional debugging information and context for the supervisor to review. Always maintain a detailed scratchpad throughout execution.

**Update Both**: You must:
1. Maintain your scratchpad with detailed progress notes
2. End with the JSON status object

## Field Descriptions

- **status**: Current execution state (success/help_needed/error)
- **session_id**: Current session identifier
- **summary**: Brief description of current state
- **execution_notes**: Detailed notes about progress, issues, or help requests
- **error_type**: Category of error (only for "error" status)
- **error_message**: Detailed error description (only for "error" status)

## Final Reminder

**END YOUR RESPONSE WITH THE JSON RESULT OBJECT ONLY.**

**NO TEXT AFTER THE JSON. THE LAST THING IN YOUR OUTPUT MUST BE THE CLOSING } OF THE JSON.**

This JSON output will be parsed by the orchestrator to determine next steps.
