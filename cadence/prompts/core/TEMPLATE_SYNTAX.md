# Template Syntax Guide

This directory contains markdown files that use Jinja2 template syntax. Understanding the dual syntax is important for developers and AI assistants working with these files.

## Template Syntax Types

### 1. Variable Substitution: `{{ variable }}`
Used for simple variable replacement.

**Examples:**
- `{{ project_path }}` - Replaced with the actual project path
- `{{ session_id }}` - Replaced with the current session ID
- `{{ max_turns }}` - Replaced with the maximum turn limit

### 2. Control Structures: `{% statement %}`
Used for logic and flow control (if/else, for loops, etc.).

**Examples:**
```jinja2
{% if has_previous_agent_result %}
  Content shown when agent has previous results
{% else %}
  Content shown for first iteration
{% endif %}

{% for item in items %}
  - {{ item }}
{% endfor %}
```

## Why Both Syntaxes?

This is **standard Jinja2 templating** - not a mistake or inconsistency:
- `{{ }}` = Output/print a value
- `{% %}` = Execute logic/control flow
- `{# #}` = Comments (rarely used in our files)

## Common Patterns in Our Files

1. **Conditional Content**
   ```jinja2
   {% if is_first_iteration %}
   ⚠️  **FIRST ITERATION NOTICE**: This is iteration {{ iteration }}.
   {% endif %}
   ```

2. **File Paths**
   ```jinja2
   Create file: {{ project_path }}/.cadence/scratchpad/session_{{ session_id }}.md
   ```

3. **Dynamic Instructions**
   ```jinja2
   {% if agent_completed_normally %}
   Great job! The agent completed successfully.
   {% else %}
   The agent needs assistance - see details below.
   {% endif %}
   ```

## For Future Developers and LLMs

When editing these files:
- Maintain the existing syntax pattern
- Use `{{ }}` for values that need to be inserted
- Use `{% %}` for conditional logic or loops
- Don't convert between the two - they serve different purposes
- All variables must be provided by the prompt rendering system

## Variable Sources

Variables come from:
- `config.yaml` - Configuration values
- Runtime context - Session IDs, paths, statuses
- Task analysis - Task numbers, completion states
- User inputs - Project paths, model selections

See `prompts.yaml` for how these templates are assembled and rendered.
