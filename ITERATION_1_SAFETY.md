# Iteration 1 Safety Mechanisms

This document summarizes all the safety mechanisms implemented to ensure iteration 1 NEVER receives prompts about processing agent work.

## 1. Session File Cleanup at Startup

**Location**: `cadence/orchestrator.py` line 299-314

```python
# CRITICAL: Clean up all old session files from previous runs
# This MUST happen before any other operations to prevent confusion
logger.info("Cleaning up old session files...")
self.cleanup_all_session_files()

# Also clean up any previous completion marker
self.cleanup_completion_marker()

# Double-check that no old agent result files exist
old_agent_results = list(self.supervisor_dir.glob("agent_result_*.json"))
if old_agent_results:
    logger.error(f"WARNING: Found {len(old_agent_results)} old agent result files after cleanup!")
else:
    logger.debug("Confirmed: No old agent result files exist")
```

## 2. Iteration-Based Agent Result Check

**Location**: `cadence/orchestrator.py` line 448-466

```python
# Check if we have previous agent results (only after first iteration)
previous_agent_result = None
if iteration > 1:
    agent_results_file = self.supervisor_dir / FilePatterns.AGENT_RESULT_FILE.format(session_id=session_id)
    logger.debug(f"Checking for agent results file: {agent_results_file}")
    if agent_results_file.exists():
        logger.debug(f"Agent results file exists, loading...")
        # Load file...
    else:
        logger.debug(f"No agent results file found at {agent_results_file}")
else:
    logger.debug(f"Iteration {iteration} - first iteration, not checking for agent results")

logger.info(f"Iteration {iteration}, has_previous_agent_result={previous_agent_result is not None}")
```

## 3. Jinja2 Template Processing

**Location**: `cadence/prompts.py` line 57-61

```python
# First, check if template contains Jinja2 syntax
if '{%' in template or '{{' in template:
    # Process with Jinja2
    jinja_template = Template(template)
    template = jinja_template.render(**context)
```

## 4. Conditional Prompt Templates

**Location**: `cadence/prompts.yaml` line 275-279

```yaml
{% if has_previous_agent_result %}
TASK: Process the agent's completed work, then analyze the current task state and decide what to do next.
{% else %}
TASK: Analyze the current task state and decide what the agent should work on first.
{% endif %}
```

## 5. Debug Logging

**Location**: `cadence/orchestrator.py` line 480-490

```python
# Log a preview of the TASK section to verify correct template
if "Process the agent's completed work" in base_prompt:
    logger.debug("Supervisor prompt includes agent work processing (iteration 2+)")
elif "Analyze the current task state and decide what the agent should work on first" in base_prompt:
    logger.debug("Supervisor prompt is for first iteration (no agent work)")
else:
    logger.warning("Supervisor prompt TASK section unclear - check template rendering")
```

## How It Works

1. **At Startup**: All old session files are deleted, ensuring no previous agent results can be found
2. **Iteration Check**: Only iterations > 1 even look for agent result files
3. **Context Setting**: `has_previous_agent_result` is explicitly set to `False` for iteration 1
4. **Template Rendering**: Jinja2 conditionals ensure different prompts based on `has_previous_agent_result`
5. **Debug Logging**: Confirms the correct prompt template is being used

## Testing

The safety mechanisms have been tested with:
1. Manual prompt generation tests
2. Debug logging to verify correct template selection
3. File system checks to ensure cleanup works

## Conclusion

With these mechanisms in place, it is impossible for iteration 1 to receive prompts about processing agent work, as:
- No old agent result files exist (cleaned at startup)
- The code explicitly doesn't look for agent results on iteration 1
- The template conditionally renders different content based on the iteration