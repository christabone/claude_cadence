# Claude Cadence Prompts Guide

## Overview

Claude Cadence uses a sophisticated YAML-based prompt management system that provides:

- **Structured Templates**: All prompts are defined in `prompts.yaml` for easy customization
- **Context Awareness**: Prompts maintain state across checkpoints
- **Dynamic Generation**: Sections are conditionally included based on context
- **Reusable Components**: Shared sections reduce duplication

## Prompt Configuration Structure

### Main Configuration File

The default prompt configuration is located at `cadence/prompts.yaml`:

```yaml
# Shared context that appears in all agent prompts
shared_agent_context:
  supervision_explanation: |
    You are working under a checkpoint supervision system...
  work_guidelines: |
    Focus on making meaningful progress...
  early_exit_protocol: |
    Exit early if all work is complete...

# Agent prompt templates
agent_prompts:
  initial:
    sections: [...]
  continuation:
    sections: [...]

# Supervisor analysis prompts
supervisor_prompts:
  analysis:
    sections: [...]
```

### Key Components

1. **Shared Context** (`shared_agent_context`)
   - Reusable sections that appear in multiple prompts
   - Reduces duplication and ensures consistency
   - Can be referenced using `{shared_agent_context.section_name}`

2. **Agent Prompts** (`agent_prompts`)
   - `initial`: First checkpoint prompt with full context
   - `continuation`: Subsequent checkpoint prompts with progress updates

3. **Supervisor Prompts** (`supervisor_prompts`)
   - `analysis`: Template for supervisor to analyze checkpoint output
   - Includes evaluation criteria and guidance generation

4. **Task Templates** (`task_templates`)
   - Formatters for task lists, progress summaries, etc.
   - Support dynamic content generation

## Dynamic Variables

Prompts support variable substitution using `{variable_name}`:

### Common Variables

- `{checkpoint_turns}`: Number of turns in current checkpoint
- `{max_checkpoints}`: Total checkpoints available
- `{current_checkpoint}`: Current checkpoint number
- `{original_task}`: The original task description
- `{supervisor_guidance}`: Guidance from previous checkpoint
- `{task_description}`: Current task or objective

### Conditional Sections

Some sections are included only when relevant:

- `{task_breakdown}`: Only if tasks are provided
- `{progress_summary}`: Only if work was completed
- `{issues_section}`: Only if issues were encountered
- `{checkpoint_warnings}`: Only for final checkpoints

## Customizing Prompts

### Method 1: Create Custom YAML Configuration

```python
from cadence import CheckpointSupervisor, ContextAwarePromptManager

# Create custom supervisor with custom prompts
class CustomSupervisor(CheckpointSupervisor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_manager = ContextAwarePromptManager(
            original_task="",
            checkpoint_turns=self.checkpoint_turns,
            max_checkpoints=self.max_checkpoints,
            config_path="my_custom_prompts.yaml"
        )
```

### Method 2: Modify Default Configuration

```python
from pathlib import Path
import yaml

# Load default config
config_path = Path("cadence/prompts.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Modify specific sections
config['shared_agent_context']['work_guidelines'] = """
My custom guidelines...
"""

# Save to new file
with open("custom_prompts.yaml", 'w') as f:
    yaml.dump(config, f)
```

## Domain-Specific Examples

### Code Review Focus

```yaml
shared_agent_context:
  work_guidelines: |
    REVIEW PRIORITIES:
    1. Security vulnerabilities (CRITICAL)
    2. Memory leaks and performance issues
    3. Code style and maintainability
    4. Test coverage gaps
```

### Data Science Tasks

```yaml
shared_agent_context:
  work_guidelines: |
    ANALYSIS GUIDELINES:
    1. Validate data quality first
    2. Document all assumptions
    3. Provide reproducible code
    4. Include visualization outputs
```

### DevOps Automation

```yaml
shared_agent_context:
  work_guidelines: |
    AUTOMATION PRINCIPLES:
    1. Idempotent operations only
    2. Dry-run before destructive changes
    3. Log all actions comprehensively
    4. Rollback strategy required
```

## Early Exit Optimization

The prompt system includes built-in support for early agent exit:

```yaml
early_exit_protocol: |
  EARLY EXIT PROTOCOL:
  - If you complete all assigned tasks before using all turns, state 'ALL TASKS COMPLETE' and exit
  - Do not continue working beyond what was requested
  - Do not wait for confirmation after completing tasks
```

This helps:
- Reduce costs by not using unnecessary turns
- Improve efficiency for simple tasks
- Prevent agents from over-engineering solutions

## Best Practices

1. **Keep Prompts Focused**
   - Each section should have a single purpose
   - Use clear headers and formatting

2. **Use Shared Sections**
   - Define common instructions once
   - Reference them across prompts

3. **Provide Clear Exit Criteria**
   - Tell agents when to stop
   - Include completion phrases

4. **Include Context Progressively**
   - Initial prompts: Full context
   - Continuations: Only changes and guidance

5. **Test Prompt Changes**
   - Start with small modifications
   - Verify behavior before major changes

## Troubleshooting

### Common Issues

1. **Variables Not Replaced**
   - Check variable names match exactly
   - Ensure context includes all required values

2. **Sections Missing**
   - Verify YAML indentation is correct
   - Check conditional logic for sections

3. **Prompts Too Long**
   - Remove redundant sections
   - Use shared context effectively

### Debug Mode

Enable verbose logging to see prompt generation:

```python
supervisor = CheckpointSupervisor(verbose=True)
# This will print generated prompts before sending to agents
```

## Advanced Features

### Nested Variable References

```yaml
{shared_agent_context.supervision_explanation}
```

This allows referencing config sections from within templates.

### Multi-Line Formatting

```yaml
work_guidelines: |
  Line 1 of guidelines
  Line 2 of guidelines
  
  New paragraph here
```

### Conditional Warnings

```yaml
checkpoint_warnings:
  final: |
    ⚠️  THIS IS YOUR FINAL CHECKPOINT!
  penultimate: |
    📍 One checkpoint remaining
```

These are automatically included based on checkpoint progress.