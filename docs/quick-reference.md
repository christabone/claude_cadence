# Claude Cadence Quick Reference

## Core Concepts

### The Cadence Pattern
```
spawn → wait → analyze → guide → repeat
```

### Key Components

1. **CheckpointSupervisor**: Core supervision engine
2. **TaskManager**: Task Master integration
3. **CheckpointResult**: Execution details
4. **SupervisorAnalysis**: Analysis and guidance

## Basic Usage

### Python API
```python
from cadence import CheckpointSupervisor

supervisor = CheckpointSupervisor(
    checkpoint_turns=15,    # Turns per checkpoint
    max_checkpoints=3,      # Maximum checkpoints
    verbose=True           # Show progress
)

success, cost = supervisor.run_supervised_task("Your task here")
```

### CLI Usage
```bash
# Basic supervision
cadence "Write a Python web scraper"

# Custom settings
cadence --turns 10 --checkpoints 5 "Complex task"

# Task Master integration
cadence --taskmaster

# Verbose output
cadence -v "Debug this code"
```

## Advanced Patterns

### Opus Supervisor Pattern
```python
class OpusSupervisor(CheckpointSupervisor):
    def __init__(self):
        super().__init__(
            model="claude-3-5-sonnet-20241022"  # Executor
        )
        self.supervisor_model = "claude-3-opus-20240229"  # Analyzer
```

### Task Master Integration
```python
from cadence import TaskManager

task_manager = TaskManager(".")
if task_manager.load_tasks():
    # Use tasks for guidance
    progress = task_manager.analyze_progress(output)
```

## Configuration

### Environment Variables
- `CLAUDE_CODE_PATH`: Path to claude CLI (optional)

### Output Structure
```
cadence_output/
├── checkpoint_1_*.log      # Raw agent output
├── checkpoint_2_*.log      
├── session_summary_*.json  # Session metadata
```

## Cost Optimization

### Model Selection
- **Execution**: Use cheaper models (Haiku, Sonnet)
- **Analysis**: Use smarter models sparingly (Opus)
- **Checkpoints**: Balance frequency vs cost

### Turn Budgets
- Short checkpoints (5-10 turns): Tight control, higher overhead
- Medium checkpoints (15-20 turns): Balanced approach
- Long checkpoints (25+ turns): More autonomy, less guidance

## Troubleshooting

### Common Issues

1. **"Agent execution timed out"**
   - Reduce checkpoint turns
   - Simplify task complexity

2. **"No Task Master tasks found"**
   - Initialize with: `claude -p "Set up task master"`
   - Check for `.taskmaster/tasks/tasks.json`

3. **High costs**
   - Use cheaper models for execution
   - Reduce checkpoint frequency
   - Optimize prompts

### Debug Mode
```python
supervisor = CheckpointSupervisor(verbose=True)
# Check logs in cadence_output/
```