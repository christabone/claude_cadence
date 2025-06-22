# Claude Cadence

A task-driven supervision system for Claude Code agent execution. Manages agent execution through Task Master integration, with agents working on TODOs until completion.

**Status: Core implementation complete. Currently in testing phase.**

## Overview

Claude Cadence provides a framework for managing Claude Code agents through task-based execution. The supervisor uses Task Master to get and track tasks, then provides TODOs to agents who work until all tasks are complete or a maximum turn limit is reached.

## Features

- **Task-Driven Execution**: Agents work on specific TODOs from the supervisor
- **Task Master Integration**: Supervisor manages tasks using Task Master
- **Natural Completion**: Agents work until tasks are done, not until turns run out
- **Safety Limits**: Maximum turn limits prevent runaway execution
- **Progress Tracking**: Monitor task completion and execution status via scratchpad files
- **Flexible Configuration**: YAML-based configuration for all settings
- **Enhanced Prompts**: Comprehensive agent context with safety-first design
- **Continuation Support**: Dynamic prompts for resumed execution with full context preservation
- **Zen MCP Integration**: Intelligent assistance when agents need help
  - Automatic stuck detection and debugging support
  - Code review and validation for critical tasks
  - Retrospective analysis for learning

## Installation

```bash
git clone https://github.com/christabone/claude_cadence.git
cd claude_cadence
pip install -r requirements.txt  # Minimal dependencies

# Check MCP servers are installed
python scripts/check_mcp_servers.py
```

## Quick Start

### Using the Orchestrator (Recommended)

```bash
# Run the orchestrator with a Task Master file
python -m cadence.orchestrator /path/to/project /path/to/tasks.json

# Or with config file
python -m cadence.orchestrator /path/to/project /path/to/tasks.json --config config.yaml
```

### Using Python API

```python
from cadence import TaskSupervisor

# Create supervisor with max turns safety limit
supervisor = TaskSupervisor(max_turns=40)

# Run with Task Master integration
success = supervisor.run_with_taskmaster("path/to/tasks.json")
```

## How It Works

1. **Load Tasks**: Supervisor loads tasks from Task Master
2. **Create TODOs**: Convert incomplete tasks into TODO items for agent
3. **Execute**: Agent works on TODOs until complete or max turns reached
4. **Track Progress**: Monitor task completion through structured markers
5. **Update Status**: Update task status in Task Master when complete

**Key Philosophy**: Task completion drives the process. Agents work naturally until TODOs are done. Turn limits exist only as safety nets to prevent infinite loops.

## Architecture

Claude Cadence implements a supervisor-agent pattern where:

- **Supervisor** uses Task Master to manage tasks and track progress
- **Agent** receives TODOs and works until completion
- **Execution** continues until all tasks complete or safety limit is hit
- **No arbitrary checkpoints** - only task completion or safety limits stop execution

This ensures agents focus on completing work rather than managing execution windows.

### Prompt System

The prompt system prioritizes safety and reliability:

- **Full Context Preservation**: Every continuation includes complete agent context
- **Scratchpad Tracking**: Agents maintain progress in `.cadence/scratchpad/` files
- **Dynamic Adaptation**: Prompts adjust based on completion status and issues
- **Safety-First Design**: Explicit warnings about `--dangerously-skip-permissions`

### Zen Integration

Claude Cadence integrates with zen MCP tools to provide assistance when agents encounter difficulties:

#### Help Protocol
Agents can request help by updating their scratchpad:
```markdown
## HELP NEEDED
Status: STUCK
Issue: [Description of the problem]
Attempted: [What has been tried]
Context: [Relevant files/errors]
```

Then declare "HELP NEEDED - STUCK" and exit. The supervisor will:
1. Detect the help request
2. Call appropriate zen tools (debug, review, etc.)
3. Generate continuation guidance
4. Resume execution with expert insights

#### Automatic Assistance
The supervisor automatically calls zen when:
- Same error occurs 3+ times (configurable)
- Agent uses >80% of max turns (configurable)
- Critical tasks match validation patterns
- Architecture/security/performance reviews are requested

## Configuration

Configuration is managed through `config.yaml`:

```yaml
execution:
  max_turns: 40         # Safety limit, not a target
  timeout: 600          # Execution timeout in seconds
  
agent:
  model: "claude-3-5-sonnet-20241022"
  tools: ["bash", "read", "write", "edit", ...]
  
task_detection:
  completion_phrase: "ALL TASKS COMPLETE"
  help_needed_phrase: "HELP NEEDED"
  stuck_status_phrase: "Status: STUCK"
  
zen_integration:
  output_lines_limit: 200     # Lines to include in zen context
  scratchpad_check_lines: 10  # Lines to check for help requests
  auto_debug_error_threshold: 3
  
processing:
  thread_join_timeout: 5      # Thread cleanup timeout
  status_check_interval: 30   # Progress check interval
  max_output_truncate_length: 3000
```

See `config.yaml` for all configuration options.

## Documentation

- [Architecture](docs/architecture.md) - System design details
- [Configuration](docs/configuration.md) - Configuration options
- [Examples](examples/) - Usage examples

## Requirements

- Python 3.8+
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Task Master MCP server for task management
- Zen MCP server for enhanced assistance (recommended)

## License

MIT License - see LICENSE file for details

## Development

See [TODO.md](TODO.md) for planned improvements and known issues.

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## Citation

If you use Claude Cadence in your research or projects, please cite:

```
@software{claude_cadence,
  title = {Claude Cadence: Task-Driven Supervision for AI Agents},
  author = {Chris Tabone},
  year = {2024},
  url = {https://github.com/christabone/claude_cadence}
}
```