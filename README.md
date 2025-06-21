# Claude Cadence

**🚧 UNDER CONSTRUCTION - NOT YET FUNCTIONAL 🚧**

A task-driven supervision system for Claude Code agent execution. Manages agent execution through Task Master integration, with agents working on TODOs until completion.

**Note: This project is actively being developed and is not ready for use. The architecture is being designed and core components are still being implemented.**

## Overview

Claude Cadence provides a framework for managing Claude Code agents through task-based execution. The supervisor uses Task Master to get and track tasks, then provides TODOs to agents who work until all tasks are complete or a maximum turn limit is reached.

## Features

- **Task-Driven Execution**: Agents work on specific TODOs from the supervisor
- **Task Master Integration**: Supervisor manages tasks using Task Master
- **Natural Completion**: Agents work until tasks are done, not until turns run out
- **Safety Limits**: Maximum turn limits prevent runaway execution
- **Progress Tracking**: Monitor task completion and execution status
- **Flexible Configuration**: YAML-based configuration for all settings

## Installation

```bash
git clone https://github.com/christabone/claude_cadence.git
cd claude_cadence
pip install -r requirements.txt  # Minimal dependencies
```

## Quick Start

```python
from cadence import TaskSupervisor

# Create supervisor with max turns safety limit
supervisor = TaskSupervisor(max_turns=40)

# Run with Task Master integration
success = supervisor.run_with_taskmaster()
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

## Configuration

Configuration is managed through `config.yaml`:

```yaml
execution:
  max_turns: 40         # Safety limit, not a target
  timeout: 600          # Execution timeout in seconds
  
agent:
  model: "claude-3-5-sonnet-20241022"
  tools: ["bash", "read", "write", "edit", ...]
  
supervisor:
  model: "heuristic"    # or "claude-3-opus-latest" for LLM supervision
  verbose: true
```

## Documentation

- [Architecture](docs/architecture.md) - System design details
- [Configuration](docs/configuration.md) - Configuration options
- [Examples](examples/) - Usage examples

## Requirements

- Python 3.8+
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Task Master MCP server for task management

## License

MIT License - see LICENSE file for details

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