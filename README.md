# Claude Cadence

**🚧 UNDER CONSTRUCTION - NOT YET FUNCTIONAL 🚧**

A checkpoint-based supervision system for Claude Code agent execution. Implements periodic turn-limited execution with inter-checkpoint analysis and guided continuation.

**Note: This project is actively being developed and is not ready for use. The architecture is being designed and core components are still being implemented.**

## Overview

Claude Cadence enables external turn management, progress monitoring, and corrective guidance injection for long-running AI agent tasks. It leverages Claude Code's `--max-turns` and `--continue` flags to enforce deterministic checkpoint intervals with supervisor-driven course correction between execution phases.

## Features

- **Checkpoint Execution**: Run agents for fixed turn intervals using `--max-turns`
- **Progress Analysis**: Analyze agent output between checkpoints to assess task completion
- **Guided Continuation**: Resume execution with corrective guidance using `--continue`
- **Task Integration**: Optional Task Master MCP integration for structured task tracking
- **Cost Tracking**: Monitor token usage and costs across checkpoints
- **Flexible Supervision**: Configurable checkpoint intervals and supervision strategies

## Installation

```bash
git clone https://github.com/christabone/claude_cadence.git
cd claude_cadence
pip install -r requirements.txt  # Minimal dependencies
```

## Quick Start

```python
from cadence import CheckpointSupervisor

# Create supervisor with 15-turn checkpoints
supervisor = CheckpointSupervisor(checkpoint_turns=15, max_checkpoints=3)

# Run supervised task
result = supervisor.run_supervised_task(
    "Validate the GeneCards database and create YAML output"
)
```

## How It Works

1. **Spawn**: Launch an agent with specific tasks (turn limit as safety net)
2. **Execute**: Agent works naturally until tasks complete OR hits turn limit
3. **Checkpoint**: Execution pauses when agent exits or reaches turn limit
4. **Analyze**: Supervisor reviews progress and completion status
5. **Guide**: Provide guidance if tasks remain incomplete
6. **Repeat**: Continue until all tasks are done or max checkpoints reached

**Key Philosophy**: Task completion drives the process, not turn counting. Agents work naturally and exit when done. Turn limits are just safety nets to prevent runaway execution.

## Architecture

Claude Cadence implements a supervisor-executor pattern where:

- **Executor** (Claude Sonnet) focuses on completing assigned tasks
- **Supervisor** (Claude Opus or custom logic) monitors progress and provides guidance
- **Checkpoints** occur when tasks complete OR safety turn limit is reached
- **Continuation** only happens if tasks remain incomplete

This ensures agents work efficiently toward task completion rather than managing turn budgets.

## Documentation

- [Architecture](docs/architecture.md) - Detailed system design
- [API Reference](docs/api.md) - Class and method documentation
- [Patterns](docs/patterns.md) - Best practices and usage patterns
- [Examples](examples/) - Working examples for common use cases

## Requirements

- Python 3.8+
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Optional: Task Master MCP for structured task management

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## Citation

If you use Claude Cadence in your research or projects, please cite:

```
@software{claude_cadence,
  title = {Claude Cadence: Checkpoint-based Supervision for AI Agents},
  author = {Chris Tabone},
  year = {2024},
  url = {https://github.com/christabone/claude_cadence}
}
```