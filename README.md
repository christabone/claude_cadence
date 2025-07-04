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
- **YAML-Based Prompts**: Unified prompt generation with Jinja2 conditional logic
- **Intelligent Continuation**: Dynamic prompts adapt to execution state, retries, and completion status
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

### YAML-Based Prompt System

Claude Cadence uses a sophisticated YAML-based prompt system with Jinja2 templating:

- **Unified Architecture**: All prompts generated from `cadence/prompts.yaml` using Jinja2 conditional logic
- **Context Preservation**: Complete agent state carried through continuation prompts
- **Dynamic Variables**: Template variables adapt based on execution state and completion status
- **Scratchpad Integration**: Agent progress tracked in `.cadence/scratchpad/` files
- **Safety-First Design**: Built-in warnings and permission handling
- **Conditional Logic**: Different prompt flows based on task status, errors, and retry attempts

#### PromptLoader System

The `PromptLoader` class provides the core YAML loading functionality with advanced features:

```python
from cadence.prompt_loader import PromptLoader

# Initialize with default config
loader = PromptLoader()

# Or with custom config path
loader = PromptLoader("path/to/prompts.yaml")

# Load templates with include support
content = loader.config['agent_prompts']['initial']

# Format templates with context variables
prompt = loader.format_template(template, context_vars)
```

**Key Features:**
- **Include Support**: Use `!include filename.md` to modularize prompt content
- **Security**: Path traversal protection prevents access outside base directory
- **Error Handling**: Graceful handling of missing files, malformed YAML, circular dependencies
- **Template Processing**: Jinja2 integration for dynamic content generation
- **Performance**: Efficient loading with validation and error reporting

**File Organization:**
```
cadence/prompts/
├── core/                  # Core prompt components
│   ├── instructions/      # Task instructions
│   ├── templates/         # Reusable templates
│   ├── safety/           # Safety notices
│   └── context/          # Context sections
└── prompts.yaml          # Main configuration with !include references
```

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

Configuration is managed through `config.yaml`. See the file for all available options including execution settings, model configuration, MCP integrations, and prompt customization.

### Troubleshooting PromptLoader Issues

**Common Issues:**

1. **FileNotFoundError: Include file not found**
   - Check that included files exist relative to the YAML file location
   - Verify file permissions and paths

2. **YAMLError: Circular dependency detected**
   - Review your !include chains for loops (file A includes B, B includes A)
   - Use `load_yaml_with_includes()` directly to debug inclusion paths

3. **YAMLError: Path outside base directory**
   - Security feature prevents `../` path traversal attacks
   - Use relative paths within the project directory only

4. **Template formatting errors**
   - Missing variables are preserved as `{variable_name}` (graceful degradation)
   - Check template syntax and available context variables

**Debug Mode:**
```python
# Enable detailed error reporting
import logging
logging.basicConfig(level=logging.DEBUG)

from cadence.prompt_loader import load_yaml_with_includes
content = load_yaml_with_includes("prompts.yaml")
```

## Documentation

- **[Prompt System Guide](docs/architecture/claude_cadence_prompt_system_guide.md)** - **Complete prompt system reference**
  - Comprehensive guide to the YAML-based prompt architecture
  - Jinja2 conditional logic and template resolution
  - Iteration-by-iteration supervisor-agent communication flow
  - Variable dependency chains and context preservation
  - Scratchpad creation system and retry mechanisms
  - Task Master and Zen MCP integration points
  - Troubleshooting guide for prompt development
  - Essential reference for understanding and modifying the prompt system
- [Development Documentation](docs/development/) - TODOs, code review findings, and agent documentation
- [Examples](examples/) - Usage examples

## Requirements

- Python 3.8+
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Task Master MCP server for task management
- Zen MCP server for enhanced assistance (recommended)

## License

MIT License - see LICENSE file for details

## Development

See [docs/development/TODO.md](docs/development/TODO.md) for planned improvements and known issues.

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
