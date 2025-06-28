# Core Prose Content Directory

This directory contains extracted prose content from prompts.yaml to improve maintainability and readability.

## Directory Structure

```
core/
├── setup/               # Activation and setup instructions
├── context/            # Agent context and explanations
├── guidelines/         # Work guidelines and rules
├── instructions/       # Specific instructions (begin work, retry, etc.)
├── templates/          # Reusable templates
├── supervisor/         # Supervisor-specific content
└── safety/            # Safety notices and warnings
```

## Naming Conventions

- **File naming**: Use kebab-case for all markdown files (e.g., `work-guidelines.md`, `safety-notice.md`)
- **Content structure**: Each file should have a clear title and maintain original content meaning
- **Path references**: All files are referenced from prompts.yaml using relative paths

## Content Categories

### setup/
- Serena MCP activation instructions
- System initialization procedures

### context/
- Supervised agent context explanations
- Agent behavior guidelines

### guidelines/
- Work execution guidelines
- Code review processes
- Tool usage guidance

### instructions/
- Begin work instructions
- Scratchpad retry procedures
- JSON output formatting requirements

### templates/
- TODO list templates
- Supervisor analysis templates
- Issue section templates
- Continuation context templates

### supervisor/
- Orchestrator task master prompts
- Analysis instructions
- Decision frameworks

### safety/
- Safety notices and warnings
- Security guidelines

## Usage

These files are referenced from prompts.yaml using the `_from_file` pattern for secure explicit loading:

```yaml
content:
  _from_file: "core/guidelines/work-guidelines.md"
```

## Migration Notes

This structure was created as part of the Split YAML migration project (Task 4) to extract 20 major prose blocks totaling ~52,000 characters from the monolithic prompts.yaml file.

Total content extracted:
- **20 prose blocks**
- **~8,500 words**
- **~52,000 characters**
- **Largest block**: Orchestrator Task Master Prompt (~18,000 characters)
