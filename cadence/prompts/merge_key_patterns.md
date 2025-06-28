# YAML Merge Key Naming Conventions and Patterns

## Analysis Date: 2025-01-26

## Overview
This document establishes naming conventions and organizational patterns for YAML merge keys within individual prompt files. All patterns use standard YAML syntax compatible with `yaml.safe_load()` to ensure security while reducing duplication.

## 1. Core Principles

### 1.1 Security First
- Use only standard YAML merge key syntax (`<<: *anchor`)
- Compatible with `yaml.safe_load()` - no custom loaders
- Anchors exist only within individual files
- No cross-file anchor references

### 1.2 Organization
- Place all anchor definitions in `_definitions:` section at file top
- Group related anchors by category
- Use descriptive, hierarchical naming
- Keep anchor scope local to file

### 1.3 Maintainability
- Self-documenting anchor names
- Consistent naming patterns
- Clear separation between definitions and usage
- Comments explaining anchor purpose

## 2. Naming Convention Standards

### 2.1 Anchor Naming Format
```
&category_subcategory_item
```

Examples:
- `&paths_cadence_scratchpad`
- `&vars_project_core`
- `&templates_status_complete`
- `&config_agent_defaults`

### 2.2 Category Prefixes
| Prefix | Usage | Example |
|--------|-------|---------|
| `paths_` | File system paths | `&paths_output_logs` |
| `vars_` | Variable collections | `&vars_session_info` |
| `templates_` | Message templates | `&templates_error_format` |
| `config_` | Configuration blocks | `&config_timeout_values` |
| `formats_` | Formatting patterns | `&formats_timestamp_iso` |
| `defaults_` | Default values | `&defaults_agent_settings` |

### 2.3 Reserved Names
Avoid these names to prevent confusion:
- `common`, `shared`, `global` (implies cross-file scope)
- `include`, `import`, `extends` (suggests loader functionality)
- Single-letter names (`a`, `x`, `tmp`)
- Python/YAML keywords

## 3. Organizational Patterns

### 3.1 File Structure Template
```yaml
# filename.yaml
# Purpose: Brief description of this prompt file

# ===== DEFINITIONS SECTION =====
# All anchors defined here for easy reference
_definitions:
  # Path definitions
  paths:
    cadence: &paths_cadence
      root: "{project_path}/.cadence"
      scratchpad: "{project_path}/.cadence/scratchpad"
      agent: "{project_path}/.cadence/agent"
      supervisor: "{project_path}/.cadence/supervisor"

    output: &paths_output
      logs: "{project_path}/.cadence/logs"
      snapshots: "{project_path}/.cadence/snapshots"
      reports: "{project_path}/.cadence/reports"

  # Variable collections
  variables:
    project: &vars_project
      project_path: "{project_path}"
      session_id: "{session_id}"
      agent_name: "{agent_name}"

    limits: &vars_limits
      max_turns: "{max_turns}"
      timeout: "{timeout}"
      retry_count: "{retry_count}"

  # Template definitions
  templates:
    headers: &templates_headers
      context: "=== SUPERVISED AGENT CONTEXT ==="
      todos: "=== YOUR TODOS ==="
      guidelines: "=== WORK GUIDELINES ==="

    status: &templates_status
      complete: "Status: COMPLETE ✅"
      in_progress: "Status: IN_PROGRESS 🔄"
      stuck: "Status: STUCK ⚠️"
      error: "Status: ERROR ❌"

  # Configuration blocks
  configuration:
    agent_defaults: &config_agent_defaults
      <<: *vars_project
      <<: *vars_limits
      paths:
        <<: *paths_cadence
        <<: *paths_output
      enable_logging: true
      debug_mode: false

# ===== MAIN CONTENT SECTION =====
# Actual prompt content using the defined anchors
agent_prompts:
  initial:
    # Merge multiple definitions
    <<: *config_agent_defaults

    # Use specific anchors
    headers: *templates_headers
    status_formats: *templates_status

    # Override merged values if needed
    max_turns: "50"  # Override the default from vars_limits

    # Add prompt-specific content
    sections:
      - header: "Initial Setup"
        content:
          _from_file: "prose/initial_setup.md"
```

### 3.2 Grouping Strategies

#### By Functionality
```yaml
_definitions:
  # Group by what they do
  path_resolution:
    project: &paths_project
      # All project-related paths
    workspace: &paths_workspace
      # All workspace paths

  data_formatting:
    json: &formats_json
      # JSON formatting rules
    yaml: &formats_yaml
      # YAML formatting rules
```

#### By Usage Context
```yaml
_definitions:
  # Group by where they're used
  agent_context:
    paths: &agent_paths
      # Paths used by agents
    vars: &agent_vars
      # Variables for agents

  supervisor_context:
    paths: &supervisor_paths
      # Paths used by supervisors
    vars: &supervisor_vars
      # Variables for supervisors
```

#### By Lifecycle Stage
```yaml
_definitions:
  # Group by when they're used
  initialization:
    config: &init_config
      # Startup configuration
    paths: &init_paths
      # Initial path setup

  runtime:
    config: &runtime_config
      # Runtime configuration
    paths: &runtime_paths
      # Dynamic paths
```

## 4. Best Practices

### 4.1 Anchor Scoping
```yaml
_definitions:
  # ✅ GOOD: Clear, descriptive scope
  paths:
    agent_output: &paths_agent_output
      logs: "{project_path}/.cadence/agent/logs"
      artifacts: "{project_path}/.cadence/agent/artifacts"

  # ❌ BAD: Too generic, unclear scope
  stuff: &stuff
    path1: "/some/path"
    thing2: "value"
```

### 4.2 Merge Key Usage
```yaml
# ✅ GOOD: Clear merge with overrides
agent_config:
  <<: *config_agent_defaults
  <<: *vars_project
  custom_timeout: 300  # Clear override

# ❌ BAD: Multiple conflicting merges
agent_config:
  <<: [*config1, *config2, *config3]  # Confusing precedence
```

### 4.3 Documentation
```yaml
_definitions:
  # Document complex anchors
  paths:
    # These paths are used for agent workspace isolation
    # Each agent gets its own subdirectory based on session_id
    agent_workspace: &paths_agent_workspace
      base: "{project_path}/.cadence/agent/{session_id}"
      input: "{project_path}/.cadence/agent/{session_id}/input"
      output: "{project_path}/.cadence/agent/{session_id}/output"
```

## 5. Common Patterns

### 5.1 Variable Collection Pattern
```yaml
_definitions:
  variables:
    # Core variables used everywhere
    core: &vars_core
      project_path: "{project_path}"
      session_id: "{session_id}"

    # Extended variables for specific contexts
    extended: &vars_extended
      <<: *vars_core  # Include core vars
      agent_id: "{agent_id}"
      task_id: "{task_id}"
```

### 5.2 Path Hierarchy Pattern
```yaml
_definitions:
  paths:
    # Base paths
    base: &paths_base
      root: "{project_path}"
      cadence: "{project_path}/.cadence"

    # Subdirectory paths
    subdirs: &paths_subdirs
      <<: *paths_base
      agent: "{project_path}/.cadence/agent"
      supervisor: "{project_path}/.cadence/supervisor"
```

### 5.3 Configuration Layering Pattern
```yaml
_definitions:
  config:
    # Base configuration
    base: &config_base
      timeout: 300
      retry: 3

    # Development overrides
    dev: &config_dev
      <<: *config_base
      timeout: 600  # Longer timeout for dev
      debug: true

    # Production settings
    prod: &config_prod
      <<: *config_base
      timeout: 120  # Shorter timeout for prod
      debug: false
```

### 5.4 Template Composition Pattern
```yaml
_definitions:
  templates:
    # Base message components
    components: &templates_components
      timestamp: "[{timestamp}]"
      session: "Session: {session_id}"
      status: "Status: {status}"

    # Composed messages
    messages: &templates_messages
      error: "{timestamp} ERROR in {session}: {error_msg}"
      success: "{timestamp} SUCCESS in {session}: {result}"
```

## 6. Migration Guidelines

### 6.1 Identifying Merge Key Candidates
Look for:
- Repeated configuration blocks
- Common variable sets
- Shared path definitions
- Template patterns

### 6.2 Refactoring Steps
1. Identify repeated structural content
2. Create anchor in `_definitions` section
3. Replace duplicates with merge key reference
4. Test with `yaml.safe_load()`
5. Verify functionality unchanged

### 6.3 Example Migration
Before:
```yaml
agent1:
  project_path: "{project_path}"
  session_id: "{session_id}"
  max_turns: "{max_turns}"
  paths:
    scratchpad: "{project_path}/.cadence/scratchpad"
    output: "{project_path}/.cadence/output"

agent2:
  project_path: "{project_path}"
  session_id: "{session_id}"
  max_turns: "{max_turns}"
  paths:
    scratchpad: "{project_path}/.cadence/scratchpad"
    output: "{project_path}/.cadence/output"
```

After:
```yaml
_definitions:
  vars:
    common: &vars_common
      project_path: "{project_path}"
      session_id: "{session_id}"
      max_turns: "{max_turns}"

  paths:
    standard: &paths_standard
      scratchpad: "{project_path}/.cadence/scratchpad"
      output: "{project_path}/.cadence/output"

agent1:
  <<: *vars_common
  paths: *paths_standard

agent2:
  <<: *vars_common
  paths: *paths_standard
```

## 7. Testing and Validation

### 7.1 Test Script
```python
import yaml

def validate_yaml_file(filepath):
    """Validate YAML file loads with safe_load and anchors work."""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Check that _definitions is not in final output
        if '_definitions' in data:
            print("✓ Definitions section present (for reference only)")

        # Verify merge keys resolved
        print(f"✓ Successfully loaded with yaml.safe_load()")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False
```

### 7.2 Common Issues
| Issue | Solution |
|-------|----------|
| Circular anchor references | Ensure anchors don't reference each other |
| Undefined anchor | Check anchor is defined before use |
| Merge conflicts | Order merge keys by precedence |
| Deep nesting | Limit anchor depth to 3 levels |

## 8. Summary

### Key Benefits
- **Security**: Uses only standard YAML features
- **Maintainability**: Clear organization and naming
- **Efficiency**: Reduces duplication within files
- **Compatibility**: Works with any YAML parser

### Remember
- Keep anchors local to individual files
- Use `_definitions:` section for organization
- Follow consistent naming patterns
- Test with `yaml.safe_load()` only
- Document complex anchors
