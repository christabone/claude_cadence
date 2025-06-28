# Secure YAML Optimization Template

## Purpose
This template provides a security-first approach to optimizing YAML prompt files by eliminating duplication while maintaining safety through standard YAML features and explicit file operations.

## Security Principles
1. **Use only `yaml.safe_load()`** - No custom loaders or tags
2. **Explicit file loading** - Use '_from_file' pattern with path validation
3. **Local scope only** - YAML anchors exist only within individual files
4. **Path traversal protection** - Validate all file paths before loading

## Template Structure

### 1. Identify Optimization Opportunities

#### 1.1 Duplication Analysis Checklist
- [ ] **Prose Content** (>100 chars or 3+ lines)
  - [ ] Instructions or documentation blocks
  - [ ] Warning messages
  - [ ] Help text or explanations
  - [ ] Code review templates

- [ ] **Structural Content** (<100 chars, configuration)
  - [ ] Path definitions
  - [ ] Variable collections
  - [ ] Status templates
  - [ ] Format patterns

#### 1.2 Metrics to Track
```yaml
metrics:
  before:
    total_lines: 0
    duplicate_blocks: 0
    file_size_kb: 0
  after:
    total_lines: 0
    unique_content_lines: 0
    file_size_kb: 0
    reduction_percentage: 0
```

### 2. Step-by-Step Optimization Guide

#### Step 1: Analyze Current File
```bash
# Count lines and identify patterns
wc -l prompts.yaml
grep -c "{project_path}" prompts.yaml
grep -c "{session_id}" prompts.yaml

# Find repeated blocks (manual review needed)
# Look for:
# - Identical instruction blocks
# - Repeated path patterns
# - Common variable sets
```

#### Step 2: Categorize Content
Use this decision tree:
```
Is the content >100 characters or 3+ lines?
├── YES → Is it human-readable prose?
│   ├── YES → Extract to prose/*.md file
│   └── NO → Keep in YAML with merge keys
└── NO → Use YAML merge keys
```

#### Step 3: Create File Structure
```
prompts/
├── secure_optimization_template.md    # This file
├── prose/                            # Extracted prose content
│   ├── serena_activation.md
│   ├── safety_notice.md
│   ├── work_guidelines.md
│   └── code_review_task.md
├── agent/                            # Agent-specific prompts
│   ├── initial.yaml
│   └── continuation.yaml
└── supervisor/                       # Supervisor prompts
    └── analysis.yaml
```

### 3. Pattern Examples

#### 3.1 YAML Merge Key Pattern

**BEFORE** (Duplicated content):
```yaml
agent1:
  project_path: "{project_path}"
  session_id: "{session_id}"
  max_turns: "{max_turns}"
  paths:
    scratchpad: "{project_path}/.cadence/scratchpad"
    output: "{project_path}/.cadence/output"
  status_complete: "Status: COMPLETE ✅"
  status_progress: "Status: IN_PROGRESS 🔄"

agent2:
  project_path: "{project_path}"
  session_id: "{session_id}"
  max_turns: "{max_turns}"
  paths:
    scratchpad: "{project_path}/.cadence/scratchpad"
    output: "{project_path}/.cadence/output"
  status_complete: "Status: COMPLETE ✅"
  status_progress: "Status: IN_PROGRESS 🔄"
```

**AFTER** (Using merge keys):
```yaml
# Define once at file top
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

  status:
    messages: &status_messages
      status_complete: "Status: COMPLETE ✅"
      status_progress: "Status: IN_PROGRESS 🔄"

# Use throughout file
agent1:
  <<: *vars_common
  paths: *paths_standard
  <<: *status_messages

agent2:
  <<: *vars_common
  paths: *paths_standard
  <<: *status_messages
```

**Reduction**: 14 lines → 8 lines (43% reduction)

#### 3.2 '_from_file' Pattern for Prose

**BEFORE** (Embedded prose):
```yaml
agent_prompts:
  initial:
    serena_activation: |
      === SERENA MCP ACTIVATION (CRITICAL) ===
      BEFORE any code analysis or file operations, you MUST activate Serena MCP:
      1. Run: mcp__serena__activate_project --project={project_path}
      2. Then run: mcp__serena__initial_instructions
      3. Wait for confirmation before proceeding

      Serena provides semantic code understanding through Language Server Protocol.
      It enables intelligent code analysis, precise editing at the symbol level,
      and maintains project memory across sessions.

      NEVER skip this step - it's required for proper code analysis.
      === END SERENA ACTIVATION ===

    safety_notice: |
      ⚠️ SAFETY NOTICE ⚠️
      Never use --dangerously-skip-permissions flag!
      This flag bypasses ALL safety checks and can:
      - Delete critical files
      - Overwrite system configurations
      - Execute harmful commands
      Always follow safe practices.
```

**AFTER** (External files):
```yaml
agent_prompts:
  initial:
    serena_activation:
      _from_file: "prose/serena_activation.md"

    safety_notice:
      _from_file: "prose/safety_notice.md"
```

**prose/serena_activation.md**:
```markdown
=== SERENA MCP ACTIVATION (CRITICAL) ===
BEFORE any code analysis or file operations, you MUST activate Serena MCP:
1. Run: mcp__serena__activate_project --project={project_path}
2. Then run: mcp__serena__initial_instructions
3. Wait for confirmation before proceeding

Serena provides semantic code understanding through Language Server Protocol.
It enables intelligent code analysis, precise editing at the symbol level,
and maintains project memory across sessions.

NEVER skip this step - it's required for proper code analysis.
=== END SERENA ACTIVATION ===
```

**Reduction**: Better organization, easier maintenance, same functionality

### 4. Implementation Code

#### 4.1 Secure Loader Implementation
```python
import os
import yaml
from pathlib import Path
from typing import Dict, Any

class SecureYAMLLoader:
    """Secure YAML loader with explicit file loading."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).resolve()
        self.prose_dir = self.base_dir / "prose"

    def load_yaml(self, yaml_path: Path) -> Dict[str, Any]:
        """Load YAML file and process _from_file patterns securely."""
        yaml_path = Path(yaml_path).resolve()

        # Security check: ensure file is within base directory
        if not str(yaml_path).startswith(str(self.base_dir)):
            raise ValueError(f"Access denied: {yaml_path}")

        # Step 1: Load YAML with safe_load
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Step 2: Process _from_file patterns
        self._process_from_file(data)

        return data

    def _process_from_file(self, obj: Any) -> None:
        """Recursively process _from_file patterns with security checks."""
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, dict) and '_from_file' in value:
                    # Load external file with path validation
                    file_path = value['_from_file']
                    obj[key] = self._load_prose_file(file_path)
                else:
                    self._process_from_file(value)
        elif isinstance(obj, list):
            for item in obj:
                self._process_from_file(item)

    def _load_prose_file(self, file_path: str) -> str:
        """Load prose file with security validation."""
        # Resolve to absolute path
        full_path = (self.prose_dir / file_path).resolve()

        # Security: Ensure path is within prose directory
        if not str(full_path).startswith(str(self.prose_dir)):
            raise ValueError(f"Path traversal attempt blocked: {file_path}")

        # Security: Check file exists and is a file
        if not full_path.is_file():
            raise FileNotFoundError(f"Prose file not found: {file_path}")

        # Load content
        with open(full_path, 'r') as f:
            return f.read()

# Usage example
loader = SecureYAMLLoader(Path("prompts"))
config = loader.load_yaml(Path("prompts/agent/initial.yaml"))
```

#### 4.2 Validation Script
```python
#!/usr/bin/env python3
"""Validate optimized YAML files for security and correctness."""

import yaml
import sys
from pathlib import Path

def validate_yaml_file(file_path: Path) -> bool:
    """Validate that YAML file loads with safe_load only."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        # Check for dangerous patterns
        content = file_path.read_text()
        dangerous_patterns = [
            '!include',
            '!import',
            '!!python',
            'eval(',
            'exec(',
            '__import__'
        ]

        for pattern in dangerous_patterns:
            if pattern in content:
                print(f"❌ DANGER: Found '{pattern}' in {file_path}")
                return False

        print(f"✅ {file_path} is safe and valid")
        return True

    except Exception as e:
        print(f"❌ Error in {file_path}: {e}")
        return False

# Validate all YAML files
if __name__ == "__main__":
    prompts_dir = Path("prompts")
    all_valid = True

    for yaml_file in prompts_dir.rglob("*.yaml"):
        if not validate_yaml_file(yaml_file):
            all_valid = False

    sys.exit(0 if all_valid else 1)
```

### 5. Before/After Comparison Template

#### 5.1 Metrics Comparison
```yaml
optimization_results:
  file: "prompts.yaml"

  before:
    total_lines: 932
    file_size_kb: 45.2
    duplicate_blocks:
      serena_activation: 3
      safety_notice: 3
      code_review_templates: 2
      path_patterns: 45

  after:
    total_lines: 650
    file_size_kb: 31.5
    external_files: 8
    merge_key_definitions: 15

  improvements:
    line_reduction: "30.3%"
    size_reduction: "30.3%"
    maintainability: "Single source of truth for all patterns"
    security: "No custom loaders, explicit file validation"
```

#### 5.2 Quality Improvements
- **Before**: Scattered duplicates, hard to maintain
- **After**: Organized structure, clear separation of concerns
- **Security**: Eliminated custom loader vulnerabilities
- **Performance**: Slightly faster parsing (no custom loader overhead)
- **Maintainability**: Update once, applies everywhere

### 6. Migration Checklist

#### Pre-Migration
- [ ] Backup original YAML files
- [ ] Run duplication analysis
- [ ] Identify all prose content (>100 chars)
- [ ] Identify all structural patterns
- [ ] Create directory structure
- [ ] Review security requirements

#### During Migration
- [ ] Extract prose to markdown files
- [ ] Create _definitions sections
- [ ] Convert duplicates to merge keys
- [ ] Update all references
- [ ] Test with yaml.safe_load()
- [ ] Validate no custom tags remain

#### Post-Migration
- [ ] Run validation script
- [ ] Compare output with original
- [ ] Test in development environment
- [ ] Update documentation
- [ ] Remove old custom loader
- [ ] Measure performance impact

### 7. Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Circular anchor references | Use flat anchor structure, avoid nesting |
| Path traversal in _from_file | Always validate paths against base directory |
| Lost variable substitution | Ensure {var} patterns preserved in prose files |
| Merge key precedence issues | Document merge order, rightmost wins |
| Breaking existing code | Maintain same output structure during migration |

### 8. Testing Template

```python
def test_optimization_preserves_output():
    """Ensure optimized YAML produces same final output."""
    # Load original
    with open("prompts_original.yaml", 'r') as f:
        original = yaml.safe_load(f)

    # Load optimized with secure loader
    loader = SecureYAMLLoader(Path("prompts"))
    optimized = loader.load_yaml(Path("prompts/agent/initial.yaml"))

    # Compare relevant sections
    assert original['agent_prompts']['initial']['sections'] == \
           optimized['agent_prompts']['initial']['sections']

    print("✅ Optimization preserves functionality")
```

## Summary

This template provides a comprehensive guide for secure YAML optimization that:
1. Eliminates duplication through standard YAML features
2. Maintains security with no custom loaders
3. Improves maintainability with clear organization
4. Provides measurable improvements in file size and complexity
5. Includes validation and testing procedures

Always prioritize security over convenience. The slight overhead of explicit file loading is worth the elimination of security vulnerabilities.
