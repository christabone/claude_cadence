# '_from_file' Explicit Loading Pattern Documentation

## Purpose
This document establishes secure patterns for explicit file loading that replace custom !include functionality while maintaining security. The '_from_file' pattern enables prose content to be stored in separate files and loaded explicitly during processing, using only standard `yaml.safe_load()` and controlled file operations.

## Security Principles

1. **No Custom YAML Tags**: Use only standard YAML syntax
2. **Explicit Loading**: Content is loaded in a separate step after YAML parsing
3. **Path Validation**: All file paths are validated before loading
4. **No Arbitrary Inclusion**: File loading is controlled and predictable
5. **Clear Boundaries**: Separation between YAML structure and prose content

## The '_from_file' Pattern

### Basic Structure
```yaml
# Instead of custom !include tags:
# content: !include prose/guidelines.md  # ❌ INSECURE

# Use '_from_file' dictionary pattern:
content:
  _from_file: "prose/guidelines.md"      # ✅ SECURE
```

### Pattern Definition
The '_from_file' pattern uses a special dictionary key to indicate that content should be loaded from an external file:

```yaml
element_name:
  _from_file: "relative/path/to/file.md"
```

This pattern is:
- Parseable by standard `yaml.safe_load()`
- Explicitly processed in a separate loading step
- Subject to security validation
- Clear in intent and implementation

## When to Use '_from_file' Pattern

### Criteria for External File Loading

Use '_from_file' pattern when content is:

1. **Long Prose** (>100 characters or 3+ lines)
   ```yaml
   # Good candidate for external file
   work_guidelines:
     _from_file: "prose/work_guidelines.md"
   ```

2. **Human-Readable Documentation**
   ```yaml
   # Instructions, explanations, help text
   serena_activation:
     _from_file: "prose/serena_activation.md"
   ```

3. **Frequently Updated Content**
   ```yaml
   # Content that changes independently of YAML structure
   safety_notice:
     _from_file: "prose/safety_notice.md"
   ```

4. **Markdown-Formatted Text**
   ```yaml
   # Content benefiting from markdown formatting
   code_review_process:
     _from_file: "prose/code_review_process.md"
   ```

### When NOT to Use '_from_file'

Keep content inline when it's:

1. **Short Configuration Values**
   ```yaml
   # Keep inline - too short for external file
   status: "COMPLETE ✅"
   max_retries: 3
   ```

2. **Structural YAML Content**
   ```yaml
   # Use merge keys instead for structural patterns
   <<: *common_paths
   <<: *default_config
   ```

3. **Tightly Coupled Data**
   ```yaml
   # Keep inline when removal would break structure
   api_endpoints:
     health: "/api/health"
     status: "/api/status"
   ```

## Implementation Guide

### Directory Structure
```
prompts/
├── agent/
│   ├── initial.yaml          # YAML with _from_file references
│   └── continuation.yaml
├── supervisor/
│   └── analysis.yaml
└── prose/                    # Prose content directory
    ├── guidelines/
    │   ├── work_guidelines.md
    │   └── safety_guidelines.md
    ├── instructions/
    │   ├── serena_activation.md
    │   └── completion_protocol.md
    └── templates/
        ├── code_review_task.md
        └── code_review_project.md
```

### Naming Conventions

#### File Naming
- Use descriptive, kebab-case names: `work-guidelines.md`
- Group by category in subdirectories
- Match YAML key names when possible
- Include `.md` extension for markdown files

#### Path Conventions
- Always use relative paths from YAML file location
- Use forward slashes even on Windows
- No leading slashes (relative, not absolute)
- Organize by content type

### YAML File Example
```yaml
# agent/initial.yaml
agent_prompts:
  initial:
    # Structural content using merge keys
    <<: *common_variables
    paths: *standard_paths

    # Prose content using _from_file pattern
    sections:
      - name: "serena_activation"
        content:
          _from_file: "prose/instructions/serena_activation.md"

      - name: "work_guidelines"
        content:
          _from_file: "prose/guidelines/work_guidelines.md"

      - name: "safety_notice"
        content:
          _from_file: "prose/guidelines/safety_guidelines.md"

    # Mixed approach for complex structures
    code_review:
      process:
        _from_file: "prose/templates/code_review_task.md"
      variables:
        <<: *review_variables
```

### Prose File Example
```markdown
<!-- prose/instructions/serena_activation.md -->
=== SERENA MCP ACTIVATION (CRITICAL) ===

BEFORE any code analysis or file operations, you MUST activate Serena MCP:

1. Run: `mcp__serena__activate_project --project={project_path}`
2. Then run: `mcp__serena__initial_instructions`
3. Wait for confirmation before proceeding

Serena provides semantic code understanding through Language Server Protocol.
It enables intelligent code analysis, precise editing at the symbol level,
and maintains project memory across sessions.

NEVER skip this step - it's required for proper code analysis.
=== END SERENA ACTIVATION ===
```

## Secure Loading Implementation

### Basic Loader
```python
import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_yaml_with_prose(yaml_path: str, prose_base_dir: str = None) -> Dict[str, Any]:
    """
    Load YAML file and process _from_file patterns securely.

    Args:
        yaml_path: Path to YAML file
        prose_base_dir: Base directory for prose files (default: same as YAML)

    Returns:
        Loaded YAML data with prose content included
    """
    yaml_path = Path(yaml_path).resolve()

    if prose_base_dir is None:
        prose_base_dir = yaml_path.parent
    else:
        prose_base_dir = Path(prose_base_dir).resolve()

    # Step 1: Load YAML with safe_load
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Step 2: Process _from_file patterns
    _process_from_file_patterns(data, prose_base_dir)

    return data

def _process_from_file_patterns(obj: Any, base_dir: Path) -> None:
    """Recursively process _from_file patterns with security validation."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, dict) and '_from_file' in value:
                # Load external file with validation
                file_path = value['_from_file']
                content = _load_prose_file(file_path, base_dir)
                obj[key] = content
            else:
                _process_from_file_patterns(value, base_dir)
    elif isinstance(obj, list):
        for item in obj:
            _process_from_file_patterns(item, base_dir)

def _load_prose_file(file_path: str, base_dir: Path) -> str:
    """Load prose file with comprehensive security validation."""
    # Resolve to absolute path
    full_path = (base_dir / file_path).resolve()

    # Security validations
    if not _is_safe_path(full_path, base_dir):
        raise ValueError(f"Security violation: Path traversal attempt - {file_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"Prose file not found: {file_path}")

    if not full_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    # Additional security checks
    if full_path.suffix not in ['.md', '.txt', '.rst']:
        raise ValueError(f"Unsupported file type: {file_path}")

    # Size limit check (prevent memory exhaustion)
    if full_path.stat().st_size > 1_000_000:  # 1MB limit
        raise ValueError(f"File too large: {file_path}")

    # Load content
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Validate path is within allowed directory."""
    try:
        # Ensure path is within base directory
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False
```

### Advanced Features

#### Caching for Performance
```python
from functools import lru_cache

class CachedProseLoader:
    def __init__(self, base_dir: Path, cache_size: int = 128):
        self.base_dir = Path(base_dir).resolve()
        self._load_cached = lru_cache(maxsize=cache_size)(self._load_prose_file)

    def load_prose(self, file_path: str) -> str:
        """Load prose with caching."""
        return self._load_cached(file_path)

    def _load_prose_file(self, file_path: str) -> str:
        """Internal method for loading (cached by lru_cache)."""
        # Implementation same as _load_prose_file above
        pass
```

#### Variable Substitution Support
```python
def process_variables(content: str, variables: Dict[str, str]) -> str:
    """Replace {variable} patterns in loaded prose."""
    for key, value in variables.items():
        content = content.replace(f"{{{key}}}", str(value))
    return content

# Usage in loader
if isinstance(value, dict) and '_from_file' in value:
    file_path = value['_from_file']
    content = _load_prose_file(file_path, base_dir)

    # Optional: Apply variable substitution
    if '_variables' in value:
        content = process_variables(content, value['_variables'])

    obj[key] = content
```

## Migration Strategy

### Phase 1: Identify Prose Content
1. Review existing YAML files
2. Identify content meeting '_from_file' criteria
3. Document current locations and usage

### Phase 2: Create Prose Files
1. Create `prose/` directory structure
2. Extract identified content to markdown files
3. Preserve formatting and variables

### Phase 3: Update YAML Files
1. Replace inline prose with '_from_file' patterns
2. Verify paths are correct
3. Test loading with secure loader

### Phase 4: Validation
1. Compare output before/after migration
2. Verify security constraints
3. Test error handling

## Security Checklist

- [ ] All file paths are relative, not absolute
- [ ] Path traversal protection is implemented
- [ ] File type validation is enforced
- [ ] File size limits are checked
- [ ] Base directory constraints are validated
- [ ] No arbitrary code execution is possible
- [ ] Error messages don't expose system paths
- [ ] Caching doesn't bypass security checks

## Common Patterns and Examples

### Pattern 1: Simple Prose Loading
```yaml
# Before
guidelines: |
  This is a long block of guidelines text
  that spans multiple lines and contains
  detailed instructions for the system...

# After
guidelines:
  _from_file: "prose/guidelines.md"
```

### Pattern 2: Conditional Loading
```yaml
# Support optional prose loading
help_text:
  _from_file: "prose/help/general.md"
  _fallback: "Default help text if file not found"
```

### Pattern 3: Templated Prose
```yaml
# Load prose with variable substitution
error_message:
  _from_file: "prose/errors/timeout.md"
  _variables:
    timeout: "{timeout_seconds}"
    retry_count: "{max_retries}"
```

### Pattern 4: Nested Structure
```yaml
# Complex nested loading
documentation:
  overview:
    _from_file: "prose/docs/overview.md"
  sections:
    - title: "Getting Started"
      content:
        _from_file: "prose/docs/getting-started.md"
    - title: "Advanced Usage"
      content:
        _from_file: "prose/docs/advanced.md"
```

## Testing and Validation

### Unit Test Example
```python
def test_from_file_loading():
    """Test _from_file pattern processing."""
    # Create test YAML
    yaml_content = """
    content:
      prose:
        _from_file: "test_prose.md"
    """

    # Create test prose file
    with open("test_prose.md", "w") as f:
        f.write("Test prose content")

    # Load and verify
    data = load_yaml_with_prose("test.yaml")
    assert data['content']['prose'] == "Test prose content"

def test_path_traversal_blocked():
    """Test security validation blocks path traversal."""
    yaml_content = """
    content:
      _from_file: "../../etc/passwd"
    """

    with pytest.raises(ValueError, match="Security violation"):
        load_yaml_with_prose("test.yaml")
```

## Summary

The '_from_file' pattern provides a secure, explicit alternative to custom YAML loaders while maintaining the benefits of external file organization. By following these patterns and implementing proper security validation, you can achieve:

1. **Security**: No arbitrary file inclusion or code execution
2. **Maintainability**: Clear separation of structure and content
3. **Flexibility**: Easy updates to prose without touching YAML
4. **Performance**: Optional caching for repeated loads
5. **Compatibility**: Works with standard YAML parsers

Always prioritize security over convenience. The explicit nature of this pattern makes the system more predictable and auditable while eliminating entire classes of security vulnerabilities.
