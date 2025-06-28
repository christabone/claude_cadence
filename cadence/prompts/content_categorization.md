# Content Categorization: Prose vs Structural Patterns

## Analysis Date: 2025-01-26

## Overview
This document categorizes the patterns identified in the common_content_analysis.md into two secure loading strategies:
1. **Prose Content** - Long text blocks suitable for '_from_file' explicit loading pattern
2. **Structural YAML Content** - Configuration and templates suitable for YAML merge keys

The categorization is based on security-first principles using only `yaml.safe_load()` and explicit file operations.

## Security Criteria for Categorization

### Prose Content Criteria (Use '_from_file' Pattern)
- Text blocks longer than 100 characters or 3+ lines
- Human-readable instructions or documentation
- Content that rarely changes structurally
- Content that benefits from markdown formatting
- Content that may be edited by non-developers

### Structural YAML Criteria (Use Merge Keys)
- Configuration data and variables
- Short templates with placeholders
- Frequently referenced structural patterns
- Content that requires YAML parsing for variable substitution
- Content that is tightly coupled with YAML structure

## 1. Prose Content for '_from_file' Pattern

### 1.1 Serena MCP Activation Instructions
- **Current Location**: Lines 5-18, 3+ occurrences
- **Size**: ~14 lines of detailed instructions
- **Recommendation**: Extract to `prose/serena_activation.md`
- **Usage Pattern**:
  ```yaml
  serena_activation:
    _from_file: "prose/serena_activation.md"
  ```

### 1.2 Safety Notice Warning
- **Current Location**: Lines 159-169, 3 identical copies
- **Size**: ~11 lines of warning text
- **Recommendation**: Extract to `prose/safety_notice.md`
- **Usage Pattern**:
  ```yaml
  safety_notice:
    _from_file: "prose/safety_notice.md"
  ```

### 1.3 Work Guidelines
- **Current Location**: Lines 44-157
- **Size**: ~113 lines of detailed guidelines
- **Recommendation**: Extract to `prose/work_guidelines.md`
- **Usage Pattern**:
  ```yaml
  guidelines:
    _from_file: "prose/work_guidelines.md"
  ```

### 1.4 Completion Protocol
- **Current Location**: Lines 151-157
- **Size**: ~7 lines of instructions
- **Recommendation**: Extract to `prose/completion_protocol.md`
- **Usage Pattern**:
  ```yaml
  completion_protocol:
    _from_file: "prose/completion_protocol.md"
  ```

### 1.5 Code Review Instructions (Task Level)
- **Current Location**: Lines 565-612
- **Size**: ~47 lines of review process
- **Recommendation**: Extract to `prose/code_review_task.md`
- **Usage Pattern**:
  ```yaml
  code_review_task:
    _from_file: "prose/code_review_task.md"
  ```

### 1.6 Code Review Instructions (Project Level)
- **Current Location**: Lines 614-667
- **Size**: ~53 lines of review process
- **Recommendation**: Extract to `prose/code_review_project.md`
- **Usage Pattern**:
  ```yaml
  code_review_project:
    _from_file: "prose/code_review_project.md"
  ```

### 1.7 Supervision Explanation
- **Current Location**: Referenced multiple times
- **Size**: Multi-paragraph explanation
- **Recommendation**: Extract to `prose/supervision_explanation.md`
- **Usage Pattern**:
  ```yaml
  supervision_explanation:
    _from_file: "prose/supervision_explanation.md"
  ```

### 1.8 Help Request Templates
- **Current Location**: Various help templates
- **Size**: 5-10 lines each
- **Recommendation**: Extract to `prose/help_templates.md`
- **Usage Pattern**:
  ```yaml
  help_request:
    _from_file: "prose/help_templates.md"
  ```

## 2. Structural YAML Content for Merge Keys

### 2.1 Path Definitions
- **Pattern**: Common directory paths
- **Recommendation**: Use merge keys at file top
- **Example**:
  ```yaml
  _definitions:
    paths: &paths
      cadence_dir: ".cadence"
      scratchpad_dir: "{project_path}/.cadence/scratchpad"
      agent_dir: "{project_path}/.cadence/agent"
      supervisor_dir: "{project_path}/.cadence/supervisor"
      logs_dir: "{project_path}/.cadence/logs"
  ```

### 2.2 Variable Collections
- **Pattern**: Commonly used variable groups
- **Recommendation**: Use merge keys for variable sets
- **Example**:
  ```yaml
  _definitions:
    project_vars: &project_vars
      project_path: "{project_path}"
      session_id: "{session_id}"
      max_turns: "{max_turns}"

    status_vars: &status_vars
      completed_count: "{completed_count}"
      remaining_count: "{remaining_count}"
      turns_used: "{turns_used}"
  ```

### 2.3 Status Message Templates
- **Pattern**: Short status formats
- **Recommendation**: Use merge keys for templates
- **Example**:
  ```yaml
  _definitions:
    status_templates: &status_templates
      complete: "Status: COMPLETE ✅"
      in_progress: "Status: IN_PROGRESS"
      stuck: "Status: STUCK"
  ```

### 2.4 Section Headers
- **Pattern**: Consistent formatting patterns
- **Recommendation**: Use merge keys for formatting
- **Example**:
  ```yaml
  _definitions:
    headers: &headers
      context: "=== SUPERVISED AGENT CONTEXT ==="
      todos: "=== YOUR TODOS ==="
      completion: "=== COMPLETION PROTOCOL ==="
  ```

### 2.5 Scratchpad Structure
- **Pattern**: File structure templates
- **Recommendation**: Use merge keys for structure
- **Example**:
  ```yaml
  _definitions:
    scratchpad_template: &scratchpad_template
      header: "# Agent Scratchpad - Session {session_id}"
      sections:
        - "## Current Task"
        - "## Analysis"
        - "## Progress"
      file_pattern: "session_{session_id}.md"
  ```

### 2.6 TODO List Formatting
- **Pattern**: Consistent list formatting
- **Recommendation**: Use merge keys for format
- **Example**:
  ```yaml
  _definitions:
    todo_format: &todo_format
      item_template: "{number}. {todo_text}"
      header: "TODO List:"
      empty_message: "No TODOs remaining"
  ```

## 3. Migration Strategy Summary

### Phase 1: Extract Prose Content
1. Create `prose/` directory structure
2. Extract all content marked for '_from_file' pattern
3. Preserve exact formatting and content
4. Add markdown headers for context

### Phase 2: Implement Merge Keys
1. Add `_definitions:` section to each YAML file
2. Define local anchors for structural content
3. Replace duplicated content with merge key references
4. Test with `yaml.safe_load()` only

### Phase 3: Update Loading Logic
1. Implement secure '_from_file' loader utility
2. Add path traversal protection
3. Remove custom !include loader completely
4. Test all patterns with standard YAML parsing

## 4. Security Benefits

### Eliminated Risks:
- No custom YAML tags or loaders
- No arbitrary file inclusion
- No path traversal vulnerabilities
- No thread safety concerns with custom loaders

### Security Guarantees:
- All YAML parsing uses `yaml.safe_load()` only
- File loading is explicit and controlled
- Path validation prevents directory traversal
- Standard YAML features ensure compatibility

## 5. Example Implementation

### YAML File Structure:
```yaml
# agents/initial.yaml
_definitions:
  # Local structural patterns using merge keys
  paths: &paths
    scratchpad: "{project_path}/.cadence/scratchpad/session_{session_id}.md"
    output: "{project_path}/.cadence/agent/output_{session_id}.log"

  vars: &vars
    project_path: "{project_path}"
    session_id: "{session_id}"
    max_turns: "{max_turns}"

# Main content using both patterns
agent_prompts:
  initial:
    # Merge structural content
    <<: *vars
    paths: *paths

    # Load prose content explicitly
    serena_activation:
      _from_file: "prose/serena_activation.md"

    guidelines:
      _from_file: "prose/work_guidelines.md"

    safety_notice:
      _from_file: "prose/safety_notice.md"
```

### Secure Loader Implementation:
```python
import os
import yaml

def load_yaml_with_prose(yaml_path, prose_base_dir):
    """Load YAML and replace _from_file patterns with prose content."""

    # Step 1: Safe YAML loading
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Step 2: Process _from_file patterns
    def process_from_file(obj, base_dir):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, dict) and '_from_file' in value:
                    # Secure path resolution
                    file_path = value['_from_file']
                    safe_path = os.path.abspath(os.path.join(base_dir, file_path))

                    # Path traversal protection
                    if not safe_path.startswith(os.path.abspath(base_dir)):
                        raise ValueError(f"Path traversal attempt: {file_path}")

                    # Load prose content
                    with open(safe_path, 'r') as f:
                        obj[key] = f.read()
                else:
                    process_from_file(value, base_dir)
        elif isinstance(obj, list):
            for item in obj:
                process_from_file(item, base_dir)

    process_from_file(data, prose_base_dir)
    return data
```

## 6. Validation Checklist

### Before Migration:
- [ ] All prose content identified for extraction
- [ ] All structural patterns identified for merge keys
- [ ] Security review completed
- [ ] Backup of original files created

### After Migration:
- [ ] All patterns load with `yaml.safe_load()`
- [ ] No custom loaders required
- [ ] Path traversal protection verified
- [ ] Content integrity maintained
- [ ] Performance acceptable
