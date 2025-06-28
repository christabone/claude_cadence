# Secure Optimization Migration Strategy

## Executive Summary
This document provides a comprehensive, phased approach for migrating from the current monolithic prompts.yaml system to a secure, optimized structure using only `yaml.safe_load()` and explicit file operations. The strategy eliminates all custom loader vulnerabilities while achieving significant duplication reduction.

## Migration Goals

1. **Security**: Eliminate all custom YAML loader vulnerabilities
2. **Maintainability**: Reduce duplication by 30%+ through secure patterns
3. **Compatibility**: Ensure all patterns work with standard YAML parsers
4. **Safety**: Maintain ability to rollback at any phase
5. **Validation**: Verify identical functionality after migration

## Pre-Migration Assessment

### Current State Analysis
- **File**: `prompts.yaml` (932 lines)
- **Security Issues**: Custom !include loader with path traversal vulnerability
- **Duplication**: 45+ repeated paths, 30+ repeated variables, 8 major prose blocks
- **Maintenance Pain**: Single large file, scattered duplicates, hard to navigate

### Target State
- **Structure**: Organized directory structure with separate YAML files
- **Security**: Only `yaml.safe_load()` with explicit file operations
- **Optimization**: 30%+ reduction through merge keys and external prose
- **Maintainability**: Clear separation of concerns, single source of truth

## Phase 1: Preparation and Backup (Days 1-2)

### 1.1 Create Full Backup
```bash
# Create timestamped backup
cp prompts.yaml prompts.yaml.backup.$(date +%Y%m%d_%H%M%S)

# Create git branch for migration
git checkout -b feature/secure-yaml-migration
git add prompts.yaml.backup.*
git commit -m "backup: Original prompts.yaml before secure migration"
```

### 1.2 Set Up Directory Structure
```bash
# Create new directory structure
mkdir -p cadence/prompts/{agent,supervisor,prose/{guidelines,instructions,templates}}

# Create initial structure documentation
cat > cadence/prompts/README.md << 'EOF'
# Secure YAML Prompt Structure

## Directory Layout
- `agent/` - Agent-specific prompts
- `supervisor/` - Supervisor prompts
- `prose/` - External prose content
  - `guidelines/` - Work guidelines and rules
  - `instructions/` - Setup and activation instructions
  - `templates/` - Code review and other templates

## Security
All YAML files use only standard yaml.safe_load()
Prose content loaded explicitly with path validation
EOF
```

### 1.3 Inventory and Categorization
Using the analysis from previous subtasks:
- Document all content to be migrated
- Categorize as structural (merge keys) or prose (_from_file)
- Create migration checklist

## Phase 2: Implement Secure Loader (Days 3-4)

### 2.1 Create SecureYAMLLoader Class
```python
# cadence/secure_yaml_loader.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SecureYAMLLoader:
    """Secure YAML loader with explicit file loading and no custom tags."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).resolve()
        self.prose_dir = self.base_dir / "prose"
        self._cache = {}

    def load_yaml(self, yaml_path: Path) -> Dict[str, Any]:
        """Load YAML file and process _from_file patterns securely."""
        yaml_path = Path(yaml_path).resolve()

        # Security check
        if not self._is_safe_path(yaml_path, self.base_dir):
            raise ValueError(f"Security violation: {yaml_path}")

        # Load with safe_load only
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Process _from_file patterns
        self._process_from_file(data)

        return data

    def _is_safe_path(self, path: Path, base: Path) -> bool:
        """Validate path is within base directory."""
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    def _process_from_file(self, obj: Any) -> None:
        """Process _from_file patterns with security validation."""
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, dict) and '_from_file' in value:
                    file_path = value['_from_file']
                    obj[key] = self._load_prose_file(file_path)
                else:
                    self._process_from_file(value)
        elif isinstance(obj, list):
            for item in obj:
                self._process_from_file(item)

    def _load_prose_file(self, file_path: str) -> str:
        """Load prose file with comprehensive security checks."""
        # Cache check
        if file_path in self._cache:
            return self._cache[file_path]

        # Security validation
        full_path = (self.prose_dir / file_path).resolve()

        if not self._is_safe_path(full_path, self.prose_dir):
            raise ValueError(f"Path traversal blocked: {file_path}")

        if not full_path.exists():
            raise FileNotFoundError(f"Prose file not found: {file_path}")

        if full_path.suffix not in ['.md', '.txt']:
            raise ValueError(f"Invalid file type: {file_path}")

        if full_path.stat().st_size > 1_000_000:  # 1MB limit
            raise ValueError(f"File too large: {file_path}")

        # Load and cache
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self._cache[file_path] = content
            return content
```

### 2.2 Create Migration Utilities
```python
# cadence/migration_utils.py
import re
from typing import Dict, List, Tuple

def identify_prose_blocks(yaml_content: str) -> List[Tuple[str, int, int]]:
    """Identify prose blocks suitable for extraction."""
    prose_blocks = []
    lines = yaml_content.split('\n')

    in_block = False
    block_start = 0
    block_key = ""

    for i, line in enumerate(lines):
        if line.strip().endswith('|') or line.strip().endswith('>'):
            in_block = True
            block_start = i
            block_key = line.split(':')[0].strip()
        elif in_block and not line.startswith(' '):
            # Block ended
            if i - block_start > 3:  # More than 3 lines
                prose_blocks.append((block_key, block_start, i))
            in_block = False

    return prose_blocks

def extract_common_patterns(yaml_content: str) -> Dict[str, List[str]]:
    """Extract common patterns for merge keys."""
    patterns = {
        'paths': [],
        'variables': [],
        'templates': []
    }

    # Find repeated path patterns
    path_pattern = r'{project_path}/[^\s}]+'
    patterns['paths'] = list(set(re.findall(path_pattern, yaml_content)))

    # Find repeated variables
    var_pattern = r'{\w+}'
    patterns['variables'] = list(set(re.findall(var_pattern, yaml_content)))

    return patterns
```

### 2.3 Test Infrastructure
```python
# tests/test_secure_migration.py
import pytest
import yaml
from pathlib import Path

def test_secure_loader_blocks_traversal():
    """Ensure path traversal is blocked."""
    loader = SecureYAMLLoader(Path("."))

    malicious_yaml = """
    content:
      _from_file: "../../etc/passwd"
    """

    with pytest.raises(ValueError, match="Path traversal blocked"):
        data = yaml.safe_load(malicious_yaml)
        loader._process_from_file(data)

def test_merge_keys_work():
    """Verify merge keys function correctly."""
    yaml_content = """
    _definitions:
      paths: &paths
        root: "/project"
        logs: "/project/logs"

    config:
      <<: *paths
      custom: "value"
    """

    data = yaml.safe_load(yaml_content)
    assert data['config']['root'] == "/project"
    assert data['config']['custom'] == "value"
```

## Phase 3: Extract Prose Content (Days 5-6)

### 3.1 Identify and Extract Prose
Based on analysis, extract these prose blocks:

```bash
# Extract major prose blocks
cat > cadence/prompts/prose/instructions/serena_activation.md << 'EOF'
=== SERENA MCP ACTIVATION (CRITICAL) ===
BEFORE any code analysis or file operations, you MUST activate Serena MCP:
1. Run: mcp__serena__activate_project --project={project_path}
2. Then run: mcp__serena__initial_instructions
3. Wait for confirmation before proceeding
[... rest of content ...]
EOF

# Continue for all 8 identified prose blocks
```

### 3.2 Verify Prose Extraction
```python
# verify_prose.py
def verify_prose_extraction():
    """Ensure all prose was extracted correctly."""
    original = load_original_prompts()

    for prose_file in Path("prose").rglob("*.md"):
        content = prose_file.read_text()
        # Verify content exists in original
        assert content in original, f"Content mismatch in {prose_file}"
```

## Phase 4: Create Optimized YAML Files (Days 7-8)

### 4.1 Create YAML with Merge Keys
```yaml
# agent/initial.yaml
_definitions:
  # Path definitions
  paths:
    cadence: &paths_cadence
      root: "{project_path}/.cadence"
      scratchpad: "{project_path}/.cadence/scratchpad"
      agent: "{project_path}/.cadence/agent"

    output: &paths_output
      logs: "{project_path}/.cadence/logs"
      snapshots: "{project_path}/.cadence/snapshots"

  # Variable collections
  variables:
    project: &vars_project
      project_path: "{project_path}"
      session_id: "{session_id}"
      max_turns: "{max_turns}"

  # Status templates
  status:
    messages: &status_messages
      complete: "Status: COMPLETE ✅"
      in_progress: "Status: IN_PROGRESS 🔄"
      stuck: "Status: STUCK ⚠️"

# Main content
agent_prompts:
  initial:
    # Merge structural content
    <<: *vars_project
    paths:
      <<: *paths_cadence
      <<: *paths_output

    # Load prose externally
    sections:
      - name: "serena_activation"
        content:
          _from_file: "instructions/serena_activation.md"

      - name: "work_guidelines"
        content:
          _from_file: "guidelines/work_guidelines.md"

    # Use status templates
    status_formats: *status_messages
```

### 4.2 Migration Script
```python
# migrate_prompts.py
def migrate_yaml_file(source_path: Path, target_dir: Path):
    """Migrate a YAML file to secure patterns."""

    # Load original
    with open(source_path, 'r') as f:
        original = yaml.safe_load(f)

    # Extract prose blocks
    prose_blocks = identify_prose_blocks(source_path.read_text())

    # Create merge key definitions
    definitions = create_merge_definitions(original)

    # Build new structure
    new_yaml = {
        '_definitions': definitions,
        'agent_prompts': migrate_prompts(original['agent_prompts'])
    }

    # Write new file
    output_path = target_dir / "agent/initial.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(new_yaml, f, default_flow_style=False)
```

## Phase 5: Integration and Testing (Days 9-10)

### 5.1 Update Import Statements
```python
# In orchestrator.py
# OLD:
# from .prompts import YAMLPromptLoader

# NEW:
from .secure_yaml_loader import SecureYAMLLoader as PromptLoader
```

### 5.2 Compatibility Layer
```python
# cadence/prompt_loader_compat.py
class PromptLoaderCompat:
    """Compatibility layer for transition period."""

    def __init__(self, base_dir: Path):
        self.secure_loader = SecureYAMLLoader(base_dir)
        self.legacy_path = base_dir / "prompts.yaml"
        self.use_legacy = False

    def load_prompt(self, prompt_key: str) -> str:
        """Load prompt with fallback to legacy."""
        try:
            # Try new system first
            return self._load_from_secure(prompt_key)
        except Exception as e:
            logger.warning(f"Falling back to legacy: {e}")
            self.use_legacy = True
            return self._load_from_legacy(prompt_key)
```

### 5.3 A/B Testing
```python
# test_migration_parity.py
def test_output_parity():
    """Ensure old and new systems produce identical output."""
    legacy_loader = YAMLPromptLoader("prompts.yaml")
    secure_loader = SecureYAMLLoader(Path("prompts"))

    test_cases = [
        ("agent.initial", {"project_path": "/test", "session_id": "123"}),
        ("agent.continuation", {"previous_session_id": "456"}),
        # ... all prompt types
    ]

    for prompt_key, variables in test_cases:
        legacy_output = legacy_loader.format_prompt(prompt_key, variables)
        secure_output = secure_loader.format_prompt(prompt_key, variables)

        assert legacy_output == secure_output, f"Mismatch in {prompt_key}"
```

## Phase 6: Validation and Cutover (Days 11-12)

### 6.1 Security Validation Checklist
- [ ] No custom YAML tags in any file
- [ ] All files load with `yaml.safe_load()` only
- [ ] Path traversal protection tested and working
- [ ] File type validation enforced
- [ ] Size limits implemented
- [ ] No hardcoded absolute paths

### 6.2 Performance Validation
```python
# benchmark_performance.py
def benchmark_loading():
    """Compare performance between old and new systems."""
    import time

    # Legacy system
    start = time.time()
    for _ in range(100):
        legacy_loader.load_prompt("agent.initial")
    legacy_time = time.time() - start

    # Secure system
    start = time.time()
    for _ in range(100):
        secure_loader.load_prompt("agent.initial")
    secure_time = time.time() - start

    print(f"Legacy: {legacy_time:.2f}s")
    print(f"Secure: {secure_time:.2f}s")
    assert secure_time < legacy_time * 1.5  # Allow 50% overhead max
```

### 6.3 Metrics Validation
```yaml
# metrics_report.yaml
migration_results:
  security:
    custom_loaders_removed: true
    path_traversal_protected: true
    arbitrary_inclusion_prevented: true

  optimization:
    original_lines: 932
    new_total_lines: 650
    reduction_percentage: 30.3%
    prose_externalized: 8
    merge_keys_created: 15

  compatibility:
    yaml_safe_load_only: true
    standard_yaml_syntax: true
    cross_platform_tested: true
```

## Phase 7: Deployment (Day 13)

### 7.1 Final Cutover Steps
```bash
# 1. Run final validation suite
pytest tests/test_secure_migration.py -v

# 2. Create deployment backup
tar -czf prompts_backup_$(date +%Y%m%d).tar.gz prompts.yaml prompts/

# 3. Update configuration
echo "PROMPT_SYSTEM=secure" >> .env

# 4. Deploy new system
mv prompts.yaml prompts.yaml.legacy
ln -s prompts/agent/initial.yaml prompts.yaml  # Temporary compatibility

# 5. Monitor for issues
tail -f logs/prompt_loader.log
```

### 7.2 Rollback Procedure
```bash
# If issues arise:
# 1. Restore legacy system
mv prompts.yaml.legacy prompts.yaml
rm prompts.yaml  # Remove symlink

# 2. Update configuration
sed -i 's/PROMPT_SYSTEM=secure/PROMPT_SYSTEM=legacy/' .env

# 3. Restart services
systemctl restart cadence-orchestrator

# 4. Investigate issues
grep ERROR logs/prompt_loader.log
```

## Phase 8: Cleanup and Documentation (Day 14)

### 8.1 Remove Legacy Code
After 1 week of stable operation:
```python
# Remove custom loader
rm cadence/prompt_loader.py

# Remove legacy prompts
mkdir -p archive/$(date +%Y%m%d)
mv prompts.yaml.legacy archive/$(date +%Y%m%d)/

# Update imports
find . -name "*.py" -exec sed -i 's/YAMLPromptLoader/SecureYAMLLoader/g' {} \;
```

### 8.2 Update Documentation
- Update README with new structure
- Document security improvements
- Create maintenance guide
- Update developer onboarding

## Risk Mitigation

### Identified Risks and Mitigations

1. **Risk**: Output differences between systems
   - **Mitigation**: Comprehensive A/B testing, byte-level comparison

2. **Risk**: Performance degradation
   - **Mitigation**: Caching layer, performance benchmarks

3. **Risk**: Missing content during migration
   - **Mitigation**: Automated inventory validation, checksums

4. **Risk**: Security vulnerabilities in new code
   - **Mitigation**: Security review, penetration testing

## Success Criteria

The migration is considered successful when:

1. ✅ All tests pass with 100% parity
2. ✅ Security scan shows no vulnerabilities
3. ✅ Performance within 10% of original
4. ✅ 30%+ reduction in duplication achieved
5. ✅ No custom YAML loaders remain
6. ✅ 1 week stable operation in production

## Timeline Summary

- **Week 1**: Preparation, loader implementation, prose extraction
- **Week 2**: YAML optimization, testing, validation, deployment
- **Week 3**: Monitoring, cleanup, documentation

Total estimated effort: 14 days with 1 engineer

## Conclusion

This migration strategy provides a secure, phased approach to eliminating custom YAML loader vulnerabilities while achieving significant optimization benefits. The emphasis on validation and rollback capability ensures a safe transition with minimal risk to production systems.
