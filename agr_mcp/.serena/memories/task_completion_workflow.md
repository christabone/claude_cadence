# Task Completion Workflow

## When a Task is Completed

### 1. Code Quality Checks (Required)
Run these commands in order:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

All checks must pass before considering the task complete.

### 2. Testing (Required)
```bash
# Run relevant tests
pytest tests/tools/test_[module_name].py -v

# Run all tests to ensure no regressions
pytest --cov=agr_mcp tests/

# Ensure coverage is adequate
pytest --cov=agr_mcp --cov-report=term-missing tests/
```

### 3. Integration Testing (When Applicable)
```bash
# Test server startup
python -m agr_mcp.server --help

# Test CLI commands
agr-mcp --help
```

### 4. Pre-commit Validation (Recommended)
```bash
# Run pre-commit hooks
pre-commit run --all-files
```

### 5. Documentation Updates (When Applicable)
- Update README.md if new tools were added
- Update docstrings if APIs changed
- Update configuration documentation if config changed

### 6. Git Operations (When Ready)
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: implement [feature] tool with comprehensive error handling"

# Push to branch
git push origin feature-branch
```

## Critical Requirements
- **ALL** linting/formatting/type checking must pass
- **ALL** existing tests must continue to pass
- **NEW** functionality must have corresponding tests
- Code must follow project conventions
- Error handling must be comprehensive
- Logging must be appropriate

## Warning Signs to Address
- MyPy type errors
- Flake8 lint warnings
- Test failures or decreased coverage
- Missing docstrings or type hints
- Hardcoded values that should be configurable
