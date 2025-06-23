# Claude Cadence Test Suite

Comprehensive test suite for Claude Cadence, including unit tests and end-to-end (E2E) tests that spawn real supervisors and agents.

## Setup

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Ensure Claude CLI is installed and configured:
```bash
claude --version
```

3. Ensure required MCP servers are available:
```bash
mcp list
```

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run only unit tests:
```bash
pytest tests/unit/
```

### Run only E2E tests:
```bash
pytest tests/e2e/
```

### Run with coverage:
```bash
pytest tests/ --cov=cadence --cov-report=html
```

### Run in parallel:
```bash
pytest tests/ -n auto
```

### Run with verbose output:
```bash
pytest tests/ -v
```

## Test Structure

### Unit Tests (`tests/unit/`)
- `test_task_supervisor.py` - Tests for TaskSupervisor core functionality
- `test_zen_integration.py` - Tests for ZenIntegration logic
- `test_prompts.py` - Tests for prompt generation system

### E2E Tests (`tests/e2e/`)
- `test_basic_execution.py` - Basic supervisor/agent execution scenarios
- `test_continuation.py` - Execution continuation and recovery
- `test_zen_integration_e2e.py` - Zen assistance with real execution
- `test_taskmaster_integration_e2e.py` - TaskMaster integration
- `test_scratchpad_monitoring_e2e.py` - Scratchpad-based communication
- `test_prompt_system_e2e.py` - Prompt system and agent guidance
- `test_error_handling_e2e.py` - Error scenarios and recovery
- `test_supervisor_features_e2e.py` - Supervisor-specific features

## Important Notes

### E2E Test Requirements
- E2E tests require Claude CLI to be installed and accessible
- Tests will skip if Claude is not available
- Some tests require specific MCP servers (filesystem, zen, serena, taskmaster-ai)
- Tests use `claude-3-haiku-20240307` model by default for speed

### Test Isolation
- Each test creates its own temporary directory
- Scratchpad and log files are created in proper directories
- All test artifacts are cleaned up after execution

### Mocking
- E2E tests mock Zen tool calls to avoid actual MCP calls
- Network operations are mocked or use invalid domains
- Dangerous operations are simulated safely

### Test Data
- Tests create their own test files and directories
- TaskMaster tests create sample task JSON files
- No production data is used or modified

## Debugging Failed Tests

### Check logs:
```bash
# Supervisor logs
ls .cadence/supervisor/

# Agent scratchpads
ls .cadence/scratchpad/

# Execution logs
ls .cadence/logs/
```

### Run specific test:
```bash
pytest tests/e2e/test_basic_execution.py::TestBasicE2E::test_simple_todo_execution -v
```

### Run with debugging:
```bash
pytest tests/ -v --pdb
```

## Writing New Tests

### Unit Test Template:
```python
class TestNewFeature:
    def test_feature_behavior(self, mock_config):
        # Arrange
        supervisor = TaskSupervisor(config=mock_config)

        # Act
        result = supervisor.some_method()

        # Assert
        assert result == expected_value
```

### E2E Test Template:
```python
def test_new_e2e_scenario(self, e2e_config, e2e_temp_dir):
    # Skip if Claude not available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("Claude CLI not available")

    # Create TODOs
    todos = ["Task 1", "Task 2"]

    # Execute
    supervisor = TaskSupervisor(config=e2e_config)
    result = supervisor.execute_with_todos(todos, session_id="test_id")

    # Assert
    assert result.success is True
```

## CI/CD Integration

Tests are designed to run in CI/CD pipelines. Set environment variables:
- `CLAUDE_API_KEY` - For Claude CLI authentication
- `MCP_SERVERS` - List of available MCP servers

Skip E2E tests in CI if needed:
```bash
pytest tests/unit/  # Only run unit tests
```
