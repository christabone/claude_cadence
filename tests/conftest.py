"""
Pytest configuration and fixtures for Claude Cadence tests
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import CadenceConfig, ConfigLoader
from cadence.task_supervisor import TaskSupervisor
from cadence.prompts import TodoPromptManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration for testing"""
    config = CadenceConfig()
    config.execution.log_dir = str(temp_dir / "logs")
    config.execution.max_turns = 10
    config.execution.timeout = 60
    config.agent.model = "test-model"
    config.agent.tools = ["bash", "read", "write"]
    config.supervisor.model = "claude-3-opus-latest"  # Use AI model, not heuristic
    config.supervisor.verbose = False
    config.supervisor.zen_integration.enabled = False
    return config


@pytest.fixture
def mock_task_file(temp_dir):
    """Create a mock task file for testing"""
    task_file = temp_dir / ".taskmaster" / "tasks" / "tasks.json"
    task_file.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "id": "1",
            "title": "Test Task 1",
            "description": "First test task",
            "status": "pending"
        },
        {
            "id": "2",
            "title": "Test Task 2",
            "description": "Second test task",
            "status": "pending"
        }
    ]

    with open(task_file, 'w') as f:
        json.dump({"tasks": tasks}, f)

    return task_file


@pytest.fixture
def mock_claude_response():
    """Mock response from Claude CLI"""
    return {
        "success": True,
        "output": [
            "Creating scratchpad file...",
            "Working on TODO #1...",
            "TODO #1 complete",
            "ALL TASKS COMPLETE"
        ],
        "errors": [],
        "exit_code": 0
    }


@pytest.fixture
def mock_subprocess(mock_claude_response):
    """Mock subprocess for Claude CLI calls"""
    with patch('subprocess.Popen') as mock_popen:
        # Create mock process
        mock_process = MagicMock()
        mock_process.returncode = mock_claude_response["exit_code"]
        mock_process.poll.side_effect = [None] * 3 + [0]  # Running, then done

        # Mock stdout
        mock_stdout = MagicMock()
        output_lines = [line + '\n' for line in mock_claude_response["output"]]
        mock_stdout.readline.side_effect = output_lines + ['']

        # Mock stderr
        mock_stderr = MagicMock()
        error_lines = [line + '\n' for line in mock_claude_response["errors"]]
        mock_stderr.readline.side_effect = error_lines + ['']

        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr

        mock_popen.return_value = mock_process

        yield mock_popen


@pytest.fixture
def supervisor(mock_config):
    """Create a TaskSupervisor instance for testing"""
    return TaskSupervisor(config=mock_config)


@pytest.fixture
def mock_scratchpad(temp_dir):
    """Create a mock scratchpad file"""
    scratchpad_dir = temp_dir / ".cadence" / "scratchpad"
    scratchpad_dir.mkdir(parents=True, exist_ok=True)

    def create_scratchpad(session_id, content):
        scratchpad_file = scratchpad_dir / f"session_{session_id}.md"
        with open(scratchpad_file, 'w') as f:
            f.write(content)
        return scratchpad_file

    return create_scratchpad


@pytest.fixture
def mock_zen_integration():
    """Mock ZenIntegration for testing"""
    with patch('cadence.task_supervisor.ZenIntegration') as mock_zen:
        mock_instance = MagicMock()
        mock_instance.should_call_zen.return_value = None
        mock_zen.return_value = mock_instance
        yield mock_instance


# Environment fixtures
@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment for testing"""
    # Remove any existing Claude-related env vars
    env_vars_to_remove = [
        'CLAUDE_API_KEY',
        'CLAUDE_MODEL',
        'TASKMASTER_PATH'
    ]
    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)

    # Set test environment
    monkeypatch.setenv('CLAUDE_CADENCE_TEST', '1')


# Async fixtures for E2E tests
@pytest.fixture
async def async_supervisor(mock_config):
    """Async supervisor for E2E tests"""
    supervisor = TaskSupervisor(config=mock_config)
    yield supervisor
    # Cleanup if needed


@pytest.fixture
def capture_logs():
    """Capture logs during tests"""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Get all cadence loggers
    loggers = [
        logging.getLogger('cadence'),
        logging.getLogger('cadence.supervisor')
    ]

    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    yield log_capture

    # Cleanup
    for logger in loggers:
        logger.removeHandler(handler)
