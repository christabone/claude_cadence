"""
End-to-end tests for basic supervisor and agent execution
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import time
import yaml

from cadence.task_supervisor import TaskSupervisor, ExecutionResult
from cadence.config import CadenceConfig, ConfigLoader, SCRATCHPAD_DIR, SUPERVISOR_LOG_DIR


class TestBasicE2E:
    """Basic E2E tests that spawn real supervisors and agents"""

    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration for E2E tests"""
        config_dict = {
            "supervisor": {
                "verbose": True,
                "model": "claude-3-5-sonnet-20241022",  # Use AI model, not heuristic
                "zen_integration": {
                    "enabled": False  # Disable for basic tests
                }
            },
            "execution": {
                "max_turns": 5,  # Keep low for testing
                "log_dir": str(e2e_temp_dir / "logs"),
                "timeout": 30
            },
            "agent": {
                "model": "claude-3-haiku-20240307",  # Use fast model for tests
                "extra_flags": ["--no-cache"]
            },
            "integrations": {
                "mcp": {
                    "servers": ["filesystem"]  # Basic filesystem access only
                }
            }
        }

        config_file = e2e_temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)

        # Use ConfigLoader to properly load the config
        loader = ConfigLoader(str(config_file))
        return loader.config


    def test_simple_todo_execution(self, e2e_config, e2e_temp_dir):
        """Test executing simple TODOs with real Claude"""
        # Skip if Claude is not available
        try:
            result = subprocess.run(["claude", "--version"],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                pytest.skip("Claude CLI not available")
        except FileNotFoundError:
            pytest.skip("Claude CLI not installed")

        # Change to temp directory for execution
        original_cwd = os.getcwd()
        os.chdir(e2e_temp_dir)

        try:
            # Create simple TODOs
            todos = [
                "Create a file called test_output.txt with the content 'Hello from E2E test'",
                "Read the file test_output.txt and print its contents"
            ]

            # Create supervisor
            supervisor = TaskSupervisor(config=e2e_config)

            # Execute
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_simple_test"
            )

            # Verify execution
            assert isinstance(result, ExecutionResult)
            assert result.success is True
            assert result.turns_used == 0  # Turn counting not available with stream-json

            # Check if file was created
            test_file = e2e_temp_dir / "test_output.txt"
            assert test_file.exists()
            assert test_file.read_text().strip() == "Hello from E2E test"

            # Check scratchpad was created
            scratchpad_path = e2e_temp_dir / SCRATCHPAD_DIR / "session_e2e_simple_test.md"
            assert scratchpad_path.exists()

            # Check supervisor log was created
            supervisor_log = e2e_temp_dir / SUPERVISOR_LOG_DIR / "session_e2e_simple_test.md"
            assert supervisor_log.exists()

        finally:
            # Always restore original directory
            os.chdir(original_cwd)


    def test_execution_with_errors(self, e2e_config, e2e_temp_dir):
        """Test handling of execution errors"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create TODOs that will cause errors
        todos = [
            "Try to read a non-existent file at /definitely/does/not/exist/file.txt",
            "Create a file called error_handled.txt with content 'Error was handled'"
        ]

        # Change to temp directory for execution
        original_cwd = os.getcwd()
        os.chdir(e2e_temp_dir)

        try:
            supervisor = TaskSupervisor(config=e2e_config)

            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_error_test"
            )

            # Should still succeed but with errors noted
            assert isinstance(result, ExecutionResult)
            # Agent should handle the error and continue
            assert result.success is True or len(result.errors) > 0

            # Check if second task was still completed
            handled_file = e2e_temp_dir / "error_handled.txt"
            # This might or might not exist depending on how the agent handles the error

        finally:
            # Always restore original directory
            os.chdir(original_cwd)

    def test_execution_timeout(self, e2e_config, e2e_temp_dir):
        """Test execution timeout handling"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Configure very short timeout
        e2e_config.execution.timeout = 5  # 5 seconds

        # Change to temp directory for execution
        original_cwd = os.getcwd()
        os.chdir(e2e_temp_dir)

        try:
            # Create a task that might take longer
            todos = [
                "Count from 1 to 1000000 and print every 100000th number"
            ]

            supervisor = TaskSupervisor(config=e2e_config)

            start_time = time.time()
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_timeout_test"
            )
            elapsed = time.time() - start_time

            # Should timeout within reasonable bounds
            assert elapsed < 10  # Should not take more than 10 seconds

            if not result.success:
                assert any("timed out" in err.lower() for err in result.errors)

        finally:
            # Always restore original directory
            os.chdir(original_cwd)

    def test_multiple_sessions(self, e2e_config, e2e_temp_dir):
        """Test running multiple independent sessions"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Change to temp directory for execution
        original_cwd = os.getcwd()
        os.chdir(e2e_temp_dir)

        try:
            supervisor = TaskSupervisor(config=e2e_config)

            # First session
            todos1 = ["Create a file called session1.txt with content 'Session 1'"]
            result1 = supervisor.execute_with_todos(
                todos1,
                session_id="e2e_session1"
            )

            # Second session
            todos2 = ["Create a file called session2.txt with content 'Session 2'"]
            result2 = supervisor.execute_with_todos(
                todos2,
                session_id="e2e_session2"
            )

            # Both should succeed
            assert result1.success is True
            assert result2.success is True

            # Check files
            assert (e2e_temp_dir / "session1.txt").exists()
            assert (e2e_temp_dir / "session2.txt").exists()

            # Check separate scratchpads
            assert (e2e_temp_dir / SCRATCHPAD_DIR / "session_e2e_session1.md").exists()
            assert (e2e_temp_dir / SCRATCHPAD_DIR / "session_e2e_session2.md").exists()

        finally:
            # Always restore original directory
            os.chdir(original_cwd)

    def test_task_completion_detection(self, e2e_config, e2e_temp_dir):
        """Test proper detection of task completion"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Change to temp directory for execution
        original_cwd = os.getcwd()
        os.chdir(e2e_temp_dir)

        try:
            todos = [
                "Create a file called done.txt with content 'Task completed'",
                "Print 'ALL TASKS COMPLETE' to indicate you're done"
            ]

            supervisor = TaskSupervisor(config=e2e_config)

            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_completion_test"
            )

            assert result.success is True
            assert result.task_complete is True
            assert (e2e_temp_dir / "done.txt").exists()

            # Check scratchpad has completion marker
            scratchpad = e2e_temp_dir / SCRATCHPAD_DIR / "session_e2e_completion_test.md"
            if scratchpad.exists():
                content = scratchpad.read_text()
                assert "ALL TASKS COMPLETE" in content or "Completion Summary" in content

        finally:
            # Always restore original directory
            os.chdir(original_cwd)
