"""
End-to-end tests for error handling and recovery scenarios
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import subprocess
import signal
import time
import json

from cadence.task_supervisor import TaskSupervisor, ExecutionResult
from cadence.config import CadenceConfig, SCRATCHPAD_DIR, SUPERVISOR_LOG_DIR


class TestErrorHandlingE2E:
    """E2E tests for various error scenarios and recovery"""

    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_error_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration"""
        config = CadenceConfig()
        config.execution.max_turns = 10
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.execution.timeout = 30
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.zen_integration.enabled = True
        config.supervisor.zen_integration.auto_debug_threshold = 3
        return config

    def test_file_permission_errors(self, e2e_config, e2e_temp_dir):
        """Test handling of file permission errors"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Try to create a file at /root/test.txt (this will fail)",
            "Document the error in error_log.txt",
            "Create a file at ./fallback.txt instead"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_permission")

        # Should complete despite permission error
        assert result.success is True

        # Check error was documented
        error_log = e2e_temp_dir / "error_log.txt"
        if error_log.exists():
            content = error_log.read_text()
            assert "permission" in content.lower() or "error" in content.lower()

        # Check fallback worked
        assert (e2e_temp_dir / "fallback.txt").exists()

    def test_network_error_handling(self, e2e_config, e2e_temp_dir):
        """Test handling of network-related errors"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Try to download a file from http://invalid.domain.that.does.not.exist/file.txt",
            "If the download fails, create network_error.txt documenting what happened",
            "Create success.txt to show task completion"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_network")

        # Should handle network error gracefully
        assert result.success is True

        # Check error handling
        assert (e2e_temp_dir / "network_error.txt").exists() or \
               (e2e_temp_dir / "success.txt").exists()

    def test_syntax_error_recovery(self, e2e_config, e2e_temp_dir):
        """Test recovery from code syntax errors"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create a file with syntax error
        bad_code = e2e_temp_dir / "broken.py"
        bad_code.write_text("""
def broken_function(
    print("This has a syntax error"
    return None
""")

        todos = [
            "Try to run or analyze broken.py",
            "Fix the syntax error in broken.py",
            "Create fixed_confirmation.txt when the file is corrected"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_syntax")

        # Check if syntax was fixed
        if result.success:
            fixed_content = bad_code.read_text()
            # Should have fixed parentheses
            assert fixed_content.count("(") == fixed_content.count(")")

    def test_execution_interrupt_handling(self, e2e_config, e2e_temp_dir):
        """Test handling of execution interrupts"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # This test simulates an interrupt but can't actually send SIGINT to Claude
        todos = [
            "Create checkpoint1.txt with 'Checkpoint 1 reached'",
            "Create checkpoint2.txt with 'Checkpoint 2 reached'",
            "Create checkpoint3.txt with 'Checkpoint 3 reached'"
        ]

        supervisor = TaskSupervisor(config=e2e_config)

        # Set very short max turns to simulate early termination
        supervisor.max_turns = 2

        result = supervisor.execute_with_todos(todos, session_id="e2e_interrupt")

        # Should not complete all tasks
        assert result.task_complete is False

        # But should have created some files
        checkpoints_created = sum(1 for i in range(1, 4)
                                 if (e2e_temp_dir / f"checkpoint{i}.txt").exists())
        assert checkpoints_created > 0
        assert checkpoints_created < 3

    def test_memory_intensive_task_handling(self, e2e_config, e2e_temp_dir):
        """Test handling of memory-intensive operations"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create a large list in memory (but don't try to store millions of items)",
            "Write a sample of the data to memory_test.txt",
            "Clean up and create completion.txt"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_memory")

        assert result.success is True

        # Should complete without memory issues
        assert (e2e_temp_dir / "memory_test.txt").exists() or \
               (e2e_temp_dir / "completion.txt").exists()

    def test_invalid_json_handling(self, e2e_config, e2e_temp_dir):
        """Test handling of invalid JSON data"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create invalid JSON file
        bad_json = e2e_temp_dir / "invalid.json"
        bad_json.write_text('{"key": "value", "broken": }')

        todos = [
            "Try to read and parse invalid.json",
            "If it fails, fix the JSON syntax",
            "Create parsed_data.txt with the fixed JSON content"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_json")

        # Should handle JSON error
        if result.success:
            # Check if JSON was fixed
            if bad_json.exists():
                try:
                    json.loads(bad_json.read_text())
                    json_fixed = True
                except:
                    json_fixed = False

            # Either JSON is fixed or error was handled
            assert json_fixed or (e2e_temp_dir / "parsed_data.txt").exists()

    def test_circular_dependency_detection(self, e2e_config, e2e_temp_dir):
        """Test detection of circular dependencies"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create files with circular imports
        file_a = e2e_temp_dir / "module_a.py"
        file_b = e2e_temp_dir / "module_b.py"

        file_a.write_text("from module_b import func_b\ndef func_a(): return func_b()")
        file_b.write_text("from module_a import func_a\ndef func_b(): return func_a()")

        todos = [
            "Analyze module_a.py and module_b.py for issues",
            "Document any circular dependency found in circular_deps.txt",
            "Suggest a fix in fix_suggestion.txt"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_circular")

        # Should detect and document the issue
        assert (e2e_temp_dir / "circular_deps.txt").exists() or \
               (e2e_temp_dir / "fix_suggestion.txt").exists()

    def test_partial_execution_recovery(self, e2e_config, e2e_temp_dir):
        """Test recovery from partial execution"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # First execution - partial completion
        todos = [
            "Create step1.txt",
            "Create step2.txt",
            "Create step3.txt",
            "Create final_report.txt summarizing all steps"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        supervisor.max_turns = 2  # Force early termination

        result1 = supervisor.execute_with_todos(todos, session_id="e2e_partial")
        assert result1.task_complete is False

        # Count completed steps
        steps_done = sum(1 for i in range(1, 4)
                        if (e2e_temp_dir / f"step{i}.txt").exists())

        # Recovery execution
        supervisor.max_turns = 10  # Allow full execution
        remaining_todos = []

        for i in range(1, 4):
            if not (e2e_temp_dir / f"step{i}.txt").exists():
                remaining_todos.append(f"Create step{i}.txt")

        if not (e2e_temp_dir / "final_report.txt").exists():
            remaining_todos.append("Create final_report.txt summarizing all steps")

        result2 = supervisor.execute_with_todos(
            remaining_todos,
            session_id="e2e_partial_recovery"
        )

        # Should complete remaining tasks
        assert result2.success is True

        # All files should now exist
        for i in range(1, 4):
            assert (e2e_temp_dir / f"step{i}.txt").exists()

    def test_error_cascade_prevention(self, e2e_config, e2e_temp_dir):
        """Test prevention of error cascades"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Try to read non_existent_1.txt",
            "Try to read non_existent_2.txt",
            "Try to read non_existent_3.txt",
            "After encountering errors, create error_summary.txt",
            "Create recovery_complete.txt to show you recovered"
        ]

        supervisor = TaskSupervisor(config=e2e_config)

        # Mock zen to prevent actual calls
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen:
            mock_zen.return_value = {
                "success": True,
                "guidance": "The files don't exist. Skip to creating the summary."
            }

            result = supervisor.execute_with_todos(todos, session_id="e2e_cascade")

        # Should not cascade into failure
        assert result.success is True

        # Should have created summary or recovery file
        assert (e2e_temp_dir / "error_summary.txt").exists() or \
               (e2e_temp_dir / "recovery_complete.txt").exists()

        # Check if zen was triggered for repeated errors
        if len(result.errors) >= e2e_config.supervisor.zen_integration.auto_debug_threshold:
            assert result.metadata.get("zen_needed") is not None
