"""
End-to-end tests for scratchpad monitoring and agent communication
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import subprocess
import time
import threading

from cadence.task_supervisor import TaskSupervisor
from cadence.config import CadenceConfig, SCRATCHPAD_DIR


class TestScratchpadMonitoringE2E:
    """E2E tests for scratchpad-based supervisor/agent communication"""

    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_scratch_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration"""
        config = CadenceConfig()
        config.execution.max_turns = 10
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.zen_integration.enabled = True
        return config

    def test_scratchpad_creation_and_updates(self, e2e_config, e2e_temp_dir):
        """Test that agent creates and updates scratchpad correctly"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create your scratchpad and document your plan",
            "Update scratchpad with progress after creating test.txt",
            "Create test.txt with content 'Scratchpad test complete'"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_scratchpad_test"

        result = supervisor.execute_with_todos(todos, session_id=session_id)

        assert result.success is True

        # Check scratchpad exists and has content
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        assert scratchpad_path.exists()

        content = scratchpad_path.read_text()
        assert len(content) > 0
        assert "Scratchpad" in content or "scratchpad" in content

        # Check test file was created
        assert (e2e_temp_dir / "test.txt").exists()

    def test_help_request_detection(self, e2e_config, e2e_temp_dir):
        """Test supervisor detects help requests in scratchpad"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Task that will likely cause agent to ask for help
        todos = [
            "Create your scratchpad first",
            "Try to fix the bug in the non-existent file /does/not/exist/buggy.py",
            "If you can't find the file, update your scratchpad with HELP NEEDED status"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_help_test"

        # Mock zen response
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen:
            mock_zen.return_value = {
                "success": True,
                "guidance": "The file doesn't exist. Create a new file instead."
            }

            result = supervisor.execute_with_todos(todos, session_id=session_id)

            # Check if help was detected
            scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
            if scratchpad_path.exists():
                content = scratchpad_path.read_text()
                # Agent might have written help request
                if "HELP NEEDED" in content:
                    assert result.metadata.get("zen_needed") is not None

    def test_completion_detection_via_scratchpad(self, e2e_config, e2e_temp_dir):
        """Test supervisor detects completion via scratchpad"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create your scratchpad",
            "Create file1.txt with 'Task 1 done'",
            "Create file2.txt with 'Task 2 done'",
            "Update your scratchpad with 'ALL TASKS COMPLETE' when finished"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_completion_scratch"

        result = supervisor.execute_with_todos(todos, session_id=session_id)

        assert result.success is True
        assert result.task_complete is True

        # Check scratchpad has completion marker
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad_path.exists():
            content = scratchpad_path.read_text()
            assert "ALL TASKS COMPLETE" in content or "complete" in content.lower()

    def test_scratchpad_progress_tracking(self, e2e_config, e2e_temp_dir):
        """Test tracking progress through scratchpad updates"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create scratchpad with sections for each task",
            "Task 1: Create dir1/file1.txt - mark complete in scratchpad when done",
            "Task 2: Create dir2/file2.txt - mark complete in scratchpad when done",
            "Task 3: Create summary.txt with task completion status"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_progress_track"

        result = supervisor.execute_with_todos(todos, session_id=session_id)

        # Check scratchpad for progress markers
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad_path.exists():
            content = scratchpad_path.read_text()
            # Should have some progress indicators
            assert "Task 1" in content or "task" in content.lower()

    def test_scratchpad_error_documentation(self, e2e_config, e2e_temp_dir):
        """Test agent documents errors in scratchpad"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create scratchpad to track your work",
            "Try to read /etc/shadow (will fail with permission error)",
            "Document the error in your scratchpad with details",
            "Create error_handled.txt to show you recovered"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_error_doc"

        result = supervisor.execute_with_todos(todos, session_id=session_id)

        # Check scratchpad documents the error
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad_path.exists():
            content = scratchpad_path.read_text()
            # Should mention error or permission
            assert "error" in content.lower() or "permission" in content.lower() or "fail" in content.lower()

        # Check recovery file
        assert (e2e_temp_dir / "error_handled.txt").exists()

    def test_scratchpad_planning_documentation(self, e2e_config, e2e_temp_dir):
        """Test agent uses scratchpad for planning"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create scratchpad and write a detailed plan for implementing a calculator",
            "Document the functions needed: add, subtract, multiply, divide",
            "Create calculator.py with the planned functions",
            "Update scratchpad with implementation notes"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_planning"

        result = supervisor.execute_with_todos(todos, session_id=session_id)

        assert result.success is True

        # Check scratchpad has planning content
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad_path.exists():
            content = scratchpad_path.read_text()
            # Should have planning details
            assert "plan" in content.lower() or "function" in content.lower()

        # Check calculator was created
        calc_file = e2e_temp_dir / "calculator.py"
        if calc_file.exists():
            calc_content = calc_file.read_text()
            # Should have some functions
            assert "def" in calc_content

    def test_supervisor_scratchpad_monitoring(self, e2e_config, e2e_temp_dir):
        """Test supervisor actively monitors scratchpad during execution"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Set up supervisor to check scratchpad
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_monitor_test"

        # Create initial scratchpad
        scratchpad_dir = Path(SCRATCHPAD_DIR)
        scratchpad_dir.mkdir(parents=True, exist_ok=True)
        scratchpad_path = scratchpad_dir / f"session_{session_id}.md"

        todos = [
            "Read the existing scratchpad if it exists",
            "Create status.txt with current timestamp",
            "Update scratchpad with completion status"
        ]

        # Pre-create scratchpad with initial content
        scratchpad_path.write_text("""
        # Agent Scratchpad
        ## Initial Status
        Starting work on tasks...
        """)

        result = supervisor.execute_with_todos(todos, session_id=session_id)

        # Check supervisor detected and used existing scratchpad
        assert result.success is True

        # Scratchpad should be updated
        final_content = scratchpad_path.read_text()
        assert len(final_content) > len("# Agent Scratchpad\n## Initial Status\nStarting work on tasks...")

    def test_scratchpad_persistence_across_sessions(self, e2e_config, e2e_temp_dir):
        """Test scratchpad persists and is reused across sessions"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        supervisor = TaskSupervisor(config=e2e_config)
        base_session = "e2e_persist"

        # First session
        todos1 = [
            "Create scratchpad and note 'Session 1 started'",
            "Create session1.txt"
        ]

        result1 = supervisor.execute_with_todos(todos1, session_id=f"{base_session}_1")
        assert result1.success is True

        # Get scratchpad path from first session
        scratchpad1 = Path(SCRATCHPAD_DIR) / f"session_{base_session}_1.md"
        assert scratchpad1.exists()

        # Second session references first
        todos2 = [
            f"Check if scratchpad from session_{base_session}_1 exists",
            "Create your own scratchpad noting you found the previous one",
            "Create session2.txt"
        ]

        result2 = supervisor.execute_with_todos(todos2, session_id=f"{base_session}_2")
        assert result2.success is True

        # Both scratchpads should exist
        scratchpad2 = Path(SCRATCHPAD_DIR) / f"session_{base_session}_2.md"
        assert scratchpad2.exists()

        # Second might reference first
        content2 = scratchpad2.read_text()
        # Might mention previous session
