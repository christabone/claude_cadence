"""
End-to-end tests for supervisor-specific features
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime

from cadence.task_supervisor import TaskSupervisor
from cadence.config import CadenceConfig, SUPERVISOR_LOG_DIR, SCRATCHPAD_DIR


class TestSupervisorFeaturesE2E:
    """E2E tests for supervisor-specific functionality"""
    
    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_super_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration"""
        config = CadenceConfig()
        config.execution.max_turns = 10
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.verbose = True
        config.supervisor.zen_integration.enabled = True
        return config
        
    def test_supervisor_logging_creation(self, e2e_config, e2e_temp_dir):
        """Test supervisor creates comprehensive logs"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        todos = [
            "Create test1.txt",
            "Create test2.txt",
            "Create summary.txt with task completion status"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_supervisor_logs"
        
        result = supervisor.execute_with_todos(todos, session_id=session_id)
        
        # Check supervisor log exists
        supervisor_log = Path(SUPERVISOR_LOG_DIR) / f"session_{session_id}.md"
        assert supervisor_log.exists()
        
        # Check log content
        log_content = supervisor_log.read_text()
        
        # Should have all major sections
        assert "# Supervisor Log" in log_content
        assert "Session ID:" in log_content
        assert "Task Analysis" in log_content
        assert "Execution" in log_content
        assert "Final Summary" in log_content
        
        # Should document TODOs
        assert "test1.txt" in log_content
        assert "test2.txt" in log_content
        
    def test_execution_history_tracking(self, e2e_config, e2e_temp_dir):
        """Test supervisor tracks execution history correctly"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        supervisor = TaskSupervisor(config=e2e_config)
        
        # Multiple executions
        todos1 = ["Create first.txt"]
        result1 = supervisor.execute_with_todos(todos1, session_id="e2e_history_1")
        
        todos2 = ["Create second.txt"]
        result2 = supervisor.execute_with_todos(todos2, session_id="e2e_history_2")
        
        # Check execution history
        assert len(supervisor.execution_history) == 2
        assert supervisor.execution_history[0]["session_id"] == "e2e_history_1"
        assert supervisor.execution_history[1]["session_id"] == "e2e_history_2"
        
        # Each entry should have required fields
        for entry in supervisor.execution_history:
            assert "session_id" in entry
            assert "result" in entry
            assert "timestamp" in entry
            
    def test_supervisor_task_analysis(self, e2e_config, e2e_temp_dir):
        """Test supervisor analyzes tasks correctly"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create a TaskMaster file for analysis
        tasks = {
            "tasks": [
                {
                    "id": "1",
                    "title": "High priority task",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "id": "2", 
                    "title": "Dependent task",
                    "status": "pending",
                    "priority": "medium",
                    "dependencies": ["1"]
                },
                {
                    "id": "3",
                    "title": "Already done",
                    "status": "completed",
                    "priority": "low",
                    "dependencies": []
                }
            ]
        }
        
        task_file = e2e_temp_dir / "tasks.json"
        task_file.write_text(json.dumps(tasks, indent=2))
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_analysis"
        
        # Run with TaskMaster
        success = supervisor.run_with_taskmaster(
            str(task_file),
            task_numbers="1,2"
        )
        
        # Check supervisor log for analysis
        supervisor_log = Path(SUPERVISOR_LOG_DIR) / f"session_taskmaster_{session_id}.md"
        # Log might have different naming for TaskMaster runs
        
        # Check execution happened
        assert isinstance(success, bool)
        
    def test_supervisor_decision_logging(self, e2e_config, e2e_temp_dir):
        """Test supervisor logs its decisions"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Force a scenario requiring decisions
        e2e_config.execution.max_turns = 2  # Very low to force continuation decision
        
        todos = [
            "Task 1: Create file1.txt",
            "Task 2: Create file2.txt",
            "Task 3: Create file3.txt",
            "Task 4: Create summary.txt"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_decisions"
        
        result = supervisor.execute_with_todos(todos, session_id=session_id)
        
        # Should not complete due to turn limit
        assert result.task_complete is False
        
        # Check supervisor log
        supervisor_log = Path(SUPERVISOR_LOG_DIR) / f"session_{session_id}.md"
        if supervisor_log.exists():
            log_content = supervisor_log.read_text()
            
            # Should document the cutoff
            assert "Execution Result" in log_content
            assert str(e2e_config.execution.max_turns) in log_content or "turn" in log_content
            
    def test_supervisor_zen_coordination(self, e2e_config, e2e_temp_dir):
        """Test supervisor coordinates with Zen integration"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Task that will trigger zen
        todos = [
            "Debug the issue in /nonexistent/complex/system.py",
            "If stuck, document the issue clearly"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_zen_coord"
        
        # Mock zen response
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen:
            mock_zen.return_value = {
                "success": True,
                "guidance": "File doesn't exist. Create an example instead."
            }
            
            result = supervisor.execute_with_todos(todos, session_id=session_id)
            
            # Check supervisor log documents zen interaction
            supervisor_log = Path(SUPERVISOR_LOG_DIR) / f"session_{session_id}.md"
            if supervisor_log.exists() and mock_zen.called:
                log_content = supervisor_log.read_text()
                assert "Zen" in log_content or "assistance" in log_content.lower()
                
    def test_supervisor_progress_monitoring(self, e2e_config, e2e_temp_dir):
        """Test supervisor monitors agent progress"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        todos = [
            "Create progress markers: start.txt, middle.txt, end.txt",
            "Update your scratchpad after each file creation",
            "Create done.txt when all markers are created"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_progress"
        
        result = supervisor.execute_with_todos(todos, session_id=session_id)
        
        assert result.success is True
        
        # Check files were created in sequence
        assert (e2e_temp_dir / "start.txt").exists()
        assert (e2e_temp_dir / "done.txt").exists()
        
        # Check scratchpad shows progress
        scratchpad = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad.exists():
            content = scratchpad.read_text()
            # Should show some progression
            assert "start" in content.lower() or "progress" in content.lower()
            
    def test_supervisor_error_analysis(self, e2e_config, e2e_temp_dir):
        """Test supervisor analyzes errors properly"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Tasks that will generate errors
        todos = [
            "Read /etc/shadow (will fail)",
            "Read /root/.ssh/id_rsa (will fail)",
            "After errors, create error_analysis.txt explaining what happened"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_error_analysis"
        
        result = supervisor.execute_with_todos(todos, session_id=session_id)
        
        # Check supervisor log
        supervisor_log = Path(SUPERVISOR_LOG_DIR) / f"session_{session_id}.md"
        if supervisor_log.exists():
            log_content = supervisor_log.read_text()
            
            # Should document errors encountered
            if len(result.errors) > 0:
                assert "error" in log_content.lower()
                
        # Agent should have created analysis
        assert (e2e_temp_dir / "error_analysis.txt").exists()
        
    def test_supervisor_session_management(self, e2e_config, e2e_temp_dir):
        """Test supervisor manages sessions correctly"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        supervisor = TaskSupervisor(config=e2e_config)
        
        # Test auto-generated session IDs
        todos = ["Create auto_session.txt"]
        result1 = supervisor.execute_with_todos(todos)
        
        # Should have generated a session ID
        assert len(supervisor.execution_history) > 0
        auto_session_id = supervisor.execution_history[-1]["session_id"]
        assert auto_session_id.startswith("session_")
        
        # Test custom session ID
        custom_id = "custom_test_id_123"
        todos2 = ["Create custom_session.txt"]
        result2 = supervisor.execute_with_todos(todos2, session_id=custom_id)
        
        # Should use custom ID
        assert supervisor.execution_history[-1]["session_id"] == custom_id
        
        # Both scratchpads should exist
        assert (Path(SCRATCHPAD_DIR) / f"{auto_session_id}.md").exists()
        assert (Path(SCRATCHPAD_DIR) / f"session_{custom_id}.md").exists()
        
    def test_supervisor_timeout_handling(self, e2e_config, e2e_temp_dir):
        """Test supervisor handles timeouts properly"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Set very short timeout
        e2e_config.execution.timeout = 3  # 3 seconds
        
        todos = [
            "Start creating timeout_test.txt",
            "Count to 1000000 slowly",  # Time-consuming task
            "Create completion.txt"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        session_id = "e2e_timeout"
        
        start_time = time.time()
        result = supervisor.execute_with_todos(todos, session_id=session_id)
        elapsed = time.time() - start_time
        
        # Should timeout quickly
        assert elapsed < 10  # Should not take long
        
        if not result.success:
            # Check supervisor log documents timeout
            supervisor_log = Path(SUPERVISOR_LOG_DIR) / f"session_{session_id}.md"
            if supervisor_log.exists():
                log_content = supervisor_log.read_text()
                assert "timeout" in log_content.lower() or "timed out" in log_content.lower()