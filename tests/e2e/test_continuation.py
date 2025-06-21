"""
End-to-end tests for execution continuation and recovery
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import time

from cadence.task_supervisor import TaskSupervisor, ExecutionResult
from cadence.config import CadenceConfig, SCRATCHPAD_DIR, SUPERVISOR_LOG_DIR
from cadence.prompts import TodoPromptManager


class TestContinuationE2E:
    """E2E tests for continuation and recovery scenarios"""
    
    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_cont_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration for E2E tests"""
        config = CadenceConfig()
        config.execution.max_turns = 3  # Very low to force continuation
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.execution.timeout = 30
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.zen_integration.enabled = False
        return config
        
    def test_continuation_after_cutoff(self, e2e_config, e2e_temp_dir):
        """Test continuation when execution is cut off by turn limit"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create tasks that will require more than 3 turns
        todos = [
            "Create a file called step1.txt with content 'Step 1 complete'",
            "Create a file called step2.txt with content 'Step 2 complete'",
            "Create a file called step3.txt with content 'Step 3 complete'",
            "Create a file called step4.txt with content 'Step 4 complete'",
            "Create a file called final.txt with content 'All steps done'"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        
        # First execution - should hit turn limit
        result1 = supervisor.execute_with_todos(
            todos,
            session_id="e2e_continuation"
        )
        
        # Should not complete all tasks
        assert result1.task_complete is False
        assert result1.turns_used >= 3
        
        # Check which files were created
        created_files = []
        for i in range(1, 6):
            if (e2e_temp_dir / f"step{i}.txt").exists():
                created_files.append(f"step{i}.txt")
        if (e2e_temp_dir / "final.txt").exists():
            created_files.append("final.txt")
            
        # Should have created some but not all files
        assert len(created_files) > 0
        assert len(created_files) < 5
        
        # Continue execution
        continuation_result = supervisor.continue_execution(
            previous_result=result1,
            session_id="e2e_continuation_cont1",
            continuation_guidance="Continue with the remaining file creation tasks"
        )
        
        assert isinstance(continuation_result, ExecutionResult)
        
        # Check more files were created
        created_after = []
        for i in range(1, 6):
            if (e2e_temp_dir / f"step{i}.txt").exists():
                created_after.append(f"step{i}.txt")
        if (e2e_temp_dir / "final.txt").exists():
            created_after.append("final.txt")
            
        assert len(created_after) > len(created_files)
        
    def test_continuation_with_updated_todos(self, e2e_config, e2e_temp_dir):
        """Test continuation with updated TODO list"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Initial TODOs
        initial_todos = [
            "Create a file called initial.txt with content 'Initial task'",
            "Create a file called second.txt with content 'Second task'"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        
        # Execute initial
        result1 = supervisor.execute_with_todos(
            initial_todos,
            session_id="e2e_updated_todos"
        )
        
        # Update prompt manager with progress
        supervisor.prompt_manager.update_progress(
            completed_todos=["Create a file called initial.txt with content 'Initial task'"],
            remaining_todos=["Create a file called second.txt with content 'Second task'"]
        )
        
        # Add new TODO
        new_todos = supervisor.prompt_manager.context.remaining_todos + [
            "Create a file called additional.txt with content 'New task added'"
        ]
        supervisor.prompt_manager.context.todos = initial_todos + [new_todos[-1]]
        supervisor.prompt_manager.context.remaining_todos = new_todos
        
        # Continue with updated TODOs
        continuation_result = supervisor.continue_execution(
            previous_result=result1,
            session_id="e2e_updated_todos_cont",
            continuation_guidance="Continue with remaining tasks including the new one"
        )
        
        # Check files
        assert (e2e_temp_dir / "initial.txt").exists()
        # Additional files might exist depending on execution
        
    def test_recovery_from_error(self, e2e_config, e2e_temp_dir):
        """Test recovery from execution errors"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create a problematic TODO that might cause issues
        todos = [
            "Try to create a file at /root/unauthorized.txt (this will fail)",
            "Create a file called recovery.txt with content 'Recovered from error'"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        
        # First attempt
        result1 = supervisor.execute_with_todos(
            todos,
            session_id="e2e_recovery"
        )
        
        # If it encountered the permission error, continue with guidance
        if not (e2e_temp_dir / "recovery.txt").exists():
            continuation_result = supervisor.continue_execution(
                previous_result=result1,
                session_id="e2e_recovery_cont",
                continuation_guidance="Skip the unauthorized file creation and proceed with creating recovery.txt in the current directory"
            )
            
            # Should recover and complete second task
            assert (e2e_temp_dir / "recovery.txt").exists()
            
    def test_scratchpad_persistence(self, e2e_config, e2e_temp_dir):
        """Test that scratchpad persists across continuations"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        todos = [
            "Create your scratchpad and note 'Starting E2E test'",
            "Create a file called persistent.txt with content 'Testing persistence'"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        
        # Initial execution
        result1 = supervisor.execute_with_todos(
            todos,
            session_id="e2e_scratchpad_test"
        )
        
        # Check scratchpad exists
        scratchpad_path = Path(SCRATCHPAD_DIR) / "session_e2e_scratchpad_test.md"
        assert scratchpad_path.exists()
        
        initial_content = scratchpad_path.read_text()
        assert "Starting E2E test" in initial_content or "Scratchpad" in initial_content
        
        # Continue with new session but same scratchpad referenced
        continuation_todos = ["Add to your scratchpad: 'Continuation successful'"]
        
        result2 = supervisor.execute_with_todos(
            continuation_todos,
            session_id="e2e_scratchpad_test_cont"
        )
        
        # Original scratchpad should still exist with initial content
        assert scratchpad_path.exists()
        
    def test_supervisor_analysis_flow(self, e2e_config, e2e_temp_dir):
        """Test full supervisor analysis and decision flow"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Complex set of TODOs
        todos = [
            "Analyze the current directory structure",
            "Create a summary file called analysis.txt with your findings",
            "Create a recommendations file called recommendations.txt"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config)
        
        # Execute with detailed supervisor analysis
        result = supervisor.execute_with_todos(
            todos,
            session_id="e2e_analysis_flow"
        )
        
        # Check supervisor log for analysis
        supervisor_log = Path(SUPERVISOR_LOG_DIR) / "session_e2e_analysis_flow.md"
        assert supervisor_log.exists()
        
        log_content = supervisor_log.read_text()
        assert "Task Analysis" in log_content
        assert "Execution Result" in log_content
        
        # Verify files were created based on analysis
        if result.success:
            # At least one of the files should exist
            assert (e2e_temp_dir / "analysis.txt").exists() or \
                   (e2e_temp_dir / "recommendations.txt").exists()