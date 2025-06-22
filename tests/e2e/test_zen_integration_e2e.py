"""
End-to-end tests for Zen integration with real execution
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import yaml
from unittest.mock import patch, Mock

from cadence.task_supervisor import TaskSupervisor, ExecutionResult
from cadence.config import CadenceConfig, ZenIntegrationConfig, SCRATCHPAD_DIR, ConfigLoader
from cadence.zen_integration import ZenIntegration


class TestZenIntegrationE2E:
    """E2E tests for Zen assistance integration"""
    
    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_zen_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def e2e_config_with_zen(self, e2e_temp_dir):
        """Create a test configuration with Zen enabled"""
        config_dict = {
            "supervisor": {
                "verbose": True,
                "zen_integration": {
                    "enabled": True,
                    "stuck_detection": True,
                    "auto_debug": True,
                    "auto_debug_threshold": 2,
                    "cutoff_detection": True,
                    "validate_on_complete": ["*critical*", "*security*"]
                }
            },
            "execution": {
                "max_turns": 5,
                "log_dir": str(e2e_temp_dir / "logs"),
                "timeout": 30
            },
            "agent": {
                "model": "claude-3-haiku-20240307",
                "claude_args": ["--no-cache"]
            },
            "mcp": {
                "servers": ["filesystem", "zen"]  # Include zen server
            }
        }
        
        config_file = e2e_temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
            
        loader = ConfigLoader(str(config_file))
        return loader.config
        
    def test_stuck_agent_triggers_zen(self, e2e_config_with_zen, e2e_temp_dir):
        """Test that a stuck agent triggers Zen assistance"""
        # Skip if Claude or MCP zen is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
            subprocess.run(["mcp", "list"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI or MCP not available")
            
        # Create a TODO that will cause the agent to get stuck
        todos = [
            "Find and fix the bug in the non-existent file /does/not/exist/bug.py"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        # Mock the zen tool call to avoid actual MCP call
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            mock_zen_call.return_value = {
                "success": True,
                "guidance": "The file doesn't exist. Create a new file instead with a bug fix example."
            }
            
            # Execute
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_stuck_zen"
            )
            
            # Check if zen was considered
            if result.turns_used >= 2:  # Agent had time to get stuck
                # Either zen was called or it was detected as needed
                assert mock_zen_call.called or \
                       (result.metadata.get("zen_needed") is not None)
                       
    def test_repeated_errors_trigger_zen(self, e2e_config_with_zen, e2e_temp_dir):
        """Test that repeated errors trigger Zen debug assistance"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create TODOs that will cause repeated errors
        todos = [
            "Read the file at /invalid/path/file1.txt",
            "Read the file at /invalid/path/file2.txt",
            "Read the file at /invalid/path/file3.txt"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        # Mock zen to provide guidance
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            mock_zen_call.return_value = {
                "success": True,
                "guidance": "The paths are invalid. Create the files locally instead."
            }
            
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_errors_zen"
            )
            
            # Should detect repeated errors
            if len(result.errors) >= e2e_config_with_zen.supervisor.zen_integration.auto_debug_threshold:
                assert result.metadata.get("zen_needed") is not None
                assert result.metadata["zen_needed"]["tool"] == "debug"
                
    def test_cutoff_triggers_zen_analysis(self, e2e_config_with_zen, e2e_temp_dir):
        """Test that execution cutoff triggers Zen analysis"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Set very low turn limit
        e2e_config_with_zen.execution.max_turns = 2
        
        # Create many TODOs that can't be completed in 2 turns
        todos = [
            f"Create file{i}.txt with content 'File {i}'" 
            for i in range(10)
        ]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            mock_zen_call.return_value = {
                "success": True,
                "guidance": "Focus on creating the first 3 files in the next continuation."
            }
            
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_cutoff_zen"
            )
            
            # Should be cut off
            assert result.task_complete is False
            assert result.turns_used >= 2
            
            # Should detect cutoff
            assert result.metadata.get("zen_needed") is not None
            assert result.metadata["zen_needed"]["tool"] == "analyze"
            
    def test_critical_task_validation(self, e2e_config_with_zen, e2e_temp_dir):
        """Test that critical tasks trigger validation on completion"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create a critical security task
        todos = [
            "Create a critical security configuration file security_config.json with proper settings"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            mock_zen_call.return_value = {
                "success": True,
                "guidance": "Security configuration looks good. Ensure proper permissions are set."
            }
            
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_critical_zen"
            )
            
            # If task completed, should trigger validation
            if result.task_complete:
                assert result.metadata.get("zen_needed") is not None
                assert result.metadata["zen_needed"]["tool"] == "precommit"
                assert "critical" in result.metadata["zen_needed"]["reason"].lower()
                
    def test_zen_continuation_guidance(self, e2e_config_with_zen, e2e_temp_dir):
        """Test Zen provides useful continuation guidance"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create a task that will need help
        todos = [
            "Debug why the async function in nonexistent.js is hanging"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        # First execution - should need help
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            mock_zen_call.return_value = {
                "success": True,
                "guidance": "File doesn't exist. Create an example async function that demonstrates proper error handling."
            }
            
            result1 = supervisor.execute_with_todos(
                todos,
                session_id="e2e_zen_guidance"
            )
            
            # Handle zen assistance if needed
            if result1.metadata.get("zen_needed"):
                zen_response = supervisor.handle_zen_assistance(
                    result1,
                    "e2e_zen_guidance"
                )
                
                assert zen_response["continuation_guidance"]
                assert "example async function" in zen_response["continuation_guidance"]
                
    def test_zen_help_request_via_scratchpad(self, e2e_config_with_zen, e2e_temp_dir):
        """Test agent requesting help via scratchpad"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Create scratchpad with help request
        session_id = "e2e_scratchpad_help"
        scratchpad_dir = Path(SCRATCHPAD_DIR)
        scratchpad_dir.mkdir(parents=True, exist_ok=True)
        scratchpad_path = scratchpad_dir / f"session_{session_id}.md"
        
        scratchpad_path.write_text("""
        # Agent Scratchpad
        
        ## Current Status
        Working on database migration task
        
        ## HELP NEEDED
        Status: STUCK
        Issue: Cannot determine the correct migration strategy
        Details: Need guidance on handling foreign key constraints
        """)
        
        todos = ["Migrate the database schema from v1 to v2"]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        # Create a custom execution result that simulates completed execution
        result = ExecutionResult(
            success=True,
            turns_used=3,
            output_lines=["Checking scratchpad", "Found help request"],
            errors=[],
            metadata={},
            task_complete=False
        )
        
        # Check if zen detects the help request
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            mock_zen_call.return_value = {
                "success": True,
                "guidance": "For foreign key constraints: 1) Drop constraints, 2) Migrate schema, 3) Re-add constraints"
            }
            
            zen_tool, reason = supervisor.zen.should_call_zen(
                result, 
                supervisor.prompt_manager.context if hasattr(supervisor, 'prompt_manager') else Mock(),
                session_id
            )
            
            assert zen_tool == "debug"
            assert "stuck" in reason.lower()
            
    def test_full_zen_workflow(self, e2e_config_with_zen, e2e_temp_dir):
        """Test complete workflow with zen assistance"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")
            
        # Complex task that might need help
        todos = [
            "Implement a rate limiter for the API",
            "Ensure it handles concurrent requests properly",
            "Add comprehensive error handling"
        ]
        
        supervisor = TaskSupervisor(config=e2e_config_with_zen)
        
        with patch.object(supervisor.zen, '_call_zen_tool') as mock_zen_call:
            # Simulate different zen responses
            call_count = 0
            def zen_response(*args):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {
                        "success": True,
                        "guidance": "Use token bucket algorithm for rate limiting. Consider using Redis for distributed systems."
                    }
                else:
                    return {
                        "success": True,
                        "guidance": "Implementation looks good. Add circuit breaker pattern for better resilience."
                    }
                    
            mock_zen_call.side_effect = zen_response
            
            # Execute
            result = supervisor.execute_with_todos(
                todos,
                session_id="e2e_full_zen"
            )
            
            # Check execution
            assert isinstance(result, ExecutionResult)
            
            # Check supervisor log for zen interactions
            supervisor_log = Path(e2e_config_with_zen.execution.log_dir) / "supervisor" / "session_e2e_full_zen.md"
            if supervisor_log.exists():
                log_content = supervisor_log.read_text()
                # Should have zen-related entries if zen was triggered
                if mock_zen_call.called:
                    assert "Zen" in log_content or "assistance" in log_content