"""
Unit tests for SupervisorOrchestrator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import json
import os

from cadence.orchestrator import SupervisorOrchestrator, SupervisorDecision, AgentResult


class TestSupervisorOrchestrator:
    """Test SupervisorOrchestrator functionality"""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure"""
        project_root = tmp_path / "test_project"
        project_root.mkdir()
        
        # Create task file
        task_file = project_root / ".taskmaster" / "tasks" / "tasks.json"
        task_file.parent.mkdir(parents=True)
        
        tasks = {
            "tasks": [
                {
                    "id": "1",
                    "title": "Test Task",
                    "status": "pending",
                    "subtasks": [
                        {"id": "1.1", "title": "Subtask 1", "status": "pending"},
                        {"id": "1.2", "title": "Subtask 2", "status": "pending"}
                    ]
                }
            ]
        }
        
        with open(task_file, 'w') as f:
            json.dump(tasks, f)
            
        return project_root, task_file
    
    def test_initialization(self, temp_project):
        """Test orchestrator initialization"""
        project_root, task_file = temp_project
        
        orchestrator = SupervisorOrchestrator(
            project_root=project_root,
            task_file=task_file
        )
        
        # Check directories created
        assert orchestrator.supervisor_dir.exists()
        assert orchestrator.agent_dir.exists()
        
        # Check state initialized
        assert orchestrator.state["first_run"] is True
        assert orchestrator.state["session_count"] == 0
        
    def test_state_persistence(self, temp_project):
        """Test state loading and saving"""
        project_root, task_file = temp_project
        
        # First orchestrator
        orch1 = SupervisorOrchestrator(project_root, task_file)
        assert orch1.state["first_run"] is True
        
        # Modify and save state
        orch1.state["first_run"] = False
        orch1.state["session_count"] = 5
        orch1.save_state()
        
        # Second orchestrator should load saved state
        orch2 = SupervisorOrchestrator(project_root, task_file)
        assert orch2.state["first_run"] is False
        assert orch2.state["session_count"] == 5
        
    def test_session_id_generation(self, temp_project):
        """Test session ID generation"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)
        
        session_id = orchestrator.generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) > 10
        assert "_" in session_id
        
    @patch('subprocess.run')
    def test_run_supervisor_analysis_first_run(self, mock_run, temp_project):
        """Test running supervisor analysis on first run"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)
        
        # Mock subprocess result
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Create mock decision file
        decision_data = {
            "action": "execute",
            "todos": ["Test TODO"],
            "guidance": "Test guidance",
            "task_id": "1",
            "session_id": "test_session",
            "reason": "Test reason"
        }
        
        decision_file = orchestrator.supervisor_dir / "decision_test_session.json"
        with open(decision_file, 'w') as f:
            json.dump(decision_data, f)
        
        # Test first run (no --continue flag)
        with patch('os.chdir') as mock_chdir, patch('os.getcwd', return_value=str(project_root)):
            decision = orchestrator.run_supervisor_analysis("test_session", use_continue=False)
            
            # Verify chdir was called (first call should be to supervisor dir)
            assert mock_chdir.call_count >= 1
            mock_chdir.assert_any_call(orchestrator.supervisor_dir)
            
            # Verify command
            args = mock_run.call_args[0][0]
            assert "python" in args[0]
            assert "--analyze" in args
            assert "--continue" not in args
            
            # Verify decision
            assert decision.action == "execute"
            assert decision.todos == ["Test TODO"]
            
    @patch('subprocess.run')
    def test_run_supervisor_analysis_continue(self, mock_run, temp_project):
        """Test running supervisor analysis with continue flag"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)
        
        # Mock subprocess result
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Create mock decision file
        decision_file = orchestrator.supervisor_dir / "decision_test_session.json"
        with open(decision_file, 'w') as f:
            json.dump({"action": "skip", "reason": "Test"}, f)
        
        # Test with continue flag
        with patch('os.chdir'), patch('os.getcwd', return_value=str(project_root)):
            orchestrator.run_supervisor_analysis("test_session", use_continue=True)
            
            # Verify --continue flag was added
            args = mock_run.call_args[0][0]
            assert "--continue" in args
            
    @patch('subprocess.run')
    def test_run_agent_first_run(self, mock_run, temp_project):
        """Test running agent on first run"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)
        
        # Mock subprocess result
        mock_run.return_value = Mock(returncode=0)
        
        todos = ["Task 1", "Task 2"]
        guidance = "Test guidance"
        
        with patch('os.chdir') as mock_chdir, patch('os.getcwd', return_value=str(project_root)):
            result = orchestrator.run_agent(
                todos=todos,
                guidance=guidance,
                session_id="test_session",
                use_continue=False
            )
            
            # Verify chdir to agent directory (checking any call, not just the last)
            assert mock_chdir.call_count >= 1
            mock_chdir.assert_any_call(orchestrator.agent_dir)
            
            # Verify command
            args = mock_run.call_args[0][0]
            assert args[0] == "claude"
            assert "-p" in args
            assert "-c" not in args  # No continue on first run
            
            # Verify result
            assert isinstance(result, AgentResult)
            assert result.success is True
            
    def test_build_agent_prompt(self, temp_project):
        """Test agent prompt building"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)
        
        todos = ["Complete task 1", "Complete task 2"]
        guidance = "Focus on quality"
        
        prompt = orchestrator.build_agent_prompt(todos, guidance)
        
        assert "SUPERVISOR GUIDANCE" in prompt
        assert guidance in prompt
        assert "YOUR TODOS" in prompt
        assert todos[0] in prompt
        assert todos[1] in prompt
        assert "ALL TASKS COMPLETE" in prompt
        assert "HELP NEEDED" in prompt
        
    def test_save_agent_results(self, temp_project):
        """Test saving agent results"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)
        
        agent_result = AgentResult(
            success=True,
            session_id="test_session",
            output_file="output.log",
            error_file="error.log",
            execution_time=10.5,
            completed_normally=True,
            requested_help=False,
            errors=[]
        )
        
        orchestrator.save_agent_results(agent_result, "test_session")
        
        # Check file created
        results_file = orchestrator.supervisor_dir / "agent_result_test_session.json"
        assert results_file.exists()
        
        # Check content
        with open(results_file) as f:
            saved_data = json.load(f)
            
        assert saved_data["success"] is True
        assert saved_data["execution_time"] == 10.5
        assert saved_data["completed_normally"] is True