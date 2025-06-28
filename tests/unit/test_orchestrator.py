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
        assert orchestrator.state["session_count"] == 0
        assert orchestrator.state["last_session_id"] is None
        assert "created_at" in orchestrator.state

    def test_state_persistence(self, temp_project):
        """Test state loading and saving"""
        project_root, task_file = temp_project

        # First orchestrator
        orch1 = SupervisorOrchestrator(project_root, task_file)
        assert orch1.state["session_count"] == 0

        # Modify and save state
        orch1.state["session_count"] = 5
        orch1.state["last_session_id"] = "test-session"
        orch1.save_state()

        # Second orchestrator should load saved state
        orch2 = SupervisorOrchestrator(project_root, task_file)
        assert orch2.state["session_count"] == 5
        assert orch2.state["last_session_id"] == "test-session"

    def test_session_id_generation(self, temp_project):
        """Test session ID generation"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)

        session_id = orchestrator.generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) > 10
        assert "_" in session_id

    @patch('cadence.orchestrator.SupervisorOrchestrator.run_claude_with_realtime_output')
    def test_run_supervisor_analysis_first_run(self, mock_run_claude, temp_project):
        """Test running supervisor analysis on first run"""
        project_root, task_file = temp_project
        config = {
            "supervisor": {
                "model": "claude-3-5-sonnet-20241022"
            },
            "agent": {
                "model": "claude-3-5-sonnet-20241022"
            }
        }
        orchestrator = SupervisorOrchestrator(project_root, task_file, config=config)

        # Mock async subprocess result with simulated output
        # Return value is (returncode, all_output_lines)
        mock_output = [
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Here is the decision:\\n{\\n  \\"action\\": \\"execute\\",\\n  \\"todos\\": [\\"Test TODO\\"],\\n  \\"guidance\\": \\"Test guidance\\",\\n  \\"task_id\\": \\"1\\",\\n  \\"session_id\\": \\"test_session\\",\\n  \\"reason\\": \\"Test reason\\"\\n}"}]}}'
        ]
        mock_run_claude.return_value = (0, mock_output)

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
            decision = orchestrator.run_supervisor_analysis("test_session", use_continue=False, iteration=1)

            # Verify chdir was called (first call should be to supervisor dir)
            assert mock_chdir.call_count >= 1
            mock_chdir.assert_any_call(orchestrator.supervisor_dir)

            # Verify command was called
            assert mock_run_claude.called
            args = mock_run_claude.call_args[0][0]  # First positional argument is cmd list
            assert "claude" in args[0]
            assert "--continue" not in args

            # Verify decision
            assert decision.action == "execute"
            assert decision.todos == ["Test TODO"]

    @patch('cadence.orchestrator.SupervisorOrchestrator.run_claude_with_realtime_output')
    def test_run_supervisor_analysis_continue(self, mock_run_claude, temp_project):
        """Test running supervisor analysis with continue flag"""
        project_root, task_file = temp_project
        config = {
            "supervisor": {
                "model": "claude-3-5-sonnet-20241022"
            },
            "agent": {
                "model": "claude-3-5-sonnet-20241022"
            }
        }
        orchestrator = SupervisorOrchestrator(project_root, task_file, config=config)

        # Mock async subprocess result with simulated output
        # Return value is (returncode, all_output_lines)
        mock_output = [
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Continuing analysis...\\n{\\n  \\"action\\": \\"skip\\",\\n  \\"reason\\": \\"Test skip reason\\",\\n  \\"task_id\\": \\"1\\",\\n  \\"session_id\\": \\"test_session\\"\\n}"}]}}'
        ]
        mock_run_claude.return_value = (0, mock_output)

        # Create mock decision file
        decision_file = orchestrator.supervisor_dir / "decision_test_session.json"
        with open(decision_file, 'w') as f:
            json.dump({"action": "skip", "reason": "Test skip reason", "task_id": "1", "session_id": "test_session"}, f)

        # Test with continue flag
        with patch('os.chdir'), patch('os.getcwd', return_value=str(project_root)):
            decision = orchestrator.run_supervisor_analysis("test_session", use_continue=True, iteration=2)

            # Verify --continue flag was added
            assert mock_run_claude.called
            args = mock_run_claude.call_args[0][0]
            assert "--continue" in args

            # Verify decision
            assert decision.action == "skip"
            assert decision.reason == "Test skip reason"

    @patch('cadence.orchestrator.SupervisorOrchestrator.build_agent_prompt')
    @patch('cadence.orchestrator.SupervisorOrchestrator.run_claude_with_realtime_output')
    def test_run_agent_first_run(self, mock_run_claude, mock_build_prompt, temp_project):
        """Test running agent on first run"""
        project_root, task_file = temp_project
        config = {
            "supervisor": {
                "model": "claude-3-5-sonnet-20241022"
            },
            "agent": {
                "model": "claude-3-5-sonnet-20241022"
            }
        }
        orchestrator = SupervisorOrchestrator(project_root, task_file, config=config)

        # Mock async subprocess result with simulated output showing completion
        mock_output = [
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Working on tasks..."}]}}',
            '{"type":"assistant","message":{"content":[{"type":"text","text":"ALL TASKS COMPLETE"}]}}'
        ]
        mock_run_claude.return_value = (0, mock_output)

        # Mock build_agent_prompt to return a simple prompt
        mock_build_prompt.return_value = "Test prompt with TODOs"

        todos = ["Task 1", "Task 2"]
        guidance = "Test guidance"

        with patch('os.chdir') as mock_chdir, patch('os.getcwd', return_value=str(project_root)):
            result = orchestrator.run_agent(
                todos=todos,
                guidance=guidance,
                session_id="test_session",
                use_continue=False,
                task_id="1",
                subtasks=[],
                project_root=str(project_root)
            )

            # Verify chdir to agent directory (checking any call, not just the last)
            assert mock_chdir.call_count >= 1
            mock_chdir.assert_any_call(orchestrator.agent_dir)

            # Verify command was called
            assert mock_run_claude.called
            args = mock_run_claude.call_args[0][0]
            assert "claude" in args[0]
            assert "-p" in args
            assert "-c" not in args  # No continue on first run

            # Verify result
            assert isinstance(result, AgentResult)
            assert result.success is True

    @patch('cadence.prompts.PromptGenerator.generate_initial_todo_prompt')
    def test_build_agent_prompt(self, mock_generate_prompt, temp_project):
        """Test agent prompt building"""
        project_root, task_file = temp_project
        orchestrator = SupervisorOrchestrator(project_root, task_file)

        # Mock the prompt generator to return a template with placeholders
        mock_generate_prompt.return_value = """
=== YOUR TODOS ===
TODO1
TODO2

ALL TASKS COMPLETE marker
HELP NEEDED marker
"""

        todos = ["Complete task 1", "Complete task 2"]
        guidance = "Focus on quality"

        prompt = orchestrator.build_agent_prompt(todos, guidance)

        # Since we're mocking generate_initial_todo_prompt, we just verify it was called
        assert mock_generate_prompt.called

        # Check that guidance was added
        assert "SUPERVISOR GUIDANCE" in prompt
        assert guidance in prompt

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
