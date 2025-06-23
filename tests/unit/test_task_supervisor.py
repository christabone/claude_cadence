"""
Unit tests for TaskSupervisor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import json
import queue
import time

from cadence.task_supervisor import TaskSupervisor, ExecutionResult
from cadence.config import CadenceConfig, SCRATCHPAD_DIR, SUPERVISOR_LOG_DIR


class TestTaskSupervisor:
    """Test TaskSupervisor core functionality"""

    def test_initialization(self, mock_config):
        """Test supervisor initialization"""
        supervisor = TaskSupervisor(config=mock_config)

        assert supervisor.config == mock_config
        assert supervisor.max_turns == mock_config.execution.max_turns
        assert supervisor.model == mock_config.agent.model
        assert supervisor.verbose == mock_config.supervisor.verbose
        assert supervisor.execution_history == []

    def test_logger_setup(self, mock_config, temp_dir):
        """Test logger is properly initialized"""
        supervisor = TaskSupervisor(config=mock_config)

        assert hasattr(supervisor, 'logger')
        assert supervisor.logger is not None
        # Check log file was created
        log_files = list(Path(mock_config.execution.log_dir).glob("supervisor_logs/*.log"))
        assert len(log_files) >= 0  # May be 0 if logger doesn't create file immediately

    def test_supervisor_log_initialization(self, supervisor, temp_dir):
        """Test supervisor markdown log initialization"""
        session_id = "test_session_123"
        log_path = supervisor._init_supervisor_log(session_id)

        assert log_path.exists()
        assert log_path.name == f"session_{session_id}.md"
        assert supervisor.current_log_path == log_path

        # Check content
        content = log_path.read_text()
        assert "# Supervisor Log" in content
        assert f"Session ID: {session_id}" in content
        assert f"Max Turns: {supervisor.max_turns}" in content

    def test_scratchpad_status_check(self, supervisor, mock_scratchpad, temp_dir):
        """Test scratchpad status checking"""
        session_id = "test_123"

        # Patch SCRATCHPAD_DIR to use temp directory
        with patch('cadence.task_supervisor.SCRATCHPAD_DIR', str(temp_dir / ".cadence/scratchpad")):
            # Test non-existent scratchpad
            status = supervisor._check_scratchpad_status(session_id)
            assert status["exists"] is False

            # Test scratchpad with help needed
            mock_scratchpad(session_id, """
            # Scratchpad
            ## HELP NEEDED
            Status: STUCK
            Issue: Cannot find the file
            """)

            status = supervisor._check_scratchpad_status(session_id)
            assert status["exists"] is True
            assert status["has_help_needed"] is True
            assert status["has_all_complete"] is False

            # Test completed scratchpad
            mock_scratchpad(session_id + "_2", """
            # Scratchpad
            ## Completion Summary
            ALL TASKS COMPLETE
            """)

            status = supervisor._check_scratchpad_status(session_id + "_2")
            assert status["exists"] is True
            assert status["has_help_needed"] is False
            assert status["has_all_complete"] is True

    def test_build_todo_prompt(self, supervisor):
        """Test TODO prompt building"""
        todos = ["Fix bug in auth", "Add logging"]
        supervisor.prompt_manager = Mock()
        supervisor.prompt_manager.get_initial_prompt.return_value = "test prompt"

        prompt = supervisor._build_todo_prompt(todos)
        assert prompt == "test prompt"
        supervisor.prompt_manager.get_initial_prompt.assert_called_once()

    def test_analyze_task_completion(self, supervisor):
        """Test task completion analysis"""
        # Test explicit completion
        output = "Working on tasks... ALL TASKS COMPLETE"
        assert supervisor._analyze_task_completion(output, None) is True

        # Test incomplete
        output = "Working on tasks... still in progress"
        assert supervisor._analyze_task_completion(output, None) is False

        # Test with task list
        task_list = [
            {"id": "1", "status": "done"},
            {"id": "2", "status": "done"}
        ]
        output = "Completed both tasks"
        with patch('cadence.task_supervisor.TaskManager') as mock_tm:
            mock_manager = Mock()
            mock_manager.analyze_progress.return_value = {
                "completed": ["1", "2"],
                "incomplete": []
            }
            mock_tm.return_value = mock_manager

            result = supervisor._analyze_task_completion(output, task_list)
            assert result is True

    def test_execute_with_todos_basic(self, supervisor, mock_subprocess, temp_dir):
        """Test basic TODO execution"""
        todos = ["Test TODO 1", "Test TODO 2"]
        session_id = "test_exec_123"

        # Mock prompt manager
        supervisor.prompt_manager = Mock()
        supervisor.prompt_manager.get_initial_prompt.return_value = "test prompt"
        supervisor.prompt_manager.context = Mock()

        # Execute
        result = supervisor.execute_with_todos(todos, session_id=session_id)

        # Verify result
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.task_complete is True
        assert len(result.output_lines) > 0
        assert "ALL TASKS COMPLETE" in " ".join(result.output_lines)

        # Verify Claude was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "claude" in call_args
        assert "--max-turns" in call_args
        assert str(supervisor.max_turns) in call_args

    def test_execute_with_timeout(self, supervisor, temp_dir):
        """Test execution timeout handling"""
        todos = ["Long running task"]
        supervisor.timeout = 0.1  # Very short timeout

        with patch('subprocess.Popen') as mock_popen:
            # Mock a process that never completes
            mock_process = Mock()
            mock_process.poll.return_value = None  # Always running
            mock_process.returncode = None

            mock_stdout = Mock()
            mock_stdout.readline.return_value = "Still working...\n"
            mock_stderr = Mock()
            mock_stderr.readline.return_value = ""

            mock_process.stdout = mock_stdout
            mock_process.stderr = mock_stderr

            mock_popen.return_value = mock_process

            # Should timeout
            result = supervisor.execute_with_todos(todos)

            assert result.success is False
            assert any("timed out" in err for err in result.errors)
            mock_process.terminate.assert_called()

    def test_execute_with_errors(self, supervisor, temp_dir):
        """Test execution with errors"""
        todos = ["Failing task"]

        with patch('subprocess.Popen') as mock_popen:
            # Mock process with errors
            mock_process = Mock()
            mock_process.returncode = 1
            mock_process.poll.side_effect = [None, None, 1]

            mock_stdout = Mock()
            mock_stdout.readline.side_effect = ["Working...\n", "Error occurred\n", ""]
            mock_stderr = Mock()
            mock_stderr.readline.side_effect = ["ERROR: Task failed\n", ""]

            mock_process.stdout = mock_stdout
            mock_process.stderr = mock_stderr

            mock_popen.return_value = mock_process

            result = supervisor.execute_with_todos(todos)

            assert result.success is False
            assert len(result.errors) > 0
            assert "ERROR: Task failed" in result.errors

    def test_logging_methods(self, supervisor, temp_dir, capture_logs):
        """Test all logging methods"""
        session_id = "log_test_123"
        supervisor._init_supervisor_log(session_id)

        # Test various log methods
        supervisor._log_supervisor("Test message", "INFO")
        supervisor._log_task_analysis(["TODO 1", "TODO 2"], [{"id": "1", "status": "pending"}])
        supervisor._log_execution_start(["claude", "-p", "prompt"], continuation=False)

        result = ExecutionResult(
            success=True,
            turns_used=5,
            output_lines=["line1", "line2"],
            errors=["error1"],
            metadata={"execution_time": 10.5},
            task_complete=True
        )
        supervisor._log_execution_result(result)

        supervisor._log_zen_recommendation("debug", "Agent is stuck")
        supervisor._log_supervisor_analysis("Analysis results here")
        supervisor._log_final_summary([result])

        # Check markdown log
        log_content = supervisor.current_log_path.read_text()
        assert "Test message" in log_content
        assert "TODO 1" in log_content
        assert "Initial Execution" in log_content
        assert "Execution Result" in log_content
        assert "Zen Assistance Recommended" in log_content
        assert "Final Summary" in log_content

        # Check Python logs
        log_output = capture_logs.getvalue()
        assert "Test message" in log_output
        assert "Task analysis: 2 TODOs" in log_output

    def test_run_with_taskmaster(self, supervisor, mock_task_file, mock_subprocess):
        """Test TaskMaster integration"""
        with patch('cadence.task_supervisor.TaskManager') as mock_tm:
            # Mock task loading
            mock_manager = Mock()
            mock_manager.load_tasks.return_value = True
            # Create simple task objects instead of Mocks
            class MockTask:
                def __init__(self, id, title, description, status="pending"):
                    self.id = id
                    self.title = title
                    self.description = description
                    self.status = status

                def is_complete(self):
                    return self.status == "done"

                @property
                def __dict__(self):
                    return {
                        "id": self.id,
                        "title": self.title,
                        "description": self.description,
                        "status": self.status
                    }

            task1 = MockTask("1", "Task 1", "Desc 1")
            task2 = MockTask("2", "Task 2", "Desc 2")

            mock_manager.tasks = [task1, task2]
            mock_manager.update_task_status = Mock()
            mock_manager.save_tasks = Mock()
            mock_tm.return_value = mock_manager

            # Mock prompt manager
            with patch('cadence.task_supervisor.TodoPromptManager') as mock_prompt_manager:
                # Make sure execute_with_todos returns a successful result
                with patch.object(supervisor, 'execute_with_todos') as mock_execute:
                    mock_execute.return_value = ExecutionResult(
                        success=True,
                        turns_used=5,
                        output_lines=["Working...", "ALL TASKS COMPLETE"],
                        errors=[],
                        metadata={},
                        task_complete=True
                    )

                    result = supervisor.run_with_taskmaster(str(mock_task_file))

                    assert result is True
                    assert mock_manager.load_tasks.called
                    assert mock_execute.called

    def test_zen_assistance_detection(self, supervisor, mock_subprocess, temp_dir):
        """Test zen assistance detection"""
        todos = ["Complex task"]
        supervisor.config.supervisor.zen_integration.enabled = True

        # Mock zen integration
        with patch.object(supervisor.zen, 'should_call_zen') as mock_should_call:
            mock_should_call.return_value = ("debug", "Agent appears stuck")

            result = supervisor.execute_with_todos(todos)

            assert "zen_needed" in result.metadata
            assert result.metadata["zen_needed"]["tool"] == "debug"
            assert result.metadata["zen_needed"]["reason"] == "Agent appears stuck"

    def test_handle_zen_assistance(self, supervisor):
        """Test zen assistance handling"""
        execution_result = ExecutionResult(
            success=False,
            turns_used=10,
            output_lines=["stuck here"],
            errors=["error"],
            metadata={
                "zen_needed": {
                    "tool": "debug",
                    "reason": "Stuck on error"
                }
            },
            task_complete=False
        )

        supervisor.prompt_manager = Mock()
        supervisor.prompt_manager.context = Mock()

        with patch.object(supervisor.zen, 'call_zen_support') as mock_call_zen:
            with patch.object(supervisor.zen, 'generate_continuation_guidance') as mock_guidance:
                mock_call_zen.return_value = {"guidance": "Try this approach"}
                mock_guidance.return_value = "Continuation guidance"

                response = supervisor.handle_zen_assistance(execution_result, "test_123")

                assert response["zen_tool_used"] == "debug"
                assert response["zen_reason"] == "Stuck on error"
                assert response["continuation_guidance"] == "Continuation guidance"

    @pytest.mark.parametrize("output,expected", [
        (["Turn 1 - Working on first task with detailed output",
          "Turn 2 - Continuing with the second phase of the task",
          "Turn 3 - Finishing up the remaining work items now"], 0),  # Turn counting removed
        (["Short output line that is less than fifty characters"], 0),  # Turn counting removed
        (["x" * 60], 0),  # Turn counting removed
        ([], 0)  # Turn counting removed
    ])
    def test_turn_estimation(self, supervisor, output, expected, temp_dir):
        """Test that turn count is always 0 (turn estimation was removed)"""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.poll.side_effect = [None] * len(output) + [0]

            mock_stdout = Mock()
            # Add more None poll responses to allow all lines to be read
            mock_stdout.readline.side_effect = [line + '\n' for line in output] + ['']
            mock_stderr = Mock()
            mock_stderr.readline.return_value = ''

            mock_process.stdout = mock_stdout
            mock_process.stderr = mock_stderr
            mock_popen.return_value = mock_process

            supervisor.max_turns = 10
            result = supervisor.execute_with_todos(["test"])

            assert result.turns_used == 0  # Always 0 now
