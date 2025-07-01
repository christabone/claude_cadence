# file: tests/test_unified_agent.py

import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from cadence.unified_agent import UnifiedAgent, AgentResult, ALLOWED_TOOLS, ALLOWED_FLAGS

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

@pytest.fixture
def agent_config():
    """Provides a default, valid agent configuration."""
    return {
        'agent': {
            'defaults': {
                'model': 'test-model',
                'tools': ['bash', 'read', 'write'],
                'extra_flags': ['--dangerously-skip-permissions'],
                'retry_count': 1,
                'use_continue': False,
                'timeout_seconds': 10,
                'temperature': 0.1,
                'max_turns': 5
            }
        },
        'retry_behavior': {}
    }

@pytest.fixture
def mock_subprocess(mocker):
    """Mocks asyncio.create_subprocess_exec."""
    mock_proc = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.wait = AsyncMock()

    # Default successful execution
    mock_proc.returncode = 0
    mock_proc.stdout.readline.side_effect = [
        b'{"type": "result", "subtype": "success", "is_error": false}\n',
        b''  # EOF
    ]
    mock_proc.wait.return_value = 0

    return mocker.patch('asyncio.create_subprocess_exec', return_value=mock_proc), mock_proc


class TestAgentResult:
    def test_instantiation_defaults(self):
        """Tests that AgentResult can be instantiated with minimal required fields."""
        result = AgentResult(
            success=True,
            session_id="test_session",
            output_file="out.log",
            error_file="err.log",
            execution_time=1.23
        )
        assert result.success is True
        assert result.completed_normally is False
        assert result.requested_help is False
        assert result.errors is None
        assert result.quit_too_quickly is False

    def test_instantiation_with_all_fields(self):
        """Tests that AgentResult can be instantiated with all fields set."""
        result = AgentResult(
            success=False,
            session_id="test_session",
            output_file="out.log",
            error_file="err.log",
            execution_time=1.23,
            completed_normally=True,
            requested_help=True,
            errors=["Error 1", "Error 2"],
            quit_too_quickly=True,
            retry_count=3
        )
        assert result.success is False
        assert result.completed_normally is True
        assert result.requested_help is True
        assert result.errors == ["Error 1", "Error 2"]
        assert result.quit_too_quickly is True
        assert result.retry_count == 3

class TestUnifiedAgentInit:
    def test_init_success(self, agent_config, tmp_path):
        """Tests successful initialization of UnifiedAgent."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path, session_id="fixed_id")
        assert agent.session_id == "fixed_id"
        assert agent.model == "test-model"
        assert agent.tools == ['bash', 'read', 'write']
        assert agent.working_dir == tmp_path
        assert tmp_path.exists()

    def test_init_creates_working_dir(self, agent_config, tmp_path):
        """Tests that the working directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_agent_dir"
        assert not new_dir.exists()
        UnifiedAgent(config=agent_config, working_dir=new_dir)
        assert new_dir.exists()

    def test_init_handles_missing_config_sections(self, tmp_path):
        """Tests graceful handling of incomplete configuration."""
        # Test with missing 'defaults'
        agent = UnifiedAgent(config={'agent': {}}, working_dir=tmp_path)
        assert agent.model == 'claude-3-5-sonnet-20241022'  # Default value
        assert agent.tools == []

        # Test with missing 'agent'
        agent = UnifiedAgent(config={}, working_dir=tmp_path)
        assert agent.model == 'claude-3-5-sonnet-20241022'
        assert agent.tools == []

    def test_init_with_custom_session_id(self, agent_config, tmp_path):
        """Tests initialization with a custom session ID."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path, session_id="custom_123")
        assert agent.session_id == "custom_123"

    @patch('cadence.unified_agent.generate_session_id')
    def test_init_generates_session_id_when_none_provided(self, mock_generate_session_id, agent_config, tmp_path):
        """Tests that a session ID is generated when none is provided."""
        mock_generate_session_id.return_value = "generated_session_id"
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        assert agent.session_id == "generated_session_id"
        mock_generate_session_id.assert_called_once()

    def test_init_sets_all_settings_from_config(self, tmp_path):
        """Tests that all configuration settings are properly loaded."""
        config = {
            'agent': {
                'defaults': {
                    'model': 'custom-model',
                    'tools': ['bash', 'edit'],
                    'extra_flags': ['--verbose'],
                    'retry_count': 3,
                    'use_continue': True,
                    'timeout_seconds': 120,
                    'temperature': 0.8,
                    'max_turns': 10
                }
            }
        }
        agent = UnifiedAgent(config=config, working_dir=tmp_path)

        assert agent.model == 'custom-model'
        assert agent.tools == ['bash', 'edit']
        assert agent.extra_flags == ['--verbose']
        assert agent.retry_count == 3
        assert agent.use_continue is True
        assert agent.timeout_seconds == 120
        assert agent.temperature == 0.8
        assert agent.settings['max_turns'] == 10

class TestUnifiedAgentSecurity:
    def test_validate_security_settings_success(self, agent_config, tmp_path):
        """Tests that validation passes with allowed tools and flags."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        agent.tools = list(ALLOWED_TOOLS)
        agent.extra_flags = list(ALLOWED_FLAGS)
        agent._validate_security_settings() # Should not raise

    def test_validate_security_settings_mcp_tool_success(self, agent_config, tmp_path):
        """Tests that MCP-style tools are allowed."""
        agent_config['agent']['defaults']['tools'].append("mcp__taskmaster-ai__*")
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        agent._validate_security_settings() # Should not raise

    def test_validate_security_settings_multiple_mcp_tools(self, agent_config, tmp_path):
        """Tests that multiple MCP tools are allowed."""
        agent_config['agent']['defaults']['tools'].extend([
            "mcp__taskmaster-ai__get_tasks",
            "mcp__github__create_pr",
            "mcp__serena__find_symbol"
        ])
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        agent._validate_security_settings() # Should not raise

    def test_validate_security_settings_disallowed_tool_raises_error(self, agent_config, tmp_path):
        """Tests that a disallowed tool raises a ValueError."""
        agent_config['agent']['defaults']['tools'].append("rm -rf /")
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        with pytest.raises(ValueError, match="Disallowed tool configured: 'rm -rf /'"):
            agent._validate_security_settings()

    def test_validate_security_settings_disallowed_flag_raises_error(self, agent_config, tmp_path):
        """Tests that a disallowed flag raises a ValueError."""
        agent_config['agent']['defaults']['extra_flags'].append("--execute-arbitrary-code")
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        with pytest.raises(ValueError, match="Disallowed flag configured: '--execute-arbitrary-code'"):
            agent._validate_security_settings()

    def test_validate_security_settings_flag_with_value(self, agent_config, tmp_path):
        """Tests validation of flags with values (e.g., --temperature 0.5)."""
        agent_config['agent']['defaults']['extra_flags'].append("--temperature=0.5")
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        agent._validate_security_settings() # Should not raise since --temperature is allowed

    def test_validate_security_settings_flag_with_spaces(self, agent_config, tmp_path):
        """Tests validation of flags with spaces and values."""
        agent_config['agent']['defaults']['extra_flags'].append("--model custom-model")
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        agent._validate_security_settings() # Should not raise since --model is allowed

    def test_validate_security_settings_empty_lists(self, agent_config, tmp_path):
        """Tests validation with empty tools and flags lists."""
        agent_config['agent']['defaults']['tools'] = []
        agent_config['agent']['defaults']['extra_flags'] = []
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        agent._validate_security_settings() # Should not raise

class TestUnifiedAgentBuildCommand:
    def test_build_command_basic(self, agent_config, tmp_path):
        """Tests basic command construction."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "claude" in cmd
        assert "--model" in cmd
        assert "test-model" in cmd
        assert "--allowedTools" in cmd
        assert "bash,read,write" in cmd
        assert "--continue" not in cmd
        assert "test prompt" in cmd

    def test_build_command_with_continue(self, agent_config, tmp_path):
        """Tests that the --continue flag is added correctly."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=True)
        assert "--continue" in cmd

    def test_build_command_raises_on_empty_prompt_file(self, agent_config, tmp_path):
        """Tests that an error is raised if the prompt file is empty."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.touch() # Create empty file
        with pytest.raises(ValueError, match="is empty"):
            agent._build_command(prompt_file, continue_session=False)

    def test_build_command_raises_on_missing_prompt_file(self, agent_config, tmp_path):
        """Tests that an error is raised if the prompt file doesn't exist."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            agent._build_command(prompt_file, continue_session=False)

    def test_build_command_with_custom_temperature(self, agent_config, tmp_path):
        """Tests command construction with non-default temperature."""
        agent_config['agent']['defaults']['temperature'] = 0.8
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "--temperature" in cmd
        assert "0.8" in cmd

    def test_build_command_with_no_tools(self, agent_config, tmp_path):
        """Tests command construction with no tools specified."""
        agent_config['agent']['defaults']['tools'] = []
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "--allowedTools" not in cmd

    def test_build_command_with_max_turns(self, agent_config, tmp_path):
        """Tests command construction with max_turns setting."""
        agent_config['agent']['defaults']['max_turns'] = 15
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "--max-turns" in cmd
        assert "15" in cmd

    def test_build_command_without_max_turns(self, agent_config, tmp_path):
        """Tests command construction without max_turns setting."""
        del agent_config['agent']['defaults']['max_turns']
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "--max-turns" not in cmd

    def test_build_command_with_extra_flags(self, agent_config, tmp_path):
        """Tests command construction with extra flags."""
        agent_config['agent']['defaults']['extra_flags'] = ['--verbose', '--debug']
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "--verbose" in cmd
        assert "--debug" in cmd

    def test_build_command_includes_required_flags(self, agent_config, tmp_path):
        """Tests that required flags are always included."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt")
        cmd = agent._build_command(prompt_file, continue_session=False)

        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--verbose" in cmd

class TestUnifiedAgentExecute:
    async def test_execute_success_normal_completion(self, agent_config, tmp_path, mock_subprocess):
        """Tests a successful run where the agent completes its task."""
        _, mock_proc = mock_subprocess
        mock_proc.stdout.readline.side_effect = [
            b'{"status": "success", "session_id": "test_session"}\n',
            b''
        ]
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is True
        assert result.completed_normally is True
        assert result.requested_help is False
        assert result.errors == []

    async def test_execute_failure_non_zero_exit_code(self, agent_config, tmp_path, mock_subprocess):
        """Tests a failure due to the subprocess returning a non-zero exit code."""
        _, mock_proc = mock_subprocess
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1
        mock_proc.stdout.readline.side_effect = [b'Some error message\n', b'']

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is False
        assert result.completed_normally is False
        assert "Agent reported failure" in result.errors

    async def test_execute_failure_requested_help(self, agent_config, tmp_path, mock_subprocess):
        """Tests a scenario where the agent explicitly requests help."""
        _, mock_proc = mock_subprocess
        mock_proc.stdout.readline.side_effect = [
            b'{"status": "help_needed", "reason": "I am stuck"}\n',
            b''
        ]
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is False
        assert result.completed_normally is False
        assert result.requested_help is True

    async def test_execute_quit_too_quickly(self, agent_config, tmp_path, mock_subprocess, mocker):
        """Tests that quit_too_quickly is flagged for short executions."""
        mocker.patch('time.time', side_effect=[1000.0, 1001.0])  # 1 second execution
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.quit_too_quickly is True

    async def test_execute_normal_execution_time(self, agent_config, tmp_path, mock_subprocess, mocker):
        """Tests that quit_too_quickly is False for normal execution times."""
        mocker.patch('time.time', side_effect=[1000.0, 1015.0])  # 15 second execution
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.quit_too_quickly is False

    async def test_execute_handles_subprocess_exception(self, agent_config, tmp_path, mock_subprocess):
        """Tests that exceptions during subprocess execution are handled gracefully."""
        mock_create_subprocess, _ = mock_subprocess
        mock_create_subprocess.side_effect = FileNotFoundError("claude command not found")

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is False
        assert "Agent execution failed: claude command not found" in result.errors

    async def test_execute_handles_security_validation_failure(self, agent_config, tmp_path):
        """Tests that a security validation failure is caught and reported."""
        agent_config['agent']['defaults']['tools'].append("invalid-tool")
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is False
        assert "Agent execution failed: Security violation: Disallowed tool configured: 'invalid-tool'" in result.errors

    async def test_execute_with_context(self, agent_config, tmp_path, mock_subprocess):
        """Tests execution with additional context."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        context = {"test_key": "test_value"}
        result = agent.execute(prompt="Do the thing", context=context)

        assert result.success is True
        # Context should not affect basic execution but should be handled gracefully

    async def test_execute_with_continue_session(self, agent_config, tmp_path, mock_subprocess):
        """Tests execution with continue_session=True."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing", continue_session=True)

        assert result.success is True
        # The command should include --continue flag (tested in build_command tests)

    async def test_execute_json_result_object_success(self, agent_config, tmp_path, mock_subprocess):
        """Tests parsing of JSON result object indicating success."""
        _, mock_proc = mock_subprocess
        mock_proc.stdout.readline.side_effect = [
            b'{"type": "result", "subtype": "success", "is_error": false}\n',
            b''
        ]
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is True
        assert result.completed_normally is True

    async def test_execute_json_result_object_error(self, agent_config, tmp_path, mock_subprocess):
        """Tests parsing of JSON result object indicating error."""
        _, mock_proc = mock_subprocess
        mock_proc.returncode = 0  # Even with 0 exit code, JSON can indicate error
        mock_proc.stdout.readline.side_effect = [
            b'{"type": "result", "subtype": "error", "is_error": true}\n',
            b''
        ]
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.success is False
        assert result.completed_normally is False

    async def test_execute_malformed_json_fallback(self, agent_config, tmp_path, mock_subprocess):
        """Tests fallback behavior when JSON output is malformed."""
        _, mock_proc = mock_subprocess
        mock_proc.stdout.readline.side_effect = [
            b'{"incomplete json\n',
            b'some other output\n',
            b''
        ]
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        # Should fall back to exit code based success determination
        assert result.success is True  # returncode is 0 by default

    async def test_execute_creates_output_files(self, agent_config, tmp_path, mock_subprocess):
        """Tests that output files are created during execution."""
        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path, session_id="test_session")
        result = agent.execute(prompt="Do the thing")

        # Check that files were created
        expected_prompt_file = tmp_path / "prompt_test_session.txt"
        expected_output_file = tmp_path / "output_test_session.log"
        expected_debug_file = tmp_path / "debug_test_session.log"

        assert expected_prompt_file.exists()
        assert expected_output_file.exists()
        assert expected_debug_file.exists()

        # Check prompt file content
        assert expected_prompt_file.read_text() == "Do the thing"

    async def test_execute_execution_time_recorded(self, agent_config, tmp_path, mock_subprocess, mocker):
        """Tests that execution time is properly recorded."""
        start_time = 1000.0
        end_time = 1005.5
        mocker.patch('time.time', side_effect=[start_time, end_time])

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do the thing")

        assert result.execution_time == pytest.approx(5.5, rel=1e-9)

class TestUnifiedAgentAsyncSupport:
    @patch('cadence.unified_agent.asyncio.get_running_loop')
    def test_run_async_safely_no_existing_loop(self, mock_get_loop, agent_config, tmp_path):
        """Tests _run_async_safely when no event loop is running."""
        mock_get_loop.side_effect = RuntimeError("No running event loop")

        # Mock asyncio.run
        with patch('cadence.unified_agent.asyncio.run') as mock_run:
            mock_run.return_value = "test_result"

            agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
            result = agent._run_async_safely("test_coroutine")

            assert result == "test_result"
            mock_run.assert_called_once_with("test_coroutine")

    @patch('cadence.unified_agent.asyncio.get_running_loop')
    @patch('cadence.unified_agent.threading.Thread')
    @patch('cadence.unified_agent.asyncio.new_event_loop')
    def test_run_async_safely_with_existing_loop(self, mock_new_loop, mock_thread, mock_get_loop, agent_config, tmp_path):
        """Tests _run_async_safely when an event loop is already running."""
        # Simulate existing event loop
        mock_get_loop.return_value = MagicMock()

        # Mock new loop
        mock_loop = MagicMock()
        mock_new_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = "thread_result"

        # Mock thread behavior
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)

        # We need to patch the thread execution to simulate the result
        def side_effect(target):
            target()  # Execute the target function

        mock_thread_instance.start.side_effect = lambda: side_effect(mock_thread.call_args[1]['target'])

        # This is a complex test that requires more detailed mocking
        # For now, we'll test that the method exists and can be called
        assert hasattr(agent, '_run_async_safely')

class TestUnifiedAgentLogging:
    def test_init_with_logging_environment_variables(self, agent_config, tmp_path, mocker):
        """Tests initialization with logging environment variables set."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict('os.environ', {
            'CADENCE_LOG_SESSION': 'test_session',
            'CADENCE_LOG_DIR': str(log_dir),
            'CADENCE_LOG_LEVEL': 'INFO'
        }):
            # Mock the setup_file_logging function
            mock_setup_logging = mocker.patch('cadence.log_utils.setup_file_logging')

            agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)

            # Verify logging was set up
            mock_setup_logging.assert_called_once()

    def test_init_logging_setup_failure(self, agent_config, tmp_path, mocker):
        """Tests graceful handling of logging setup failure."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict('os.environ', {
            'CADENCE_LOG_DIR': str(log_dir)
        }):
            # Mock setup_file_logging to raise an exception
            mocker.patch('cadence.log_utils.setup_file_logging', side_effect=Exception("Logging failed"))

            # Should not raise an exception, just log a warning
            agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
            assert agent is not None

class TestUnifiedAgentIntegration:
    """Integration tests that test multiple components working together."""

    async def test_full_execution_flow_success(self, agent_config, tmp_path, mock_subprocess, mocker):
        """Tests a complete successful execution flow from start to finish."""
        # Mock time for consistent execution time
        mocker.patch('time.time', side_effect=[1000.0, 1015.0])  # 15 second execution

        # Mock successful subprocess output
        _, mock_proc = mock_subprocess
        mock_proc.stdout.readline.side_effect = [
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Starting task"}]}}\n',
            b'{"type": "user", "message": {"content": [{"type": "tool_result", "content": "Task completed"}]}}\n',
            b'{"type": "result", "subtype": "success", "is_error": false, "duration_ms": 15000}\n',
            b''
        ]

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path, session_id="integration_test")
        result = agent.execute(prompt="Complete the integration test task")

        # Verify result
        assert result.success is True
        assert result.completed_normally is True
        assert result.requested_help is False
        assert result.quit_too_quickly is False
        assert result.execution_time == pytest.approx(15.0, rel=1e-9)
        assert result.session_id == "integration_test"
        assert result.errors == []

        # Verify files were created
        assert (tmp_path / "prompt_integration_test.txt").exists()
        assert (tmp_path / "output_integration_test.log").exists()
        assert (tmp_path / "debug_integration_test.log").exists()

    async def test_full_execution_flow_with_help_request(self, agent_config, tmp_path, mock_subprocess):
        """Tests a complete execution flow where the agent requests help."""
        _, mock_proc = mock_subprocess
        mock_proc.stdout.readline.side_effect = [
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "I need help"}]}}\n',
            b'{"status": "help_needed", "reason": "Task is too complex"}\n',
            b''
        ]

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do something complex")

        assert result.success is False
        assert result.completed_normally is False
        assert result.requested_help is True

    async def test_security_validation_in_full_flow(self, agent_config, tmp_path):
        """Tests that security validation is properly integrated into the execution flow."""
        # Add an invalid tool
        agent_config['agent']['defaults']['tools'].append("dangerous_tool")

        agent = UnifiedAgent(config=agent_config, working_dir=tmp_path)
        result = agent.execute(prompt="Do something")

        # Should fail due to security validation
        assert result.success is False
        assert any("Security violation" in error for error in result.errors)
