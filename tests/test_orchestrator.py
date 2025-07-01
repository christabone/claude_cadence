import pytest
import json
import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call

from cadence.orchestrator import SupervisorDecision, SupervisorOrchestrator
from cadence.unified_agent import AgentResult
from cadence.retry_utils import RetryError

# A mock config class to simulate the nested structure of the real config
class MockConfig:
    def __init__(self, config_dict):
        self._config = config_dict

    def __getattr__(self, name):
        value = self._config.get(name)
        if isinstance(value, dict):
            return MockConfig(value)
        return value

    def get(self, key, default=None):
        # Handle nested gets like 'agent.defaults'
        if '.' in key:
            keys = key.split('.')
            val = self._config
            for k in keys:
                val = val.get(k)
                if val is None:
                    return default
            return val
        return self._config.get(key, default)

# Default config for tests
@pytest.fixture
def default_config_dict():
    """Provides the default configuration as a dictionary for testing."""
    return {
        "supervisor": {
            "model": "test-supervisor-model",
            "use_continue": True,
            "tools": ["tool1", "tool2"],
            "zen_integration": {
                "code_review_frequency": "task"
            }
        },
        "agent": {
            "defaults": {
                "model": "test-agent-model",
                "tools": ["agent_tool1"],
                "retry_count": 2,
                "use_continue": False,
                "timeout_seconds": 60,
                "temperature": 0.1
            }
        },
        "execution": {
            "log_level": "DEBUG",
            "max_supervisor_turns": 5,
            "max_agent_turns": 10,
            "timeout": 120,
            "clean_logs_on_startup": False,
            "max_log_size_mb": 20
        },
        "orchestration": {
            "max_iterations": 3,
            "quick_quit_seconds": 2,
            "cleanup_keep_sessions": 5
        },
        "retry_behavior": {
            "max_json_retries": 2,
            "base_delay": 0.1
        },
        "file_patterns": {
            "agent_result_file": "agent_result_{session_id}.json"
        },
        "integrations": {
            "mcp": {
                "supervisor_servers": ["taskmaster-ai", "zen"],
                "agent_servers": ["serena"]
            }
        },
        "zen_processing_config": {
            "primary_review_model": "zen-primary-model",
            "secondary_review_model": "zen-secondary-model",
            "debug_model": "zen-debug-model",
            "analyze_model": "zen-analyze-model"
        },
        "scratchpad_retry": {
            "max_retries": 2,
            "allowed_tools": "Write,Read",
            "max_turns": 3
        }
    }

@pytest.fixture
def orchestrator(tmp_path, default_config_dict):
    """Fixture for a SupervisorOrchestrator instance with mocked dependencies."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()

    mock_config_loader = MagicMock()
    mock_config_loader.config = MockConfig(default_config_dict)

    mock_prompt_loader = MagicMock()
    mock_prompt_loader.get_template.return_value = "template: {placeholder}"
    mock_prompt_loader.format_template.side_effect = lambda template, context: template.format(**context) if isinstance(template, str) else ""

    with patch('cadence.orchestrator.ConfigLoader', return_value=mock_config_loader), \
         patch('cadence.orchestrator.PromptLoader', return_value=mock_prompt_loader):

        orch = SupervisorOrchestrator(project_root=project_root)
        # Replace generated session ID for deterministic tests
        orch.generate_session_id = lambda: "test_session_001"
        orch.current_session_id = "test_session_001"
        yield orch


class TestSupervisorDecision:
    """Tests for the SupervisorDecision dataclass."""

    def test_creation_execute_action(self):
        """
        Tests basic creation of a SupervisorDecision with the 'execute' action.
        """
        decision = SupervisorDecision(
            action="execute",
            task_id="1",
            task_title="Test Task",
            subtasks=[{"id": "1.1", "title": "Subtask 1", "description": "Do something"}],
            project_path="/path/to/project",
            guidance="Some guidance",
            session_id="sess_123",
            reason="Time to work"
        )
        assert decision.action == "execute"
        assert decision.task_id == "1"
        assert decision.guidance == "Some guidance"
        assert len(decision.subtasks) == 1

    def test_creation_skip_action(self):
        """
        Tests basic creation of a SupervisorDecision with the 'skip' action.
        """
        decision = SupervisorDecision(
            action="skip",
            session_id="sess_123",
            reason="Nothing to do"
        )
        assert decision.action == "skip"
        assert decision.reason == "Nothing to do"
        assert decision.subtasks is None

    def test_creation_with_all_fields(self):
        """
        Tests creation with all possible fields, including code review fields.
        """
        decision = SupervisorDecision(
            action="execute",
            task_id="2",
            task_title="Review Task",
            subtasks=[],
            project_path="/path/to/project",
            guidance="Review this code",
            session_id="sess_456",
            reason="Code review needed",
            execution_time=1.23,
            quit_too_quickly=False,
            review_scope="task",
            files_to_review=["file1.py"],
            supervisor_findings="Found an issue.",
            code_review_has_critical_or_high_issues=True
        )
        assert decision.code_review_has_critical_or_high_issues is True
        assert decision.review_scope == "task"
        assert decision.execution_time == 1.23


class TestSupervisorOrchestrator:
    """Tests for the SupervisorOrchestrator class."""

    def test_init(self, orchestrator, tmp_path):
        """
        Verifies that the orchestrator initializes correctly, creating necessary
        directories and loading configuration.
        """
        project_root = tmp_path / "test_project"
        assert orchestrator.project_root == project_root
        assert orchestrator.supervisor_dir.exists() and orchestrator.supervisor_dir.is_dir()
        assert orchestrator.agent_dir.exists() and orchestrator.agent_dir.is_dir()
        assert orchestrator.state["session_count"] == 0
        assert orchestrator.cadence_config.supervisor.model == "test-supervisor-model"

    def test_init_with_default_task_file(self, tmp_path):
        """
        Verifies that if no task_file is provided, it defaults to the
        correct path within the project root.
        """
        project_root = tmp_path / "test_project_2"
        project_root.mkdir()

        with patch('cadence.orchestrator.ConfigLoader'), patch('cadence.orchestrator.PromptLoader'):
            orch = SupervisorOrchestrator(project_root=project_root)
            expected_task_file = project_root / ".taskmaster" / "tasks" / "tasks.json"
            assert orch.task_file == expected_task_file

    def test_validate_path_security(self, orchestrator):
        """
        Ensures validate_path prevents path traversal attacks.
        """
        base_dir = orchestrator.project_root

        # Test directory traversal
        with pytest.raises(ValueError, match="is outside allowed directory"):
            orchestrator.validate_path(base_dir / "../../../etc/passwd", base_dir)

        # Test absolute path outside base
        with pytest.raises(ValueError, match="is outside allowed directory"):
            orchestrator.validate_path(Path("/etc/hosts"), base_dir)

    def test_validate_path_valid(self, orchestrator):
        """
        Ensures validate_path allows valid paths within the base directory.
        """
        base_dir = orchestrator.project_root
        valid_path = base_dir / "subdir" / "file.txt"

        # We don't need the file to exist, just for the path to be valid
        resolved = orchestrator.validate_path(valid_path, base_dir)
        assert resolved == valid_path.resolve()

    def test_state_management(self, orchestrator):
        """
        Tests saving and loading of the orchestrator's state.
        """
        assert orchestrator.state["session_count"] == 0

        orchestrator.state["session_count"] = 1
        orchestrator.state["last_session_id"] = "test_session_001"
        orchestrator.save_state()

        assert orchestrator.state_file.exists()

        # Create a new orchestrator to test loading the saved state
        with patch('cadence.orchestrator.ConfigLoader'), patch('cadence.orchestrator.PromptLoader'):
            new_orch = SupervisorOrchestrator(project_root=orchestrator.project_root)
            assert new_orch.state["session_count"] == 1
            assert new_orch.state["last_session_id"] == "test_session_001"

    @patch.object(SupervisorOrchestrator, '_run_claude_with_json_retry')
    def test_run_supervisor_analysis_happy_path_execute(self, mock_run_retry, orchestrator):
        """
        Tests the happy path for run_supervisor_analysis where it decides to 'execute'.
        """
        decision_data = {
            "action": "execute",
            "task_id": "1",
            "task_title": "Implement Feature X",
            "subtasks": [{"id": "1.1", "title": "Subtask 1", "description": "Do it"}],
            "project_root": str(orchestrator.project_root),
            "guidance": "Focus on performance.",
            "session_id": "test_session_001",
            "reason": "Ready to start work."
        }
        mock_run_retry.return_value = decision_data

        decision = orchestrator.run_supervisor_analysis("test_session_001", use_continue=False, iteration=1)

        assert isinstance(decision, SupervisorDecision)
        assert decision.action == "execute"
        assert decision.task_id == "1"
        assert decision.guidance == "Focus on performance."
        assert decision.quit_too_quickly is False

    @patch.object(SupervisorOrchestrator, '_run_claude_with_json_retry')
    def test_run_supervisor_analysis_task_id_correction(self, mock_run_retry, orchestrator, caplog):
        """
        Tests that a task_id with a subtask notation (e.g., '1.1') is corrected
        to the main task ID ('1').
        """
        decision_data = {
            "action": "execute",
            "task_id": "1.1", # Incorrect format
            "task_title": "Implement Feature X",
            "subtasks": [],
            "project_root": str(orchestrator.project_root),
            "guidance": "Some guidance",
            "session_id": "test_session_001",
            "reason": "Starting subtask."
        }
        mock_run_retry.return_value = decision_data

        decision = orchestrator.run_supervisor_analysis("test_session_001", use_continue=False, iteration=1)

        assert decision.task_id == "1"
        assert "Supervisor provided subtask ID '1.1' in task_id field. Correcting to main task ID '1'." in caplog.text

    @patch.object(SupervisorOrchestrator, '_run_claude_with_json_retry')
    def test_run_supervisor_analysis_json_missing_fields(self, mock_run_retry, orchestrator):
        """
        Tests that an error is raised if the supervisor's JSON response is
        missing required fields.
        """
        # Missing 'task_id' for an 'execute' action
        decision_data = {
            "action": "execute",
            "task_title": "Implement Feature X",
            "subtasks": [],
            "project_root": str(orchestrator.project_root),
            "guidance": "Some guidance",
            "session_id": "test_session_001",
            "reason": "Ready to start work."
        }

        # The retry logic will catch the parsing error. We simulate the final failure.
        mock_run_retry.side_effect = RetryError("Failed after retries: Decision JSON missing required fields: ['task_id']")

        with pytest.raises(RuntimeError, match="Decision JSON missing required fields:.*task_id"):
            orchestrator.run_supervisor_analysis("test_session_001", use_continue=False, iteration=1)

    @patch.object(SupervisorOrchestrator, '_run_async_safely', new_callable=AsyncMock)
    def test_run_supervisor_analysis_parses_last_json(self, mock_run_async, orchestrator):
        """
        Ensures that if the supervisor output contains multiple JSON objects,
        the last valid decision object is used.
        """
        # The output contains stream-json lines. The parser should extract the text
        # content and find JSON objects within it.
        mock_output = [
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Thinking..."}]}}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Here is an old decision:\\n{\\n  \\"action\\": \\"skip\\", \\"session_id\\": \\"test_session_001\\", \\"reason\\": \\"old decision\\"\\n}"}]}}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Wait, I changed my mind. Here is the final decision:\\n{\\n  \\"action\\": \\"execute\\", \\"task_id\\": \\"1\\", \\"task_title\\": \\"Final Task\\", \\"subtasks\\": [], \\"project_root\\": \\"path\\", \\"session_id\\": \\"test_session_001\\", \\"reason\\": \\"final decision\\"\\n}"}]}}',
            '{"type": "result", "duration_ms": 1000}'
        ]
        mock_run_async.return_value = (0, mock_output)

        decision = orchestrator.run_supervisor_analysis("test_session_001", use_continue=False, iteration=1)

        assert decision.action == "execute"
        assert decision.reason == "final decision"
        assert decision.task_id == "1"

    def test_cleanup_old_sessions(self, orchestrator):
        """
        Tests the cleanup logic for old session files and directories.
        """
        log_dir = orchestrator.project_root / ".cadence" / "logs"

        # Create files and dirs for 6 old sessions
        session_ids = [f"20230101_0{i}0000_abcdef1{i}" for i in range(6)]
        for sid in session_ids:
            # Create files in supervisor dir
            (orchestrator.supervisor_dir / f"decision_snapshot_{sid}.json").touch()
            # Create log directory
            (log_dir / sid).mkdir(parents=True)
            (log_dir / sid / "orchestrator.log").touch()

        # Config keeps 5 sessions, so the oldest one should be removed.
        orchestrator.cleanup_old_sessions(keep_last_n=5)

        oldest_sid = "20230101_000000_abcdef10"
        newest_sid = "20230101_050000_abcdef15"

        assert not (orchestrator.supervisor_dir / f"decision_snapshot_{oldest_sid}.json").exists()
        assert not (log_dir / oldest_sid).exists()
        assert (orchestrator.supervisor_dir / f"decision_snapshot_{newest_sid}.json").exists()
        assert (log_dir / newest_sid).exists()

    async def _run_in_loop(self, orchestrator):
        """Helper coroutine to test calling a sync method from an async context."""
        with pytest.raises(RuntimeError, match="Cannot call blocking async wrapper"):
            async def dummy_coro():
                await asyncio.sleep(0)
            # This call is the anti-pattern _run_async_safely is designed to prevent.
            orchestrator._run_async_safely(dummy_coro())

    def test_run_async_safely_in_loop_raises_error(self, orchestrator):
        """
        Verifies that _run_async_safely raises a RuntimeError if called from
        within a running event loop, preventing deadlocks.
        """
        asyncio.run(self._run_in_loop(orchestrator))

    @patch('cadence.orchestrator.UnifiedAgent')
    def test_run_agent(self, MockUnifiedAgent, orchestrator):
        """
        Tests that the run_agent method correctly initializes and executes
        the UnifiedAgent with the right parameters.
        """
        mock_agent_instance = MockUnifiedAgent.return_value
        mock_agent_instance.execute.return_value = AgentResult(
            success=True,
            session_id="test_session_001",
            execution_time=10.5,
            completed_normally=True,
            requested_help=False
        )

        todos = ["Implement the thing."]
        guidance = "Be careful."

        result = orchestrator.run_agent(
            todos=todos,
            guidance=guidance,
            session_id="test_session_001",
            use_continue=False,
            task_id="3",
            subtasks=[{"id": "3.1", "title": "sub", "description": "desc"}],
            project_root=str(orchestrator.project_root)
        )

        # Verify UnifiedAgent was initialized correctly
        MockUnifiedAgent.assert_called_once()
        init_kwargs = MockUnifiedAgent.call_args.kwargs
        assert init_kwargs['working_dir'] == orchestrator.agent_dir
        assert init_kwargs['session_id'] == "test_session_001"

        # Verify execute was called correctly
        mock_agent_instance.execute.assert_called_once()
        execute_kwargs = mock_agent_instance.execute.call_args.kwargs
        assert "Implement the thing." in execute_kwargs['prompt']
        assert "Be careful." in execute_kwargs['prompt']
        assert execute_kwargs['continue_session'] is False
        assert execute_kwargs['context']['task_id'] == "3"

        # Verify result is passed through
        assert result.success is True
        assert result.execution_time == 10.5
