# tests/test_zen_integration.py

import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from cadence.config import ZenIntegrationConfig
from cadence.zen_integration import ZenIntegration, SCRATCHPAD_DIR

# Constants from the module, redefined here for clarity in tests
CUTOFF_INDICATOR_THRESHOLD = 3
AUTO_DEBUG_ERROR_THRESHOLD = 3


@pytest.fixture
def zen_config() -> ZenIntegrationConfig:
    """Provides a default ZenIntegrationConfig for tests."""
    return ZenIntegrationConfig(
        enabled=True,
        stuck_detection=True,
        auto_debug_threshold=AUTO_DEBUG_ERROR_THRESHOLD,
        cutoff_detection=True,
        validate_on_complete=[
            "*security*", "*payment*"
        ]
    )


@pytest.fixture
def zen_integration(zen_config: ZenIntegrationConfig) -> ZenIntegration:
    """Provides a ZenIntegration instance with a default config."""
    return ZenIntegration(config=zen_config, verbose=False)


class TestShouldCallZen:
    """Tests for the main decision-making method `should_call_zen`."""

    def test_should_call_zen__when_disabled__returns_none(self, zen_config: ZenIntegrationConfig):
        """Ensures no tool is called if the integration is disabled."""
        # Arrange
        zen_config.enabled = False
        integration = ZenIntegration(config=zen_config)
        mock_result = SimpleNamespace(errors=[], output_lines=[])
        mock_context = SimpleNamespace(current_task="some task")

        # Act
        result = integration.should_call_zen(mock_result, mock_context, "session-1")

        # Assert
        assert result is None

    def test_should_call_zen__when_stuck_detected__returns_debug_tool(self, zen_integration: ZenIntegration, mocker):
        """Ensures 'debug' tool is returned when stuck status is detected."""
        # Arrange
        mocker.patch.object(zen_integration, '_detect_stuck_status', return_value="Agent is stuck")
        mock_result = SimpleNamespace()
        mock_context = SimpleNamespace()

        # Act
        tool, reason = zen_integration.should_call_zen(mock_result, mock_context, "session-1")

        # Assert
        assert tool == "debug"
        assert reason == "Agent is stuck"

    def test_should_call_zen__when_error_pattern_detected__returns_debug_tool(self, zen_integration: ZenIntegration, mocker):
        """Ensures 'debug' tool is returned when a repeated error pattern is found."""
        # Arrange
        mocker.patch.object(zen_integration, '_detect_stuck_status', return_value=None)
        mocker.patch.object(zen_integration, '_detect_error_pattern', return_value="Repeated error")
        mock_result = SimpleNamespace()
        mock_context = SimpleNamespace()

        # Act
        tool, reason = zen_integration.should_call_zen(mock_result, mock_context, "session-1")

        # Assert
        assert tool == "debug"
        assert reason == "Repeated error"

    def test_should_call_zen__when_task_needs_validation__returns_precommit_tool(self, zen_integration: ZenIntegration, mocker):
        """Ensures 'precommit' tool is returned for tasks requiring validation."""
        # Arrange
        mocker.patch.object(zen_integration, '_detect_stuck_status', return_value=None)
        mocker.patch.object(zen_integration, '_detect_error_pattern', return_value=None)
        mocker.patch.object(zen_integration, '_task_requires_validation', return_value="Matches pattern: *security*")
        mock_result = SimpleNamespace()
        mock_context = SimpleNamespace(current_task="Implement security login flow")

        # Act
        tool, reason = zen_integration.should_call_zen(mock_result, mock_context, "session-1")

        # Assert
        assert tool == "precommit"
        assert reason == "Critical task validation: Matches pattern: *security*"

    def test_should_call_zen__when_cutoff_detected__returns_analyze_tool(self, zen_integration: ZenIntegration, mocker):
        """Ensures 'analyze' tool is returned when a cutoff is detected."""
        # Arrange
        mocker.patch.object(zen_integration, '_detect_stuck_status', return_value=None)
        mocker.patch.object(zen_integration, '_detect_error_pattern', return_value=None)
        mocker.patch.object(zen_integration, '_task_requires_validation', return_value=None)
        mocker.patch.object(zen_integration, '_detect_cutoff', return_value=True)
        mock_result = SimpleNamespace()
        mock_context = SimpleNamespace(current_task="some task")

        # Act
        tool, reason = zen_integration.should_call_zen(mock_result, mock_context, "session-1")

        # Assert
        assert tool == "analyze"
        assert reason == "Task appears to have been cut off at turn limit"

    def test_should_call_zen__when_no_issues__returns_none(self, zen_integration: ZenIntegration, mocker):
        """Ensures nothing is returned when no issues are detected."""
        # Arrange
        mocker.patch.object(zen_integration, '_detect_stuck_status', return_value=None)
        mocker.patch.object(zen_integration, '_detect_error_pattern', return_value=None)
        mocker.patch.object(zen_integration, '_task_requires_validation', return_value=None)
        mocker.patch.object(zen_integration, '_detect_cutoff', return_value=False)
        mock_result = SimpleNamespace()
        mock_context = SimpleNamespace(current_task="some task")

        # Act
        result = zen_integration.should_call_zen(mock_result, mock_context, "session-1")

        # Assert
        assert result is None


class TestDetectStuckStatus:
    """Tests for `_detect_stuck_status` involving filesystem and output checks."""

    def test_detect_stuck_status__with_stuck_phrase_in_scratchpad__returns_reason(self, zen_integration: ZenIntegration, fs):
        """Verify stuck detection from scratchpad file with 'STUCK' status."""
        # Arrange
        session_id = "session-stuck"
        scratchpad_content = "Plan: Do a thing.\nStatus: STUCK\nIssue: The thing is hard."
        fs.create_file(Path(SCRATCHPAD_DIR) / f"session_{session_id}.md", contents=scratchpad_content)
        mock_result = SimpleNamespace(output_lines=[])

        # Act
        result = zen_integration._detect_stuck_status(mock_result, session_id)

        # Assert
        assert result == "Agent stuck: The thing is hard."

    def test_detect_stuck_status__with_help_needed_in_scratchpad__returns_reason(self, zen_integration: ZenIntegration, fs):
        """Verify stuck detection from scratchpad file with 'HELP NEEDED'."""
        # Arrange
        session_id = "session-help"
        scratchpad_content = "I am confused. HELP NEEDED"
        fs.create_file(Path(SCRATCHPAD_DIR) / f"session_{session_id}.md", contents=scratchpad_content)
        mock_result = SimpleNamespace(output_lines=[])

        # Act
        result = zen_integration._detect_stuck_status(mock_result, session_id)

        # Assert
        assert result == "Agent stuck: Agent requested help"

    def test_detect_stuck_status__with_help_pattern_in_output__returns_reason(self, zen_integration: ZenIntegration, fs):
        """Verify stuck detection from agent's output lines."""
        # Arrange
        session_id = "session-output-help"
        mock_result = SimpleNamespace(output_lines=["Some output", "SECURITY_REVIEW_NEEDED", "More output"])
        # Ensure scratchpad does not exist to isolate the test
        assert not Path(SCRATCHPAD_DIR).exists()

        # Act
        result = zen_integration._detect_stuck_status(mock_result, session_id)

        # Assert
        assert result == "Agent requested: SECURITY_REVIEW_NEEDED"

    def test_detect_stuck_status__when_no_indicators__returns_none(self, zen_integration: ZenIntegration, fs):
        """Verify it returns None when no stuck indicators are present."""
        # Arrange
        session_id = "session-ok"
        fs.create_file(Path(SCRATCHPAD_DIR) / f"session_{session_id}.md", contents="Everything is fine.")
        mock_result = SimpleNamespace(output_lines=["All good here."])

        # Act
        result = zen_integration._detect_stuck_status(mock_result, session_id)

        # Assert
        assert result is None


class TestDetectErrorPattern:
    """Tests for `_detect_error_pattern` and its stateful error counting."""

    def test_detect_error_pattern__when_errors_below_threshold__returns_none(self, zen_integration: ZenIntegration):
        """Error counts below threshold should not trigger a call."""
        # Arrange
        session_id = "session-err-1"
        errors = ["ModuleNotFoundError: no module named 'foo'"] * (AUTO_DEBUG_ERROR_THRESHOLD - 1)
        mock_result = SimpleNamespace(errors=errors)

        # Act
        result = zen_integration._detect_error_pattern(mock_result, session_id)

        # Assert
        assert result is None
        assert zen_integration.error_counts[session_id]['import_error'] == AUTO_DEBUG_ERROR_THRESHOLD - 1

    def test_detect_error_pattern__when_errors_reach_threshold__returns_reason(self, zen_integration: ZenIntegration):
        """A repeated error reaching the threshold should trigger a call."""
        # Arrange
        session_id = "session-err-2"
        errors = ["SyntaxError: invalid syntax"] * AUTO_DEBUG_ERROR_THRESHOLD
        mock_result = SimpleNamespace(errors=errors)

        # Act
        result = zen_integration._detect_error_pattern(mock_result, session_id)

        # Assert
        assert result == f"Repeated error ({AUTO_DEBUG_ERROR_THRESHOLD}x): syntax_error"
        assert zen_integration.error_counts[session_id]['syntax_error'] == AUTO_DEBUG_ERROR_THRESHOLD

    def test_detect_error_pattern__with_multiple_sessions__tracks_counts_separately(self, zen_integration: ZenIntegration):
        """Error counts should be tracked independently for each session."""
        # Arrange
        session_1 = "session-a"
        session_2 = "session-b"
        errors_1 = ["PermissionError: access denied"]
        errors_2 = ["TimeoutError: connection timed out"]

        # Act & Assert
        # Process for session 1
        res_1 = zen_integration._detect_error_pattern(SimpleNamespace(errors=errors_1), session_1)
        assert res_1 is None
        assert zen_integration.error_counts[session_1]['permission_error'] == 1
        assert session_2 not in zen_integration.error_counts

        # Process for session 2
        res_2 = zen_integration._detect_error_pattern(SimpleNamespace(errors=errors_2), session_2)
        assert res_2 is None
        assert zen_integration.error_counts[session_2]['timeout_error'] == 1
        assert zen_integration.error_counts[session_1]['permission_error'] == 1 # Unchanged


@pytest.mark.parametrize("error_string, expected_category", [
    ("ModuleNotFoundError: No module named 'requests'", "import_error"),
    ("cannot import name 'MyClass' from 'mymodule'", "import_error"),
    ("FileNotFoundError: [Errno 2] No such file or directory: 'test.txt'", "file_not_found"),
    ("SyntaxError: invalid syntax", "syntax_error"),
    ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", "type_error"),
    ("AttributeError: 'NoneType' object has no attribute 'foo'", "type_error"),
    ("PermissionError: [Errno 13] Permission denied: '/root/file'", "permission_error"),
    ("ConnectionError: [Errno 111] Connection refused", "connection_error"),
    ("TimeoutError: The read operation timed out", "timeout_error"),
    ("ValueError: invalid literal for int() with base 10: 'abc'", "value_error"),
    ("KeyError: 'missing_key'", "key_error"),
    ("RuntimeError: something went wrong during runtime", "runtime_error"),
    ("An unknown and bizarre cosmic ray error occurred", "unknown_error"),
])
def test_categorize_error(zen_integration: ZenIntegration, error_string: str, expected_category: str):
    """Tests the `_categorize_error` method with various error strings."""
    # Act
    category = zen_integration._categorize_error(error_string)
    # Assert
    assert category == expected_category


@pytest.mark.parametrize("task_description, patterns, expected_match", [
    ("Implement new security login flow", ["*security*", "*payment*"], "Matches pattern: *security*"),
    ("Refactor the payment processing module", ["*security*", "*payment*"], "Matches pattern: *payment*"),
    ("Update the UI documentation", ["*security*", "*payment*"], None),
    ("Fix a bug in the auth system", ["*auth*"], "Matches pattern: *auth*"),
    ("A task with no keywords", ["*security*"], None),
    (None, ["*security*"], None),
    ("", ["*security*"], None),
])
def test_task_requires_validation(zen_config: ZenIntegrationConfig, task_description: str, patterns: list, expected_match: str):
    """Tests the `_task_requires_validation` method."""
    # Arrange
    zen_config.validate_on_complete = patterns
    integration = ZenIntegration(config=zen_config)

    # Act
    result = integration._task_requires_validation(task_description)

    # Assert
    assert result == expected_match


class TestDetectCutoff:
    """Tests for the `_detect_cutoff` logic."""

    def test_detect_cutoff__when_stopped_unexpectedly_is_true__returns_true(self, zen_integration: ZenIntegration):
        """The `stopped_unexpectedly` flag should take precedence."""
        # Arrange
        mock_result = SimpleNamespace(stopped_unexpectedly=True, task_complete=False)
        mock_context = SimpleNamespace()

        # Act
        result = zen_integration._detect_cutoff(mock_result, mock_context)

        # Assert
        assert result is True

    def test_detect_cutoff__when_task_is_complete__returns_false(self, zen_integration: ZenIntegration):
        """If the task is marked as complete, it's not a cutoff."""
        # Arrange
        mock_result = SimpleNamespace(task_complete=True, output_lines=[])
        mock_context = SimpleNamespace()

        # Act
        result = zen_integration._detect_cutoff(mock_result, mock_context)

        # Assert
        assert result is False

    def test_detect_cutoff__when_indicators_meet_threshold__returns_true(self, zen_integration: ZenIntegration):
        """A combination of indicators should signal a cutoff."""
        # Arrange
        output_lines = ["I am still working on this.", "next, i'll do the other thing"]
        # Indicators:
        # 1. "ALL TASKS COMPLETE" not in last_lines -> True
        # 2. "HELP NEEDED" not in last_lines -> True
        # 3. has_remaining_todos -> True
        # 4. "working on" in last_lines -> True
        # Total = 4, which is >= CUTOFF_INDICATOR_THRESHOLD
        mock_result = SimpleNamespace(task_complete=False, output_lines=output_lines)
        mock_context = SimpleNamespace(remaining_todos=["one more thing"])

        # Act
        result = zen_integration._detect_cutoff(mock_result, mock_context)

        # Assert
        assert result is True

    def test_detect_cutoff__when_indicators_below_threshold__returns_false(self, zen_integration: ZenIntegration):
        """Fewer indicators should not signal a cutoff."""
        # Arrange
        output_lines = ["I have finished the task."]
        # Indicators:
        # 1. "ALL TASKS COMPLETE" not in last_lines -> True
        # 2. "HELP NEEDED" not in last_lines -> True
        # 3. has_remaining_todos -> False
        # Total = 2, which is < CUTOFF_INDICATOR_THRESHOLD
        mock_result = SimpleNamespace(task_complete=False, output_lines=output_lines)
        mock_context = SimpleNamespace(remaining_todos=[])

        # Act
        result = zen_integration._detect_cutoff(mock_result, mock_context)

        # Assert
        assert result is False


class TestCleanupSession:
    """Tests for session data cleanup."""

    def test_cleanup_session__removes_data_for_existing_session(self, zen_integration: ZenIntegration):
        """Should remove the error count data for a given session ID."""
        # Arrange
        session_id = "session-to-clean"
        zen_integration.error_counts[session_id] = {"some_error": 1}
        assert session_id in zen_integration.error_counts

        # Act
        zen_integration.cleanup_session(session_id)

        # Assert
        assert session_id not in zen_integration.error_counts

    def test_cleanup_session__does_not_fail_for_nonexistent_session(self, zen_integration: ZenIntegration):
        """Should not raise an error if the session ID does not exist."""
        # Arrange
        session_id = "nonexistent-session"
        assert session_id not in zen_integration.error_counts

        # Act & Assert
        try:
            zen_integration.cleanup_session(session_id)
        except Exception as e:
            pytest.fail(f"cleanup_session raised an unexpected exception: {e}")
