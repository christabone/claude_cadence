# tests/test_retry_utils.py

import json
import logging
import time
from pathlib import Path
from unittest.mock import Mock, call, MagicMock

import pytest

from cadence.retry_utils import (
    parse_json_with_retry,
    run_claude_with_realtime_retry,
    save_json_with_retry,
    RetryError,
)


class TestParseJSONWithRetry:
    """Tests for the parse_json_with_retry utility."""

    def test_success_on_first_attempt(self, monkeypatch):
        """Should parse valid JSON immediately without retries."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)
        json_string = '{"key": "value"}'

        result = parse_json_with_retry(json_string)

        assert result == {"key": "value"}
        mock_sleep.assert_not_called()

    def test_success_on_retry_with_callback(self, monkeypatch, caplog):
        """Should succeed on the second attempt after a callback fixes the JSON."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)

        malformed_string = '{"key": "value",}'
        fixed_string = '{"key": "value"}'

        def fix_json_callback(s, attempt):
            # A simple callback that removes a trailing comma
            return s.rstrip(',')

        with caplog.at_level(logging.WARNING):
            result = parse_json_with_retry(
                malformed_string,
                max_retries=3,
                retry_callback=fix_json_callback,
                base_delay=1.0
            )

        assert result == {"key": "value"}
        mock_sleep.assert_called_once_with(1.0) # delay = base_delay * attempt (1.0 * 1)
        assert "Parse error on attempt 1" in caplog.text

    def test_failure_after_all_retries(self, monkeypatch, caplog):
        """Should raise RetryError after all attempts fail."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)
        malformed_string = '{"key": "incomplete"'

        with pytest.raises(RetryError) as excinfo:
            parse_json_with_retry(malformed_string, max_retries=3, base_delay=0.1)

        assert "Failed to parse JSON after 3 attempts" in str(excinfo.value)
        assert excinfo.value.last_output == malformed_string
        assert mock_sleep.call_count == 2 # Sleeps before attempt 2 and 3
        assert "Parse error on attempt 1" in caplog.text
        assert "Parse error on attempt 2" in caplog.text
        assert "Parse error on attempt 3" in caplog.text

    def test_correct_delays_on_retries(self, monkeypatch):
        """Ensures linear backoff is applied correctly."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)

        with pytest.raises(RetryError):
            parse_json_with_retry("invalid", max_retries=4, base_delay=1.5)

        assert mock_sleep.call_args_list == [
            call(1.5),  # Before attempt 2
            call(3.0),  # Before attempt 3
            call(4.5),  # Before attempt 4
        ]

    def test_zero_retries(self, monkeypatch):
        """Should fail immediately with max_retries=0."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)

        with pytest.raises(RetryError) as excinfo:
            parse_json_with_retry("invalid", max_retries=0)

        assert "Failed to parse JSON after 0 attempts" in str(excinfo.value)
        mock_sleep.assert_not_called()

    def test_empty_string_input(self, monkeypatch):
        """Should handle empty string appropriately."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)

        with pytest.raises(RetryError) as excinfo:
            parse_json_with_retry("", max_retries=2)

        assert excinfo.value.last_output == ""
        mock_sleep.assert_called_once()

    def test_callback_modifies_string_progressively(self, monkeypatch):
        """Should handle callbacks that progressively fix JSON."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)

        def progressive_fix(s, attempt):
            if attempt == 1:
                return s.replace("'", '"')  # Fix quotes
            elif attempt == 2:
                return s.rstrip(',') + '}'  # Fix structure
            return s

        malformed_string = "{'key': 'value',"

        result = parse_json_with_retry(
            malformed_string,
            max_retries=3,
            retry_callback=progressive_fix
        )

        assert result == {"key": "value"}
        assert mock_sleep.call_count == 2


class TestRunClaudeWithRealtimeRetry:
    """Tests for the run_claude_with_realtime_retry orchestrator helper."""

    @pytest.fixture
    def mock_dependencies(self, monkeypatch):
        """Provides mocks for all callable dependencies."""
        monkeypatch.setattr(time, "sleep", Mock())
        mocks = {
            "build_cmd": Mock(),
            "parse_output": Mock(),
            "runner": Mock(),
        }
        return mocks

    def test_success_on_first_attempt(self, mock_dependencies):
        """Should succeed immediately when all components work."""
        # Arrange
        mock_dependencies["build_cmd"].return_value = ["claude", "-p", "prompt"]
        mock_dependencies["runner"].return_value = (0, ['{"status": "success"}'])
        mock_dependencies["parse_output"].return_value = {"status": "success"}

        # Act
        result = run_claude_with_realtime_retry(
            build_command_func=mock_dependencies["build_cmd"],
            parse_output_func=mock_dependencies["parse_output"],
            realtime_runner_func=mock_dependencies["runner"],
            working_dir="/tmp",
            process_name="TEST"
        )

        # Assert
        assert result == {"status": "success"}
        mock_dependencies["build_cmd"].assert_called_once()
        mock_dependencies["runner"].assert_called_once()
        mock_dependencies["parse_output"].assert_called_once()
        time.sleep.assert_not_called()

    def test_success_on_retry_due_to_parse_error(self, mock_dependencies, caplog):
        """Should succeed on the second attempt if parsing fails once."""
        # Arrange
        mock_dependencies["build_cmd"].side_effect = [["claude"], ["claude", "--continue"]]
        mock_dependencies["runner"].return_value = (0, ['{"status": "success"}'])
        mock_dependencies["parse_output"].side_effect = [ValueError("bad json"), {"status": "success"}]

        # Act
        with caplog.at_level(logging.INFO):
            result = run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=2
            )

        # Assert
        assert result == {"status": "success"}
        assert mock_dependencies["build_cmd"].call_count == 2
        assert mock_dependencies["runner"].call_count == 2
        assert mock_dependencies["parse_output"].call_count == 2
        time.sleep.assert_called_once()
        assert "Retrying TEST due to parsing error (attempt 2)" in caplog.text

    def test_failure_after_all_runner_retries(self, mock_dependencies):
        """Should raise RetryError if the runner consistently fails."""
        # Arrange
        mock_dependencies["build_cmd"].return_value = ["claude"]
        mock_dependencies["runner"].return_value = (1, ["Error output"])

        # Act & Assert
        with pytest.raises(RetryError) as excinfo:
            run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=3
            )

        assert "TEST failed with exit code 1" in str(excinfo.value)
        assert mock_dependencies["runner"].call_count == 3
        mock_dependencies["parse_output"].assert_not_called() # Should not be called if runner fails

    def test_failure_after_all_parse_retries(self, mock_dependencies):
        """Should raise RetryError if parsing consistently fails."""
        # Arrange
        mock_dependencies["build_cmd"].return_value = ["claude"]
        mock_dependencies["runner"].return_value = (0, ["invalid output"])
        mock_dependencies["parse_output"].side_effect = ValueError("bad json")

        # Act & Assert
        with pytest.raises(RetryError) as excinfo:
            run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=3
            )

        assert "TEST failed to produce valid output after 3 attempts" in str(excinfo.value)
        assert excinfo.value.last_output == "invalid output"
        assert excinfo.value.last_error == "bad json"
        assert mock_dependencies["runner"].call_count == 3
        assert mock_dependencies["parse_output"].call_count == 3

    def test_runner_exception_handling(self, mock_dependencies, caplog):
        """Should handle exceptions from the runner function."""
        # Arrange
        mock_dependencies["build_cmd"].return_value = ["claude"]
        mock_dependencies["runner"].side_effect = [RuntimeError("Connection failed"), (0, ['{"status": "ok"}'])]
        mock_dependencies["parse_output"].return_value = {"status": "ok"}

        # Act
        with caplog.at_level(logging.WARNING):
            result = run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=2
            )

        # Assert
        assert result == {"status": "ok"}
        assert "Retrying TEST due to runner error" in caplog.text
        assert "Connection failed" in caplog.text

    def test_empty_output_handling(self, mock_dependencies):
        """Should handle empty output from runner."""
        # Arrange
        mock_dependencies["build_cmd"].return_value = ["claude"]
        mock_dependencies["runner"].return_value = (0, [])
        mock_dependencies["parse_output"].side_effect = ValueError("No output to parse")

        # Act & Assert
        with pytest.raises(RetryError) as excinfo:
            run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=2
            )

        assert "TEST failed to produce valid output" in str(excinfo.value)

    def test_build_command_receives_retry_count(self, mock_dependencies):
        """Should pass retry count to build_command_func."""
        # Arrange
        mock_dependencies["build_cmd"].side_effect = [
            ["claude", "v1"],
            ["claude", "v2"],
            ["claude", "v3"]
        ]
        mock_dependencies["runner"].return_value = (1, ["Failed"])

        # Act
        with pytest.raises(RetryError):
            run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=3
            )

        # Assert
        assert mock_dependencies["build_cmd"].call_count == 3
        # Verify each call received the correct attempt number
        calls = mock_dependencies["build_cmd"].call_args_list
        assert len(calls) == 3

    def test_correct_delay_calculation(self, mock_dependencies, monkeypatch):
        """Should apply correct linear backoff delays."""
        mock_sleep = Mock()
        monkeypatch.setattr(time, "sleep", mock_sleep)

        mock_dependencies["build_cmd"].return_value = ["claude"]
        mock_dependencies["runner"].return_value = (1, ["Failed"])

        with pytest.raises(RetryError):
            run_claude_with_realtime_retry(
                build_command_func=mock_dependencies["build_cmd"],
                parse_output_func=mock_dependencies["parse_output"],
                realtime_runner_func=mock_dependencies["runner"],
                working_dir="/tmp",
                process_name="TEST",
                max_retries=4,
                base_delay=2.0
            )

        assert mock_sleep.call_args_list == [
            call(2.0),   # Before attempt 2
            call(4.0),   # Before attempt 3
            call(6.0),   # Before attempt 4
        ]


class TestSaveJSONWithRetry:
    """Tests for the save_json_with_retry utility."""

    def test_success_on_first_attempt(self, tmp_path: Path):
        """Should correctly write a serializable object to a file."""
        # Arrange
        data = {"a": 1, "b": [2, 3]}
        file_path = tmp_path / "test.json"

        # Act
        save_json_with_retry(data, str(file_path))

        # Assert
        assert file_path.exists()
        content = file_path.read_text()
        assert json.loads(content) == data

    def test_raises_retry_error_immediately_on_type_error(self, tmp_path: Path, caplog):
        """Should fail immediately without retrying on a serialization error."""
        # Arrange
        data = {"a": 1, "b": {2, 3}}  # A set is not JSON serializable
        file_path = tmp_path / "test.json"

        # Act & Assert
        with pytest.raises(RetryError) as excinfo:
            save_json_with_retry(data, str(file_path), max_retries=3)

        assert "data contains non-serializable objects" in str(excinfo.value)
        assert not file_path.exists()
        assert "JSON serialization error" in caplog.text
        # This test confirms the function does not retry on TypeError, as expected from the implementation.

    def test_raises_io_error_immediately_on_write_error(self, monkeypatch, caplog):
        """Should fail immediately without retrying on a file write error."""
        # Arrange
        data = {"a": 1}

        # Mock open to raise IOError
        mock_open = MagicMock()
        mock_open.side_effect = IOError("Permission denied")
        monkeypatch.setattr("builtins.open", mock_open)

        # Act & Assert
        with pytest.raises(IOError) as excinfo:
            save_json_with_retry(data, "/read-only/test.json", max_retries=3)

        assert "Permission denied" in str(excinfo.value)
        assert "File write error" in caplog.text
        # This test confirms the function does not retry on IOError, as expected.

    def test_pretty_printing_format(self, tmp_path: Path):
        """Should save JSON with pretty printing (4-space indent)."""
        # Arrange
        data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        file_path = tmp_path / "test.json"

        # Act
        save_json_with_retry(data, str(file_path))

        # Assert
        content = file_path.read_text()
        # Check for pretty printing indicators
        assert "    " in content  # 4-space indent
        assert "{\n" in content   # Newlines after braces
        assert json.loads(content) == data

    def test_handles_pathlib_path(self, tmp_path: Path):
        """Should accept pathlib.Path objects."""
        # Arrange
        data = {"test": True}
        file_path = tmp_path / "test.json"

        # Act
        save_json_with_retry(data, file_path)  # Pass Path object directly

        # Assert
        assert file_path.exists()
        assert json.loads(file_path.read_text()) == data

    def test_creates_parent_directories(self, tmp_path: Path):
        """Should create parent directories if they don't exist."""
        # Arrange
        data = {"test": True}
        nested_path = tmp_path / "nested" / "dirs" / "test.json"

        # Act
        save_json_with_retry(data, str(nested_path))

        # Assert
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert json.loads(nested_path.read_text()) == data

    def test_overwrites_existing_file(self, tmp_path: Path):
        """Should overwrite existing file content."""
        # Arrange
        file_path = tmp_path / "test.json"
        file_path.write_text('{"old": "data"}')
        new_data = {"new": "data"}

        # Act
        save_json_with_retry(new_data, str(file_path))

        # Assert
        assert json.loads(file_path.read_text()) == new_data

    def test_unicode_handling(self, tmp_path: Path):
        """Should correctly handle Unicode characters."""
        # Arrange
        data = {"emoji": "🚀", "chinese": "你好", "special": "café"}
        file_path = tmp_path / "unicode.json"

        # Act
        save_json_with_retry(data, str(file_path))

        # Assert
        content = file_path.read_text(encoding='utf-8')
        loaded = json.loads(content)
        assert loaded == data

    def test_none_and_boolean_values(self, tmp_path: Path):
        """Should correctly serialize None and boolean values."""
        # Arrange
        data = {"null_value": None, "true_val": True, "false_val": False}
        file_path = tmp_path / "special_values.json"

        # Act
        save_json_with_retry(data, str(file_path))

        # Assert
        loaded = json.loads(file_path.read_text())
        assert loaded["null_value"] is None
        assert loaded["true_val"] is True
        assert loaded["false_val"] is False


class TestRetryError:
    """Tests for the RetryError exception class."""

    def test_basic_instantiation(self):
        """Should create RetryError with just a message."""
        error = RetryError("Something failed")
        assert str(error) == "Something failed"
        assert error.last_output is None
        assert error.last_error is None

    def test_with_last_output(self):
        """Should store last_output attribute."""
        error = RetryError("Failed", last_output='{"partial": "data"}')
        assert str(error) == "Failed"
        assert error.last_output == '{"partial": "data"}'
        assert error.last_error is None

    def test_with_last_error(self):
        """Should store last_error attribute."""
        error = RetryError("Failed", last_error="Connection timeout")
        assert str(error) == "Failed"
        assert error.last_output is None
        assert error.last_error == "Connection timeout"

    def test_with_both_attributes(self):
        """Should store both last_output and last_error."""
        error = RetryError(
            "Complete failure",
            last_output="some output",
            last_error="some error"
        )
        assert str(error) == "Complete failure"
        assert error.last_output == "some output"
        assert error.last_error == "some error"

    def test_inheritance(self):
        """Should be a proper Exception subclass."""
        error = RetryError("Test")
        assert isinstance(error, Exception)

        # Should be raisable and catchable
        with pytest.raises(RetryError):
            raise error

        # Should be catchable as generic Exception
        try:
            raise RetryError("Test")
        except Exception as e:
            assert isinstance(e, RetryError)
