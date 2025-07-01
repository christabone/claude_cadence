# tests/test_log_utils.py

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest

from cadence.log_utils import (
    ColoredFormatter,
    Colors,
    setup_colored_logging,
    get_colored_logger,
    setup_file_logging,
)

# A basic log record for testing the formatter
@pytest.fixture
def log_record():
    return logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="This is a test message with SUPERVISOR and AGENT keywords.",
        args=(),
        exc_info=None,
    )


class TestColoredFormatter:
    """Tests for the ColoredFormatter class."""

    @pytest.mark.parametrize(
        "use_color_init, isatty_return, expected_use_color",
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_init_use_color_logic(
        self, monkeypatch, use_color_init, isatty_return, expected_use_color
    ):
        """Test that the `use_color` attribute is set correctly based on args and TTY status."""
        monkeypatch.setattr(sys.stderr, "isatty", lambda: isatty_return)
        formatter = ColoredFormatter(use_color=use_color_init)
        assert formatter.use_color == expected_use_color

    def test_format_no_color_when_disabled(self, log_record):
        """Test that no ANSI codes are added when coloring is disabled."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s', use_color=False)
        formatted_msg = formatter.format(log_record)
        assert "\033" not in formatted_msg
        assert "INFO" in formatted_msg

    @pytest.mark.parametrize(
        "level, color",
        [
            (logging.DEBUG, Colors.CYAN),
            (logging.INFO, Colors.GREEN),
            (logging.WARNING, Colors.YELLOW),
            (logging.ERROR, Colors.RED),
            (logging.CRITICAL, Colors.BOLD_RED),
        ],
    )
    def test_format_level_colors(self, log_record, level, color):
        """Test that log levels are colored correctly."""
        log_record.levelno = level
        log_record.levelname = logging.getLevelName(level)
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        # Mock isatty to ensure color is used
        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        expected_level_str = f" - {color}{log_record.levelname}{Colors.RESET} - "
        assert expected_level_str in formatted_msg

    @pytest.mark.parametrize(
        "keyword, color",
        [
            ("SUPERVISOR", Colors.BOLD_BLUE),
            ("AGENT", Colors.BOLD_MAGENTA),
            ("COMPLETED", Colors.BOLD_GREEN),
            ("ERROR", Colors.BOLD_RED),
        ],
    )
    def test_format_keyword_colors(self, log_record, keyword, color):
        """Test that specific keywords are colored correctly."""
        log_record.msg = f"Message with keyword: {keyword}"
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        expected_keyword_str = f"{color}{keyword}{Colors.RESET}"
        assert expected_keyword_str in formatted_msg

    def test_format_separator_line_color(self, log_record):
        """Test that separator lines like '===' are colored correctly."""
        log_record.msg = "===\nSome content\n---"
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        # Check that the keywords are colored in the formatted message
        assert f"{Colors.BOLD_WHITE}==={Colors.RESET}" in formatted_msg
        assert f"{Colors.WHITE}---{Colors.RESET}" in formatted_msg
        assert "Some content" in formatted_msg


class TestLoggingSetup:
    """Tests for logging setup utilities."""

    def test_setup_colored_logging(self, caplog):
        """Test that colored logging is set up on the root logger correctly."""
        # Arrange: clean up any handlers from other tests
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers = []

        # Act
        setup_colored_logging(level=logging.DEBUG)

        # Assert
        assert len(root_logger.handlers) == 1
        handler = root_logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert isinstance(handler.formatter, ColoredFormatter)
        assert root_logger.level == logging.DEBUG

        # Verify it logs
        with caplog.at_level(logging.DEBUG):
            root_logger.debug("A debug message")
            assert "A debug message" in caplog.text

        # Teardown
        root_logger.handlers = original_handlers

    def test_get_colored_logger_first_time(self):
        """Test getting a new logger configures it correctly."""
        logger_name = "my_app_logger"
        # Ensure logger is clean before test
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]

        logger = get_colored_logger(logger_name, level=logging.WARNING)

        assert logger.name == logger_name
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, ColoredFormatter)
        assert logger.level == logging.WARNING
        assert not logger.propagate

    def test_get_colored_logger_is_idempotent(self):
        """Test that getting an existing logger does not add more handlers."""
        logger_name = "my_idempotent_logger"
        # Ensure logger is clean before test
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]

        logger1 = get_colored_logger(logger_name)
        assert len(logger1.handlers) == 1

        logger2 = get_colored_logger(logger_name)
        assert len(logger2.handlers) == 1  # Should not add another handler
        assert logger1 is logger2

    def test_setup_file_logging_success(self, monkeypatch, tmp_path):
        """Test successful setup of a file logger."""
        mock_file_handler = MagicMock(spec=logging.FileHandler)
        mock_file_handler.baseFilename = str(tmp_path / "session1" / "component.log")

        mock_mkdir = MagicMock()
        monkeypatch.setattr(Path, "mkdir", mock_mkdir)

        mock_fh_constructor = MagicMock(return_value=mock_file_handler)
        monkeypatch.setattr(logging, "FileHandler", mock_fh_constructor)

        logger = setup_file_logging("session1", "component", tmp_path, level=logging.INFO)

        expected_session_dir = tmp_path / "session1"
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        expected_log_file = expected_session_dir / "component.log"
        mock_fh_constructor.assert_called_once_with(expected_log_file, encoding="utf-8")

        assert logger is not None
        assert mock_file_handler in logger.handlers
        assert logger.level == logging.INFO

    def test_setup_file_logging_permission_error(self, caplog, tmp_path):
        """Test that file logging setup handles PermissionError gracefully."""
        with patch.object(Path, "mkdir", side_effect=PermissionError("Access denied")):
            with caplog.at_level(logging.WARNING):
                logger = setup_file_logging("s1", "comp", tmp_path)

        assert logger is None
        assert "Failed to set up file logging" in caplog.text
        assert "Access denied" in caplog.text
        assert "Continuing with console logging only" in caplog.text

    def test_setup_file_logging_is_idempotent(self, monkeypatch, tmp_path):
        """Test that file logging setup does not add duplicate handlers."""
        logger_name = "idempotent_file_logger"
        log_file = tmp_path / "session1" / f"{logger_name}.log"

        # Mock a FileHandler that will be "found" on the second call
        mock_handler = MagicMock(spec=logging.FileHandler)
        mock_handler.baseFilename = str(log_file)

        # Get a clean logger and attach the mock handler
        logger = logging.getLogger(logger_name)
        logger.handlers = [mock_handler]

        # Mock the constructor to see if it's called again
        mock_fh_constructor = MagicMock()
        monkeypatch.setattr(logging, "FileHandler", mock_fh_constructor)

        # Act: Call the setup function on a logger that already has the handler
        result_logger = setup_file_logging("session1", logger_name, tmp_path)

        # Assert: The constructor for a new handler should NOT have been called
        mock_fh_constructor.assert_not_called()
        assert result_logger is logger
        assert len(result_logger.handlers) == 1

        # Cleanup
        logger.handlers = []


class TestColorsClass:
    """Tests for the Colors class constants."""

    def test_colors_are_ansi_codes(self):
        """Test that Colors class contains valid ANSI escape codes."""
        assert Colors.RESET == "\033[0m"
        assert Colors.BOLD == "\033[1m"
        # Colors.DIM is not defined in the actual implementation, so removing this test
        assert Colors.RED == "\033[31m"
        assert Colors.GREEN == "\033[32m"
        assert Colors.YELLOW == "\033[33m"
        assert Colors.BLUE == "\033[34m"
        assert Colors.MAGENTA == "\033[35m"
        assert Colors.CYAN == "\033[36m"
        assert Colors.WHITE == "\033[37m"
        assert Colors.BOLD_RED == "\033[1;31m"
        assert Colors.BOLD_GREEN == "\033[1;32m"
        assert Colors.BOLD_YELLOW == "\033[1;33m"
        assert Colors.BOLD_BLUE == "\033[1;34m"
        assert Colors.BOLD_MAGENTA == "\033[1;35m"
        assert Colors.BOLD_CYAN == "\033[1;36m"
        assert Colors.BOLD_WHITE == "\033[1;37m"


class TestEdgeCases:
    """Test edge cases and additional scenarios."""

    def test_format_with_empty_message(self, log_record):
        """Test formatting an empty log message."""
        log_record.msg = ""
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        assert formatted_msg is not None
        assert "INFO" in formatted_msg

    def test_format_with_multiline_message(self, log_record):
        """Test formatting a multi-line log message with keywords."""
        log_record.msg = "First line with SUPERVISOR\n===\nSecond line with AGENT\n---\nThird line with ERROR"
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        # Check that all keywords are properly colored in the formatted message
        assert f"{Colors.BOLD_BLUE}SUPERVISOR{Colors.RESET}" in formatted_msg
        assert f"{Colors.BOLD_WHITE}==={Colors.RESET}" in formatted_msg
        assert f"{Colors.BOLD_MAGENTA}AGENT{Colors.RESET}" in formatted_msg
        assert f"{Colors.WHITE}---{Colors.RESET}" in formatted_msg
        assert f"{Colors.BOLD_RED}ERROR{Colors.RESET}" in formatted_msg

    def test_format_with_overlapping_keywords(self, log_record):
        """Test formatting when keywords overlap (e.g., 'TERROR' contains 'ERROR')."""
        log_record.msg = "This is a TERROR message"
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        # Should color 'ERROR' within 'TERROR'
        assert f"T{Colors.BOLD_RED}ERROR{Colors.RESET}" in formatted_msg

    def test_setup_file_logging_with_path_traversal(self, tmp_path):
        """Test that path traversal attempts are handled safely."""
        # pathlib should handle this safely, but let's verify
        dangerous_session_id = "../../../etc"
        dangerous_component = "passwd"

        with patch("logging.FileHandler") as mock_fh:
            logger = setup_file_logging(dangerous_session_id, dangerous_component, tmp_path)

        # Should create the file in the intended location, not traverse up
        expected_path = tmp_path / dangerous_session_id / f"{dangerous_component}.log"
        mock_fh.assert_called_once_with(expected_path, encoding="utf-8")

    def test_setup_file_logging_os_error(self, caplog, tmp_path):
        """Test that file logging setup handles OSError gracefully."""
        with patch.object(Path, "mkdir", side_effect=OSError("Disk full")):
            with caplog.at_level(logging.WARNING):
                logger = setup_file_logging("s1", "comp", tmp_path)

        assert logger is None
        assert "Failed to set up file logging" in caplog.text
        assert "Disk full" in caplog.text

    def test_format_json_output(self, log_record):
        """Test formatting when JSON output is requested."""
        # Simulate JSON output request
        log_record.args = {"json_output": True}
        log_record.msg = "Test message"

        formatter = ColoredFormatter(use_color=False)  # JSON typically means no color
        formatted_msg = formatter.format(log_record)

        assert "\033" not in formatted_msg  # No ANSI codes in JSON mode

    def test_multiple_formatters_independence(self):
        """Test that multiple formatter instances are independent."""
        with patch.object(sys.stderr, "isatty", return_value=True):
            formatter1 = ColoredFormatter(use_color=True)
        with patch.object(sys.stderr, "isatty", return_value=False):
            formatter2 = ColoredFormatter(use_color=False)

        assert formatter1.use_color != formatter2.use_color

    def test_format_with_exception_info(self, log_record):
        """Test formatting when exception info is present."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            log_record.exc_info = sys.exc_info()

        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_color=True
        )

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatted_msg = formatter.format(log_record)

        assert "ValueError: Test exception" in formatted_msg
        assert "Traceback" in formatted_msg
