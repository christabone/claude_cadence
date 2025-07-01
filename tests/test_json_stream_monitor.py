# filename: tests/test_json_stream_monitor.py

import pytest
import json
from typing import List, Optional, Dict, Any

from cadence.json_stream_monitor import SimpleJSONStreamMonitor

@pytest.fixture
def monitor() -> SimpleJSONStreamMonitor:
    """Provides a fresh SimpleJSONStreamMonitor instance for each test."""
    return SimpleJSONStreamMonitor()

def process_lines(monitor: SimpleJSONStreamMonitor, lines: List[str]) -> Optional[Dict[str, Any]]:
    """Helper function to process multiple lines and return the last result."""
    result = None
    for line in lines:
        res = monitor.process_line(line)
        if res:
            result = res
    return result

class TestSimpleJSONStreamMonitor:
    """Tests for the SimpleJSONStreamMonitor class."""

    def test_initial_state_and_reset(self, monitor: SimpleJSONStreamMonitor):
        """
        Verifies the initial state of the monitor and its state after a reset.
        """
        # Arrange: Initial state verification
        assert monitor.buffer == []
        assert not monitor.in_json
        assert monitor.last_json_object is None

        # Act: Process some data to change the state
        monitor.process_line('{"key": "value",')
        assert monitor.buffer != []
        assert monitor.in_json

        # Act: Reset the monitor
        monitor.reset()

        # Assert: State is back to initial
        assert monitor.buffer == []
        assert not monitor.in_json
        assert monitor.last_json_object is None

    def test_get_last_json_object(self, monitor: SimpleJSONStreamMonitor):
        """
        Verifies the behavior of get_last_json_object.
        """
        # Assert: Initially returns None
        assert monitor.get_last_json_object() is None

        # Act: Process a complete JSON object
        process_lines(monitor, ['{"status": "complete"}'])

        # Assert: Returns the parsed object
        assert monitor.get_last_json_object() == {"status": "complete"}

        # Act: Process incomplete data
        monitor.process_line('{"next":')

        # Assert: Still returns the last *complete* object
        assert monitor.get_last_json_object() == {"status": "complete"}

        # Act: Reset
        monitor.reset()

        # Assert: Returns None after reset
        assert monitor.get_last_json_object() is None

    @pytest.mark.parametrize("lines, expected_json", [
        pytest.param(['{"key": "value"}'], {"key": "value"}, id="single_line_object"),
        pytest.param(['[1, 2, "three"]'], [1, 2, "three"], id="single_line_array"),
        pytest.param(['{', '"key": "value",', '"num": 123', '}'], {"key": "value", "num": 123}, id="multi_line_object"),
        pytest.param(['[', '1,', '2', ']'], [1, 2], id="multi_line_array"),
        pytest.param(['{}'], {}, id="empty_object"),
        pytest.param(['[]'], [], id="empty_array"),
        pytest.param(['{"a": {"b": {}}}'], {"a": {"b": {}}}, id="nested_object"),
        pytest.param(['{"msg": "hello world 👋"}'], {"msg": "hello world 👋"}, id="unicode_emoji"),
    ])
    def test_basic_json_parsing(self, monitor: SimpleJSONStreamMonitor, lines: List[str], expected_json: Any):
        """
        Tests parsing of various valid JSON structures.
        """
        # Act
        result = process_lines(monitor, lines)

        # Assert
        assert result == expected_json
        assert monitor.get_last_json_object() == expected_json
        assert not monitor.in_json # Should reset after successful parse

    def test_incomplete_json_is_buffered(self, monitor: SimpleJSONStreamMonitor):
        """
        Verifies that incomplete JSON is buffered and does not return a result.
        """
        # Arrange
        lines = ['{"key": "value",', '"another_key":']

        # Act
        result = process_lines(monitor, lines)

        # Assert
        assert result is None
        assert monitor.in_json
        assert monitor.buffer == lines
        assert monitor.get_last_json_object() is None

    def test_json_with_surrounding_text(self, monitor: SimpleJSONStreamMonitor):
        """
        Tests that JSON can be extracted even with surrounding non-JSON text.
        """
        # Arrange
        lines = [
            "Some introductory text.",
            'Here is the JSON: {"data": "important"}',
            "Some trailing text."
        ]

        # Act
        result = process_lines(monitor, lines)

        # Assert
        assert result == {"data": "important"}
        assert monitor.get_last_json_object() == {"data": "important"}

    def test_multiple_json_objects_in_stream(self, monitor: SimpleJSONStreamMonitor):
        """
        Tests processing a stream containing multiple, separate JSON objects.
        """
        # Arrange
        results = []
        lines = [
            '{"id": 1, "status": "first"}',
            'Some text in between',
            '{"id": 2, "status": "second"}'
        ]

        # Act
        for line in lines:
            res = monitor.process_line(line)
            if res:
                results.append(res)

        # Assert
        assert len(results) == 2
        assert results[0] == {"id": 1, "status": "first"}
        assert results[1] == {"id": 2, "status": "second"}
        assert monitor.get_last_json_object() == {"id": 2, "status": "second"}

    def test_json_with_trailing_content_on_same_line(self, monitor: SimpleJSONStreamMonitor):
        """
        Verifies that a JSON object is parsed correctly even with trailing text on the same line.
        The current implementation should parse the JSON and ignore the trailing part.
        """
        # Arrange
        line = '{"status": "done"}... and some other text'

        # Act
        result = monitor.process_line(line)

        # Assert
        assert result == {"status": "done"}
        assert monitor.get_last_json_object() == {"status": "done"}
        # The monitor should be reset, ready for the next JSON.
        assert not monitor.in_json
        assert monitor.buffer == []

    @pytest.mark.parametrize("max_lines, error_msg", [
        (SimpleJSONStreamMonitor.MAX_BUFFER_SIZE, f"JSON buffer size exceeded ({SimpleJSONStreamMonitor.MAX_BUFFER_SIZE} lines). The JSON object is too large or malformed."),
    ])
    def test_max_buffer_size_exceeded(self, monitor: SimpleJSONStreamMonitor, max_lines: int, error_msg: str):
        """
        Tests that a ValueError is raised when the buffer size limit is exceeded.
        """
        # Arrange: Fill the buffer up to the limit with incomplete JSON
        for i in range(max_lines):
            monitor.process_line(f'"line": {i},')

        # Act & Assert: The next line should raise a ValueError
        with pytest.raises(ValueError, match=error_msg) as excinfo:
            monitor.process_line('"last_line": "too much"')

        # Assert that state is reset after the error
        assert not monitor.in_json
        assert monitor.buffer == []

    def test_max_line_length_exceeded(self, monitor: SimpleJSONStreamMonitor):
        """
        Tests that a ValueError is raised when a single line exceeds the length limit.
        """
        # Arrange
        long_line = "a" * (SimpleJSONStreamMonitor.MAX_LINE_LENGTH + 1)
        error_msg = f"Line exceeds maximum length ({len(long_line)} > {SimpleJSONStreamMonitor.MAX_LINE_LENGTH}). This could indicate corrupt data or a security issue."

        # Act & Assert
        with pytest.raises(ValueError, match=error_msg):
            monitor.process_line(long_line)

class TestAssistantMessageHandling:
    """
    Tests focused on the special handling of 'assistant' type messages
    and the extraction of embedded JSON.
    """

    ASSISTANT_WRAPPER_TEMPLATE = """
{{
    "type": "assistant",
    "message": {{
        "content": [
            {{
                "type": "text",
                "text": "{text_content}"
            }}
        ]
    }}
}}
"""

    @pytest.fixture
    def monitor(self) -> SimpleJSONStreamMonitor:
        return SimpleJSONStreamMonitor()

    def test_extract_from_markdown_block(self, monitor: SimpleJSONStreamMonitor):
        """
        Tests extraction of a valid agent result from a markdown code block.
        """
        # Arrange
        embedded_json_str = '{"status": "success", "session_id": "123"}'
        text_content = f"Here is the result: ```json\\n{embedded_json_str}\\n```"
        assistant_message = self.ASSISTANT_WRAPPER_TEMPLATE.format(text_content=text_content)

        # Act
        result = process_lines(monitor, assistant_message.splitlines())

        # Assert: The outer wrapper is returned by process_line
        assert result is not None
        assert result['type'] == 'assistant'
        # Assert: The inner, embedded JSON is stored as the last object
        assert monitor.get_last_json_object() == json.loads(embedded_json_str)

    def test_extract_from_raw_json(self, monitor: SimpleJSONStreamMonitor):
        """
        Tests extraction of a valid agent result from a raw JSON string in the text.
        """
        # Arrange
        embedded_json_str = '{"status": "help_needed", "session_id": "456", "reason": "API key missing"}'
        text_content = f"I encountered an issue. {embedded_json_str}"
        assistant_message = self.ASSISTANT_WRAPPER_TEMPLATE.format(text_content=text_content)

        # Act
        result = process_lines(monitor, assistant_message.splitlines())

        # Assert
        assert monitor.get_last_json_object() == json.loads(embedded_json_str)

    def test_extract_from_escaped_json(self, monitor: SimpleJSONStreamMonitor):
        """
        Tests extraction from text containing an escaped JSON string.
        """
        # Arrange
        embedded_json = {"status": "error", "session_id": "789", "error_message": "Failed to write file"}
        escaped_json_str = json.dumps(json.dumps(embedded_json)) # Double-encoded
        text_content = f"An error occurred: {escaped_json_str}"
        assistant_message = self.ASSISTANT_WRAPPER_TEMPLATE.format(text_content=text_content.replace('"', '\\"'))

        # Act
        # This is tricky because the monitor's own parser will handle the outer JSON.
        # We need to test the internal _extract_json_candidates logic.
        candidates = monitor._extract_json_candidates(text_content)
        parsed_candidates = []
        for c in candidates:
            try:
                # The candidate should be the unescaped JSON string
                parsed = json.loads(c)
                if 'status' in parsed and 'session_id' in parsed:
                    parsed_candidates.append(parsed)
            except json.JSONDecodeError:
                continue

        # Assert
        assert embedded_json in parsed_candidates

    def test_no_valid_embedded_json_returns_wrapper(self, monitor: SimpleJSONStreamMonitor):
        """
        If no valid embedded agent JSON is found, the last object should be the wrapper itself.
        A "valid" embedded JSON requires 'status' and 'session_id' keys.
        """
        # Arrange
        # This embedded JSON is syntactically correct but lacks the required keys.
        embedded_json_str = '{"message": "Just a regular JSON"}'
        text_content = f"Some text with some JSON. {embedded_json_str}"
        assistant_message = self.ASSISTANT_WRAPPER_TEMPLATE.format(text_content=text_content)
        wrapper_json = json.loads(assistant_message)

        # Act
        result = process_lines(monitor, assistant_message.splitlines())

        # Assert
        assert result == wrapper_json
        assert monitor.get_last_json_object() == wrapper_json

    def test_multiple_candidates_selects_last_valid(self, monitor: SimpleJSONStreamMonitor):
        """
        If multiple JSON objects are embedded, it should select the last one that
        matches the agent result criteria.
        """
        # Arrange
        first_json = '{"status": "working", "session_id": "abc"}' # Valid
        second_json = '{"info": "intermediate data"}' # Invalid (no required keys)
        third_json = '{"status": "success", "session_id": "abc"}' # Valid and final
        text_content = f"Progress update: {first_json}. Some data: {second_json}. Final result: {third_json}"
        assistant_message = self.ASSISTANT_WRAPPER_TEMPLATE.format(text_content=text_content)

        # Act
        process_lines(monitor, assistant_message.splitlines())

        # Assert
        assert monitor.get_last_json_object() == json.loads(third_json)

    def test_non_assistant_message_is_unaffected(self, monitor: SimpleJSONStreamMonitor):
        """
        Messages not of type 'assistant' should be processed normally without
        attempting to extract embedded JSON.
        """
        # Arrange
        message_str = '{"type": "tool_result", "content": "File written successfully."}'
        expected_json = json.loads(message_str)

        # Act
        result = process_lines(monitor, [message_str])

        # Assert
        assert result == expected_json
        assert monitor.get_last_json_object() == expected_json
