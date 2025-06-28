"""
Unit tests for SimpleJSONStreamMonitor
"""
import pytest
from cadence.json_stream_monitor import SimpleJSONStreamMonitor


class TestSimpleJSONStreamMonitor:
    """Test the simple JSON stream monitor"""

    def test_single_line_json(self):
        """Test parsing single-line JSON"""
        monitor = SimpleJSONStreamMonitor()

        # Single line JSON
        result = monitor.process_line('{"type": "assistant", "content": "hello"}')

        assert result is not None
        assert result['type'] == 'assistant'
        assert result['content'] == 'hello'

        # Buffer should be clear
        assert monitor.buffer == []
        assert monitor.in_json is False

    def test_multi_line_json(self):
        """Test parsing multi-line JSON"""
        monitor = SimpleJSONStreamMonitor()

        # Multi-line JSON
        assert monitor.process_line('{') is None
        assert monitor.process_line('  "type": "assistant",') is None
        assert monitor.process_line('  "content": "hello"') is None
        result = monitor.process_line('}')

        assert result is not None
        assert result['type'] == 'assistant'
        assert result['content'] == 'hello'

        # Buffer should be clear after successful parse
        assert monitor.buffer == []
        assert monitor.in_json is False

    def test_buffer_reset_after_success(self):
        """Test that buffer resets after successful parse"""
        monitor = SimpleJSONStreamMonitor()

        # First JSON
        monitor.process_line('{"id": 1}')
        assert monitor.buffer == []

        # Second JSON
        monitor.process_line('{')
        monitor.process_line('  "id": 2')
        result = monitor.process_line('}')

        assert result is not None
        assert result['id'] == 2
        assert monitor.buffer == []

    def test_non_json_lines_ignored(self):
        """Test that non-JSON lines are ignored"""
        monitor = SimpleJSONStreamMonitor()

        # Non-JSON lines
        assert monitor.process_line('This is plain text') is None
        assert monitor.process_line('Another line') is None

        # Buffer should remain empty
        assert monitor.buffer == []
        assert monitor.in_json is False

    def test_empty_lines_handled(self):
        """Test that empty lines are handled gracefully"""
        monitor = SimpleJSONStreamMonitor()

        assert monitor.process_line('') is None
        assert monitor.process_line('   ') is None
        assert monitor.process_line('\n') is None

    def test_reset_method(self):
        """Test the reset method"""
        monitor = SimpleJSONStreamMonitor()

        # Add some data
        monitor.process_line('{')
        monitor.process_line('  "test": true')

        # Verify buffer has data
        assert len(monitor.buffer) > 0
        assert monitor.in_json is True

        # Reset
        monitor.reset()

        # Verify clean state
        assert monitor.buffer == []
        assert monitor.in_json is False

    def test_json_array_parsing(self):
        """Test parsing JSON arrays"""
        monitor = SimpleJSONStreamMonitor()

        # Single line array
        result = monitor.process_line('["item1", "item2", "item3"]')
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == "item1"

        # Multi-line array
        monitor.reset()
        assert monitor.process_line('[') is None
        assert monitor.process_line('  "item1",') is None
        assert monitor.process_line('  "item2",') is None
        assert monitor.process_line('  "item3"') is None
        result = monitor.process_line(']')

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[2] == "item3"

    def test_nested_json_structures(self):
        """Test parsing nested JSON objects and arrays"""
        monitor = SimpleJSONStreamMonitor()

        # Nested object
        assert monitor.process_line('[') is None
        assert monitor.process_line('  {') is None
        assert monitor.process_line('    "name": "test",') is None
        assert monitor.process_line('    "values": [1, 2, 3]') is None
        assert monitor.process_line('  }') is None
        result = monitor.process_line(']')

        assert result is not None
        assert isinstance(result, list)
        assert result[0]["name"] == "test"
        assert result[0]["values"] == [1, 2, 3]

    def test_large_json_handling(self):
        """Test handling of large JSON objects"""
        monitor = SimpleJSONStreamMonitor()

        # Start a JSON object
        monitor.process_line('{')

        # Add many fields to create a large JSON
        for i in range(1000):
            monitor.process_line(f'  "field{i}": "This is field number {i}",')

        # Close the JSON properly
        monitor.process_line('  "final": "last field"')
        result = monitor.process_line('}')

        # Should successfully parse the large JSON
        assert result is not None
        assert isinstance(result, dict)
        assert "field0" in result
        assert "field999" in result
        assert "final" in result
        assert result["final"] == "last field"

        # Buffer should be cleared after successful parse
        assert monitor.buffer == []
        assert monitor.in_json is False

    def test_incomplete_json_handling(self):
        """Test handling of incomplete JSON that never completes"""
        monitor = SimpleJSONStreamMonitor()

        # Start JSON but don't complete it
        assert monitor.process_line('{') is None
        assert monitor.process_line('  "incomplete": true') is None

        # Buffer should still contain the incomplete JSON
        assert len(monitor.buffer) == 2
        assert monitor.in_json is True

        # Process a non-JSON line - should be added to buffer
        assert monitor.process_line('  some text') is None
        assert len(monitor.buffer) == 3

        # Reset clears incomplete JSON
        monitor.reset()
        assert monitor.buffer == []
        assert monitor.in_json is False
