"""
Unit tests for ReviewTriggerDetector
"""

import pytest
import re
import json
from unittest.mock import Mock, patch
from datetime import datetime

from cadence.review_trigger_detector import (
    ReviewTriggerDetector, TriggerType, TriggerContext, TriggerPattern
)


class TestReviewTriggerDetector:
    """Test cases for ReviewTriggerDetector"""

    @pytest.fixture
    def detector(self):
        """Create a basic detector instance"""
        return ReviewTriggerDetector()

    @pytest.fixture
    def custom_config(self):
        """Create custom configuration for testing"""
        return {
            "triggers": {
                "patterns": [
                    {
                        "name": "custom_complete",
                        "type": "task_complete",
                        "pattern": r"CUSTOM COMPLETE",
                        "confidence": 0.8,
                        "priority": 15,
                        "description": "Custom completion pattern"
                    }
                ]
            }
        }

    def test_initialization_default(self, detector):
        """Test default initialization"""
        assert len(detector.patterns) > 0
        assert detector.buffer == ""
        assert detector.line_count == 0

        # Check that default patterns are loaded
        pattern_names = [p.name for p in detector.patterns]
        assert "all_tasks_complete" in pattern_names
        assert "help_needed" in pattern_names
        assert "json_decision" in pattern_names

    def test_initialization_with_config(self, custom_config):
        """Test initialization with custom config"""
        detector = ReviewTriggerDetector(custom_config)

        # Should have default patterns plus custom ones
        pattern_names = [p.name for p in detector.patterns]
        assert "custom_complete" in pattern_names
        assert "all_tasks_complete" in pattern_names

    def test_register_callback(self, detector):
        """Test callback registration"""
        callback = Mock()

        detector.register_callback(TriggerType.TASK_COMPLETE, callback)

        assert TriggerType.TASK_COMPLETE in detector.callbacks
        assert callback in detector.callbacks[TriggerType.TASK_COMPLETE]

    def test_process_line_all_tasks_complete(self, detector):
        """Test detection of ALL TASKS COMPLETE"""
        callback = Mock()
        detector.register_callback(TriggerType.ALL_TASKS_COMPLETE, callback)

        triggers = detector.process_line("Everything is done - ALL TASKS COMPLETE!")

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.ALL_TASKS_COMPLETE
        assert trigger.confidence == 0.95
        assert "ALL TASKS COMPLETE" in trigger.matched_text
        assert trigger.line_number == 1

        # Callback should be triggered
        callback.assert_called_once()

    def test_process_line_help_needed(self, detector):
        """Test detection of HELP NEEDED"""
        triggers = detector.process_line("I'm stuck and need assistance - HELP NEEDED")

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.HELP_NEEDED
        assert trigger.confidence == 0.9
        assert "HELP NEEDED" in trigger.matched_text

    def test_process_line_status_stuck(self, detector):
        """Test detection of Status: STUCK"""
        triggers = detector.process_line("Status: STUCK - cannot proceed")

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.HELP_NEEDED
        assert trigger.confidence == 0.85

    def test_process_line_json_task_complete(self, detector):
        """Test detection of structured JSON task completion"""
        json_line = '{"task_complete": {"task_id": "5", "title": "Fix authentication"}}'

        triggers = detector.process_line(json_line)

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.TASK_COMPLETE
        assert trigger.task_id == "5"
        assert trigger.task_title == "Fix authentication"
        assert trigger.json_data is not None

    def test_process_line_json_decision(self, detector):
        """Test detection of JSON decision object"""
        json_line = '{"action": "execute", "task_id": "3", "session_id": "abc123"}'

        triggers = detector.process_line(json_line)

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.JSON_DECISION
        assert trigger.task_id == "3"
        assert trigger.session_id == "abc123"

    def test_process_line_zen_mcp_call(self, detector):
        """Test detection of Zen MCP calls"""
        triggers = detector.process_line("Calling mcp__zen__codereview with parameters")

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.ZEN_MCP_CALL
        assert trigger.confidence == 0.95

    def test_process_line_import_error(self, detector):
        """Test detection of import errors"""
        triggers = detector.process_line("ModuleNotFoundError: No module named 'missing_module'")

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.ERROR_PATTERN
        assert trigger.error_details["error_type"] == "ModuleNotFoundError"

    def test_process_line_file_error(self, detector):
        """Test detection of file errors"""
        triggers = detector.process_line("FileNotFoundError: [Errno 2] No such file or directory: 'test.py'")

        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger.trigger_type == TriggerType.ERROR_PATTERN
        assert "test.py" in trigger.modified_files

    def test_process_line_no_match(self, detector):
        """Test line with no pattern matches"""
        triggers = detector.process_line("This is just a normal line of output")

        assert len(triggers) == 0

    def test_process_line_multiple_matches(self, detector):
        """Test line with multiple pattern matches"""
        # Create a line that matches multiple patterns
        triggers = detector.process_line("HELP NEEDED - ModuleNotFoundError occurred")

        # Should detect both help needed and error pattern
        assert len(triggers) == 2
        trigger_types = [t.trigger_type for t in triggers]
        assert TriggerType.HELP_NEEDED in trigger_types
        assert TriggerType.ERROR_PATTERN in trigger_types

    def test_process_buffer_multiple_lines(self, detector):
        """Test processing multiple lines at once"""
        text = """
        Starting task execution
        Implementing feature X
        ALL TASKS COMPLETE
        Shutting down
        """

        triggers = detector.process_buffer(text)

        # Should find the completion trigger
        completion_triggers = [t for t in triggers if t.trigger_type == TriggerType.ALL_TASKS_COMPLETE]
        assert len(completion_triggers) == 1

    def test_buffer_management(self, detector):
        """Test that buffer is managed properly"""
        # Process many lines to test buffer size management
        for i in range(1000):
            detector.process_line(f"Line {i} of test output")

        # Buffer should be limited in size
        assert len(detector.buffer) <= 50000
        assert detector.line_count == 1000

    def test_extract_json_task_data_valid(self, detector):
        """Test JSON task data extraction with valid JSON"""
        context = TriggerContext(TriggerType.TASK_COMPLETE, 1.0, "")
        match = Mock()
        match.group.return_value = '{"task_complete": {"task_id": "10", "title": "Test Task"}}'

        detector.extract_json_task_data(context, match, "", "")

        assert context.task_id == "10"
        assert context.task_title == "Test Task"
        assert context.json_data is not None

    def test_extract_json_task_data_invalid(self, detector):
        """Test JSON task data extraction with invalid JSON"""
        context = TriggerContext(TriggerType.TASK_COMPLETE, 1.0, "")
        match = Mock()
        match.group.return_value = '{"task_complete": invalid json}'

        # Should not raise exception
        detector.extract_json_task_data(context, match, "", "")

        assert context.task_id is None
        assert context.json_data is None

    def test_extract_mcp_context(self, detector):
        """Test MCP context extraction"""
        context = TriggerContext(TriggerType.ZEN_MCP_CALL, 1.0, "mcp__zen__codereview")
        match = Mock()
        match.group.return_value = "mcp__zen__codereview"

        buffer = '''
        Previous line
        "file_path": "/path/to/file.py"
        Calling mcp__zen__codereview
        task 5 analysis
        Next line
        '''

        detector.extract_mcp_context(context, match, "", buffer)

        assert "/path/to/file.py" in context.modified_files
        assert context.task_id == "5"

    def test_extract_error_details(self, detector):
        """Test error details extraction"""
        context = TriggerContext(TriggerType.ERROR_PATTERN, 1.0, "")
        match = Mock()
        match.group.return_value = "ImportError"
        match.group.side_effect = lambda x: "ImportError" if x == 1 else match.group.return_value

        line = 'ImportError: cannot import name "missing_func" from "test_module.py"'

        detector.extract_error_details(context, match, line, "")

        assert context.error_details["error_type"] == "ImportError"
        assert "test_module.py" in context.modified_files

    def test_callback_exception_handling(self, detector):
        """Test that callback exceptions are handled gracefully"""
        def failing_callback(context):
            raise Exception("Callback failed")

        detector.register_callback(TriggerType.TASK_COMPLETE, failing_callback)

        # Should not raise exception despite failing callback
        triggers = detector.process_line('{"task_complete": {"task_id": "1"}}')

        assert len(triggers) == 1

    def test_pattern_priority_ordering(self, detector):
        """Test that patterns are processed in priority order"""
        # Add a high-priority custom pattern
        high_priority_pattern = TriggerPattern(
            name="high_priority_test",
            trigger_type=TriggerType.TASK_COMPLETE,
            pattern=re.compile(r"HIGH PRIORITY"),
            priority=1  # Higher priority than default patterns
        )
        detector.patterns.append(high_priority_pattern)

        # The patterns should be sorted by priority when processing
        sorted_patterns = sorted(detector.patterns, key=lambda p: p.priority)
        assert sorted_patterns[0].name == "high_priority_test"

    def test_disabled_pattern(self, detector):
        """Test that disabled patterns are not processed"""
        # Disable a pattern
        for pattern in detector.patterns:
            if pattern.name == "all_tasks_complete":
                pattern.enabled = False
                break

        triggers = detector.process_line("ALL TASKS COMPLETE")

        # Should not trigger because pattern is disabled
        completion_triggers = [t for t in triggers if t.trigger_type == TriggerType.ALL_TASKS_COMPLETE]
        assert len(completion_triggers) == 0

    def test_get_pattern_stats(self, detector):
        """Test pattern statistics"""
        stats = detector.get_pattern_stats()

        assert "total_patterns" in stats
        assert "enabled_patterns" in stats
        assert "patterns_by_type" in stats
        assert "registered_callbacks" in stats

        assert stats["total_patterns"] > 0
        assert stats["enabled_patterns"] > 0

    def test_clear_buffer(self, detector):
        """Test buffer clearing"""
        detector.process_line("Some text")
        assert detector.buffer != ""
        assert detector.line_count == 1

        detector.clear_buffer()

        assert detector.buffer == ""
        assert detector.line_count == 0


class TestTriggerContext:
    """Test cases for TriggerContext dataclass"""

    def test_trigger_context_creation(self):
        """Test TriggerContext creation"""
        context = TriggerContext(
            trigger_type=TriggerType.TASK_COMPLETE,
            confidence=0.9,
            matched_text="test match"
        )

        assert context.trigger_type == TriggerType.TASK_COMPLETE
        assert context.confidence == 0.9
        assert context.matched_text == "test match"
        assert isinstance(context.timestamp, datetime)
        assert context.modified_files == []
        assert context.error_details == {}
        assert context.metadata == {}

    def test_trigger_context_with_data(self):
        """Test TriggerContext with additional data"""
        context = TriggerContext(
            trigger_type=TriggerType.HELP_NEEDED,
            confidence=1.0,
            matched_text="HELP NEEDED",
            task_id="5",
            modified_files=["file1.py", "file2.py"],
            help_category="architecture"
        )

        assert context.task_id == "5"
        assert len(context.modified_files) == 2
        assert context.help_category == "architecture"


class TestTriggerPattern:
    """Test cases for TriggerPattern dataclass"""

    def test_trigger_pattern_creation(self):
        """Test TriggerPattern creation"""
        pattern = TriggerPattern(
            name="test_pattern",
            trigger_type=TriggerType.ERROR_PATTERN,
            pattern=re.compile(r"ERROR"),
            confidence=0.8
        )

        assert pattern.name == "test_pattern"
        assert pattern.trigger_type == TriggerType.ERROR_PATTERN
        assert pattern.confidence == 0.8
        assert pattern.enabled is True
        assert pattern.priority == 50
        assert pattern.context_extractors == []


if __name__ == "__main__":
    pytest.main([__file__])
