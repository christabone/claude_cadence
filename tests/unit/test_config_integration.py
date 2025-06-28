"""
Tests for ReviewTriggerDetector integration with config.yaml
"""

import pytest
import yaml
from pathlib import Path
from cadence.review_trigger_detector import ReviewTriggerDetector, TriggerType


class TestConfigIntegration:
    """Test cases for config.yaml integration"""

    @pytest.fixture
    def config_path(self):
        """Get path to the project config.yaml"""
        return Path(__file__).parent.parent.parent / "config.yaml"

    @pytest.fixture
    def config_data(self, config_path):
        """Load config.yaml data"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def test_config_file_has_triggers_section(self, config_data):
        """Test that config.yaml has triggers section"""
        assert "triggers" in config_data
        assert "patterns" in config_data["triggers"]
        assert isinstance(config_data["triggers"]["patterns"], list)

    def test_config_trigger_patterns_valid(self, config_data):
        """Test that config trigger patterns have required fields"""
        patterns = config_data["triggers"]["patterns"]

        for pattern in patterns:
            # Required fields
            assert "name" in pattern
            assert "type" in pattern
            assert "pattern" in pattern

            # Type must be valid
            assert pattern["type"] in [t.value for t in TriggerType]

            # Optional fields should have valid types if present
            if "confidence" in pattern:
                assert isinstance(pattern["confidence"], (int, float))
                assert 0.0 <= pattern["confidence"] <= 1.0

            if "priority" in pattern:
                assert isinstance(pattern["priority"], int)
                assert pattern["priority"] >= 0

            if "enabled" in pattern:
                assert isinstance(pattern["enabled"], bool)

    def test_detector_loads_custom_patterns_from_config(self, config_data):
        """Test that ReviewTriggerDetector can load custom patterns from config"""
        detector = ReviewTriggerDetector(config_data)

        # Should have both default and custom patterns
        pattern_names = [p.name for p in detector.patterns]

        # Check that custom patterns from config are loaded
        expected_custom_patterns = [
            "caps_task_complete",
            "need_assistance",
            "permission_error"
        ]

        for custom_pattern in expected_custom_patterns:
            assert custom_pattern in pattern_names

        # Should also have default patterns
        assert "all_tasks_complete" in pattern_names
        assert "help_needed" in pattern_names

    def test_custom_pattern_functionality(self, config_data):
        """Test that custom patterns from config work correctly"""
        detector = ReviewTriggerDetector(config_data)

        # Test caps task complete pattern
        triggers = detector.process_line("TASK COMPLETED successfully!")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == TriggerType.TASK_COMPLETE
        assert triggers[0].confidence == 0.9

        # Test custom help pattern
        triggers = detector.process_line("I NEED ASSISTANCE with this implementation")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == TriggerType.HELP_NEEDED
        assert triggers[0].confidence == 0.85

        # Test custom error pattern
        triggers = detector.process_line("PermissionError: access denied to file")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == TriggerType.ERROR_PATTERN
        assert triggers[0].confidence == 0.8

    def test_pattern_priority_from_config(self, config_data):
        """Test that pattern priorities from config are respected"""
        detector = ReviewTriggerDetector(config_data)

        # Find the custom caps_task_complete pattern
        caps_pattern = None
        for pattern in detector.patterns:
            if pattern.name == "caps_task_complete":
                caps_pattern = pattern
                break

        assert caps_pattern is not None
        assert caps_pattern.priority == 12

        # Test that it has higher priority than the default all_tasks_complete (priority 10)
        default_pattern = None
        for pattern in detector.patterns:
            if pattern.name == "all_tasks_complete":
                default_pattern = pattern
                break

        assert default_pattern is not None
        assert caps_pattern.priority > default_pattern.priority

    def test_pattern_enable_disable_from_config(self, config_data):
        """Test that patterns can be enabled/disabled via config"""
        # Modify config to disable a pattern
        modified_config = config_data.copy()
        for pattern in modified_config["triggers"]["patterns"]:
            if pattern["name"] == "permission_error":
                pattern["enabled"] = False
                break

        detector = ReviewTriggerDetector(modified_config)

        # Should not trigger the disabled pattern
        triggers = detector.process_line("PermissionError: access denied")

        # Should not match the disabled permission_error pattern
        permission_triggers = [t for t in triggers if "permission" in t.matched_text.lower()]
        assert len(permission_triggers) == 0

    def test_invalid_config_pattern_handling(self):
        """Test that invalid config patterns are handled gracefully"""
        invalid_config = {
            "triggers": {
                "patterns": [
                    {
                        "name": "invalid_pattern",
                        "type": "invalid_type",  # Invalid trigger type
                        "pattern": "test pattern"
                    },
                    {
                        "name": "missing_fields"
                        # Missing required type and pattern fields
                    },
                    {
                        "name": "valid_pattern",
                        "type": "task_complete",
                        "pattern": "VALID PATTERN"
                    }
                ]
            }
        }

        # Should create detector without crashing
        detector = ReviewTriggerDetector(invalid_config)

        # Should have default patterns plus any valid custom ones
        pattern_names = [p.name for p in detector.patterns]
        assert "all_tasks_complete" in pattern_names  # Default pattern
        assert "valid_pattern" in pattern_names  # Valid custom pattern

        # Should not have invalid patterns
        assert "invalid_pattern" not in pattern_names
        assert "missing_fields" not in pattern_names


if __name__ == "__main__":
    pytest.main([__file__])
