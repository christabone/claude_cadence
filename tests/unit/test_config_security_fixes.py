"""Test security fixes for configuration validation"""

import pytest
import sys
import tempfile
import yaml
import os
from pathlib import Path

from cadence.config import FixAgentDispatcherConfig, ConfigLoader


class TestConfigSecurityFixes:
    """Test security-related validation fixes"""

    def test_max_attempts_upper_bound_validation(self):
        """Test that max_attempts has proper upper bound validation"""
        # Test valid values
        valid_values = [1, 50, 100]
        for val in valid_values:
            config = FixAgentDispatcherConfig(max_attempts=val)
            assert config.max_attempts == val

        # Test invalid values (too high)
        invalid_high_values = [101, 1000, sys.maxsize]
        for val in invalid_high_values:
            with pytest.raises(ValueError, match="max_attempts must be between 1 and 100"):
                FixAgentDispatcherConfig(max_attempts=val)

        # Test invalid values (too low)
        invalid_low_values = [0, -1, -100]
        for val in invalid_low_values:
            with pytest.raises(ValueError, match="max_attempts must be between 1 and 100"):
                FixAgentDispatcherConfig(max_attempts=val)

    def test_severity_threshold_runtime_validation(self):
        """Test that severity_threshold is validated at runtime"""
        # Test valid values
        valid_severities = ["low", "medium", "high", "critical"]
        for severity in valid_severities:
            config = FixAgentDispatcherConfig(severity_threshold=severity)
            assert config.severity_threshold == severity

        # Test invalid values
        invalid_severities = ["invalid", "critcal", "LOW", "HIGH", "none", ""]
        for severity in invalid_severities:
            with pytest.raises(ValueError, match="severity_threshold must be one of"):
                FixAgentDispatcherConfig(severity_threshold=severity)

    def test_severity_threshold_yaml_validation(self):
        """Test that invalid severity_threshold in YAML is handled gracefully"""
        config_data = {
            "fix_agent_dispatcher": {
                "severity_threshold": "invalid_severity",
                "max_attempts": 5
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Should fall back to defaults due to validation error
            loader = ConfigLoader(config_path)
            config = loader.config

            # Should use default severity due to validation failure
            assert config.fix_agent_dispatcher.severity_threshold == "high"  # default value
            assert config.fix_agent_dispatcher.max_attempts == 3  # default value, not the invalid config
        finally:
            os.unlink(config_path)

    def test_max_attempts_yaml_validation(self):
        """Test that invalid max_attempts in YAML is handled gracefully"""
        config_data = {
            "fix_agent_dispatcher": {
                "max_attempts": sys.maxsize,  # Should trigger validation error
                "severity_threshold": "high"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Should fall back to defaults due to validation error
            loader = ConfigLoader(config_path)
            config = loader.config

            # Should use default values due to validation failure
            assert config.fix_agent_dispatcher.max_attempts == 3  # default value
            assert config.fix_agent_dispatcher.severity_threshold == "high"  # default value
        finally:
            os.unlink(config_path)

    def test_combined_validation_edge_cases(self):
        """Test combinations of valid and invalid values"""
        # Valid combination
        config = FixAgentDispatcherConfig(
            max_attempts=10,
            severity_threshold="medium",
            timeout_ms=5000
        )
        assert config.max_attempts == 10
        assert config.severity_threshold == "medium"
        assert config.timeout_ms == 5000

        # Invalid severity with valid max_attempts should fail
        with pytest.raises(ValueError, match="severity_threshold must be one of"):
            FixAgentDispatcherConfig(
                max_attempts=10,
                severity_threshold="invalid"
            )

        # Valid severity with invalid max_attempts should fail
        with pytest.raises(ValueError, match="max_attempts must be between 1 and 100"):
            FixAgentDispatcherConfig(
                max_attempts=1000,
                severity_threshold="high"
            )

    def test_literal_type_introspection(self):
        """Test that the Literal type introspection works correctly"""
        from typing import get_args

        # Get the valid severities from the type annotation
        valid_severities = get_args(FixAgentDispatcherConfig.__annotations__['severity_threshold'])
        expected_severities = ('low', 'medium', 'high', 'critical')

        assert valid_severities == expected_severities

        # Test that our validation uses the same values
        for severity in valid_severities:
            config = FixAgentDispatcherConfig(severity_threshold=severity)
            assert config.severity_threshold == severity
