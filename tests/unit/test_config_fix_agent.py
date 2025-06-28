"""Test configuration loading for fix_agent_dispatcher"""

import os
import yaml
import tempfile
import pytest
from pathlib import Path

from cadence.config import ConfigLoader, CadenceConfig


class TestFixAgentDispatcherConfig:
    """Test fix_agent_dispatcher configuration loading"""

    def test_default_fix_agent_config(self):
        """Test default fix_agent_dispatcher configuration"""
        config = CadenceConfig()

        assert config.fix_agent_dispatcher.max_attempts == 3
        assert config.fix_agent_dispatcher.timeout_ms == 300000
        assert config.fix_agent_dispatcher.enable_auto_fix is True
        assert config.fix_agent_dispatcher.severity_threshold == "high"
        assert config.fix_agent_dispatcher.enable_verification is True
        assert config.fix_agent_dispatcher.verification_timeout_ms == 60000
        assert config.fix_agent_dispatcher.circular_dependency.max_file_modifications == 3
        assert config.fix_agent_dispatcher.circular_dependency.min_attempts_before_check == 5

    def test_load_fix_agent_config_from_yaml(self):
        """Test loading fix_agent_dispatcher configuration from YAML"""
        config_data = {
            "fix_agent_dispatcher": {
                "max_attempts": 5,
                "timeout_ms": 600000,
                "enable_auto_fix": False,
                "severity_threshold": "critical",
                "enable_verification": False,
                "verification_timeout_ms": 120000,
                "circular_dependency": {
                    "max_file_modifications": 5,
                    "min_attempts_before_check": 10
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            config = loader.config

            assert config.fix_agent_dispatcher.max_attempts == 5
            assert config.fix_agent_dispatcher.timeout_ms == 600000
            assert config.fix_agent_dispatcher.enable_auto_fix is False
            assert config.fix_agent_dispatcher.severity_threshold == "critical"
            assert config.fix_agent_dispatcher.enable_verification is False
            assert config.fix_agent_dispatcher.verification_timeout_ms == 120000
            assert config.fix_agent_dispatcher.circular_dependency.max_file_modifications == 5
            assert config.fix_agent_dispatcher.circular_dependency.min_attempts_before_check == 10
        finally:
            os.unlink(config_path)

    def test_partial_fix_agent_config(self):
        """Test partial fix_agent_dispatcher configuration with defaults"""
        config_data = {
            "fix_agent_dispatcher": {
                "max_attempts": 10,
                "severity_threshold": "medium"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            config = loader.config

            # Overridden values
            assert config.fix_agent_dispatcher.max_attempts == 10
            assert config.fix_agent_dispatcher.severity_threshold == "medium"

            # Default values should still be present
            assert config.fix_agent_dispatcher.timeout_ms == 300000
            assert config.fix_agent_dispatcher.enable_auto_fix is True
            assert config.fix_agent_dispatcher.enable_verification is True
        finally:
            os.unlink(config_path)

    def test_fix_agent_config_with_other_configs(self):
        """Test fix_agent_dispatcher configuration alongside other configurations"""
        config_data = {
            "execution": {
                "max_supervisor_turns": 100,
                "max_agent_turns": 150
            },
            "fix_agent_dispatcher": {
                "max_attempts": 7,
                "enable_auto_fix": True
            },
            "development": {
                "debug": True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            config = loader.config

            # Check all configs loaded correctly
            assert config.execution.max_supervisor_turns == 100
            assert config.execution.max_agent_turns == 150
            assert config.fix_agent_dispatcher.max_attempts == 7
            assert config.fix_agent_dispatcher.enable_auto_fix is True
            assert config.development["debug"] is True
        finally:
            os.unlink(config_path)
