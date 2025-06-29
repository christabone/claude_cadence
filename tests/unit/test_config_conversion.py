"""
Tests for config conversion logic in FixAgentDispatcher

These tests verify that configuration objects are properly converted
between different dispatcher classes during initialization.
"""

import unittest
from cadence.config import FixAgentDispatcherConfig, CircularDependencyConfig
from cadence.fix_agent_dispatcher import FixAgentDispatcher


class TestConfigConversion(unittest.TestCase):
    """Test config conversion logic for dispatchers"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = FixAgentDispatcherConfig(
            max_attempts=5,
            timeout_ms=120000,
            enable_auto_fix=True,
            severity_threshold="medium",
            enable_verification=False,
            verification_timeout_ms=30000,
            circular_dependency=CircularDependencyConfig(
                max_file_modifications=2,
                min_attempts_before_check=3
            )
        )

    def test_fix_agent_dispatcher_config_conversion(self):
        """Test that FixAgentDispatcherConfig is properly converted for parent dispatcher"""
        dispatcher = FixAgentDispatcher(self.test_config)

        # Check that fix-specific config is preserved
        self.assertEqual(dispatcher.max_attempts, 5)
        self.assertEqual(dispatcher.timeout_ms, 120000)
        self.assertEqual(dispatcher.enable_auto_fix, True)
        self.assertEqual(dispatcher.severity_threshold, "medium")
        self.assertEqual(dispatcher.enable_verification, False)
        self.assertEqual(dispatcher.verification_timeout_ms, 30000)

        # Check that circular dependency config is preserved
        self.assertEqual(dispatcher._max_file_modifications, 2)
        self.assertEqual(dispatcher._min_attempts_before_check, 3)

    def test_config_defaults_when_none_provided(self):
        """Test that default config is used when None is provided"""
        dispatcher = FixAgentDispatcher(None)

        # Should use defaults from FixAgentDispatcherConfig
        self.assertEqual(dispatcher.max_attempts, 3)
        self.assertEqual(dispatcher.timeout_ms, 300000)  # DEFAULT_FIX_TIMEOUT_MS
        self.assertEqual(dispatcher.enable_auto_fix, True)
        self.assertEqual(dispatcher.severity_threshold, "high")
        self.assertEqual(dispatcher.enable_verification, True)
        self.assertEqual(dispatcher.verification_timeout_ms, 60000)  # DEFAULT_VERIFICATION_TIMEOUT_MS

    def test_config_validation_during_conversion(self):
        """Test that config validation occurs during dispatcher initialization"""
        # Test invalid timeout (too small)
        invalid_config = FixAgentDispatcherConfig(
            timeout_ms=500  # Below minimum of 1000ms
        )

        with self.assertRaises(ValueError) as context:
            FixAgentDispatcher(invalid_config)

        self.assertIn("timeout_ms must be at least", str(context.exception))

    def test_circular_dependency_config_validation(self):
        """Test that circular dependency config is properly validated"""
        # Test invalid circular dependency config
        invalid_circular_config = CircularDependencyConfig(
            max_file_modifications=0  # Should be positive
        )

        with self.assertRaises(ValueError) as context:
            invalid_config = FixAgentDispatcherConfig(
                circular_dependency=invalid_circular_config
            )

        self.assertIn("max_file_modifications must be positive", str(context.exception))


if __name__ == '__main__':
    unittest.main()
