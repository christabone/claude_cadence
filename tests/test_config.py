# tests/test_config.py

import os
import yaml
import pytest
from pathlib import Path
from typing import Generator

from cadence.config import (
    CadenceConfig,
    ConfigLoader,
    ExecutionConfig,
    SupervisorConfig,
    ZenIntegrationConfig,
)

# A sample valid YAML configuration for testing overrides
SAMPLE_YAML_CONFIG = """
execution:
  max_agent_turns: 150
  log_level: "INFO"

supervisor:
  model: "claude-3-opus-20240229"
  zen_integration:
    enabled: false
    auto_debug_threshold: 5

project:
  root_directory: "/tmp/cadence_test"
  extra_key: "extra_value"

development:
  debug: true
"""

# A YAML configuration with an invalid data type
INVALID_TYPE_YAML_CONFIG = """
execution:
  max_agent_turns: "one hundred" # Invalid type: should be int
"""

# A malformed YAML configuration
MALFORMED_YAML_CONFIG = """
execution:
  max_agent_turns: 100
  log_level: "INFO"
supervisor:
  model: "test-model"
    zen_integration: # Bad indentation
      enabled: true
"""


@pytest.fixture
def config_fs(fs) -> Generator[Path, None, None]:
    """
    Sets up a fake filesystem with a home directory and a current working directory.
    `fs` is the pyfakefs fixture.
    """
    # pyfakefs doesn't create home dir by default on non-Windows
    home_dir = Path.home()
    if not home_dir.exists():
        fs.create_dir(home_dir)

    # Create a consistent CWD for tests
    cwd = "/test_project"
    fs.create_dir(cwd)
    os.chdir(cwd)
    yield Path(cwd)


class TestConfigDataClasses:
    """Tests the default values and structure of the configuration dataclasses."""

    def test_cadence_config_defaults(self):
        """
        Verify that a default CadenceConfig instance is created with nested
        dataclasses containing their own correct defaults.
        """
        config = CadenceConfig()
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.supervisor, SupervisorConfig)
        assert isinstance(config.supervisor.zen_integration, ZenIntegrationConfig)
        assert config.execution.max_agent_turns == 120  # Default from ExecutionConfig
        assert config.supervisor.model == "heuristic"
        assert config.supervisor.zen_integration.enabled is True

    def test_project_root_defaults_to_cwd(self):
        """
        Ensures the project root directory defaults to the current working directory
        at the time of instantiation.
        """
        config = CadenceConfig()
        assert config.project["root_directory"] == os.getcwd()


class TestConfigLoaderFinding:
    """Tests the logic for finding configuration files."""

    def test_find_config_no_file(self, config_fs: Path):
        """
        When no config file exists, _find_config_file should return None.
        """
        loader = ConfigLoader()
        assert loader.config_path is None

    def test_find_config_explicit_path_exists(self, config_fs: Path):
        """
        The loader should find the file at the explicitly provided path.
        """
        config_path = config_fs / "my_cadence.yaml"
        config_path.touch()
        loader = ConfigLoader(str(config_path))
        assert loader.config_path == config_path

    def test_find_config_explicit_path_not_found(self, config_fs: Path):
        """
        The loader should raise FileNotFoundError for a non-existent explicit path.
        """
        with pytest.raises(FileNotFoundError):
            ConfigLoader("non_existent_file.yaml")

    def test_find_config_from_env_var(self, config_fs: Path, monkeypatch):
        """
        The loader should prioritize and find the file specified by the
        CADENCE_CONFIG environment variable.
        """
        env_config_path = config_fs / "env_config.yaml"
        env_config_path.touch()
        monkeypatch.setenv("CADENCE_CONFIG", str(env_config_path))

        loader = ConfigLoader()
        assert loader.config_path == env_config_path

    def test_find_config_from_default_locations(self, config_fs: Path):
        """
        The loader should find a config file in one of the default search paths,
        like cadence.yaml in the current directory.
        """
        default_config_path = config_fs / "cadence.yaml"
        default_config_path.touch()

        loader = ConfigLoader()
        assert loader.config_path == default_config_path

    def test_find_config_priority_order(self, config_fs: Path, monkeypatch):
        """
        Verify the search priority: explicit path > environment variable > default path.
        """
        # 1. Create all three potential config files
        explicit_path = config_fs / "explicit.yaml"
        explicit_path.touch()

        env_path = config_fs / "env.yaml"
        env_path.touch()
        monkeypatch.setenv("CADENCE_CONFIG", str(env_path))

        default_path = config_fs / "cadence.yaml"
        default_path.touch()

        # 2. Test that explicit path wins
        loader_explicit = ConfigLoader(str(explicit_path))
        assert loader_explicit.config_path == explicit_path

        # 3. Test that env var wins over default
        loader_env = ConfigLoader()
        assert loader_env.config_path == env_path

        # 4. Test that default is found when others are absent
        monkeypatch.delenv("CADENCE_CONFIG")
        loader_default = ConfigLoader()
        assert loader_default.config_path == default_path


class TestConfigLoaderLoading:
    """Tests the logic for loading and merging YAML configuration."""

    def test_load_with_no_config_file(self, config_fs: Path):
        """
        When no config file is found, the loader should return a default CadenceConfig.
        """
        loader = ConfigLoader()
        assert loader.config == CadenceConfig()

    def test_load_and_override_values(self, config_fs: Path):
        """
        Verify that values from a loaded YAML file correctly override the defaults
        in both top-level and nested dataclasses.
        """
        config_path = config_fs / "cadence.yaml"
        config_path.write_text(SAMPLE_YAML_CONFIG)

        loader = ConfigLoader()
        config = loader.config

        # Test overridden dataclass value
        assert config.execution.max_agent_turns == 150
        assert config.execution.log_level == "INFO"

        # Test overridden nested dataclass value
        assert config.supervisor.model == "claude-3-opus-20240229"
        assert config.supervisor.zen_integration.enabled is False
        assert config.supervisor.zen_integration.auto_debug_threshold == 5

        # Test deep-merged dictionary
        assert config.project["root_directory"] == "/tmp/cadence_test"
        assert config.project["extra_key"] == "extra_value"
        assert config.project["taskmaster_file"] is None  # Default preserved

        # Test another overridden value
        assert config.development["debug"] is True

    def test_load_empty_yaml_file(self, config_fs: Path):
        """
        An empty YAML file should result in a default configuration.
        """
        config_path = config_fs / "empty.yaml"
        config_path.touch()

        loader = ConfigLoader(str(config_path))
        assert loader.config == CadenceConfig()

    def test_load_yaml_with_null_section(self, config_fs: Path):
        """
        If a section in YAML is null, it should fall back to the default
        dataclass for that section.
        """
        yaml_content = "execution: null\n"
        config_path = config_fs / "null_section.yaml"
        config_path.write_text(yaml_content)

        loader = ConfigLoader(str(config_path))
        # The execution section should be a default ExecutionConfig, not None
        assert isinstance(loader.config.execution, ExecutionConfig)
        assert loader.config.execution == ExecutionConfig()


class TestConfigLoaderEdgeCases:
    """Tests resilience against malformed or problematic config files."""

    def test_load_malformed_yaml(self, config_fs: Path, caplog):
        """
        Malformed YAML should be caught, log a warning, and return a default config.
        """
        config_path = config_fs / "malformed.yaml"
        config_path.write_text(MALFORMED_YAML_CONFIG)

        loader = ConfigLoader(str(config_path))

        # Should fall back to default config
        assert loader.config == CadenceConfig()
        # Should log a warning about the failure
        assert "Failed to load config" in caplog.text
        assert "Using default configuration" in caplog.text

    def test_load_invalid_data_type(self, config_fs: Path, caplog):
        """
        A field with an incorrect data type should cause that specific dataclass
        to fall back to its defaults, without affecting other sections.
        """
        config_path = config_fs / "invalid_type.yaml"
        config_path.write_text(INVALID_TYPE_YAML_CONFIG)

        # Add another valid section to ensure it's not affected
        valid_yaml = "supervisor:\n  model: 'overridden-model'"
        config_path.write_text(INVALID_TYPE_YAML_CONFIG + valid_yaml)

        loader = ConfigLoader(str(config_path))
        config = loader.config

        # The 'execution' section with the error should revert to defaults
        assert config.execution == ExecutionConfig()
        assert "Invalid execution config" in caplog.text

        # Other valid sections should still be loaded correctly
        assert config.supervisor.model == "overridden-model"

    def test_load_file_permission_error(self, config_fs: Path, caplog):
        """
        A file with no read permissions should be handled gracefully.
        """
        config_path = config_fs / "unreadable.yaml"
        config_path.write_text("content")
        config_path.chmod(0o000) # No permissions

        loader = ConfigLoader(str(config_path))

        assert loader.config == CadenceConfig()
        assert "Failed to load config" in caplog.text
        assert "Permission denied" in caplog.text


class TestConfigLoaderSaving:
    """Tests the config saving functionality."""

    def test_save_and_reload_config(self, config_fs: Path):
        """
        Verify that saving a modified config and reloading it produces an
        equivalent configuration object.
        """
        # 1. Load a config and modify it
        loader1 = ConfigLoader()
        loader1.config.execution.max_agent_turns = 999
        loader1.config.project["new_key"] = "new_value"

        # 2. Save it to a new file
        save_path = config_fs / "saved_config.yaml"
        loader1.save(str(save_path))
        assert save_path.exists()

        # 3. Load the saved file with a new loader
        loader2 = ConfigLoader(str(save_path))

        # 4. Compare the configs
        assert loader1.config.execution.max_agent_turns == loader2.config.execution.max_agent_turns
        assert loader2.config.project["new_key"] == "new_value"

    def test_save_to_default_path(self, config_fs: Path):
        """
        When no path is given and none was loaded, it should save to 'cadence.yaml'.
        """
        loader = ConfigLoader() # No file loaded
        loader.save() # No path provided

        expected_path = config_fs / "cadence.yaml"
        assert expected_path.exists()
        with open(expected_path, 'r') as f:
            data = yaml.safe_load(f)
            assert "execution" in data
            assert data["execution"]["max_agent_turns"] == 80 # Default value


class TestConfigLoaderOverrides:
    """Tests the command-line argument override functionality."""

    def test_override_from_args_valid(self, config_fs: Path):
        """
        Verify that a valid, nested key can be overridden.
        """
        loader = ConfigLoader()
        original_value = loader.config.execution.max_agent_turns
        assert original_value != 999

        loader.override_from_args(**{'execution.max_agent_turns': 999})

        assert loader.config.execution.max_agent_turns == 999

    def test_override_from_args_invalid_key(self, config_fs: Path, caplog):
        """
        An attempt to override an unknown key should be ignored and logged.
        """
        loader = ConfigLoader()
        loader.override_from_args(**{'execution.non_existent_key': 'value'})

        assert "Ignoring unknown config override" in caplog.text

    def test_override_from_args_invalid_type(self, config_fs: Path, caplog):
        """
        An attempt to override with an invalid type should be ignored and logged.
        """
        loader = ConfigLoader()
        original_value = loader.config.execution.max_agent_turns

        loader.override_from_args(**{'execution.max_agent_turns': 'not-a-number'})

        assert "Invalid value for execution.max_agent_turns" in caplog.text
        # Value should remain unchanged
        assert loader.config.execution.max_agent_turns == original_value

    def test_override_from_args_top_level_not_allowed(self, config_fs: Path, caplog):
        """
        An attempt to override a top-level attribute should be disallowed.
        """
        loader = ConfigLoader()
        loader.override_from_args(**{'execution': {'max_agent_turns': 500}})

        assert "Top-level config overrides not allowed" in caplog.text
