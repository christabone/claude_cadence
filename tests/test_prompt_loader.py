# tests/test_prompt_loader.py

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

# Attempt to import jinja2 to conditionally skip tests
try:
    import jinja2
    JINJA2_INSTALLED = True
except ImportError:
    JINJA2_INSTALLED = False

# Assuming the file to be tested is in cadence/prompt_loader.py
# Adjust the import path if your structure is different.
from cadence.prompt_loader import (
    load_yaml_with_includes,
    load_yaml_string_with_includes,
    PromptLoader,
    YAMLIncludeLoader
)

@pytest.fixture
def create_files(tmp_path: Path):
    """A fixture to create a standard set of test files and directories."""
    # Main files
    (tmp_path / "main.yml").write_text("""
key: value
include_yaml: !include included.yml
include_text: !include included.txt
nested:
  level1: !include sub/nested.yml
    """)

    (tmp_path / "included.yml").write_text("item: included_item")
    (tmp_path / "included.txt").write_text("This is plain text.")

    # Nested files
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.yml").write_text("nested_key: nested_value")

    # Files for error cases
    (tmp_path / "malformed.yml").write_text("key: [missing_quote")
    (tmp_path / "no_permission.yml").write_text("secret")
    (tmp_path / "no_permission.yml").chmod(0o000)

    # Files for circular dependency tests
    (tmp_path / "circ_a.yml").write_text("a: !include circ_b.yml")
    (tmp_path / "circ_b.yml").write_text("b: !include circ_a.yml")

    # Files for security tests
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("sensitive_data")
    (tmp_path / "security.yml").write_text("data: !include ../outside/secret.txt")

    return tmp_path


class TestYAMLIncludeLoader:
    """Tests for the core !include functionality."""

    def test_load_simple_yaml(self, create_files):
        """Tests loading a YAML file with no includes."""
        # ARRANGE
        p = create_files / "included.yml"

        # ACT
        data = load_yaml_with_includes(p)

        # ASSERT
        assert data == {"item": "included_item"}

    def test_include_yaml_and_text_files(self, create_files):
        """Tests !include for both YAML and plain text files."""
        # ARRANGE
        p = create_files / "main.yml"

        # ACT
        data = load_yaml_with_includes(p)

        # ASSERT
        assert data["key"] == "value"
        assert data["include_yaml"] == {"item": "included_item"}
        assert data["include_text"] == "This is plain text."

    def test_nested_include_with_relative_paths(self, create_files):
        """Tests that nested includes resolve paths relative to the including file."""
        # ARRANGE
        p = create_files / "main.yml"

        # ACT
        data = load_yaml_with_includes(p)

        # ASSERT
        assert data["nested"]["level1"] == {"nested_key": "nested_value"}

    def test_load_from_string_with_base_dir(self, create_files):
        """Tests loading from a string, with includes resolved relative to a base_dir."""
        # ARRANGE
        yaml_string = "data: !include included.yml"

        # ACT
        data = load_yaml_string_with_includes(yaml_string, base_dir=create_files)

        # ASSERT
        assert data == {"data": {"item": "included_item"}}

    def test_error_file_not_found(self, create_files):
        """Tests that a clear error is raised for a missing include file."""
        # ARRANGE
        p = create_files / "test.yml"
        p.write_text("key: !include missing.yml")

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"Include file not found: missing.yml"):
            load_yaml_with_includes(p)

    def test_error_include_is_directory(self, create_files):
        """Tests that including a directory raises an error."""
        # ARRANGE
        p = create_files / "test.yml"
        p.write_text("key: !include sub")

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"Include path is not a file: sub"):
            load_yaml_with_includes(p)

    def test_error_circular_dependency(self, create_files):
        """Tests that a circular dependency (A->B->A) is detected and raises an error."""
        # ARRANGE
        p = create_files / "circ_a.yml"

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"Circular include dependency detected"):
            load_yaml_with_includes(p)

    def test_error_permission_denied(self, create_files):
        """Tests that a permission error on an included file is handled gracefully."""
        # ARRANGE
        p = create_files / "test.yml"
        p.write_text("key: !include no_permission.yml")

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"Permission denied reading include file"):
            load_yaml_with_includes(p)

        # Clean up file permissions to allow deletion
        (create_files / "no_permission.yml").chmod(0o644)

    def test_error_malformed_include(self, create_files):
        """Tests that an error in an included YAML file is reported correctly."""
        # ARRANGE
        p = create_files / "test.yml"
        p.write_text("key: !include malformed.yml")

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"YAML error in included file: malformed.yml"):
            load_yaml_with_includes(p)

    def test_security_path_traversal_fails(self, create_files):
        """
        CRITICAL: Ensures that !include cannot be used for path traversal to
        access files outside the base directory. This is a check on line 81.
        """
        # ARRANGE
        p = create_files / "sub" / "security_test.yml"
        # This path tries to go from sub/ up to root and then into outside/
        p.write_text("data: !include ../../outside/secret.txt")

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"attempts to access a file outside of the base directory"):
            load_yaml_with_includes(p)

    def test_security_path_traversal_from_string_load_fails(self, create_files):
        """
        CRITICAL: Ensures path traversal with '..' is blocked when loading from a string.
        This is a check on line 91.
        """
        # ARRANGE
        yaml_string = "data: !include ../secret.txt"

        # ACT & ASSERT
        with pytest.raises(yaml.YAMLError, match=r"contains parent directory references"):
            # When base_dir is not provided, it defaults to cwd, so this check is active
            load_yaml_string_with_includes(yaml_string)


class TestPromptLoader:
    """Tests for the PromptLoader class that consumes the YAML loader."""

    @pytest.fixture
    def prompt_config_path(self, tmp_path: Path) -> Path:
        """Creates a complex prompt config for testing."""
        (tmp_path / "shared.yml").write_text("""
common_rule: "Always be helpful."
jinja_section: |
  {% if is_expert %}Expert mode enabled.{% else %}User mode.{% endif %}
        """)

        config_path = tmp_path / "prompts.yml"
        config_path.write_text("""
agent:
  role: "You are a helpful AI assistant."
  rules:
    - !include shared.yml
  dynamic_rule: "{extra_rule}"

expert_agent:
  role: "{agent.role}" # Config reference
  rules:
    - "{agent.rules.0.common_rule}" # Nested config reference
    - "{extra_rule}" # Context reference

recursive_ref_a: "Start -> {recursive_ref_b}"
recursive_ref_b: "Middle -> {recursive_ref_c}"
recursive_ref_c: "End"

circular_ref_a: "{circular_ref_b}"
circular_ref_b: "{circular_ref_a}"

jinja_template: !include shared.yml
        """)
        return config_path

    def test_init_and_get_template(self, prompt_config_path):
        """Tests successful initialization and template retrieval."""
        # ARRANGE & ACT
        loader = PromptLoader(prompt_config_path)
        template = loader.get_template("agent.role")

        # ASSERT
        assert template == "You are a helpful AI assistant."
        assert loader.config["agent"]["rules"][0]["common_rule"] == "Always be helpful."

    def test_get_template_missing_path(self, prompt_config_path):
        """Tests that getting a non-existent path returns an empty string."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)

        # ACT
        template = loader.get_template("agent.non_existent.key")

        # ASSERT
        assert template == ""

    def test_format_template_with_context(self, prompt_config_path):
        """Tests simple .format() style replacement from the context dict."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = loader.get_template("agent.dynamic_rule")
        context = {"extra_rule": "Be concise."}

        # ACT
        formatted = loader.format_template(template, context)

        # ASSERT
        assert formatted == "Be concise."

    def test_format_template_with_config_refs(self, prompt_config_path):
        """Tests replacement of {key.path} style references from the config itself."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = loader.get_template("expert_agent.role")
        context = {"extra_rule": "Be concise."}

        # ACT
        formatted = loader.format_template(template, context)

        # ASSERT
        assert formatted == "You are a helpful AI assistant."

    def test_format_template_with_nested_config_refs(self, prompt_config_path):
        """Tests deeply nested config references."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = loader.get_template("expert_agent.rules.0")

        # ACT
        formatted = loader.format_template(template, {})

        # ASSERT
        assert formatted == "Always be helpful."

    def test_format_template_recursive_refs(self, prompt_config_path):
        """Tests that config references can be recursively formatted."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = "{recursive_ref_a}"

        # ACT
        formatted = loader.format_template(template, {})

        # ASSERT
        assert formatted == "Start -> Middle -> End"

    def test_format_template_circular_ref_detection(self, prompt_config_path):
        """Tests that circular config references are detected and handled."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = "{circular_ref_a}"

        # ACT
        formatted = loader.format_template(template, {})

        # ASSERT
        assert "CYCLE_DETECTED: circular_ref_a" in formatted

    @pytest.mark.skipif(not JINJA2_INSTALLED, reason="jinja2 is not installed")
    def test_format_template_with_jinja(self, prompt_config_path):
        """Tests that Jinja2 templates are rendered correctly."""
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = loader.get_template("jinja_template.jinja_section")

        # ACT (expert mode)
        formatted_expert = loader.format_template(template, {"is_expert": True})

        # ACT (user mode)
        formatted_user = loader.format_template(template, {"is_expert": False})

        # ASSERT
        assert formatted_expert.strip() == "Expert mode enabled."
        assert formatted_user.strip() == "User mode."

    def test_format_template_graceful_key_error(self, prompt_config_path):
        """
        Tests that formatting gracefully handles missing keys, which can be
        valid for optional prompt sections. The check is on line 357.
        """
        # ARRANGE
        loader = PromptLoader(prompt_config_path)
        template = "This section is optional: {optional_key}"

        # ACT
        formatted = loader.format_template(template, {})

        # ASSERT
        # It should not raise a KeyError and should leave the placeholder
        assert formatted == "This section is optional: {optional_key}"
