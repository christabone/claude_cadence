"""
Comprehensive unit tests for PromptLoader class

This test suite provides complete coverage of the PromptLoader class functionality
including template formatting, config navigation, and integration with the custom YAML loader.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from cadence.prompt_loader import PromptLoader


class TestPromptLoaderInitialization:
    """Test PromptLoader initialization and configuration loading"""

    def test_default_initialization(self):
        """Test initialization with default prompts.yaml"""
        loader = PromptLoader()

        assert loader.config is not None
        assert isinstance(loader.config, dict)
        # Should load the actual prompts.yaml structure
        assert "core_agent_context" in loader.config
        assert "agent_prompts" in loader.config

    def test_custom_config_path_string(self, temp_dir):
        """Test initialization with custom config path as string"""
        custom_config = {
            "test_section": {
                "template": "Hello {name}",
                "value": "test"
            }
        }

        config_file = temp_dir / "custom.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(custom_config, f)

        loader = PromptLoader(str(config_file))
        assert loader.config["test_section"]["template"] == "Hello {name}"

    def test_custom_config_path_pathlib(self, temp_dir):
        """Test initialization with custom config path as Path object"""
        custom_config = {
            "test_section": {
                "template": "Hello {name}",
                "value": "test"
            }
        }

        config_file = temp_dir / "custom.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(custom_config, f)

        loader = PromptLoader(config_file)
        assert loader.config["test_section"]["template"] == "Hello {name}"

    def test_missing_config_file(self):
        """Test error handling for missing config file"""
        with pytest.raises(IOError, match="Prompt configuration file not found"):
            PromptLoader("/nonexistent/config.yaml")

    def test_malformed_config_file(self, temp_dir):
        """Test error handling for malformed YAML config"""
        bad_config = temp_dir / "bad.yaml"
        bad_config.write_text("{ invalid yaml :")

        with pytest.raises(ValueError, match="Error parsing YAML file"):
            PromptLoader(bad_config)

    def test_config_with_includes(self, temp_dir):
        """Test config loading with !include tags"""
        # Create included content
        included_file = temp_dir / "included.md"
        included_file.write_text("# Included Content\n\nThis is included content.")

        # Create main config with include
        config = {
            "section": {
                "included_content": "!include included.md"
            }
        }

        config_file = temp_dir / "config.yaml"
        config_file.write_text("section:\n  included_content: !include included.md")

        loader = PromptLoader(config_file)
        assert "Included Content" in loader.config["section"]["included_content"]


class TestTemplateFormatting:
    """Test template formatting functionality"""

    def test_simple_format(self):
        """Test simple template variable replacement"""
        loader = PromptLoader()

        template = "Hello {name}, welcome to {place}!"
        context = {"name": "Alice", "place": "Wonderland"}

        result = loader.format_template(template, context)
        assert result == "Hello Alice, welcome to Wonderland!"

    def test_jinja2_variables(self):
        """Test Jinja2 variable syntax"""
        loader = PromptLoader()

        template = "Hello {{ name }}, you have {{ count }} messages."
        context = {"name": "Bob", "count": 5}

        result = loader.format_template(template, context)
        assert result == "Hello Bob, you have 5 messages."

    def test_jinja2_control_structures(self):
        """Test Jinja2 control structures"""
        loader = PromptLoader()

        template = """
{%- if show_greeting %}
Hello {{ name }}!
{%- endif %}
{%- for item in items %}
- {{ item }}
{%- endfor %}
"""
        context = {
            "show_greeting": True,
            "name": "Charlie",
            "items": ["apple", "banana", "cherry"]
        }

        result = loader.format_template(template, context)
        assert "Hello Charlie!" in result
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result

    def test_config_reference_resolution(self):
        """Test resolution of config references like {section.key}"""
        loader = PromptLoader()
        loader.config = {
            "messages": {
                "greeting": "Hello {name}",
                "farewell": "Goodbye {name}"
            },
            "templates": {
                "full_message": "{messages.greeting} and {messages.farewell}"
            }
        }

        template = "{templates.full_message}"
        context = {"name": "Dave"}

        result = loader.format_template(template, context)
        assert "Hello Dave" in result
        assert "Goodbye Dave" in result

    def test_nested_reference_resolution(self):
        """Test deeply nested config reference resolution"""
        loader = PromptLoader()
        loader.config = {
            "level1": {
                "level2": {
                    "level3": {
                        "message": "Deep message: {value}"
                    }
                }
            }
        }

        template = "{level1.level2.level3.message}"
        context = {"value": "found"}

        result = loader.format_template(template, context)
        assert result == "Deep message: found"

    def test_circular_reference_detection(self):
        """Test circular reference detection and handling"""
        loader = PromptLoader()
        loader.config = {
            "ref_a": "{ref_b}",
            "ref_b": "{ref_a}"
        }

        template = "{ref_a}"
        context = {}

        result = loader.format_template(template, context)
        assert "CYCLE_DETECTED" in result

    def test_missing_variable_handling(self):
        """Test handling of missing template variables"""
        loader = PromptLoader()

        template = "Hello {name}, you have {missing_var} messages."
        context = {"name": "Eve"}

        # Should not raise error, missing variables stay as placeholders
        result = loader.format_template(template, context)
        assert "Hello Eve" in result
        assert "{missing_var}" in result

    def test_missing_config_reference(self):
        """Test handling of missing config references"""
        loader = PromptLoader()
        loader.config = {"existing": "value"}

        template = "{nonexistent.reference}"
        context = {}

        result = loader.format_template(template, context)
        assert result == "{nonexistent.reference}"

    def test_mixed_jinja2_and_format_syntax(self):
        """Test mixing Jinja2 and format syntax"""
        loader = PromptLoader()
        loader.config = {
            "greeting": "Hello {{ name }}"
        }

        template = "{greeting} - Welcome to {place}!"
        context = {"name": "Frank", "place": "Paris"}

        result = loader.format_template(template, context)
        assert "Hello Frank" in result
        assert "Welcome to Paris" in result

    def test_visited_set_isolation(self):
        """Test that visited set is properly isolated between calls"""
        loader = PromptLoader()
        loader.config = {
            "msg1": "Message 1: {value}",
            "msg2": "Message 2: {value}"
        }

        # First call
        result1 = loader.format_template("{msg1}", {"value": "first"})

        # Second call should not be affected by first call's visited set
        result2 = loader.format_template("{msg2}", {"value": "second"})

        assert result1 == "Message 1: first"
        assert result2 == "Message 2: second"


class TestGetTemplate:
    """Test get_template method for config navigation"""

    def test_simple_path(self):
        """Test getting template with simple path"""
        loader = PromptLoader()
        loader.config = {
            "section": {
                "template": "Simple template"
            }
        }

        result = loader.get_template("section.template")
        assert result == "Simple template"

    def test_deep_path(self):
        """Test getting template with deep path"""
        loader = PromptLoader()
        loader.config = {
            "level1": {
                "level2": {
                    "level3": {
                        "template": "Deep template"
                    }
                }
            }
        }

        result = loader.get_template("level1.level2.level3.template")
        assert result == "Deep template"

    def test_missing_path(self):
        """Test getting template with missing path"""
        loader = PromptLoader()
        loader.config = {"existing": "value"}

        result = loader.get_template("missing.path")
        assert result == ""

    def test_partial_missing_path(self):
        """Test getting template with partially missing path"""
        loader = PromptLoader()
        loader.config = {
            "section": {
                "existing": "value"
            }
        }

        result = loader.get_template("section.missing.path")
        assert result == ""

    def test_none_value_handling(self):
        """Test handling of None values in config"""
        loader = PromptLoader()
        loader.config = {
            "section": {
                "null_value": None
            }
        }

        result = loader.get_template("section.null_value")
        assert result == ""

    def test_non_string_value_conversion(self):
        """Test conversion of non-string values to string"""
        loader = PromptLoader()
        loader.config = {
            "numbers": {
                "integer": 42,
                "float": 3.14
            },
            "boolean": True,
            "list": [1, 2, 3]
        }

        assert loader.get_template("numbers.integer") == "42"
        assert loader.get_template("numbers.float") == "3.14"
        assert loader.get_template("boolean") == "True"
        assert loader.get_template("list") == "[1, 2, 3]"


class TestBuildPromptFromSections:
    """Test build_prompt_from_sections method"""

    def test_simple_sections(self):
        """Test building prompt from simple sections"""
        loader = PromptLoader()

        sections = [
            "Header: {title}",
            "Content: {content}",
            "Footer"
        ]
        context = {"title": "Test", "content": "Main content"}

        result = loader.build_prompt_from_sections(sections, context)
        lines = result.split('\n')

        assert "Header: Test" in lines
        assert "Content: Main content" in lines
        assert "Footer" in lines

    def test_empty_sections_filtering(self):
        """Test filtering of empty sections"""
        loader = PromptLoader()

        sections = [
            "Valid content",
            "",
            "   ",  # Whitespace only
            "More content"
        ]
        context = {}

        result = loader.build_prompt_from_sections(sections, context)
        lines = [line for line in result.split('\n') if line.strip()]

        assert len(lines) == 2
        assert "Valid content" in lines
        assert "More content" in lines

    def test_unresolved_variables_filtering(self):
        """Test filtering of sections with unresolved variables"""
        loader = PromptLoader()

        sections = [
            "Resolved: {name}",
            "Unresolved: {missing_var}",
            "Also resolved: {value}"
        ]
        context = {"name": "Alice", "value": "test"}

        result = loader.build_prompt_from_sections(sections, context)
        lines = result.split('\n')

        # Should include resolved sections, skip unresolved
        assert "Resolved: Alice" in result
        assert "Also resolved: test" in result
        assert "Unresolved:" not in result

    def test_key_error_handling(self):
        """Test handling of KeyError during template formatting"""
        loader = PromptLoader()

        sections = [
            "Good section",
            "Bad section: {required_missing}",
            "Another good section"
        ]
        context = {}

        # Should not raise error, just skip problematic sections
        result = loader.build_prompt_from_sections(sections, context)

        assert "Good section" in result
        assert "Another good section" in result
        assert "Bad section:" not in result

    def test_complex_template_sections(self):
        """Test sections with complex templates"""
        loader = PromptLoader()
        loader.config = {
            "templates": {
                "greeting": "Hello {name}!"
            }
        }

        sections = [
            "{templates.greeting}",
            "{% if show_extra %}Extra content{% endif %}",
            "Standard content"
        ]
        context = {"name": "Bob", "show_extra": True}

        result = loader.build_prompt_from_sections(sections, context)

        assert "Hello Bob!" in result
        assert "Extra content" in result
        assert "Standard content" in result

    def test_jinja2_sections(self):
        """Test sections with Jinja2 control structures"""
        loader = PromptLoader()

        sections = [
            "Header",
            """
{%- for item in items %}
- {{ item }}
{%- endfor %}
""",
            "Footer"
        ]
        context = {"items": ["apple", "banana"]}

        result = loader.build_prompt_from_sections(sections, context)

        assert "Header" in result
        assert "- apple" in result
        assert "- banana" in result
        assert "Footer" in result


class TestIntegrationWithYAMLLoader:
    """Test integration with the custom YAML loader"""

    def test_loading_with_includes(self, temp_dir):
        """Test PromptLoader with YAML includes"""
        # Create included content
        included_content = temp_dir / "greeting.md"
        included_content.write_text("Hello {{ name }}!")

        # Create config with includes
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
templates:
  greeting: !include greeting.md
  message: "{templates.greeting} Welcome to our service."
""")

        loader = PromptLoader(config_file)

        result = loader.format_template("{templates.message}", {"name": "Alice"})
        assert "Hello Alice!" in result
        assert "Welcome to our service" in result

    def test_nested_includes_with_templates(self, temp_dir):
        """Test nested includes with template processing"""
        # Create deep content
        deep_content = temp_dir / "deep.txt"
        deep_content.write_text("Deep: {value}")

        # Create middle file
        middle_file = temp_dir / "middle.yaml"
        middle_file.write_text("deep_template: !include deep.txt")

        # Create main config
        main_config = temp_dir / "main.yaml"
        main_config.write_text("""
nested: !include middle.yaml
message: "Start -> {nested.deep_template} <- End"
""")

        loader = PromptLoader(main_config)

        result = loader.format_template("{message}", {"value": "found"})
        assert result == "Start -> Deep: found <- End"


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_actual_prompts_structure(self):
        """Test with actual prompts.yaml structure"""
        loader = PromptLoader()

        # Test that we can access actual config sections
        assert loader.get_template("core_agent_context.supervised_context") != ""
        assert loader.get_template("safety_notice_section") != ""

        # Test template formatting with actual structure
        context = {"session_id": "test123", "max_turns": 10}

        # Should not raise errors with actual config structure
        safety_notice = loader.get_template("safety_notice_section")
        formatted = loader.format_template(safety_notice, context)
        assert len(formatted) > 0

    def test_agent_prompt_generation(self):
        """Test generating agent prompts like real usage"""
        loader = PromptLoader()

        # Test building sections like PromptGenerator does
        if "agent_prompts" in loader.config and "initial" in loader.config["agent_prompts"]:
            sections = loader.config["agent_prompts"]["initial"].get("sections", [])
            if sections:
                context = {
                    "session_id": "test456",
                    "max_turns": 15,
                    "todo_list": "1. Test task\n2. Another task",
                    "project_path": "/test/path"
                }

                result = loader.build_prompt_from_sections(sections, context)
                # Should generate substantial content
                assert len(result) > 100

    def test_supervisor_prompt_generation(self):
        """Test generating supervisor prompts like real usage"""
        loader = PromptLoader()

        # Test supervisor prompts if they exist
        if "supervisor_prompts" in loader.config:
            context = {
                "execution_output": "Test output",
                "max_turns": 20,
                "turns_used": 5
            }

            # Should be able to format supervisor-related templates
            result = loader.format_template("Analysis for {max_turns} turns", context)
            assert "Analysis for 20 turns" in result


class TestErrorHandling:
    """Test comprehensive error handling"""

    def test_template_formatting_with_invalid_jinja2(self):
        """Test handling of invalid Jinja2 syntax"""
        loader = PromptLoader()

        # Invalid Jinja2 syntax
        template = "Hello {{ name"  # Missing closing brace
        context = {"name": "Alice"}

        # Should handle gracefully without crashing
        with pytest.raises(Exception):  # Jinja2 will raise TemplateError
            loader.format_template(template, context)

    def test_config_modification_safety(self):
        """Test that config modifications don't affect loader state"""
        loader = PromptLoader()
        original_config = loader.config.copy()

        # Modify config externally
        if "test_key" not in loader.config:
            loader.config["test_key"] = "test_value"

        # Create new loader - should not be affected
        new_loader = PromptLoader()
        assert "test_key" not in new_loader.config

    def test_large_context_handling(self):
        """Test handling of large context dictionaries"""
        loader = PromptLoader()

        # Create large context
        large_context = {f"key_{i}": f"value_{i}" for i in range(1000)}
        large_context["name"] = "Alice"

        template = "Hello {name}!"
        result = loader.format_template(template, large_context)

        assert result == "Hello Alice!"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
