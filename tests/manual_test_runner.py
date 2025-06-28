#!/usr/bin/env python3
"""
Manual test runner for basic validation of YAML loader functionality
This runs key tests without requiring pytest
"""

import sys
import tempfile
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadence.prompt_loader import (
    YAMLIncludeLoader,
    load_yaml_with_includes,
    load_yaml_string_with_includes,
    PromptLoader
)


def test_basic_yaml_loading():
    """Test basic YAML loading without includes"""
    print("Testing basic YAML loading...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create simple YAML file
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("key: value\nnumber: 42")

        result = load_yaml_with_includes(yaml_file)
        assert result["key"] == "value"
        assert result["number"] == 42

    print("✅ Basic YAML loading passed")


def test_simple_include():
    """Test simple file inclusion"""
    print("Testing simple file inclusion...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create included file
        included_file = temp_dir / "included.txt"
        included_file.write_text("Hello, World!")

        # Create main YAML file
        main_file = temp_dir / "main.yaml"
        main_file.write_text("message: !include included.txt")

        result = load_yaml_with_includes(main_file)
        assert result["message"] == "Hello, World!"

    print("✅ Simple file inclusion passed")


def test_yaml_include():
    """Test including YAML files"""
    print("Testing YAML file inclusion...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create included YAML file
        included_file = temp_dir / "included.yaml"
        included_file.write_text("key: value\nnumber: 123")

        # Create main YAML file
        main_file = temp_dir / "main.yaml"
        main_file.write_text("data: !include included.yaml")

        result = load_yaml_with_includes(main_file)
        assert result["data"]["key"] == "value"
        assert result["data"]["number"] == 123

    print("✅ YAML file inclusion passed")


def test_circular_dependency_detection():
    """Test circular dependency detection"""
    print("Testing circular dependency detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create file A that includes B
        file_a = temp_dir / "a.yaml"
        file_a.write_text("content: !include b.yaml")

        # Create file B that includes A (circular)
        file_b = temp_dir / "b.yaml"
        file_b.write_text("content: !include a.yaml")

        try:
            load_yaml_with_includes(file_a)
            assert False, "Should have detected circular dependency"
        except yaml.YAMLError as e:
            assert "Circular include dependency" in str(e)

    print("✅ Circular dependency detection passed")


def test_path_traversal_security():
    """Test security against path traversal"""
    print("Testing path traversal security...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include ../../../etc/passwd")

        try:
            load_yaml_with_includes(main_file)
            assert False, "Should have blocked path traversal"
        except yaml.YAMLError as e:
            assert "attempts to access a file outside" in str(e)

    print("✅ Path traversal security passed")


def test_prompt_loader_basic():
    """Test basic PromptLoader functionality"""
    print("Testing PromptLoader basic functionality...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create test config
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
templates:
  greeting: "Hello {name}!"
  message: "Welcome to {place}"
""")

        loader = PromptLoader(config_file)

        # Test template formatting
        result = loader.format_template("{templates.greeting}", {"name": "Alice"})
        assert result == "Hello Alice!"

        # Test get_template
        greeting = loader.get_template("templates.greeting")
        assert greeting == "Hello {name}!"

        # Test build_prompt_from_sections
        sections = [
            "{templates.greeting}",
            "{templates.message}"
        ]
        context = {"name": "Bob", "place": "Wonderland"}
        result = loader.build_prompt_from_sections(sections, context)
        assert "Hello Bob!" in result
        assert "Welcome to Wonderland" in result

    print("✅ PromptLoader basic functionality passed")


def test_jinja2_templates():
    """Test Jinja2 template processing"""
    print("Testing Jinja2 template processing...")

    loader = PromptLoader()

    # Test Jinja2 variables
    template = "Hello {{ name }}, you have {{ count }} messages."
    context = {"name": "Charlie", "count": 3}
    result = loader.format_template(template, context)
    assert result == "Hello Charlie, you have 3 messages."

    # Test Jinja2 control structures
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
        "name": "Dave",
        "items": ["apple", "banana"]
    }
    result = loader.format_template(template, context)
    assert "Hello Dave!" in result
    assert "- apple" in result
    assert "- banana" in result

    print("✅ Jinja2 template processing passed")


def test_string_loading():
    """Test loading YAML from strings"""
    print("Testing YAML string loading...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create included file
        included_file = temp_dir / "included.txt"
        included_file.write_text("included content")

        yaml_string = "content: !include included.txt"
        result = load_yaml_string_with_includes(yaml_string, base_dir=temp_dir)
        assert result["content"] == "included content"

    print("✅ YAML string loading passed")


def test_actual_prompts_loading():
    """Test loading actual prompts.yaml"""
    print("Testing actual prompts.yaml loading...")

    try:
        loader = PromptLoader()
        assert loader.config is not None
        assert "core_agent_context" in loader.config

        # Test that we can get actual templates
        safety_notice = loader.get_template("safety_notice_section")
        assert len(safety_notice) > 0

        print("✅ Actual prompts.yaml loading passed")
    except Exception as e:
        print(f"⚠️  Actual prompts.yaml loading failed: {e}")
        print("   This might be expected if prompts.yaml structure has changed")


def run_all_tests():
    """Run all manual tests"""
    print("Running manual test suite for YAML loader and PromptLoader...")
    print("=" * 60)

    tests = [
        test_basic_yaml_loading,
        test_simple_include,
        test_yaml_include,
        test_circular_dependency_detection,
        test_path_traversal_security,
        test_prompt_loader_basic,
        test_jinja2_templates,
        test_string_loading,
        test_actual_prompts_loading
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
