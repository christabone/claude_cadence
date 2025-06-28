#!/usr/bin/env python3
"""
Error Handling Validation Test (Task 8.3)

This test validates that our YAML loader and PromptLoader handle errors gracefully
and provide appropriate error messages. Focus is on verifying existing error handling
works correctly rather than comprehensive edge case testing.
"""

import sys
import tempfile
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cadence.prompt_loader import (
    load_yaml_with_includes,
    load_yaml_string_with_includes,
    PromptLoader
)


def test_missing_file_error():
    """Test that missing files produce clear error messages"""
    print("Testing missing file error handling...")

    try:
        load_yaml_with_includes("/nonexistent/file.yaml")
        print("❌ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            print("✅ Missing file error message is clear")
            return True
        else:
            print(f"❌ Missing file error message unclear: {error_msg}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        return False


def test_malformed_yaml_error():
    """Test that malformed YAML produces clear error messages"""
    print("Testing malformed YAML error handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        bad_file = temp_dir / "bad.yaml"
        bad_file.write_text("{ invalid yaml content :")

        try:
            load_yaml_with_includes(bad_file)
            print("❌ Should have raised YAML error")
            return False
        except yaml.YAMLError as e:
            error_msg = str(e)
            if len(error_msg) > 0:
                print("✅ Malformed YAML error message provided")
                return True
            else:
                print("❌ Malformed YAML error message is empty")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
            return False


def test_missing_include_error():
    """Test that missing included files produce clear error messages"""
    print("Testing missing include file error handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include nonexistent.txt")

        try:
            load_yaml_with_includes(main_file)
            print("❌ Should have raised YAML error for missing include")
            return False
        except yaml.YAMLError as e:
            error_msg = str(e)
            if "include file not found" in error_msg.lower():
                print("✅ Missing include file error message is clear")
                return True
            else:
                print(f"❌ Missing include error message unclear: {error_msg}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
            return False


def test_circular_dependency_error():
    """Test that circular dependencies produce clear error messages"""
    print("Testing circular dependency error handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create circular dependency
        file_a = temp_dir / "a.yaml"
        file_a.write_text("content: !include b.yaml")

        file_b = temp_dir / "b.yaml"
        file_b.write_text("content: !include a.yaml")

        try:
            load_yaml_with_includes(file_a)
            print("❌ Should have detected circular dependency")
            return False
        except yaml.YAMLError as e:
            error_msg = str(e)
            if "circular" in error_msg.lower():
                print("✅ Circular dependency error message is clear")
                return True
            else:
                print(f"❌ Circular dependency error message unclear: {error_msg}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
            return False


def test_path_traversal_security_error():
    """Test that path traversal attempts produce clear security error messages"""
    print("Testing path traversal security error handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include ../../../etc/passwd")

        try:
            load_yaml_with_includes(main_file)
            print("❌ Should have blocked path traversal")
            return False
        except yaml.YAMLError as e:
            error_msg = str(e)
            if "outside" in error_msg.lower() or "security" in error_msg.lower():
                print("✅ Path traversal security error message is clear")
                return True
            else:
                print(f"❌ Path traversal error message unclear: {error_msg}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
            return False


def test_prompt_loader_missing_config_error():
    """Test that PromptLoader handles missing config files gracefully"""
    print("Testing PromptLoader missing config error handling...")

    try:
        PromptLoader("/nonexistent/config.yaml")
        print("❌ Should have raised IOError")
        return False
    except IOError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            print("✅ PromptLoader missing config error message is clear")
            return True
        else:
            print(f"❌ PromptLoader missing config error message unclear: {error_msg}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        return False


def test_prompt_loader_malformed_config_error():
    """Test that PromptLoader handles malformed config files gracefully"""
    print("Testing PromptLoader malformed config error handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        bad_config = temp_dir / "bad.yaml"
        bad_config.write_text("{ invalid yaml :")

        try:
            PromptLoader(bad_config)
            print("❌ Should have raised ValueError")
            return False
        except ValueError as e:
            error_msg = str(e)
            if "parsing yaml" in error_msg.lower():
                print("✅ PromptLoader malformed config error message is clear")
                return True
            else:
                print(f"❌ PromptLoader malformed config error message unclear: {error_msg}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
            return False


def test_template_formatting_graceful_handling():
    """Test that template formatting handles missing variables gracefully"""
    print("Testing template formatting graceful error handling...")

    try:
        loader = PromptLoader()

        # Template with missing variable - should not crash
        template = "Hello {name}, you have {missing_var} messages"
        context = {"name": "Alice"}

        result = loader.format_template(template, context)

        if "Alice" in result and "{missing_var}" in result:
            print("✅ Template formatting handles missing variables gracefully")
            return True
        else:
            print(f"❌ Template formatting result unexpected: {result}")
            return False

    except Exception as e:
        print(f"❌ Template formatting should not raise error: {type(e).__name__}: {e}")
        return False


def test_string_yaml_with_security():
    """Test that string YAML loading handles security issues gracefully"""
    print("Testing string YAML security error handling...")

    yaml_string = "content: !include ../secret.txt"

    try:
        load_yaml_string_with_includes(yaml_string)
        print("❌ Should have blocked parent directory access")
        return False
    except yaml.YAMLError as e:
        error_msg = str(e)
        if "outside" in error_msg.lower() or "parent directory" in error_msg.lower():
            print("✅ String YAML security error message is clear")
            return True
        else:
            print(f"❌ String YAML security error message unclear: {error_msg}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        return False


def run_error_handling_validation():
    """Run all error handling validation tests"""
    print("Error Handling Validation Test (Task 8.3)")
    print("=" * 50)
    print("Verifying graceful error handling and appropriate error messages...")
    print()

    tests = [
        test_missing_file_error,
        test_malformed_yaml_error,
        test_missing_include_error,
        test_circular_dependency_error,
        test_path_traversal_security_error,
        test_prompt_loader_missing_config_error,
        test_prompt_loader_malformed_config_error,
        test_template_formatting_graceful_handling,
        test_string_yaml_with_security
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Error Handling Validation Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All error handling tests passed!")
        print("✅ Error handling is graceful with appropriate error messages")
        return True
    else:
        print("💥 Some error handling tests failed!")
        print("❌ Error handling needs improvement")
        return False


if __name__ == "__main__":
    success = run_error_handling_validation()
    sys.exit(0 if success else 1)
