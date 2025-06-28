"""
Comprehensive unit tests for custom YAML loader with !include support

This test suite provides complete coverage of the custom YAML loader functionality
including file inclusion, error handling, security validation, and edge cases.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO

from cadence.prompt_loader import (
    YAMLIncludeLoader,
    include_constructor,
    load_yaml_with_includes,
    load_yaml_string_with_includes
)


class TestYAMLIncludeLoader:
    """Test the custom YAML loader class"""

    def test_loader_initialization_with_file(self, temp_dir):
        """Test loader initialization with file stream"""
        test_file = temp_dir / "test.yaml"
        test_file.write_text("test: value")

        with open(test_file, 'r') as f:
            loader = YAMLIncludeLoader(f)

        assert loader._root_dir == test_file.parent
        assert str(test_file) in loader._include_stack

    def test_loader_initialization_with_string(self):
        """Test loader initialization with StringIO"""
        stream = StringIO("test: value")
        loader = YAMLIncludeLoader(stream)

        assert loader._root_dir == Path.cwd()
        assert '<stream>' in loader._include_stack

    def test_get_include_stack(self, temp_dir):
        """Test include stack tracking"""
        test_file = temp_dir / "test.yaml"
        test_file.write_text("test: value")

        with open(test_file, 'r') as f:
            loader = YAMLIncludeLoader(f)
            stack = loader.get_include_stack()

        assert isinstance(stack, list)
        assert str(test_file) in stack
        assert len(stack) == 1


class TestIncludeConstructor:
    """Test the !include constructor functionality"""

    def test_simple_text_include(self, temp_dir):
        """Test including a simple text file"""
        # Create included file
        included_file = temp_dir / "included.txt"
        included_file.write_text("Hello, World!")

        # Create main YAML file
        main_file = temp_dir / "main.yaml"
        main_file.write_text(f"message: !include included.txt")

        result = load_yaml_with_includes(main_file)
        assert result["message"] == "Hello, World!"

    def test_yaml_include(self, temp_dir):
        """Test including a YAML file"""
        # Create included YAML file
        included_file = temp_dir / "included.yaml"
        included_file.write_text("key: value\nnumber: 42")

        # Create main YAML file
        main_file = temp_dir / "main.yaml"
        main_file.write_text("data: !include included.yaml")

        result = load_yaml_with_includes(main_file)
        assert result["data"]["key"] == "value"
        assert result["data"]["number"] == 42

    def test_nested_includes(self, temp_dir):
        """Test nested file inclusion"""
        # Create deepest file
        deep_file = temp_dir / "deep.txt"
        deep_file.write_text("deep content")

        # Create middle file that includes deep file
        middle_file = temp_dir / "middle.yaml"
        middle_file.write_text("content: !include deep.txt")

        # Create main file that includes middle file
        main_file = temp_dir / "main.yaml"
        main_file.write_text("nested: !include middle.yaml")

        result = load_yaml_with_includes(main_file)
        assert result["nested"]["content"] == "deep content"

    def test_relative_path_resolution(self, temp_dir):
        """Test relative path resolution in subdirectories"""
        # Create subdirectory structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        # Create file in subdirectory
        sub_file = subdir / "sub.txt"
        sub_file.write_text("subdirectory content")

        # Create main file that includes from subdirectory
        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include subdir/sub.txt")

        result = load_yaml_with_includes(main_file)
        assert result["content"] == "subdirectory content"

    def test_circular_dependency_detection(self, temp_dir):
        """Test circular dependency detection"""
        # Create file A that includes B
        file_a = temp_dir / "a.yaml"
        file_a.write_text("content: !include b.yaml")

        # Create file B that includes A (circular)
        file_b = temp_dir / "b.yaml"
        file_b.write_text("content: !include a.yaml")

        with pytest.raises(yaml.YAMLError, match="Circular include dependency"):
            load_yaml_with_includes(file_a)

    def test_file_not_found_error(self, temp_dir):
        """Test error handling for missing files"""
        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include nonexistent.txt")

        with pytest.raises(yaml.YAMLError, match="Include file not found"):
            load_yaml_with_includes(main_file)

    def test_directory_include_error(self, temp_dir):
        """Test error when trying to include a directory"""
        # Create a subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include subdir")

        with pytest.raises(yaml.YAMLError, match="Include path is not a file"):
            load_yaml_with_includes(main_file)

    def test_permission_error_handling(self, temp_dir):
        """Test permission error handling"""
        # Create a file
        restricted_file = temp_dir / "restricted.txt"
        restricted_file.write_text("restricted content")

        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            main_file = temp_dir / "main.yaml"
            main_file.write_text("content: !include restricted.txt")

            with pytest.raises(yaml.YAMLError, match="Permission denied"):
                load_yaml_with_includes(main_file)

    def test_unicode_decode_error_handling(self, temp_dir):
        """Test Unicode decode error handling"""
        # Create a file with invalid UTF-8
        bad_file = temp_dir / "bad.txt"
        bad_file.write_bytes(b'\xff\xfe\x00\x00')  # Invalid UTF-8

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include bad.txt")

        with pytest.raises(yaml.YAMLError, match="Unicode decode error"):
            load_yaml_with_includes(main_file)

    def test_malformed_included_yaml(self, temp_dir):
        """Test error handling for malformed included YAML"""
        # Create malformed YAML file
        bad_yaml = temp_dir / "bad.yaml"
        bad_yaml.write_text("{ invalid yaml content :")

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include bad.yaml")

        with pytest.raises(yaml.YAMLError, match="YAML error in included file"):
            load_yaml_with_includes(main_file)

    def test_path_traversal_security(self, temp_dir):
        """Test security against path traversal attacks"""
        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include ../../../etc/passwd")

        with pytest.raises(yaml.YAMLError, match="attempts to access a file outside"):
            load_yaml_with_includes(main_file)

    def test_parent_directory_security_string_mode(self):
        """Test security against parent directory access in string mode"""
        yaml_content = "content: !include ../secret.txt"

        with pytest.raises(yaml.YAMLError, match="parent directory references"):
            load_yaml_string_with_includes(yaml_content)

    def test_include_stack_cleanup_on_error(self, temp_dir):
        """Test that include stack is properly cleaned up on errors"""
        # Create a file that will cause an error
        bad_file = temp_dir / "bad.txt"
        bad_file.write_bytes(b'\xff\xfe')  # Invalid UTF-8

        main_file = temp_dir / "main.yaml"
        main_file.write_text("content: !include bad.txt")

        with open(main_file, 'r') as f:
            loader = YAMLIncludeLoader(f)

            # Before error, stack should have main file
            assert len(loader.get_include_stack()) == 1

            try:
                loader.get_single_data()
            except yaml.YAMLError:
                pass  # Expected error

            # After error, stack should be cleaned up (only main file)
            assert len(loader.get_include_stack()) == 1


class TestLoadYamlWithIncludes:
    """Test the main loading function"""

    def test_simple_yaml_loading(self, temp_dir):
        """Test loading simple YAML without includes"""
        yaml_file = temp_dir / "simple.yaml"
        yaml_file.write_text("key: value\nnumber: 42")

        result = load_yaml_with_includes(yaml_file)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_file_not_found_handling(self):
        """Test error handling for missing files"""
        with pytest.raises(FileNotFoundError):
            load_yaml_with_includes("/nonexistent/file.yaml")

    def test_directory_instead_of_file(self, temp_dir):
        """Test error when path points to directory"""
        with pytest.raises(IOError, match="Path is not a file"):
            load_yaml_with_includes(temp_dir)

    def test_malformed_yaml_handling(self, temp_dir):
        """Test error handling for malformed YAML"""
        bad_file = temp_dir / "bad.yaml"
        bad_file.write_text("{ invalid yaml :")

        with pytest.raises(yaml.YAMLError):
            load_yaml_with_includes(bad_file)

    def test_pathlib_path_input(self, temp_dir):
        """Test that function accepts Path objects"""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("test: success")

        # Test with Path object (should work)
        result = load_yaml_with_includes(yaml_file)
        assert result["test"] == "success"

        # Test with string path (should also work)
        result = load_yaml_with_includes(str(yaml_file))
        assert result["test"] == "success"

    def test_encoding_parameter(self, temp_dir):
        """Test custom encoding parameter"""
        # Create file with specific encoding
        yaml_file = temp_dir / "encoded.yaml"
        content = "message: 'Hello World'"
        yaml_file.write_text(content, encoding='utf-8')

        result = load_yaml_with_includes(yaml_file, encoding='utf-8')
        assert result["message"] == "Hello World"


class TestLoadYamlStringWithIncludes:
    """Test loading YAML from strings with includes"""

    def test_string_loading_without_includes(self):
        """Test loading YAML string without includes"""
        yaml_string = "key: value\nnumber: 42"
        result = load_yaml_string_with_includes(yaml_string)

        assert result["key"] == "value"
        assert result["number"] == 42

    def test_string_loading_with_base_dir(self, temp_dir):
        """Test loading YAML string with base directory for includes"""
        # Create included file
        included_file = temp_dir / "included.txt"
        included_file.write_text("included content")

        yaml_string = "content: !include included.txt"
        result = load_yaml_string_with_includes(yaml_string, base_dir=temp_dir)

        assert result["content"] == "included content"

    def test_string_loading_without_base_dir(self):
        """Test loading YAML string without base directory"""
        yaml_string = "simple: content"
        result = load_yaml_string_with_includes(yaml_string)

        assert result["simple"] == "content"

    def test_string_malformed_yaml(self):
        """Test error handling for malformed YAML string"""
        yaml_string = "{ invalid yaml :"

        with pytest.raises(yaml.YAMLError):
            load_yaml_string_with_includes(yaml_string)

    def test_pathlib_base_dir(self, temp_dir):
        """Test that base_dir accepts Path objects"""
        # Create included file
        included_file = temp_dir / "test.txt"
        included_file.write_text("test content")

        yaml_string = "data: !include test.txt"

        # Test with Path object
        result = load_yaml_string_with_includes(yaml_string, base_dir=temp_dir)
        assert result["data"] == "test content"

        # Test with string path
        result = load_yaml_string_with_includes(yaml_string, base_dir=str(temp_dir))
        assert result["data"] == "test content"


class TestComplexScenarios:
    """Test complex real-world scenarios"""

    def test_multiple_includes_same_file(self, temp_dir):
        """Test including the same file multiple times"""
        # Create shared content
        shared_file = temp_dir / "shared.txt"
        shared_file.write_text("shared content")

        main_file = temp_dir / "main.yaml"
        main_file.write_text("""
first: !include shared.txt
second: !include shared.txt
""")

        result = load_yaml_with_includes(main_file)
        assert result["first"] == "shared content"
        assert result["second"] == "shared content"

    def test_mixed_yaml_and_text_includes(self, temp_dir):
        """Test mixing YAML and text file includes"""
        # Create text file
        text_file = temp_dir / "content.txt"
        text_file.write_text("Plain text content")

        # Create YAML file
        yaml_file = temp_dir / "data.yaml"
        yaml_file.write_text("structured: data\nvalue: 123")

        main_file = temp_dir / "main.yaml"
        main_file.write_text("""
text_content: !include content.txt
yaml_content: !include data.yaml
""")

        result = load_yaml_with_includes(main_file)
        assert result["text_content"] == "Plain text content"
        assert result["yaml_content"]["structured"] == "data"
        assert result["yaml_content"]["value"] == 123

    def test_deep_nesting_includes(self, temp_dir):
        """Test deeply nested includes"""
        # Create chain: main -> level1 -> level2 -> level3 -> content
        content_file = temp_dir / "content.txt"
        content_file.write_text("final content")

        level3_file = temp_dir / "level3.yaml"
        level3_file.write_text("final: !include content.txt")

        level2_file = temp_dir / "level2.yaml"
        level2_file.write_text("level3: !include level3.yaml")

        level1_file = temp_dir / "level1.yaml"
        level1_file.write_text("level2: !include level2.yaml")

        main_file = temp_dir / "main.yaml"
        main_file.write_text("level1: !include level1.yaml")

        result = load_yaml_with_includes(main_file)
        assert result["level1"]["level2"]["level3"]["final"] == "final content"

    def test_include_with_complex_yaml_structures(self, temp_dir):
        """Test includes with complex YAML structures"""
        # Create complex included file
        complex_file = temp_dir / "complex.yaml"
        complex_file.write_text("""
list_data:
  - item1
  - item2
  - nested:
      key: value
dict_data:
  key1: value1
  key2:
    nested_key: nested_value
mixed:
  - name: first
    value: 1
  - name: second
    value: 2
""")

        main_file = temp_dir / "main.yaml"
        main_file.write_text("complex_structure: !include complex.yaml")

        result = load_yaml_with_includes(main_file)
        complex_data = result["complex_structure"]

        assert len(complex_data["list_data"]) == 3
        assert complex_data["list_data"][2]["nested"]["key"] == "value"
        assert complex_data["dict_data"]["key2"]["nested_key"] == "nested_value"
        assert complex_data["mixed"][1]["name"] == "second"

    def test_error_propagation_in_nested_includes(self, temp_dir):
        """Test error propagation through nested includes"""
        # Create a bad file that will cause an error
        bad_file = temp_dir / "bad.yaml"
        bad_file.write_text("{ malformed yaml :")

        # Create middle file that includes the bad file
        middle_file = temp_dir / "middle.yaml"
        middle_file.write_text("bad_content: !include bad.yaml")

        # Create main file that includes the middle file
        main_file = temp_dir / "main.yaml"
        main_file.write_text("middle: !include middle.yaml")

        with pytest.raises(yaml.YAMLError) as exc_info:
            load_yaml_with_includes(main_file)

        # Error message should include the include chain
        error_msg = str(exc_info.value)
        assert "bad.yaml" in error_msg
        assert "Include chain:" in error_msg


class TestPerformance:
    """Test performance characteristics"""

    def test_large_file_handling(self, temp_dir):
        """Test handling of reasonably large files"""
        # Create a moderately large included file (not too big for CI)
        large_content = "line content\n" * 1000  # ~12KB
        large_file = temp_dir / "large.txt"
        large_file.write_text(large_content)

        main_file = temp_dir / "main.yaml"
        main_file.write_text("large_content: !include large.txt")

        import time
        start_time = time.time()
        result = load_yaml_with_includes(main_file)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(result["large_content"]) > 10000

    def test_many_small_includes(self, temp_dir):
        """Test performance with many small includes"""
        # Create multiple small files
        num_files = 50
        for i in range(num_files):
            small_file = temp_dir / f"small_{i}.txt"
            small_file.write_text(f"content {i}")

        # Create main file that includes all of them
        includes = "\n".join([f"file_{i}: !include small_{i}.txt" for i in range(num_files)])
        main_file = temp_dir / "main.yaml"
        main_file.write_text(includes)

        import time
        start_time = time.time()
        result = load_yaml_with_includes(main_file)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 2.0
        assert len(result) == num_files
        assert result["file_25"] == "content 25"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
