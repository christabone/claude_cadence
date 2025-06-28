"""
Custom YAML loader with !include tag support for Claude Cadence

This module provides a minimal, clean YAML loader that supports !include tags
for file inclusion with relative path handling and comprehensive error reporting.
"""

import os
import yaml
import re
from pathlib import Path
from typing import Any, Dict, Set, Optional, Union, List


class YAMLIncludeLoader(yaml.SafeLoader):
    """
    A custom YAML loader that extends SafeLoader to support !include tags.

    This loader allows YAML files to include other YAML or text files using
    the !include tag, with proper relative path resolution and circular
    dependency detection.
    """

    def __init__(self, stream):
        """
        Initialize the loader with a stream and set up file tracking.

        Args:
            stream: The YAML stream to load from (file-like object)
        """
        self._root_dir = None
        self._include_stack = []

        # Determine the root directory for relative path resolution
        if hasattr(stream, 'name'):
            file_path = Path(stream.name).resolve()
            self._root_dir = file_path.parent
            self._include_stack.append(str(file_path))
        else:
            # If stream has no name (e.g., StringIO), use current working directory
            self._root_dir = Path.cwd()
            self._include_stack.append('<stream>')

        super().__init__(stream)

    def get_include_stack(self) -> list:
        """Get the current include stack for error reporting."""
        return self._include_stack.copy()


# Register the include constructor with the custom loader
def include_constructor(loader: YAMLIncludeLoader, node: yaml.ScalarNode) -> Any:
    """
    Constructor for !include tags that handles file inclusion.

    This function is called when the YAML parser encounters an !include tag.
    It resolves the file path relative to the including file and loads the
    content, with circular dependency detection.

    Args:
        loader: The YAMLIncludeLoader instance
        node: The YAML node containing the include path

    Returns:
        The loaded content from the included file

    Raises:
        yaml.YAMLError: For various error conditions including file not found,
                       circular dependencies, or YAML parsing errors
    """
    # Get the include path from the node
    include_path = loader.construct_scalar(node)

    # Resolve the path relative to the current file's directory
    if loader._root_dir:
        full_path = (loader._root_dir / include_path).resolve()
    else:
        full_path = Path(include_path).resolve()

    # Security check: ensure the path doesn't escape the base directory
    if loader._root_dir:
        try:
            # Check if the resolved path is within the allowed directory
            full_path.relative_to(loader._root_dir.resolve())
        except ValueError:
            raise yaml.YAMLError(
                f"Include path '{include_path}' attempts to access a file "
                f"outside of the base directory '{loader._root_dir}'"
            )
    else:
        # For paths without root_dir, ensure they don't contain parent references
        if '..' in include_path:
            raise yaml.YAMLError(
                f"Include path '{include_path}' contains parent directory references (..) "
                f"which are not allowed for security reasons"
            )

    # Convert to string for consistency in tracking
    full_path_str = str(full_path)

    # Check for circular dependencies
    if full_path_str in loader._include_stack:
        cycle_chain = ' -> '.join(loader._include_stack[loader._include_stack.index(full_path_str):])
        cycle_chain += f' -> {full_path_str}'
        raise yaml.YAMLError(
            f"Circular include dependency detected:\n{cycle_chain}"
        )

    # Check if file exists
    if not full_path.exists():
        include_chain = ' -> '.join(loader._include_stack)
        raise yaml.YAMLError(
            f"Include file not found: {include_path}\n"
            f"Resolved path: {full_path}\n"
            f"Include chain: {include_chain}"
        )

    # Check if it's a file (not a directory)
    if not full_path.is_file():
        include_chain = ' -> '.join(loader._include_stack)
        raise yaml.YAMLError(
            f"Include path is not a file: {include_path}\n"
            f"Resolved path: {full_path}\n"
            f"Include chain: {include_chain}"
        )

    # Add to include stack to track circular dependencies
    loader._include_stack.append(full_path_str)

    try:
        # Determine file type and load accordingly
        if full_path.suffix.lower() in ['.yaml', '.yml']:
            # Load as YAML file with the same loader type
            with open(full_path, 'r', encoding='utf-8') as f:
                # Create a new loader instance for the included file
                # This preserves the include stack but updates the root directory
                sub_loader = YAMLIncludeLoader(f)
                sub_loader._include_stack = loader._include_stack
                sub_loader._root_dir = full_path.parent
                result = sub_loader.get_single_data()
        else:
            # Load as plain text (for .md, .txt, or other files)
            with open(full_path, 'r', encoding='utf-8') as f:
                result = f.read()

    except PermissionError as e:
        include_chain = ' -> '.join(loader._include_stack)
        raise yaml.YAMLError(
            f"Permission denied reading include file: {include_path}\n"
            f"Resolved path: {full_path}\n"
            f"Include chain: {include_chain}\n"
            f"Error: {e}"
        )
    except UnicodeDecodeError as e:
        include_chain = ' -> '.join(loader._include_stack)
        raise yaml.YAMLError(
            f"Unicode decode error in include file: {include_path}\n"
            f"Resolved path: {full_path}\n"
            f"Include chain: {include_chain}\n"
            f"Error: {e}"
        )
    except yaml.YAMLError as e:
        # Re-raise YAML errors with additional context
        include_chain = ' -> '.join(loader._include_stack)
        raise yaml.YAMLError(
            f"YAML error in included file: {include_path}\n"
            f"Resolved path: {full_path}\n"
            f"Include chain: {include_chain}\n"
            f"Error: {e}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        include_chain = ' -> '.join(loader._include_stack)
        raise yaml.YAMLError(
            f"Unexpected error loading include file: {include_path}\n"
            f"Resolved path: {full_path}\n"
            f"Include chain: {include_chain}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {e}"
        )
    finally:
        # Always remove from stack, even if an error occurred
        loader._include_stack.pop()

    return result


# Register the constructor with the loader
YAMLIncludeLoader.add_constructor('!include', include_constructor)


def load_yaml_with_includes(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    **kwargs
) -> Any:
    """
    Load a YAML file with support for !include tags.

    This is the main entry point for loading YAML files with include support.
    It provides a drop-in replacement for yaml.safe_load() with added !include
    functionality.

    Args:
        file_path: Path to the YAML file to load
        encoding: File encoding (default: utf-8)
        **kwargs: Additional keyword arguments (reserved for future use)

    Returns:
        The loaded YAML content with all includes resolved

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: For YAML parsing errors or include errors
        IOError: For file reading errors
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    if not file_path.is_file():
        raise IOError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return yaml.load(f, YAMLIncludeLoader)
    except yaml.YAMLError:
        # Re-raise YAML errors as-is (they already have context)
        raise
    except Exception as e:
        # Wrap other errors with context
        raise yaml.YAMLError(
            f"Error loading YAML file: {file_path}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {e}"
        )


def load_yaml_string_with_includes(
    yaml_string: str,
    base_dir: Optional[Union[str, Path]] = None
) -> Any:
    """
    Load YAML from a string with support for !include tags.

    This function allows loading YAML content from a string while still
    supporting !include tags with relative path resolution.

    Args:
        yaml_string: The YAML content as a string
        base_dir: Base directory for resolving relative include paths
                 (default: current working directory)

    Returns:
        The loaded YAML content with all includes resolved

    Raises:
        yaml.YAMLError: For YAML parsing errors or include errors
    """
    from io import StringIO

    stream = StringIO(yaml_string)

    # Create a loader and set the base directory if provided
    loader = YAMLIncludeLoader(stream)
    if base_dir:
        loader._root_dir = Path(base_dir).resolve()

    try:
        # Note: This unconventional pattern (lambda s: loader) works because:
        # 1. yaml.load() expects a Loader class as the second argument
        # 2. The lambda ignores the stream parameter 's' and returns our pre-configured loader instance
        # 3. This allows us to use an instance with custom _root_dir set, rather than just a class
        # This pattern enables directory-relative !include resolution for YAML loaded from strings
        return yaml.load(stream, lambda s: loader)
    except yaml.YAMLError:
        # Re-raise YAML errors as-is
        raise
    except Exception as e:
        # Wrap other errors
        raise yaml.YAMLError(
            f"Error loading YAML from string\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {e}"
        )


class PromptLoader:
    """
    Loads and manages prompts from YAML configuration with !include tag support.

    This class uses our custom YAML loader to handle !include tags for modular prompts,
    enabling better organization and reusability of prompt content.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize with YAML config file"""
        if config_path is None:
            # Default to prompts.yaml in same directory
            config_path = Path(__file__).parent / "prompts.yaml"
        else:
            config_path = Path(config_path)

        try:
            self.config = load_yaml_with_includes(config_path)
        except FileNotFoundError:
            raise IOError(f"Prompt configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    def format_template(self, template: str, context: Dict[str, Any], visited: Optional[Set[str]] = None) -> str:
        """Format a template string with context variables"""
        if visited is None:
            visited = set()

        # Import Jinja2 only when needed to maintain compatibility
        from jinja2 import Template

        # First, check if template contains Jinja2 control structures or variables
        if '{%' in template or '{{' in template:
            # Process with Jinja2
            jinja_template = Template(template)
            template = jinja_template.render(**context)

        # Handle nested references like {shared_agent_context.supervision_explanation}
        def replace_ref(match):
            ref = match.group(1)

            # Check for cycles
            if ref in visited:
                return f"{{CYCLE_DETECTED: {ref}}}"

            parts = ref.split('.')

            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    # Not a config reference, check context
                    return match.group(0) if ref not in context else str(context[ref])

            # If we found a config value, format it recursively
            if isinstance(value, str):
                visited.add(ref)
                result = self.format_template(value, context, visited)
                visited.remove(ref)
                return result
            return str(value)

        # Second pass: replace config references
        result = re.sub(r'\{([^}]+)\}', replace_ref, template)

        # Third pass: simple format with context
        try:
            result = result.format(**context)
        except KeyError:
            # Some keys might be missing, that's OK for optional sections
            pass

        return result

    def get_template(self, path: str) -> str:
        """Get a template by dot-separated path"""
        parts = path.split('.')
        value = self.config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return ""

        return str(value) if value is not None else ""

    def build_prompt_from_sections(self, sections: List[str], context: Dict[str, Any]) -> str:
        """Build a prompt from a list of sections"""
        formatted_sections = []

        for section in sections:
            # Skip empty sections or those with missing required context
            if section.strip():
                try:
                    formatted = self.format_template(section, context)
                    if formatted.strip() and '{' not in formatted:  # No unresolved vars
                        formatted_sections.append(formatted)
                except KeyError:
                    continue

        return "\n".join(formatted_sections)
