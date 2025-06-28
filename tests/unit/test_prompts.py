"""
Unit tests for prompt generation system
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import yaml

from cadence.prompts import (
    ExecutionContext, PromptGenerator, TodoPromptManager
)
from cadence.prompt_loader import PromptLoader


class TestExecutionContext:
    """Test ExecutionContext dataclass"""

    def test_initialization(self):
        """Test ExecutionContext initialization"""
        todos = ["TODO 1", "TODO 2"]
        context = ExecutionContext(todos=todos, max_turns=10)

        assert context.todos == todos
        assert context.max_turns == 10
        assert context.completed_todos == []
        assert context.remaining_todos == []
        assert context.issues_encountered == []
        assert context.previous_guidance == []
        assert context.continuation_context is None

    def test_with_values(self):
        """Test ExecutionContext with all values"""
        context = ExecutionContext(
            todos=["TODO 1", "TODO 2"],
            max_turns=20,
            completed_todos=["TODO 1"],
            remaining_todos=["TODO 2"],
            issues_encountered=["Error 1"],
            previous_guidance=["Try X"],
            continuation_context="Previous execution"
        )

        assert len(context.todos) == 2
        assert len(context.completed_todos) == 1
        assert len(context.remaining_todos) == 1
        assert context.continuation_context == "Previous execution"


class TestPromptLoader:
    """Test YAML prompt loading"""

    def test_default_loader(self):
        """Test loading default prompts.yaml"""
        loader = PromptLoader()
        assert loader.config is not None
        assert "core_agent_context" in loader.config
        assert "agent_prompts" in loader.config

    def test_custom_loader(self, temp_dir):
        """Test loading custom YAML file"""
        custom_yaml = temp_dir / "custom_prompts.yaml"
        custom_content = {
            "test_template": "Hello {name}",
            "nested": {
                "template": "Nested {value}"
            }
        }

        with open(custom_yaml, 'w') as f:
            yaml.dump(custom_content, f)

        loader = PromptLoader(str(custom_yaml))
        assert loader.config["test_template"] == "Hello {name}"
        assert loader.config["nested"]["template"] == "Nested {value}"

    def test_missing_file(self):
        """Test error on missing file"""
        with pytest.raises(IOError, match="not found"):
            PromptLoader("/nonexistent/file.yaml")

    def test_invalid_yaml(self, temp_dir):
        """Test error on invalid YAML"""
        bad_yaml = temp_dir / "bad.yaml"
        bad_yaml.write_text("{ invalid yaml content :")

        with pytest.raises(ValueError, match="Error parsing YAML"):
            PromptLoader(str(bad_yaml))

    def test_format_template(self):
        """Test template formatting"""
        loader = PromptLoader()

        # Simple formatting
        result = loader.format_template("Hello {name}", {"name": "World"})
        assert result == "Hello World"

        # Missing variable - should leave placeholder
        result = loader.format_template("Hello {missing}", {})
        assert result == "Hello {missing}"

    def test_format_template_nested(self):
        """Test nested template formatting"""
        loader = PromptLoader()
        loader.config["greeting"] = "Hello {name}"
        loader.config["message"] = "{greeting}, welcome!"

        # Set up context with nested reference
        context = {
            "greeting": "{greeting}",
            "name": "Alice"
        }

        result = loader.format_template("{message}", context)
        # Should handle nested references
        assert "Hello" in result or "{greeting}" in result

    def test_format_template_circular_reference(self):
        """Test circular reference handling"""
        loader = PromptLoader()
        loader.config["a"] = "{b}"
        loader.config["b"] = "{a}"

        context = {"a": "{a}", "b": "{b}"}

        # Should not infinite loop
        result = loader.format_template("{a}", context)
        assert "{" in result  # Should contain unresolved reference

    def test_get_template(self):
        """Test getting templates by path"""
        loader = PromptLoader()

        # Direct path
        template = loader.get_template("core_agent_context.supervised_context")
        assert "SUPERVISED AGENT CONTEXT" in template

        # Missing template
        assert loader.get_template("nonexistent.path") == ""

    def test_build_prompt_from_sections(self):
        """Test building prompts from sections"""
        loader = PromptLoader()

        sections = [
            "Header: {title}",
            "Content: {content}",
            "Footer"
        ]

        context = {
            "title": "Test",
            "content": "Main content"
        }

        result = loader.build_prompt_from_sections(sections, context)
        assert "Header: Test" in result
        assert "Content: Main content" in result
        assert "Footer" in result


class TestPromptGenerator:
    """Test PromptGenerator base class"""

    def test_initialization(self):
        """Test PromptGenerator initialization"""
        loader = PromptLoader()
        generator = PromptGenerator(loader)

        assert generator.loader == loader
        assert hasattr(generator, "get_initial_prompt")
        assert hasattr(generator, "get_continuation_prompt")


class TestTodoPromptManager:
    """Test TodoPromptManager"""

    def test_initialization(self):
        """Test TodoPromptManager initialization"""
        todos = ["TODO 1", "TODO 2", "TODO 3"]
        manager = TodoPromptManager(todos, max_turns=20)

        assert manager.context.todos == todos
        assert manager.context.max_turns == 20
        assert len(manager.context.remaining_todos) == 3
        assert manager.session_id == "unknown"
        assert manager.task_numbers == ""

    def test_update_progress(self):
        """Test progress updates"""
        todos = ["TODO 1", "TODO 2", "TODO 3"]
        manager = TodoPromptManager(todos, max_turns=20)

        # Mark some as complete
        completed = ["TODO 1", "TODO 3"]
        remaining = ["TODO 2"]

        manager.update_progress(completed, remaining)

        assert manager.context.completed_todos == completed
        assert manager.context.remaining_todos == remaining

    def test_add_issue(self):
        """Test adding issues"""
        manager = TodoPromptManager(["TODO 1"], max_turns=10)

        manager.add_issue("Connection timeout")
        manager.add_issue("File not found")

        assert len(manager.context.issues_encountered) == 2
        assert "Connection timeout" in manager.context.issues_encountered

    def test_get_initial_prompt(self):
        """Test initial prompt generation"""
        todos = ["Fix bug", "Add feature"]
        manager = TodoPromptManager(todos, max_turns=15)
        manager.session_id = "test_123"
        manager.task_numbers = "1, 2"

        prompt = manager.get_initial_prompt()

        # Check key elements
        assert "SUPERVISED AGENT CONTEXT" in prompt
        assert "Fix bug" in prompt
        assert "Add feature" in prompt
        assert "test_123" in prompt
        assert "Maximum 15 turns" in prompt
        assert "scratchpad" in prompt.lower()

    def test_get_continuation_prompt_incomplete(self):
        """Test continuation prompt for incomplete execution"""
        todos = ["TODO 1", "TODO 2", "TODO 3"]
        manager = TodoPromptManager(todos, max_turns=20)
        manager.session_id = "test_456"

        # Update progress
        manager.update_progress(["TODO 1"], ["TODO 2", "TODO 3"])
        manager.add_issue("Error occurred")

        # Create supervisor analysis
        supervisor_analysis = {
            "session_id": "test_456",
            "previous_session_id": "test_123",
            "task_numbers": "1, 2, 3"
        }

        prompt = manager.get_continuation_prompt(
            analysis_guidance="Focus on TODO 2 first",
            continuation_context="Previous work done",
            supervisor_analysis=supervisor_analysis
        )

        # Check elements
        assert "CONTINUATION CONTEXT" in prompt
        assert "TODO 2" in prompt
        assert "TODO 3" in prompt
        assert "TODO 1" not in prompt  # Completed
        assert "Focus on TODO 2 first" in prompt
        assert "test_456" in prompt

    def test_get_continuation_prompt_complete(self):
        """Test continuation prompt when all complete"""
        todos = ["TODO 1", "TODO 2"]
        manager = TodoPromptManager(todos, max_turns=20)
        manager.session_id = "test_789"

        # All complete
        manager.update_progress(todos, [])

        supervisor_analysis = {
            "session_id": "test_789",
            "task_numbers": "1, 2"
        }

        prompt = manager.get_continuation_prompt(
            analysis_guidance="Great work! Here are new tasks",
            continuation_context="Previous execution completed successfully",
            supervisor_analysis=supervisor_analysis
        )

        assert "NO REMAINING TODOS" in prompt
        assert "Great work!" in prompt

    def test_session_tracking(self):
        """Test session ID tracking"""
        manager = TodoPromptManager(["TODO 1"], max_turns=10)

        # Set session info
        manager.session_id = "session_123"
        manager.task_numbers = "5, 6, 7"

        prompt = manager.get_initial_prompt()

        assert "session_123" in prompt
        assert "5, 6, 7" in prompt

    def test_safety_notices(self):
        """Test safety notices in prompts"""
        manager = TodoPromptManager(["Dangerous task"], max_turns=10)

        prompt = manager.get_initial_prompt()

        # Check safety elements
        assert "--dangerously-skip-permissions" in prompt
        assert "NEVER perform destructive operations" in prompt
        assert "rm -rf" in prompt
        assert "DROP TABLE" in prompt

    def test_help_protocol(self):
        """Test help request protocol in prompts"""
        manager = TodoPromptManager(["Complex task"], max_turns=10)

        prompt = manager.get_initial_prompt()

        # Check help protocol
        assert "HELP NEEDED" in prompt
        assert "Status: STUCK" in prompt
        assert "REQUESTING HELP" in prompt

    def test_completion_protocol(self):
        """Test completion protocol in prompts"""
        manager = TodoPromptManager(["Task 1"], max_turns=10)

        prompt = manager.get_initial_prompt()

        # Check completion protocol
        assert "ALL TASKS COMPLETE" in prompt
        assert "Exit immediately" in prompt
        assert "COMPLETION PROTOCOL" in prompt

    def test_scratchpad_instructions(self):
        """Test scratchpad instructions"""
        manager = TodoPromptManager(["Task 1"], max_turns=10)
        manager.session_id = "test_scratch"

        prompt = manager.get_initial_prompt()

        # Check scratchpad elements
        assert ".cadence/scratchpad/session_test_scratch.md" in prompt
        assert "FIRST ACTION: Create your scratchpad" in prompt
        assert "Update your scratchpad IMMEDIATELY" in prompt

    def test_code_navigation_guidance(self):
        """Test Serena code navigation guidance"""
        manager = TodoPromptManager(["Refactor code"], max_turns=10)

        prompt = manager.get_initial_prompt()

        # Check Serena guidance
        assert "CODE NAVIGATION" in prompt
        assert "mcp__serena__find_symbol" in prompt
        assert "semantic tools" in prompt
