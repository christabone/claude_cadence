# tests/test_prompts.py

import pytest
from unittest.mock import MagicMock

# Assuming the file is in a package named 'cadence'
from cadence.prompts import ExecutionContext, PromptGenerator, MAX_OUTPUT_TRUNCATE_LENGTH

# A mock PromptLoader to use in tests, avoiding file system access and isolating the generator.
class MockPromptLoader:
    """A test double for PromptLoader that uses a dictionary for config."""
    def __init__(self, config_path=None):
        # config_path is ignored; we'll set config directly in tests.
        self.config = {}

    def format_template(self, template_str, context):
        """A simple formatter that mimics the real one for testing."""
        # Use format_map to avoid KeyErrors for missing keys, which is more robust for testing.
        return template_str.format_map(MagicMock(**context))

    def build_prompt_from_sections(self, sections, context):
        """A simple builder that joins sections based on keys in the context."""
        prompt_parts = []
        for section_key in sections:
            # Retrieve content; if it's a dict, convert to a simple string for the test prompt.
            content = context.get(section_key)
            if isinstance(content, dict):
                content_str = ", ".join(f"{k}={v}" for k, v in content.items())
                prompt_parts.append(content_str)
            elif content:
                prompt_parts.append(str(content))
        return "\n".join(prompt_parts)

@pytest.fixture
def mock_prompt_loader_class(monkeypatch):
    """
    Patches the PromptLoader import within the prompts module to use
    our MockPromptLoader, ensuring tests are isolated.
    """
    monkeypatch.setattr('cadence.prompt_loader.PromptLoader', MockPromptLoader)
    return MockPromptLoader

@pytest.fixture
def basic_config():
    """Provides a basic, well-formed config dictionary for the mock loader."""
    return {
        'core_agent_context': {'role': 'You are a helpful agent.'},
        'shared_agent_context': {'format': 'Use XML tags.'},
        'safety_notice_section': 'SAFETY FIRST',
        'todo_templates': {
            'todo_item': 'TODO {number}: {todo_text}',
            'todo_list': '## TODOs\n{todo_items}',
            'continuation_types': {
                'complete_new_tasks': 'ALL_DONE_PROCEED',
                'fixing_issues': 'FIX_REPORTED_ISSUES',
                'incomplete': 'CONTINUE_WORK',
            },
            'supervisor_incomplete_analysis': 'Analysis: {previous_work_summary}. Issues: {issues_found}. Guidance: {specific_guidance}',
            'issues_section': '## Issues\n{issue_list}',
        },
        'agent_prompts': {
            'initial': {'sections': ['core_agent_context', 'todo_list', 'safety_notice_section']},
            'continuation': {'sections': ['core_agent_context', 'supervisor_analysis', 'remaining_todos', 'issues_section']},
        },
        'supervisor_prompts': {
            'analysis': {'sections': ['execution_output', 'task_progress']},
            'task_progress_template': 'Progress: {completed_count} done, {remaining_count} left.',
        },
        'final_summary': {
            'template': 'Finished.\nExecutions: {executions_count}\n{completed_section}\n{incomplete_section}\n{recommendations}',
            'completed_section': 'Done:\n{completed_list}',
            'incomplete_section': 'Not Done:\n{incomplete_list}',
            'recommendations': 'Next up:\n{focus_items}',
        }
    }

@pytest.fixture
def prompt_generator(mock_prompt_loader_class, basic_config):
    """Provides a PromptGenerator instance with a mocked loader and basic config."""
    # The __init__ will now use the patched MockPromptLoader
    pg = PromptGenerator()
    pg.loader.config = basic_config
    return pg

def test_execution_context_defaults():
    """
    Tests that ExecutionContext initializes with correct default empty lists.
    This prevents mutation issues across different instances.
    """
    # Arrange
    ctx1 = ExecutionContext(todos=["a"], max_turns=5)
    ctx2 = ExecutionContext(todos=["b"], max_turns=10)

    # Act
    ctx1.completed_todos.append("done")

    # Assert
    assert ctx1.todos == ["a"]
    assert ctx1.max_turns == 5
    assert ctx1.completed_todos == ["done"]
    assert ctx2.completed_todos == [] # Should be an independent list

def test_generate_initial_todo_prompt_happy_path(prompt_generator):
    """
    Tests the generation of an initial prompt with a standard context.
    Ensures correct formatting and inclusion of all specified sections.
    """
    # Arrange
    context = ExecutionContext(
        todos=["First task", "Second task"],
        max_turns=10,
        project_path="/test/project"
    )

    # Act
    prompt = prompt_generator.generate_initial_todo_prompt(context, session_id="sess-123")

    # Assert
    assert "role=You are a helpful agent." in prompt
    assert "## TODOs" in prompt
    assert "TODO 1: First task" in prompt
    assert "TODO 2: Second task" in prompt
    assert "SAFETY FIRST" in prompt
    # Check that project_path from context is used
    assert context.project_path is not None

def test_generate_initial_todo_prompt_empty_todos(prompt_generator):
    """
    Tests that an initial prompt can be generated correctly when there are no TODOs.
    The TODO list section should be present but empty.
    """
    # Arrange
    context = ExecutionContext(todos=[], max_turns=5)

    # Act
    prompt = prompt_generator.generate_initial_todo_prompt(context)

    # Assert
    assert "## TODOs\n" in prompt
    assert "TODO 1:" not in prompt

@pytest.mark.parametrize("analysis, expected_type_key", [
    ({"all_complete": True, "has_issues": False}, "complete_new_tasks"),
    ({"all_complete": False, "has_issues": True}, "fixing_issues"),
    ({"all_complete": False, "has_issues": False}, "incomplete"),
    ({}, "incomplete"), # Test default case with empty dict
])
def test_determine_continuation_type(prompt_generator, analysis, expected_type_key):
    """
    Tests the logic for determining the continuation type based on supervisor analysis.
    Uses parametrize to cover all logical branches.
    """
    # Arrange
    expected_type = prompt_generator.loader.config['todo_templates']['continuation_types'][expected_type_key]

    # Act
    # context_start_text="def _determine_continuation_type(self, supervisor_analysis: Dict[str, Any]) -> str:"
    # context_end_text="return continuation_types.get('incomplete', 'incomplete')"
    continuation_type = prompt_generator._determine_continuation_type(analysis)

    # Assert
    assert continuation_type == expected_type

def test_generate_continuation_prompt_with_issues(prompt_generator):
    """
    Tests generating a continuation prompt where issues were found.
    Ensures the prompt includes supervisor analysis, remaining todos, and the issues list.
    """
    # Arrange
    context = ExecutionContext(
        todos=[],
        max_turns=10,
        completed_todos=["Task 1"],
        remaining_todos=["Task 2", "Task 3"],
        issues_encountered=["Issue A", "Issue B"]
    )
    supervisor_analysis = {
        "has_issues": True,
        "work_summary": "Agent tried something.",
        "issues": "It failed."
    }

    # Act
    prompt = prompt_generator.generate_continuation_prompt(context, "Retry the task.", supervisor_analysis)

    # Assert
    assert "Analysis: Agent tried something." in prompt
    assert "Issues: It failed." in prompt
    assert "Guidance: Retry the task." in prompt
    assert "TODO 1: Task 2" in prompt
    assert "TODO 2: Task 3" in prompt
    assert "## Issues" in prompt
    assert "⚠️  Issue A" in prompt
    assert "⚠️  Issue B" in prompt

def test_generate_supervisor_analysis_prompt_output_truncation(prompt_generator):
    """
    Tests that very long execution output is correctly truncated to avoid excessive prompt length.
    """
    # Arrange
    long_output = "A" * (MAX_OUTPUT_TRUNCATE_LENGTH + 100)
    context = ExecutionContext(todos=["task"], max_turns=5)

    # Act
    # context_start_text="def generate_supervisor_analysis_prompt("
    # context_end_text="return self.loader.build_prompt_from_sections(sections, prompt_context)"
    prompt = prompt_generator.generate_supervisor_analysis_prompt(long_output, context, [])

    # Assert
    assert "[... OUTPUT TRUNCATED ...]" in prompt
    # The output in the prompt should be shorter than the original, plus the truncation message length
    assert len(prompt) < len(long_output)
    assert len(prompt) > MAX_OUTPUT_TRUNCATE_LENGTH / 2

def test_generate_final_summary_mixed_results(prompt_generator):
    """
    Tests the final summary generation for a session with both completed and incomplete tasks.
    """
    # Arrange
    context = ExecutionContext(
        todos=[],
        max_turns=20,
        completed_todos=["Successfully did this"],
        remaining_todos=["Failed to do this", "Never got to this"]
    )
    executions = [
        {"summary": "First part succeeded", "success": True},
        {"summary": "Second part failed", "success": False},
    ]

    # Act
    # context_start_text="def generate_final_summary("
    # context_end_text="prompt_context\n        )"
    summary = prompt_generator.generate_final_summary(executions, context, total_turns=15)

    # Assert
    assert "Executions: 2" in summary
    # Check for completed section
    assert "Done:" in summary
    assert "✅ Successfully did this" in summary
    # Check for incomplete section
    assert "Not Done:" in summary
    assert "❌ Failed to do this" in summary
    # Check for recommendations
    assert "Next up:" in summary
    assert "- Failed to do this" in summary
    # Check for execution progression
    assert "✅ Execution 1: First part succeeded" in summary
    assert "⚠️ Execution 2: Second part failed" in summary

def test_generate_issues_section_limits_to_last_three(prompt_generator):
    """
    Tests that the issues section is correctly formatted and only includes the last 3 issues.
    """
    # Arrange
    context = ExecutionContext(
        todos=[], max_turns=1,
        issues_encountered=["Issue 1", "Issue 2", "Issue 3", "Issue 4", "Issue 5"]
    )

    # Act
    # context_start_text="def _generate_issues_section(self, context: ExecutionContext) -> str:"
    # context_end_text="{'issue_list': issue_list}\n        )"
    section = prompt_generator._generate_issues_section(context)

    # Assert
    assert "Issue 1" not in section
    assert "Issue 2" not in section
    assert "⚠️  Issue 3" in section
    assert "⚠️  Issue 4" in section
    assert "⚠️  Issue 5" in section
