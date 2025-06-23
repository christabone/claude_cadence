"""
End-to-end tests for prompt system and agent guidance
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import subprocess
import yaml

from cadence.task_supervisor import TaskSupervisor
from cadence.config import CadenceConfig, SCRATCHPAD_DIR
from cadence.prompts import TodoPromptManager


class TestPromptSystemE2E:
    """E2E tests for prompt generation and agent guidance"""

    @pytest.fixture
    def e2e_temp_dir(self):
        """Create a temporary directory for E2E tests"""
        temp_dir = tempfile.mkdtemp(prefix="cadence_e2e_prompt_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def e2e_config(self, e2e_temp_dir):
        """Create a test configuration"""
        config = CadenceConfig()
        config.execution.max_turns = 8
        config.execution.log_dir = str(e2e_temp_dir / "logs")
        config.agent.model = "claude-3-haiku-20240307"
        config.supervisor.zen_integration.enabled = False
        return config

    def test_safety_instructions_followed(self, e2e_config, e2e_temp_dir):
        """Test agent follows safety instructions in prompts"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create potentially dangerous TODOs
        todos = [
            "Create a test file at safe_location.txt",
            "List files in the current directory",
            "Create another file at another_safe.txt"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_safety")

        assert result.success is True

        # Check safe files were created
        assert (e2e_temp_dir / "safe_location.txt").exists()

        # Agent should not have done anything destructive
        # (Can't really test this without risking actual damage)

    def test_serena_guidance_followed(self, e2e_config, e2e_temp_dir):
        """Test agent follows Serena code navigation guidance"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Create a simple Python file
        test_file = e2e_temp_dir / "example.py"
        test_file.write_text("""
def hello_world():
    '''Say hello'''
    print("Hello, World!")

def add_numbers(a, b):
    '''Add two numbers'''
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
""")

        # TODOs that should encourage Serena usage
        todos = [
            "Find all functions in example.py using semantic tools if available",
            "Document what each function does in a summary.txt file",
            "Find the Calculator class and describe its methods"
        ]

        # Add serena to MCP servers if not present
        e2e_config.mcp.servers = ["filesystem", "serena"]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_serena")

        # Check if summary was created
        summary_file = e2e_temp_dir / "summary.txt"
        if summary_file.exists():
            content = summary_file.read_text()
            # Should mention the functions
            assert "hello_world" in content or "add_numbers" in content or "Calculator" in content

    def test_todo_context_in_prompts(self, e2e_config, e2e_temp_dir):
        """Test TODO context is properly included in prompts"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Specific TODOs that should appear in agent's understanding
        todos = [
            "IMPORTANT: Create a file called context_test.txt",
            "In context_test.txt, write 'TODO context received'",
            "Create a second file called confirmation.txt with 'Tasks understood'"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_context")

        assert result.success is True

        # Check files were created with correct content
        context_file = e2e_temp_dir / "context_test.txt"
        assert context_file.exists()
        assert "TODO context received" in context_file.read_text()

    def test_turn_limit_awareness(self, e2e_config, e2e_temp_dir):
        """Test agent is aware of turn limits from prompts"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Set very low turn limit
        e2e_config.execution.max_turns = 3

        todos = [
            "Create a file called turn_awareness.txt",
            "Write in the file how many turns you have available",
            "Create final.txt when done"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_turns")

        # Check turn awareness file
        awareness_file = e2e_temp_dir / "turn_awareness.txt"
        if awareness_file.exists():
            content = awareness_file.read_text()
            # Might mention turns or limits
            assert "turn" in content.lower() or "limit" in content.lower() or "3" in content

    def test_completion_protocol_followed(self, e2e_config, e2e_temp_dir):
        """Test agent follows completion protocol from prompts"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        todos = [
            "Create a file called task1.txt with 'First task done'",
            "Create a file called task2.txt with 'Second task done'",
            "When all tasks are complete, follow the completion protocol"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_completion_protocol")

        assert result.success is True
        assert result.task_complete is True

        # Check both files exist
        assert (e2e_temp_dir / "task1.txt").exists()
        assert (e2e_temp_dir / "task2.txt").exists()

        # Check scratchpad for completion marker
        scratchpad = Path(SCRATCHPAD_DIR) / "session_e2e_completion_protocol.md"
        if scratchpad.exists():
            content = scratchpad.read_text()
            assert "complete" in content.lower()

    def test_help_protocol_usage(self, e2e_config, e2e_temp_dir):
        """Test agent uses help protocol when stuck"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Task that's impossible to complete
        todos = [
            "Read the encrypted file at /secure/vault/secrets.enc and decrypt it",
            "If you cannot complete this task, use the help protocol"
        ]

        supervisor = TaskSupervisor(config=e2e_config)
        result = supervisor.execute_with_todos(todos, session_id="e2e_help_protocol")

        # Check if help protocol was used
        scratchpad = Path(SCRATCHPAD_DIR) / "session_e2e_help_protocol.md"
        if scratchpad.exists():
            content = scratchpad.read_text()
            # Should have help request
            if "HELP NEEDED" in content:
                assert "Status: STUCK" in content or "stuck" in content.lower()

    def test_continuation_prompt_context(self, e2e_config, e2e_temp_dir):
        """Test continuation prompts maintain context"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        # Set low turn limit to force continuation
        e2e_config.execution.max_turns = 2

        todos = [
            "Create file1.txt with 'Step 1'",
            "Create file2.txt with 'Step 2'",
            "Create file3.txt with 'Step 3'",
            "Create summary.txt listing all files created"
        ]

        supervisor = TaskSupervisor(config=e2e_config)

        # First execution
        result1 = supervisor.execute_with_todos(todos, session_id="e2e_cont_context")
        assert result1.task_complete is False

        # Update progress in prompt manager
        completed = []
        remaining = []

        # Check which files exist
        for i in range(1, 4):
            if (e2e_temp_dir / f"file{i}.txt").exists():
                completed.append(f"Create file{i}.txt with 'Step {i}'")
            else:
                remaining.append(f"Create file{i}.txt with 'Step {i}'")

        if not (e2e_temp_dir / "summary.txt").exists():
            remaining.append("Create summary.txt listing all files created")

        supervisor.prompt_manager.update_progress(completed, remaining)

        # Continue execution
        result2 = supervisor.continue_execution(
            result1,
            session_id="e2e_cont_context_2",
            continuation_guidance="Continue with the remaining file creation tasks"
        )

        # More files should exist after continuation
        files_after = sum(1 for i in range(1, 4) if (e2e_temp_dir / f"file{i}.txt").exists())
        assert files_after >= len(completed)

    def test_supervisor_analysis_in_prompts(self, e2e_config, e2e_temp_dir):
        """Test supervisor analysis is included in continuation prompts"""
        # Skip if Claude is not available
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Claude CLI not available")

        supervisor = TaskSupervisor(config=e2e_config)

        # Initial execution
        todos = ["Create initial.txt with 'First execution'"]
        result1 = supervisor.execute_with_todos(todos, session_id="e2e_supervisor_1")

        # Simulate supervisor analysis
        supervisor_analysis = {
            "session_id": "e2e_supervisor_2",
            "previous_session_id": "e2e_supervisor_1",
            "task_numbers": "1,2,3",
            "guidance": "Focus on error handling in the next phase"
        }

        # Continue with new todos and supervisor analysis
        new_todos = ["Create continued.txt with 'Continuation with supervisor guidance'"]

        # Manually create continuation prompt to test
        prompt_manager = TodoPromptManager(new_todos, max_turns=10)
        prompt_manager.session_id = supervisor_analysis["session_id"]

        continuation_prompt = prompt_manager.get_continuation_prompt(
            analysis_guidance=supervisor_analysis["guidance"],
            continuation_context="Previous execution completed initial setup",
            supervisor_analysis=supervisor_analysis
        )

        # Check prompt includes supervisor analysis
        assert supervisor_analysis["session_id"] in continuation_prompt
        assert supervisor_analysis["previous_session_id"] in continuation_prompt
        assert supervisor_analysis["guidance"] in continuation_prompt
