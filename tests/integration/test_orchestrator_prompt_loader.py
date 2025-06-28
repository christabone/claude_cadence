#!/usr/bin/env python3
"""
Integration test for orchestrator using new PromptLoader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cadence.orchestrator import SupervisorOrchestrator
from cadence.task_supervisor import TaskSupervisor


def test_orchestrator_prompt_loader():
    """Test that orchestrator and task supervisor can use the new PromptLoader"""
    print("Testing Orchestrator and TaskSupervisor integration with new PromptLoader...")

    # Initialize task_file to None for proper cleanup
    task_file = None

    try:
        # Create a dummy task file
        task_file = Path("test_task.txt")
        task_file.write_text("Test task for integration")

        # Test SupervisorOrchestrator
        print("\n1. Testing SupervisorOrchestrator...")
        orch = SupervisorOrchestrator(
            project_root=Path.cwd(),
            task_file=task_file
        )
        assert hasattr(orch, 'prompt_loader'), "Orchestrator should have prompt_loader"
        assert orch.prompt_loader.__class__.__name__ == 'PromptLoader', "Should use PromptLoader"
        assert orch.prompt_loader.config is not None, "Config should be loaded"
        print("   ✅ SupervisorOrchestrator uses PromptLoader successfully")

        # Test template formatting
        test_template = "Session: {{ session_id }}"
        result = orch.prompt_loader.format_template(test_template, {"session_id": "test123"})
        assert result == "Session: test123", f"Template formatting failed: {result}"
        print("   ✅ Template formatting works correctly")

        # Test get_template method
        safety_notice = orch.prompt_loader.get_template("safety_notice_section")
        assert safety_notice and "Important Safety Notice" in safety_notice
        print("   ✅ get_template method works correctly")

        # Test TaskSupervisor
        print("\n2. Testing TaskSupervisor...")
        supervisor = TaskSupervisor()
        assert hasattr(supervisor, 'prompt_loader'), "TaskSupervisor should have prompt_loader"
        assert supervisor.prompt_loader.__class__.__name__ == 'PromptLoader', "Should use PromptLoader"
        print("   ✅ TaskSupervisor uses PromptLoader successfully")

        print("\n✅ All integration tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always clean up the task file
        if task_file and task_file.exists():
            try:
                task_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    success = test_orchestrator_prompt_loader()
    sys.exit(0 if success else 1)
