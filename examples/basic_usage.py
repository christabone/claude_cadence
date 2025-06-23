#!/usr/bin/env python3
"""
Basic usage example of Claude Cadence task-driven supervision
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadence import TaskSupervisor


def main():
    # Create supervisor with default settings
    supervisor = TaskSupervisor(verbose=True)

    # Example 1: Run with Task Master integration
    print("=== Example 1: Task Master Integration ===")
    success = supervisor.run_with_taskmaster()
    print(f"Task Master execution completed: {success}")

    # Example 2: Run with manual TODOs
    print("\n=== Example 2: Manual TODOs ===")
    todos = [
        "Create a Python function to calculate fibonacci numbers",
        "Add comprehensive docstrings",
        "Write unit tests with edge cases",
        "Add type hints"
    ]

    result = supervisor.execute_with_todos(todos=todos)
    print(f"Manual TODO execution completed: {result.task_complete}")
    print(f"Turns used: {result.turns_used}")

    # Example 3: Custom configuration
    print("\n=== Example 3: Custom Configuration ===")
    custom_supervisor = TaskSupervisor(
        max_turns=20,  # Lower safety limit
        model="claude-3-opus-latest",  # Different model
        verbose=True
    )

    todos = ["Refactor the codebase to use async/await patterns"]
    result = custom_supervisor.execute_with_todos(todos=todos)
    print(f"Custom execution completed: {result.task_complete}")


if __name__ == "__main__":
    main()
