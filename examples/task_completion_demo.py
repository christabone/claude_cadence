#!/usr/bin/env python3
"""
Task Completion Demo - Shows how agents complete TODOs and exit

This example demonstrates how agents work through TODOs naturally
and automatically exit when all tasks are complete.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadence import TaskSupervisor


def main():
    """Run task completion demonstration"""

    print("🎯 Task Completion Demonstration")
    print("=" * 50)
    print("\nThis demo shows how agents:")
    print("1. Work through TODOs systematically")
    print("2. Declare 'ALL TASKS COMPLETE' when done")
    print("3. Exit immediately without using unnecessary turns")
    print("=" * 50)

    # Create simple TODOs that should complete quickly
    simple_todos = [
        "Create a file named hello.txt with 'Hello World' content",
        "Create a file named goodbye.txt with 'Goodbye World' content",
        "Use the ls command to verify both files exist"
    ]

    # Create supervisor with a generous turn limit
    supervisor = TaskSupervisor(
        max_turns=30,  # Safety limit - agent should finish well before this
        verbose=True
    )

    print(f"\n📋 TODOs to complete: {len(simple_todos)}")
    print(f"🔢 Max turns (safety limit): {supervisor.max_turns}")
    print("\n🚀 Starting execution...")
    print("-" * 50)

    # Execute the TODOs
    result = supervisor.execute_with_todos(todos=simple_todos)

    print("-" * 50)
    print("\n📊 Results:")
    print(f"   - Success: {result.success}")
    print(f"   - Tasks complete: {result.task_complete}")
    print(f"   - Turns used: {result.turns_used}/{supervisor.max_turns}")

    # Calculate efficiency
    efficiency = (1 - result.turns_used / supervisor.max_turns) * 100
    print(f"   - Efficiency: {efficiency:.1f}% turns saved")

    # Check if files were created
    hello_exists = Path("hello.txt").exists()
    goodbye_exists = Path("goodbye.txt").exists()

    if hello_exists and goodbye_exists:
        print("   - Files created: ✅")

        # Read and display content
        with open("hello.txt") as f:
            print(f"   - hello.txt: '{f.read().strip()}'")
        with open("goodbye.txt") as f:
            print(f"   - goodbye.txt: '{f.read().strip()}'")

        # Clean up
        Path("hello.txt").unlink()
        Path("goodbye.txt").unlink()
        print("   - Cleanup: ✅")
    else:
        print("   - Files created: ❌")

    # Demonstrate more complex tasks
    print("\n" + "=" * 50)
    print("🎯 Complex Task Demonstration")
    print("=" * 50)

    complex_todos = [
        "Create a Python function called 'calculate_fibonacci' that returns the nth Fibonacci number",
        "Add comprehensive docstring with examples",
        "Create unit tests covering edge cases (n=0, n=1, n=10)",
        "Ensure all tests pass"
    ]

    # New supervisor for complex tasks
    complex_supervisor = TaskSupervisor(
        max_turns=40,
        verbose=True
    )

    print(f"\n📋 Complex TODOs: {len(complex_todos)}")
    print("\n🚀 Starting complex execution...")
    print("-" * 50)

    complex_result = complex_supervisor.execute_with_todos(todos=complex_todos)

    print("-" * 50)
    print("\n📊 Complex Task Results:")
    print(f"   - Success: {complex_result.success}")
    print(f"   - Tasks complete: {complex_result.task_complete}")
    print(f"   - Turns used: {complex_result.turns_used}/{complex_supervisor.max_turns}")

    # Summary
    print("\n" + "=" * 50)
    print("💡 Key Observations:")
    print("1. Agents complete tasks naturally without turn counting")
    print("2. Simple tasks use fewer turns automatically")
    print("3. Complex tasks may use more turns but still exit when done")
    print("4. The max_turns limit is only a safety net, not a target")


if __name__ == "__main__":
    main()
