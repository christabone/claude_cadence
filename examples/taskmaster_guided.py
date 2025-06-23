#!/usr/bin/env python3
"""
Task Master guided execution example

This example shows how Claude Cadence integrates with Task Master
for structured task management and completion tracking.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadence import TaskSupervisor


def create_sample_tasks(task_file: str):
    """Create a sample tasks.json file for demonstration"""
    tasks = {
        "project": "Sample Project",
        "tasks": [
            {
                "id": "1",
                "title": "Set up project structure",
                "description": "Create the basic directory structure and initial files",
                "status": "pending",
                "priority": "high"
            },
            {
                "id": "2",
                "title": "Implement core functionality",
                "description": "Build the main features of the application",
                "status": "pending",
                "priority": "high",
                "dependencies": ["1"]
            },
            {
                "id": "3",
                "title": "Add unit tests",
                "description": "Write comprehensive unit tests for all components",
                "status": "pending",
                "priority": "medium",
                "dependencies": ["2"]
            },
            {
                "id": "4",
                "title": "Create documentation",
                "description": "Write README and API documentation",
                "status": "pending",
                "priority": "medium",
                "dependencies": ["2"]
            },
            {
                "id": "5",
                "title": "Set up CI/CD",
                "description": "Configure continuous integration and deployment",
                "status": "pending",
                "priority": "low",
                "dependencies": ["3"]
            }
        ]
    }

    # Ensure directory exists
    Path(task_file).parent.mkdir(parents=True, exist_ok=True)

    # Write tasks file
    with open(task_file, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"✅ Created sample tasks file: {task_file}")
    return tasks


def main():
    """Run a Task Master guided execution session"""

    # Set up paths
    project_root = Path.cwd()
    tasks_file = project_root / ".taskmaster" / "tasks" / "tasks.json"

    # Create sample tasks if file doesn't exist
    if not tasks_file.exists():
        print("Creating sample Task Master tasks...")
        tasks = create_sample_tasks(str(tasks_file))
    else:
        print(f"Using existing tasks file: {tasks_file}")
        with open(tasks_file) as f:
            tasks = json.load(f)

    # Display initial task status
    print("\n📋 Initial Task Status:")
    for task in tasks["tasks"]:
        status_icon = "✅" if task["status"] == "done" else "⏳"
        print(f"{status_icon} Task {task['id']}: {task['title']} [{task['status']}]")

    # Create supervisor with Task Master integration
    supervisor = TaskSupervisor(
        max_turns=50,  # Higher limit for complex project
        verbose=True,
        config_path="config.yaml"  # Use project config if available
    )

    print("\n🚀 Starting Task Master guided execution...")
    print("=" * 60)

    # Run with Task Master
    success = supervisor.run_with_taskmaster(task_file=str(tasks_file))

    print("\n" + "=" * 60)
    print("📊 Final Summary:")

    # Check updated task status
    with open(tasks_file) as f:
        updated_tasks = json.load(f)

    completed_count = 0
    for task in updated_tasks["tasks"]:
        if task["status"] == "done":
            completed_count += 1
            print(f"✅ Task {task['id']}: {task['title']}")
        else:
            print(f"❌ Task {task['id']}: {task['title']} [{task['status']}]")

    print(f"\nCompletion: {completed_count}/{len(updated_tasks['tasks'])} tasks")

    # Show execution history
    if hasattr(supervisor, 'execution_history'):
        print(f"\nExecutions: {len(supervisor.execution_history)}")
        total_turns = sum(r.turns_used for r in supervisor.execution_history)
        print(f"Total turns used: {total_turns}")

    # Recommend next steps if incomplete
    if completed_count < len(updated_tasks["tasks"]):
        print("\n💡 Recommendation: Run again to complete remaining tasks")
        print("   The supervisor will pick up where it left off")


if __name__ == "__main__":
    main()
