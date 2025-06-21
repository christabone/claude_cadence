#!/usr/bin/env python3
"""
Early Exit Demo - Shows how agents can complete tasks and exit early

This example demonstrates the early exit capability where agents
automatically exit when they complete all assigned tasks, potentially
saving turns and reducing costs.
"""

import json
from pathlib import Path
from cadence import CheckpointSupervisor, TaskManager


def create_demo_tasks():
    """Create simple demo tasks for testing early exit"""
    demo_tasks = {
        "tasks": [
            {
                "id": "1",
                "title": "Create hello.txt file",
                "description": "Create a file named hello.txt with 'Hello World' content",
                "status": "pending"
            },
            {
                "id": "2", 
                "title": "Create goodbye.txt file",
                "description": "Create a file named goodbye.txt with 'Goodbye World' content",
                "status": "pending"
            },
            {
                "id": "3",
                "title": "List created files",
                "description": "Use ls command to show the two created files",
                "status": "pending"
            }
        ]
    }
    
    # Create demo task file
    task_dir = Path(".taskmaster/tasks")
    task_dir.mkdir(parents=True, exist_ok=True)
    
    with open(task_dir / "tasks.json", 'w') as f:
        json.dump(demo_tasks, f, indent=2)
        
    print("✅ Created demo tasks in .taskmaster/tasks/tasks.json")


def main():
    """Run early exit demonstration"""
    
    print("🎯 Early Exit Demonstration")
    print("=" * 50)
    
    # Create demo tasks
    create_demo_tasks()
    
    # Create supervisor with generous turn budget
    supervisor = CheckpointSupervisor(
        checkpoint_turns=20,  # Give plenty of turns
        max_checkpoints=2,    # But agent should finish in 1
        verbose=True
    )
    
    # Load tasks
    task_manager = TaskManager(".")
    if task_manager.load_tasks():
        task_list = [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status
            }
            for task in task_manager.tasks
        ]
        
        print(f"\n📋 Loaded {len(task_list)} demo tasks")
        
        # Run with task list
        task_prompt = """Complete all the tasks listed above.
These are simple file creation tasks that should be quick to complete.
Mark each task as done and exit when finished."""
        
        success, cost = supervisor.run_supervised_task(
            initial_prompt=task_prompt,
            task_list=task_list
        )
        
        print("\n" + "=" * 50)
        print("📊 Results:")
        print(f"   - Success: {success}")
        print(f"   - Cost: ${cost:.4f}")
        
        # Check if files were created
        if Path("hello.txt").exists() and Path("goodbye.txt").exists():
            print("   - Files created: ✅")
            
            # Clean up
            Path("hello.txt").unlink()
            Path("goodbye.txt").unlink()
            print("   - Cleanup: ✅")
        else:
            print("   - Files created: ❌")
            
        # Analyze turn usage
        if supervisor.checkpoint_history:
            first_checkpoint = supervisor.checkpoint_history[0]
            turns_used = first_checkpoint['result'].turns_used
            turns_allocated = supervisor.checkpoint_turns
            
            print(f"   - Turns used: {turns_used}/{turns_allocated}")
            print(f"   - Turn savings: {turns_allocated - turns_used} turns")
            
            if turns_used < turns_allocated:
                print("   - Early exit: ✅ Agent completed tasks and exited early!")
            else:
                print("   - Early exit: ❌ Agent used all allocated turns")
    else:
        print("❌ Failed to load demo tasks")


if __name__ == "__main__":
    main()