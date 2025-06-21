#!/usr/bin/env python3
"""
Basic checkpoint supervision example
"""

from cadence import CheckpointSupervisor


def main():
    """Run a basic supervised task"""
    
    # Create supervisor with 10-turn checkpoints
    supervisor = CheckpointSupervisor(
        checkpoint_turns=10,
        max_checkpoints=3,
        verbose=True
    )
    
    # Define the task
    task_prompt = """
    Write a Python function that calculates the Fibonacci sequence.
    Include proper documentation and test cases.
    """
    
    # Run with supervision
    success, cost = supervisor.run_supervised_task(task_prompt)
    
    if success:
        print(f"✅ Task completed successfully!")
        print(f"💰 Total cost: ${cost:.4f}")
    else:
        print(f"❌ Task incomplete after {supervisor.max_checkpoints} checkpoints")


if __name__ == "__main__":
    main()