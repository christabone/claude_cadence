#!/usr/bin/env python3
"""
Task Master guided supervision example

This example shows how to use Task Master to track and guide
agent execution across checkpoints.
"""

from pathlib import Path
from cadence import CheckpointSupervisor, TaskManager


class TaskGuidedSupervisor(CheckpointSupervisor):
    """Extended supervisor that uses Task Master for guidance"""
    
    def __init__(self, project_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.task_manager = TaskManager(project_dir)
        self.tasks_loaded = self.task_manager.load_tasks()
        
    def analyze_checkpoint(self, result, checkpoint_num):
        """Enhanced analysis using Task Master"""
        # Get base analysis
        analysis = super().analyze_checkpoint(result, checkpoint_num)
        
        if self.tasks_loaded:
            # Analyze task progress
            output_text = "\n".join(result.output_lines)
            progress = self.task_manager.analyze_progress(output_text)
            
            # Update completed tasks
            for task in progress["completed"]:
                self.task_manager.update_task_status(task.id, "done")
                
            # Enhanced guidance based on tasks
            if progress["incomplete"]:
                incomplete_summary = "\n".join([
                    f"- Task {t.id}: {t.title}"
                    for t in progress["incomplete"][:3]
                ])
                
                analysis.guidance = f"""Task-based guidance:

Completed tasks this checkpoint: {len(progress['completed'])}
Remaining tasks: {len(progress['incomplete'])}

Focus on these incomplete tasks:
{incomplete_summary}

{analysis.guidance}"""
                
        return analysis
        
    def run_supervised_task(self, initial_prompt: str = None):
        """Run with Task Master integration"""
        
        # If no prompt provided, generate from tasks
        if initial_prompt is None and self.tasks_loaded:
            task_summary = self.task_manager.generate_task_summary()
            initial_prompt = f"""Complete the following tasks:

{task_summary}

Work through each task systematically and mark your progress clearly."""
            
        return super().run_supervised_task(initial_prompt)


def main():
    """Run a task-guided supervision session"""
    
    # Assume we're in a project with .taskmaster/tasks/tasks.json
    supervisor = TaskGuidedSupervisor(
        project_dir=".",
        checkpoint_turns=15,
        max_checkpoints=4,
        verbose=True
    )
    
    if not supervisor.tasks_loaded:
        print("⚠️  No Task Master tasks found. Running without task guidance.")
        print("💡 Tip: Initialize Task Master with 'claude -p \"Set up task master\"'")
        
        # Fall back to basic supervision
        task_prompt = """
        Create a Python module for data validation with:
        1. Email validation
        2. Phone number validation  
        3. URL validation
        4. Comprehensive test suite
        """
        
        success, cost = supervisor.run_supervised_task(task_prompt)
    else:
        # Run with Task Master guidance
        print("✅ Task Master tasks loaded!")
        success, cost = supervisor.run_supervised_task()
        
    print(f"\n{'✅' if success else '❌'} Final result: {'Success' if success else 'Incomplete'}")
    print(f"💰 Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()