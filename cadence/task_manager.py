"""
Task Master integration for structured task tracking
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Task:
    """Represents a Task Master task"""
    id: str
    title: str
    description: str
    status: str
    subtasks: List['Task'] = None

    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Create Task from Task Master JSON format"""
        subtasks = []
        if 'subtasks' in data and data['subtasks']:
            subtasks = [cls.from_dict(st) for st in data['subtasks']]

        return cls(
            id=str(data.get('id', '')),
            title=data.get('title', ''),
            description=data.get('description', ''),
            status=data.get('status', 'pending'),
            subtasks=subtasks
        )

    def is_complete(self) -> bool:
        """Check if task is marked as done"""
        return self.status == 'done'


class TaskManager:
    """
    Manages Task Master integration for guided supervision
    """

    def __init__(self, project_dir: str = "."):
        """
        Initialize Task Manager

        Args:
            project_dir: Project directory containing .taskmaster folder
        """
        self.project_dir = Path(project_dir)
        self.taskmaster_dir = self.project_dir / ".taskmaster"
        self.tasks_file = self.taskmaster_dir / "tasks" / "tasks.json"
        self.tasks = []

    def load_tasks(self) -> bool:
        """
        Load tasks from Task Master

        Returns:
            True if tasks loaded successfully
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading tasks from: {self.tasks_file}")

        if not self.tasks_file.exists():
            logger.error(f"Tasks file does not exist: {self.tasks_file}")
            return False

        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)

            # Extract tasks from Task Master format
            # Handle different Task Master formats
            if isinstance(data, dict):
                if 'master' in data and 'tasks' in data['master']:
                    # New format: {"master": {"tasks": [...]}}
                    task_list = data['master']['tasks']
                elif 'tasks' in data:
                    # Old format: {"tasks": [...]}
                    task_list = data['tasks']
                else:
                    logger.error(f"Unknown task format. Keys found: {list(data.keys())}")
                    return False
            elif isinstance(data, list):
                task_list = data
            else:
                logger.error(f"Unknown data type: {type(data)}")
                return False

            # Convert to Task objects
            self.tasks = []
            for task_data in task_list:
                task = Task.from_dict(task_data)
                self.tasks.append(task)

            return True

        except (json.JSONDecodeError, IOError) as e:
            # Log the error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load tasks: {type(e).__name__}: {e}")
            return False
        except Exception as e:
            # Catch any other exceptions
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error loading tasks: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def analyze_progress(self, output: str) -> Dict[str, List[Task]]:
        """
        Analyze output to determine task completion

        Args:
            output: Agent output to analyze

        Returns:
            Dict with 'completed' and 'incomplete' task lists
        """
        completed = []
        incomplete = []

        output_lower = output.lower()

        # First check for structured task completion markers
        task_complete_pattern = r'\{"task_complete":\s*\{"task_id":\s*"([^"]+)"[^}]*\}\}'
        structured_completions = re.findall(task_complete_pattern, output)
        completed_task_ids = set(structured_completions)

        # Also check for "ALL TASKS COMPLETE" declaration
        all_tasks_complete = "all tasks complete" in output_lower

        for task in self.tasks:
            # Skip already completed tasks
            if task.is_complete():
                completed.append(task)
                continue

            # Check if this task was marked complete via structured output
            if task.id in completed_task_ids:
                completed.append(task)
                continue

            # If agent declared all tasks complete, mark all as complete
            if all_tasks_complete:
                completed.append(task)
                continue

            # Otherwise, look for task-specific completion indicators
            task_completed = False

            # Check if task title or ID is mentioned with completion words
            task_refs = [
                task.title.lower(),
                f"task {task.id}",
                f"#{task.id}"
            ]

            completion_words = [
                "completed", "done", "finished", "implemented",
                "fixed", "resolved", "added", "created"
            ]

            for ref in task_refs:
                if ref in output_lower:
                    # Check if completion word appears near the reference
                    ref_index = output_lower.find(ref)
                    context = output_lower[max(0, ref_index-50):ref_index+50]

                    for word in completion_words:
                        if word in context:
                            task_completed = True
                            break

                if task_completed:
                    break

            # Also check task description keywords
            if not task_completed and task.description:
                desc_keywords = [w.lower() for w in task.description.split()
                               if len(w) > 4][:3]  # First 3 significant words

                if all(keyword in output_lower for keyword in desc_keywords):
                    # Strong indicator that task was addressed
                    for word in completion_words:
                        if word in output_lower:
                            task_completed = True
                            break

            if task_completed:
                completed.append(task)
            else:
                incomplete.append(task)

        return {
            "completed": completed,
            "incomplete": incomplete
        }

    def update_task_status(self, task_id: str, status: str):
        """
        Update status of a specific task

        Args:
            task_id: Task ID to update
            status: New status value
        """
        # Find and update the task
        updated = False

        def update_task_recursive(tasks: List[Task]) -> bool:
            for task in tasks:
                if task.id == task_id:
                    task.status = status
                    return True
                # Check subtasks
                if task.subtasks and update_task_recursive(task.subtasks):
                    return True
            return False

        updated = update_task_recursive(self.tasks)

        if updated:
            # Save back to Task Master format
            self._save_tasks()

    def _save_tasks(self):
        """Save tasks back to Task Master file"""
        # Convert tasks back to dict format
        def task_to_dict(task: Task) -> dict:
            result = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status
            }
            if task.subtasks:
                result["subtasks"] = [task_to_dict(st) for st in task.subtasks]
            return result

        task_data = {
            "tasks": [task_to_dict(task) for task in self.tasks]
        }

        # Ensure directory exists
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(self.tasks_file, 'w') as f:
            json.dump(task_data, f, indent=2)

    def get_incomplete_tasks(self) -> List[Task]:
        """Get list of incomplete tasks"""
        incomplete = []

        def collect_incomplete(tasks: List[Task]):
            for task in tasks:
                if not task.is_complete():
                    incomplete.append(task)
                if task.subtasks:
                    collect_incomplete(task.subtasks)

        collect_incomplete(self.tasks)
        return incomplete

    def generate_task_summary(self) -> str:
        """Generate a summary of current task status"""
        total_tasks = len(self.tasks)
        completed_count = sum(1 for task in self.tasks if task.is_complete())

        summary = f"Task Progress: {completed_count}/{total_tasks} completed\n\n"

        for task in self.tasks:
            status_icon = "✅" if task.is_complete() else "⏳"
            summary += f"{status_icon} Task {task.id}: {task.title}\n"

            if task.subtasks:
                for subtask in task.subtasks:
                    sub_icon = "✅" if subtask.is_complete() else "⏳"
                    summary += f"  {sub_icon} {subtask.id}: {subtask.title}\n"

        return summary
