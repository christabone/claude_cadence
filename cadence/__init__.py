"""
Claude Cadence - Task-driven supervision for Claude Code agents

This package provides a framework for managing Claude Code agent execution
with Task Master integration. Agents work on TODOs until completion or
until reaching a maximum turn limit.
"""

from .task_supervisor import TaskSupervisor, ExecutionResult
from .config import ConfigLoader, CadenceConfig
from .task_manager import TaskManager, Task

__version__ = "0.1.0"

__all__ = [
    "TaskSupervisor",
    "ExecutionResult",
    "ConfigLoader", 
    "CadenceConfig",
    "TaskManager",
    "Task"
]