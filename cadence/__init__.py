"""
Claude Cadence - Task-driven supervision for Claude Code agents

This package provides a framework for managing Claude Code agent execution
with Task Master integration through the orchestrator.
"""

from .config import ConfigLoader, CadenceConfig
from .task_manager import TaskManager, Task
from .zen_integration import ZenIntegration, ZenRequest

__version__ = "0.1.0"

__all__ = [
    "ConfigLoader",
    "CadenceConfig",
    "TaskManager",
    "Task",
    "ZenIntegration",
    "ZenRequest"
]
