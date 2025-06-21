"""
Claude Cadence - Checkpoint-based supervision for Claude Code agents
"""

from .supervisor import CheckpointSupervisor
from .task_manager import TaskManager
from .prompts import ContextAwarePromptManager, PromptGenerator

__version__ = "0.1.0"
__all__ = ["CheckpointSupervisor", "TaskManager", "ContextAwarePromptManager", "PromptGenerator"]