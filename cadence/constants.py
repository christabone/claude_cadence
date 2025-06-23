"""
Constants for Claude Cadence Orchestrator and Supervisor

This module contains all configuration constants and magic numbers
used throughout the Claude Cadence system.
"""

from pathlib import Path


class OrchestratorDefaults:
    """Default values for orchestrator configuration"""
    MAX_ITERATIONS = 100
    MAX_AGENT_TURNS = 40
    SESSION_TIMEOUT = 300  # seconds
    SUBPROCESS_TIMEOUT = 300  # seconds, configurable via config.yaml
    CLEANUP_KEEP_SESSIONS = 5
    QUICK_QUIT_SECONDS = 10.0  # Consider it a quick quit if process exits in under 10 seconds

    

class SupervisorDefaults:
    """Default values for supervisor configuration"""
    MAX_TURNS = 40
    ANALYSIS_TIMEOUT = 600  # seconds (10 minutes)
    EXECUTION_TIMEOUT = 600  # seconds (10 minutes)
    MAX_CONSECUTIVE_ERRORS = 3
    STATUS_CHECK_INTERVAL = 30  # seconds
    MAX_OUTPUT_LINES = 10000  # Limit output lines kept in memory
    
    # Streaming output settings
    STREAM_BUFFER_SIZE = 1024 * 1024  # 1MB
    OUTPUT_UPDATE_INTERVAL = 0.1  # seconds
    
    # File size limits
    MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_PROMPT_SIZE = 100 * 1024  # 100KB


class ZenIntegrationDefaults:
    """Default values for Zen integration"""
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    REQUEST_TIMEOUT = 300  # seconds
    
    # Model-specific defaults
    DEFAULT_THINKING_MODE = "high"
    DEFAULT_TEMPERATURE = 0.7
    
    # Error thresholds
    HELP_NEEDED_THRESHOLD = 3  # consecutive errors before requesting help


class AgentPromptDefaults:
    """Default values for agent prompts"""
    DEFAULT_MAX_TURNS = 40
    SAFETY_LIMIT_MESSAGE = "You have up to {max_turns} turns as a safety limit (not a target)"
    
    # Standard messages
    COMPLETION_SIGNAL = "ALL TASKS COMPLETE"
    HELP_SIGNAL = "HELP NEEDED"
    
    # Prompt sections
    SUPERVISOR_GUIDANCE_HEADER = "=== SUPERVISOR GUIDANCE ==="
    TASK_GUIDELINES_HEADER = "=== TASK EXECUTION GUIDELINES ==="
    TODO_LIST_HEADER = "=== YOUR TODOS ==="


class FilePatterns:
    """File naming patterns"""
    DECISION_FILE = "decision_{session_id}.json"
    AGENT_RESULT_FILE = "agent_result_{session_id}.json"
    PROMPT_FILE = "prompt_{session_id}.txt"
    OUTPUT_FILE = "output_{session_id}.log"
    ERROR_FILE = "error_{session_id}.log"
    SUPERVISOR_LOG_FILE = "supervisor_{timestamp}.log"