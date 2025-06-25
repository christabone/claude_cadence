"""
Constants for Claude Cadence Orchestrator and Supervisor

This module contains all configuration constants and magic numbers
used throughout the Claude Cadence system.
"""

from pathlib import Path


class OrchestratorDefaults:
    """Default values for orchestrator configuration"""
    # These are now in config.yaml:
    # MAX_ITERATIONS → orchestration.max_iterations
    # MAX_AGENT_TURNS → execution.max_agent_turns
    # SUBPROCESS_TIMEOUT → execution.subprocess_timeout
    # QUICK_QUIT_SECONDS → orchestration.quick_quit_seconds
    SESSION_TIMEOUT = 300  # seconds (not in config.yaml)
    CLEANUP_KEEP_SESSIONS = 5  # (not in config.yaml)



class SupervisorDefaults:
    """Default values for supervisor configuration"""
    MAX_TURNS = 40  # Default supervisor turns (not in config.yaml currently)
    ANALYSIS_TIMEOUT = 600  # seconds (10 minutes) - not in config.yaml
    # EXECUTION_TIMEOUT → execution.timeout in config.yaml
    MAX_CONSECUTIVE_ERRORS = 3  # not in config.yaml
    STATUS_CHECK_INTERVAL = 30  # seconds - not in config.yaml
    MAX_OUTPUT_LINES = 10000  # Limit output lines kept in memory - not in config.yaml

    # Streaming output settings
    STREAM_BUFFER_SIZE = 1024 * 1024  # 1MB - not in config.yaml
    OUTPUT_UPDATE_INTERVAL = 0.1  # seconds - not in config.yaml

    # File size limits
    MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB - not in config.yaml
    MAX_PROMPT_SIZE = 100 * 1024  # 100KB - not in config.yaml


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
    # DEFAULT_MAX_TURNS → execution.max_agent_turns in config.yaml
    SAFETY_LIMIT_MESSAGE = "You have up to {max_turns} turns as a safety limit (not a target)"

    # Standard messages (these match config.yaml task_detection.completion_phrase and help_needed_phrase)
    COMPLETION_SIGNAL = "ALL TASKS COMPLETE"
    HELP_SIGNAL = "HELP NEEDED"

    # Prompt sections - not in config.yaml
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
