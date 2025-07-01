"""
Configuration management for Claude Cadence
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, get_args
from dataclasses import dataclass, field, asdict, is_dataclass

# Constants for system-wide strings (defaults, can be overridden by config)
COMPLETION_PHRASE = "ALL TASKS COMPLETE"
HELP_NEEDED_PHRASE = "HELP NEEDED"
STUCK_STATUS_PHRASE = "Status: STUCK"
SCRATCHPAD_DIR = ".cadence/scratchpad"
SUPERVISOR_LOG_DIR = ".cadence/supervisor"

# Timeout constants (in milliseconds)
DEFAULT_AGENT_TIMEOUT_MS = 600000  # 10 minutes
DEFAULT_FIX_TIMEOUT_MS = 300000     # 5 minutes
DEFAULT_VERIFICATION_TIMEOUT_MS = 60000  # 1 minute
MAX_TIMEOUT_MS = 3600000           # 1 hour
MIN_TIMEOUT_MS = 1000              # 1 second

# Processing constants (defaults, can be overridden by config)
ZEN_OUTPUT_LINES_LIMIT = 200
SCRATCHPAD_CHECK_LINES = 10
CUTOFF_INDICATOR_THRESHOLD = 3
LAST_LINES_TO_CHECK = 50
AUTO_DEBUG_ERROR_THRESHOLD = 3
THREAD_JOIN_TIMEOUT = 5
STATUS_CHECK_INTERVAL = 30
MAX_OUTPUT_TRUNCATE_LENGTH = 3000
SECONDS_PER_TURN_ESTIMATE = 30

# Session constants (defaults, can be overridden by config)
SESSION_ID_FORMAT = "%Y%m%d_%H%M%S"
SESSION_FILE_PREFIX = "session_"


@dataclass
class ExecutionConfig:
    """Execution-related configuration"""
    max_supervisor_turns: int = 80  # For supervisor direct execution mode
    max_agent_turns: int = 120  # For agent execution (orchestrated mode)
    timeout: int = 600
    save_logs: bool = True
    log_dir: str = ".cadence/logs"
    log_level: str = "DEBUG"  # Log level for file logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    clean_logs_on_startup: bool = True  # Clean up all logs on startup
    max_log_size_mb: int = 20  # Maximum total size of log directories before cleanup (in MB)
    subprocess_timeout: int = 300  # Timeout for subprocess calls in seconds
    max_scratchpad_retries: int = 5  # Maximum retries for scratchpad creation



@dataclass
class AgentConfig:
    """Agent execution configuration"""
    # Default settings for all agents
    defaults: Dict[str, Any] = field(default_factory=lambda: {
        "model": "claude-3-5-sonnet-20241022",
        "tools": ["bash", "read", "write", "edit", "grep", "glob", "search", "mcp"],
        "extra_flags": ["--dangerously-skip-permissions"],
        "retry_count": 1,
        "use_continue": False,
        "timeout_seconds": 300,
        "temperature": 0.2
    })


@dataclass
class ZenIntegrationConfig:
    """Zen MCP integration configuration"""
    enabled: bool = True
    stuck_detection: bool = True
    auto_debug_threshold: int = 3  # Number of errors before calling zen
    cutoff_detection: bool = True  # Detect if task was cut off at turn limit
    code_review_frequency: str = "task"  # "none", "task", "project"

    # Selective task validation patterns
    validate_on_complete: List[str] = field(default_factory=lambda: [
        "*security*", "*database*", "*critical*", "*auth*", "*payment*"
    ])

    # Model configurations for different scenarios
    models: Dict[str, List[str]] = field(default_factory=lambda: {
        "debug": ["o3", "pro"],           # For stuck/errors
        "review": ["pro"],                # For code review
        "consensus": ["o3", "pro", "flash"],  # For decisions
        "precommit": ["pro"],             # For validation
        "analyze": ["pro"]                # For retrospectives
    })

    # Thinking modes for each model type
    thinking_modes: Dict[str, str] = field(default_factory=lambda: {
        "debug": "high",
        "review": "high",
        "consensus": "medium",
        "precommit": "high",
        "analyze": "medium"
    })

    # New fields moved from ZenIntegrationDefaults
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    request_timeout: int = 300  # seconds
    default_thinking_mode: str = "high"
    default_temperature: float = 0.7
    help_needed_threshold: int = 3

    # Fields from config.yaml
    output_lines_limit: int = 200
    scratchpad_check_lines: int = 10
    cutoff_indicator_threshold: int = 3
    last_lines_to_check: int = 50
    auto_debug_error_threshold: int = 3
    primary_review_model: str = "gemini-2.5-pro"
    secondary_review_model: str = "o3"
    debug_model: str = "o3"
    analyze_model: str = "o3"
    consensus_models: List[str] = field(default_factory=lambda: ["o3", "gemini-2.5-pro"])


@dataclass
class SupervisorConfig:
    """Supervisor analysis configuration"""
    model: str = "heuristic"
    tools: List[str] = field(default_factory=lambda: [
        "bash", "read", "write", "edit", "grep", "glob", "search", "WebFetch"
    ])
    intervention_threshold: float = 0.7
    verbose: bool = True
    analysis: Dict[str, bool] = field(default_factory=lambda: {
        "check_errors": True,
        "check_task_completion": True,
        "check_progress_indicators": True,
        "check_blockers": True
    })
    zen_integration: ZenIntegrationConfig = field(default_factory=ZenIntegrationConfig)
    use_continue: bool = False  # Whether to use --continue flag for supervisor sessions
    # New fields moved from SupervisorDefaults
    max_turns: int = 40
    analysis_timeout: int = 600  # seconds
    max_consecutive_errors: int = 3
    max_output_lines: int = 10000
    stream_buffer_size: int = 1048576  # 1MB
    output_update_interval: float = 0.1  # seconds
    max_log_size: int = 52428800  # 50MB
    max_prompt_size: int = 102400  # 100KB  # Whether to use --continue flag for supervisor sessions


@dataclass
class OrchestrationConfig:
    """Orchestration configuration"""
    supervisor_dir: str = ".cadence/supervisor"
    agent_dir: str = ".cadence/agent"
    decision_format: str = "json"
    enable_zen_assistance: bool = True
    max_iterations: int = 100
    mode: str = "orchestrated"
    quick_quit_seconds: float = 10.0
    session_timeout: int = 300  # seconds - moved from OrchestratorDefaults
    cleanup_keep_sessions: int = 5  # moved from OrchestratorDefaults
    workflow_max_history_size: int = 1000  # Maximum state transitions to keep in workflow history
    code_review_fix_retries: int = 5  # Maximum retries when supervisor fails to handle critical code review issues


@dataclass
class TaskDetectionConfig:
    """Task completion detection configuration"""
    methods: List[str] = field(default_factory=lambda: [
        "structured_json", "all_tasks_phrase", "keyword_analysis"
    ])
    completion_keywords: List[str] = field(default_factory=lambda: [
        "completed", "done", "finished", "implemented",
        "fixed", "resolved", "added", "created"
    ])
    completion_phrase: str = COMPLETION_PHRASE
    help_needed_phrase: str = HELP_NEEDED_PHRASE
    stuck_status_phrase: str = STUCK_STATUS_PHRASE

@dataclass
class FilePatternConfig:
    """File naming patterns configuration"""
    decision_file: str = "decision_{session_id}.json"
    agent_result_file: str = "agent_result_{session_id}.json"
    prompt_file: str = "prompt_{session_id}.txt"
    output_file: str = "output_{session_id}.log"
    error_file: str = "error_{session_id}.log"
    supervisor_log_file: str = "supervisor_{timestamp}.log"


@dataclass
class SessionConfig:
    """Session management configuration"""
    save_summaries: bool = True
    summary_dir: str = ".cadence/sessions"
    include_checkpoint_details: bool = True
    id_format: str = SESSION_ID_FORMAT
    file_prefix: str = "session_"


@dataclass
class PromptsConfig:
    """Prompt customization configuration"""
    custom_prompts_file: str = ""
    include_timestamp: bool = True
    safety_limit_message: str = "You have up to {max_turns} turns as a safety limit (not a target)"
    supervisor_guidance_header: str = "=== SUPERVISOR GUIDANCE ==="
    task_guidelines_header: str = "=== TASK EXECUTION GUIDELINES ==="
    todo_list_header: str = "=== YOUR TODOS ==="




@dataclass
class CadenceConfig:
    """Complete Cadence configuration"""
    project: Dict[str, Any] = field(default_factory=lambda: {
        "root_directory": os.getcwd(),  # Default to current directory
        "taskmaster_file": None  # Will use default if None
    })
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    task_detection: TaskDetectionConfig = field(default_factory=TaskDetectionConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    integrations: Dict[str, Any] = field(default_factory=lambda: {
        "taskmaster": {
            "enabled": True,
            "default_task_file": ".taskmaster/tasks/tasks.json",
            "auto_update_status": True
        },
        "mcp": {
            "auto_discover": True,
            "servers": ["taskmaster-ai", "github", "filesystem"]
        }
    })
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    file_patterns: FilePatternConfig = field(default_factory=FilePatternConfig)
    zen_processing_config: Dict[str, Any] = field(default_factory=lambda: {
        "output_lines_limit": ZEN_OUTPUT_LINES_LIMIT,
        "scratchpad_check_lines": SCRATCHPAD_CHECK_LINES,
        "cutoff_indicator_threshold": CUTOFF_INDICATOR_THRESHOLD,
        "last_lines_to_check": LAST_LINES_TO_CHECK,
        "auto_debug_error_threshold": AUTO_DEBUG_ERROR_THRESHOLD
    })
    processing: Dict[str, Any] = field(default_factory=lambda: {
        "thread_join_timeout": THREAD_JOIN_TIMEOUT,
        "status_check_interval": STATUS_CHECK_INTERVAL,
        "max_output_truncate_length": MAX_OUTPUT_TRUNCATE_LENGTH,
        "seconds_per_turn_estimate": SECONDS_PER_TURN_ESTIMATE
    })
    development: Dict[str, Any] = field(default_factory=lambda: {
        "debug": False,
        "save_raw_output": False,
        "pretty_json": True
    })
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    scratchpad_retry: Dict[str, Any] = field(default_factory=dict)
    retry_behavior: Dict[str, Any] = field(default_factory=lambda: {
        "max_json_retries": 3,
        "backoff_strategy": "linear",
        "base_delay": 2,
        "max_delay": 60,
        "use_continue_on_failure": True,
        "subprocess_timeout": 300,
        "max_file_retries": 3,
        "verbose_logging": False
    })


class ConfigLoader:
    """Loads and manages Cadence configuration"""

    DEFAULT_CONFIG_PATHS = [
        Path.cwd() / "cadence.yaml",
        Path.cwd() / "config.yaml",
        Path.cwd() / ".cadence" / "config.yaml",
        Path.home() / ".config" / "claude-cadence" / "config.yaml",
        Path(__file__).parent.parent / "config.yaml"
    ]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_path: Optional path to config file. If not provided,
                        searches default locations.
        """
        self.config_path = self._find_config_file(config_path)
        self.config = self._load_config()

    def _find_config_file(self, config_path: Optional[str]) -> Optional[Path]:
        """Find configuration file"""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Check environment variable
        env_path = os.environ.get("CADENCE_CONFIG")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path

        # Check default locations
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path

        return None

    def _deep_merge_dict(self, base: dict, override: dict) -> dict:
        """Deep merge override dictionary into base dictionary"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        return result

    def _load_config(self) -> CadenceConfig:
        """
        Load configuration from YAML file if it exists, otherwise use defaults.

        Returns:
            CadenceConfig with loaded or default values
        """
        try:
            if not self.config_path or not self.config_path.exists():
                logger = logging.getLogger(__name__)
                logger.debug("No config file found, using defaults")
                return CadenceConfig()

            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Start with default config
            config = CadenceConfig()

            # Update simple fields (dict configs)
            self._update_dict_configs(config, data)

            # Handle dataclass configs with proper instantiation
            self._update_dataclass_configs(config, data)

            return config

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            logger.warning("Using default configuration")
            return CadenceConfig()

    def _update_dict_configs(self, config: CadenceConfig, data: dict) -> None:
        """Update dictionary configurations with deep merge"""
        dict_keys = ['project', 'integrations', 'zen_processing_config', 'processing', 'development', 'retry_behavior', 'scratchpad_retry']
        for key in dict_keys:
            if key in data:
                # For dictionaries, deep merge with defaults instead of replacing
                default_value = getattr(config, key)
                if isinstance(default_value, dict):
                    merged = self._deep_merge_dict(default_value, data[key])
                    setattr(config, key, merged)
                else:
                    setattr(config, key, data[key])

    def _update_dataclass_configs(self, config: CadenceConfig, data: dict) -> None:
        """Update dataclass configurations with validation"""
        # Define dataclass config mappings
        dataclass_configs = {
            'execution': (ExecutionConfig, 'execution'),
            'agent': (AgentConfig, 'agent'),
            'supervisor': (SupervisorConfig, 'supervisor'),
            'task_detection': (TaskDetectionConfig, 'task_detection'),
            'session': (SessionConfig, 'session'),
            'prompts': (PromptsConfig, 'prompts'),
            'file_patterns': (FilePatternConfig, 'file_patterns'),
            'orchestration': (OrchestrationConfig, 'orchestration'),
        }

        # Update each dataclass config
        for key, (config_class, attr_name) in dataclass_configs.items():
            if key in data:
                self._update_single_dataclass(config, attr_name, config_class, data[key])


    def _update_single_dataclass(self, config: CadenceConfig, attr_name: str,
                                  config_class: type, data: dict) -> None:
        """Update a single dataclass configuration with error handling"""
        try:
            # Handle nested dataclasses (e.g., SupervisorConfig with ZenIntegrationConfig)
            processed_data = data.copy()

            # Special handling for SupervisorConfig with nested zen_integration
            if config_class == SupervisorConfig and 'zen_integration' in processed_data:
                zen_data = processed_data['zen_integration']
                if isinstance(zen_data, dict):
                    processed_data['zen_integration'] = ZenIntegrationConfig(**zen_data)

            setattr(config, attr_name, config_class(**processed_data))
        except (TypeError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid {attr_name} config: {e}. Using defaults.")
            setattr(config, attr_name, config_class())


    def get_tool_command_flags(self) -> List[str]:
        """Get tool-related command line flags for claude CLI"""
        flags = []

        # Add tool flags
        for tool in self.config.agent.tools:
            flags.extend(["--tool", tool])

        # Add extra flags
        flags.extend(self.config.agent.extra_flags)

        return flags


    def _dataclass_to_dict(self, obj) -> dict:
        """Convert a dataclass to dictionary using Python's built-in asdict()"""
        if is_dataclass(obj):
            return asdict(obj)
        return obj

    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            save_path = Path.cwd() / "cadence.yaml"

        # Convert to dictionary
        data = {
            "execution": self._dataclass_to_dict(self.config.execution),
            "agent": self._dataclass_to_dict(self.config.agent),
            "supervisor": self._dataclass_to_dict(self.config.supervisor),
            "task_detection": self._dataclass_to_dict(self.config.task_detection),
            "orchestration": self._dataclass_to_dict(self.config.orchestration),
            "session": self.config.session,
            "integrations": self.config.integrations,
            "prompts": self.config.prompts,
            "zen_processing_config": self.config.zen_processing_config,
            "processing": self.config.processing,
            "development": self.config.development,
            "retry_behavior": self.config.retry_behavior
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def override_from_args(self, **kwargs):
        """Override config values from command line arguments with validation"""
        # Define allowed override keys and their types
        allowed_overrides = {
            # Execution settings
            'execution.max_agent_turns': int,
            'execution.max_supervisor_turns': int,
            'execution.turn_timeout': int,
            'execution.supervisor_timeout': int,
            'execution.cleanup_policy': str,
            'execution.session_retention_count': int,
            'execution.zen_code_review_frequency': str,
            'execution.max_scratchpad_retries': int,

            # Agent settings
            'agent.process_name': str,
            'agent.wait_timeout': int,

            # Task detection
            'task_detection.max_retries': int,
            'task_detection.retry_delay': float,

            # Orchestration settings
            'orchestration.orchestrator_dir': str,
            'orchestration.supervisor_dir': str,
            'orchestration.agent_dir': str,

        }

        # Handle nested overrides with validation
        for key, value in kwargs.items():
            if value is None:
                continue

            # Check if key is allowed
            if key not in allowed_overrides and not any(key.startswith(allowed) for allowed in allowed_overrides):
                logger.warning(f"Ignoring unknown config override: {key}")
                continue

            if '.' in key:
                # Nested key like "execution.max_agent_turns"
                if key in allowed_overrides:
                    # Validate type
                    expected_type = allowed_overrides[key]
                    try:
                        if expected_type == bool:
                            # Handle boolean conversion
                            if isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes', 'on')
                            else:
                                value = bool(value)
                        else:
                            # Convert to expected type
                            value = expected_type(value)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid value for {key}: {value} (expected {expected_type.__name__})")
                        continue

                # Apply the override
                parts = key.split('.')
                obj = self.config
                try:
                    for part in parts[:-1]:
                        if not hasattr(obj, part):
                            logger.error(f"Invalid config path: {key} (missing {part})")
                            break
                        obj = getattr(obj, part)
                    else:
                        # Only set if we successfully traversed the path
                        if hasattr(obj, parts[-1]):
                            setattr(obj, parts[-1], value)
                            logger.info(f"Config override: {key} = {value}")
                        else:
                            logger.error(f"Invalid config attribute: {key}")
                except AttributeError as e:
                    logger.error(f"Error setting config override {key}: {e}")
            else:
                # Top-level key - not allowed for safety
                logger.warning(f"Top-level config overrides not allowed: {key}")
