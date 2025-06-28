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
    subprocess_timeout: int = 300  # Timeout for subprocess calls in seconds



@dataclass
class AgentConfig:
    """Agent execution configuration"""
    model: str = "claude-3-5-sonnet-20241022"
    tools: List[str] = field(default_factory=lambda: [
        "bash", "read", "write", "edit", "grep", "glob", "search",
        "todo_read", "todo_write", "mcp"
    ])
    extra_flags: List[str] = field(default_factory=lambda: ["--dangerously-skip-permissions"])
    use_continue: bool = False  # Whether to use --continue flag for agent sessions


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
class CircularDependencyConfig:
    """Configuration for circular dependency detection"""
    max_file_modifications: int = 3
    min_attempts_before_check: int = 5

    def __post_init__(self):
        """Validate configuration values"""
        if self.max_file_modifications < 1:
            raise ValueError("max_file_modifications must be positive")
        if self.min_attempts_before_check < 1:
            raise ValueError("min_attempts_before_check must be positive")


@dataclass
class FixAgentDispatcherConfig:
    """Configuration for the Fix Agent Dispatcher"""
    max_attempts: int = 3
    timeout_ms: int = 300000
    enable_auto_fix: bool = True
    severity_threshold: Literal["low", "medium", "high", "critical"] = "high"
    enable_verification: bool = True
    verification_timeout_ms: int = 60000
    circular_dependency: CircularDependencyConfig = field(default_factory=CircularDependencyConfig)
    validation: Dict[str, int] = field(default_factory=lambda: {
        "min_timeout_ms": 100,
        "max_timeout_ms": 3600000
    })

    def __post_init__(self):
        """Validate configuration values"""
        # Validate severity_threshold at runtime
        valid_severities = get_args(self.__annotations__['severity_threshold'])
        if self.severity_threshold not in valid_severities:
            raise ValueError(f"severity_threshold must be one of {valid_severities}, but got '{self.severity_threshold}'")

        # Validate max_attempts with upper bound to prevent DoS
        if not 1 <= self.max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")

        min_timeout = self.validation.get("min_timeout_ms", 100)
        max_timeout = self.validation.get("max_timeout_ms", 3600000)

        if self.timeout_ms < min_timeout:
            raise ValueError(f"timeout_ms must be at least {min_timeout}ms")
        if self.timeout_ms > max_timeout:
            raise ValueError(f"timeout_ms must not exceed {max_timeout}ms")
        if self.verification_timeout_ms < min_timeout:
            raise ValueError(f"verification_timeout_ms must be at least {min_timeout}ms")
        if self.verification_timeout_ms > max_timeout:
            raise ValueError(f"verification_timeout_ms must not exceed {max_timeout}ms")


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
    session: Dict[str, Any] = field(default_factory=lambda: {
        "save_summaries": True,
        "summary_dir": ".cadence/sessions",
        "include_checkpoint_details": True,
        "id_format": SESSION_ID_FORMAT,
        "file_prefix": SESSION_FILE_PREFIX
    })
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
    prompts: Dict[str, Any] = field(default_factory=lambda: {
        "custom_prompts_file": "",
        "include_timestamp": True
    })
    zen_integration: Dict[str, Any] = field(default_factory=lambda: {
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
    fix_agent_dispatcher: FixAgentDispatcherConfig = field(default_factory=FixAgentDispatcherConfig)


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
        """Load configuration from file or use defaults"""
        if not self.config_path:
            return CadenceConfig()

        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Parse nested configs
            config = CadenceConfig()

            # Execution config
            if 'execution' in data:
                config.execution = ExecutionConfig(**data['execution'])

            # Agent config
            if 'agent' in data:
                config.agent = AgentConfig(**data['agent'])

            # Supervisor config
            if 'supervisor' in data:
                config.supervisor = SupervisorConfig(**data['supervisor'])

            # Task detection config
            if 'task_detection' in data:
                config.task_detection = TaskDetectionConfig(**data['task_detection'])

            # Orchestration config
            if 'orchestration' in data:
                config.orchestration = OrchestrationConfig(**data['orchestration'])

            # Fix Agent Dispatcher config
            if 'fix_agent_dispatcher' in data:
                try:
                    fad_data = data['fix_agent_dispatcher'].copy()
                    if 'circular_dependency' in fad_data:
                        # Handle nested dataclass
                        try:
                            fad_data['circular_dependency'] = CircularDependencyConfig(**fad_data['circular_dependency'])
                        except (TypeError, ValueError) as e:
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Invalid circular_dependency config: {e}. Using defaults.")
                            fad_data['circular_dependency'] = CircularDependencyConfig()
                    config.fix_agent_dispatcher = FixAgentDispatcherConfig(**fad_data)
                except (TypeError, ValueError) as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Invalid fix_agent_dispatcher config: {e}. Using defaults.")
                    config.fix_agent_dispatcher = FixAgentDispatcherConfig()

            # Direct dictionary configs
            for key in ['session', 'integrations', 'prompts', 'zen_integration', 'processing', 'development']:
                if key in data:
                    # For dictionaries, deep merge with defaults instead of replacing
                    default_value = getattr(config, key)
                    if isinstance(default_value, dict):
                        merged = self._deep_merge_dict(default_value, data[key])
                        setattr(config, key, merged)
                    else:
                        setattr(config, key, data[key])

            return config

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            logger.warning("Using default configuration")
            return CadenceConfig()

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
            "fix_agent_dispatcher": self._dataclass_to_dict(self.config.fix_agent_dispatcher),
            "session": self.config.session,
            "integrations": self.config.integrations,
            "prompts": self.config.prompts,
            "zen_integration": self.config.zen_integration,
            "processing": self.config.processing,
            "development": self.config.development
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def override_from_args(self, **kwargs):
        """Override config values from command line arguments"""
        # Handle nested overrides
        for key, value in kwargs.items():
            if value is None:
                continue

            if '.' in key:
                # Nested key like "checkpoint.turns"
                parts = key.split('.')
                obj = self.config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                # Top-level key
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
