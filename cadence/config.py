"""
Configuration management for Claude Cadence
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

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
    max_turns: int = 40
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
class CadenceConfig:
    """Complete Cadence configuration"""
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
                
            # Direct dictionary configs
            for key in ['session', 'integrations', 'prompts', 'zen_integration', 'processing', 'development']:
                if key in data:
                    setattr(config, key, data[key])
                    
            return config
            
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            print("Using default configuration")
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
        
        
    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            save_path = Path.cwd() / "cadence.yaml"
            
        # Convert to dictionary
        data = {
            "execution": self.config.execution.__dict__,
            "agent": self.config.agent.__dict__,
            "supervisor": self.config.supervisor.__dict__,
            "task_detection": self.config.task_detection.__dict__,
            "session": self.config.session,
            "integrations": self.config.integrations,
            "prompts": self.config.prompts,
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