"""
Task-driven supervisor implementation

This supervisor works with Task Master to manage agent execution.
The supervisor gets tasks from Task Master and gives TODOs to the agent.
Execution continues until all tasks are complete or max turns is reached.
"""

import subprocess
import json
import time
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
from .prompts import TodoPromptManager
from .task_manager import TaskManager, Task
from .config import ConfigLoader, CadenceConfig, SUPERVISOR_LOG_DIR, SCRATCHPAD_DIR
from .zen_integration import ZenIntegration
from .constants import SupervisorDefaults, AgentPromptDefaults
from .prompt_utils import PromptBuilder
from .utils import generate_session_id


@dataclass
class ExecutionResult:
    """Results from an agent execution"""
    success: bool
    turns_used: int  # Always 0 - turn counting not possible with stream-json
    output_lines: List[str]
    errors: List[str]
    metadata: Dict[str, Any]
    task_complete: bool
    # New fields for completion detection
    completed_normally: bool = False  # Agent said AgentPromptDefaults.COMPLETION_SIGNAL
    stopped_unexpectedly: bool = False  # Stopped without completion signal
    requested_help: bool = False  # Agent said AgentPromptDefaults.HELP_SIGNAL



class TaskSupervisor:
    """
    Manages task-driven supervised agent execution
    
    Philosophy: 
    - Supervisor uses Task Master to get and track tasks
    - Agent receives TODOs from supervisor and works until done
    - Max turns is a safety limit, not a target
    """
    
    def __init__(self, 
                 config: Optional[CadenceConfig] = None,
                 max_turns: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 model: Optional[str] = None,
                 verbose: Optional[bool] = None,
                 allowed_tools: Optional[List[str]] = None,
                 timeout: Optional[int] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the task supervisor
        
        Args:
            config: CadenceConfig object (if provided, overrides other params)
            max_turns: Maximum turns before forcing completion
            output_dir: Directory for output logs
            model: Claude model to use for execution
            verbose: Enable verbose logging
            allowed_tools: List of allowed tools
            timeout: Timeout in seconds for execution
            config_path: Path to config file (used if config not provided)
        """
        # Load config if not provided
        if config is None:
            loader = ConfigLoader(config_path)
            config = loader.config
            
        # Use config values, with parameter overrides
        self.config = config
        self.max_turns = max_turns or config.execution.max_turns
        self.output_dir = Path(output_dir or config.execution.log_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model = model or config.agent.model
        self.verbose = verbose if verbose is not None else config.supervisor.verbose
        self.allowed_tools = allowed_tools or config.agent.tools
        self.timeout = timeout or config.execution.timeout
        self.execution_history = []
        
        # Initialize zen integration if enabled
        self.zen = ZenIntegration(
            config.supervisor.zen_integration, 
            verbose=self.verbose
        )
        
        # Initialize supervisor log directory
        self.supervisor_log_dir = Path(SUPERVISOR_LOG_DIR)
        self.supervisor_log_dir.mkdir(exist_ok=True, parents=True)
        self.current_log_path = None
        
        # Initialize Python logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
            """Set up Python logger for detailed supervisor logging"""
            logger = logging.getLogger(f"cadence.supervisor.{id(self)}")
            logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
            
            # Remove any existing handlers
            logger.handlers.clear()
            
            # Create logs directory for Python logs
            python_logs_dir = self.output_dir / "supervisor_logs"
            python_logs_dir.mkdir(exist_ok=True, parents=True)
            
            # File handler for detailed logs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = python_logs_dir / f"supervisor_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler for important messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
            
            # Create formatter
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
            
            file_handler.setFormatter(detailed_formatter)
            console_handler.setFormatter(simple_formatter)
            
            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            logger.info(f"Supervisor logger initialized - Log file: {log_file}")
            return logger
            
    def _check_scratchpad_status(self, session_id: str) -> Dict[str, Any]:
        """Check the agent's scratchpad for status updates"""
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        
        try:
            with open(scratchpad_path, 'r') as f:
                content = f.read()
                
            # Check for key indicators
            has_help_needed = AgentPromptDefaults.HELP_SIGNAL.upper() in content.upper() and "STATUS: STUCK" in content.upper()
            has_all_complete = AgentPromptDefaults.COMPLETION_SIGNAL.upper() in content.upper()
            
            # Extract last few lines for context
            lines = content.strip().split('\n')
            last_lines = '\n'.join(lines[-10:]) if len(lines) > 10 else content
            
            status = {
                "exists": True,
                "has_help_needed": has_help_needed,
                "has_all_complete": has_all_complete,
                "last_lines": last_lines,
                "total_lines": len(lines)
            }
            
            self.logger.debug(f"Scratchpad status: help_needed={has_help_needed}, all_complete={has_all_complete}")
            
            return status
            
        except FileNotFoundError:
            self.logger.warning(f"Scratchpad not found: {scratchpad_path}")
            return {"exists": False}
        except Exception as e:
            self.logger.error(f"Error reading scratchpad: {e}")
            return {"exists": True, "error": str(e)}
        
    def _init_supervisor_log(self, session_id: str) -> Path:
        """Initialize supervisor markdown log for this session"""
        log_path = self.supervisor_log_dir / f"session_{session_id}.md"
        self.current_log_path = log_path
        
        with open(log_path, 'w') as f:
            f.write(f"""# Supervisor Log
Session ID: {session_id}
Started: {datetime.now().isoformat()}
Max Turns: {self.max_turns}
Model: {self.model}

## Configuration
- Zen Integration: {'Enabled' if self.config.supervisor.zen_integration.enabled else 'Disabled'}
- Verbose: {self.verbose}
- Timeout: {self.timeout}s

## Execution Timeline
""")
        
        return log_path
        
    def _log_supervisor(self, message: str, level: str = "INFO"):
        """Add entry to supervisor log (both Python and markdown)"""
        # Python logging
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        # Markdown logging
        if not self.current_log_path:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.current_log_path, 'a') as f:
            f.write(f"\n### [{timestamp}] {level}: {message}\n")
            
    def _log_task_analysis(self, todos: List[str], task_list: Optional[List[Dict]] = None):
        """Log task analysis details"""
        # Python logging
        self.logger.info(f"Task analysis: {len(todos)} TODOs to assign")
        if task_list:
            incomplete_tasks = [t for t in task_list if t.get('status') != 'done']
            self.logger.info(f"Task Master: {len(task_list)} total tasks, {len(incomplete_tasks)} incomplete")
        
        # Log TODO items
        for i, todo in enumerate(todos, 1):
            self.logger.debug(f"TODO {i}: {todo}")
            
        # Markdown logging
        if not self.current_log_path:
            return
            
        with open(self.current_log_path, 'a') as f:
            f.write("\n## Task Analysis\n")
            f.write(f"Total TODOs: {len(todos)}\n\n")
            
            if task_list:
                incomplete_tasks = [t for t in task_list if t.get('status') != 'done']
                f.write(f"Task Master Tasks: {len(task_list)} total, {len(incomplete_tasks)} incomplete\n\n")
                
            f.write("### TODOs Assigned:\n")
            for i, todo in enumerate(todos, 1):
                f.write(f"{i}. {todo}\n")
                
    def _log_execution_start(self, cmd: List[str], continuation: bool = False):
        """Log execution start details"""
        # Python logging
        exec_type = 'Continuation' if continuation else 'Initial'
        self.logger.info(f"{exec_type} execution starting")
        self.logger.debug(f"Command: {' '.join(cmd)}")
        
        # Markdown logging
        if not self.current_log_path:
            return
            
        with open(self.current_log_path, 'a') as f:
            f.write(f"\n## {exec_type} Execution\n")
            f.write(f"Command: `{' '.join(cmd)}`\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            
    def _log_execution_result(self, result: ExecutionResult):
        """Log execution result details"""
        # Python logging
        self.logger.info(f"Execution result: Success={result.success}, Turns={result.turns_used}/{self.max_turns}, Complete={result.task_complete}")
        self.logger.info(f"Execution time: {result.metadata.get('execution_time', 0):.2f}s")
        
        if result.errors:
            self.logger.warning(f"Execution had {len(result.errors)} errors")
            for error in result.errors[:5]:
                self.logger.error(f"Error: {error}")
            if len(result.errors) > 5:
                self.logger.error(f"... and {len(result.errors) - 5} more errors")
                
        # Markdown logging
        if not self.current_log_path:
            return
            
        with open(self.current_log_path, 'a') as f:
            f.write(f"\n### Execution Result\n")
            f.write(f"- Success: {result.success}\n")
            f.write(f"- Turns Used: {result.turns_used}/{self.max_turns}\n")
            f.write(f"- Task Complete: {result.task_complete}\n")
            f.write(f"- Execution Time: {result.metadata.get('execution_time', 0):.2f}s\n")
            
            if result.errors:
                f.write(f"\n#### Errors ({len(result.errors)}):\n")
                for error in result.errors[:5]:  # First 5 errors
                    f.write(f"- {error}\n")
                if len(result.errors) > 5:
                    f.write(f"- ... and {len(result.errors) - 5} more errors\n")
                    
    def _log_zen_recommendation(self, tool: str, reason: str):
        """Log zen assistance recommendation"""
        # Python logging
        self.logger.info(f"Zen assistance recommended: {tool} - {reason}")
        
        # Markdown logging
        if not self.current_log_path:
            return
            
        with open(self.current_log_path, 'a') as f:
            f.write(f"\n### 🔮 Zen Assistance Recommended\n")
            f.write(f"- Tool: `{tool}`\n")
            f.write(f"- Reason: {reason}\n")
            f.write(f"- Time: {datetime.now().isoformat()}\n")
            
    def _log_supervisor_analysis(self, analysis: str):
        """Log supervisor analysis results"""
        # Python logging
        self.logger.info("Supervisor analysis completed")
        # Log first few lines of analysis
        analysis_lines = analysis.strip().split('\n')
        for line in analysis_lines[:5]:
            self.logger.debug(f"Analysis: {line}")
        if len(analysis_lines) > 5:
            self.logger.debug(f"... and {len(analysis_lines) - 5} more lines")
            
        # Markdown logging
        if not self.current_log_path:
            return
            
        with open(self.current_log_path, 'a') as f:
            f.write(f"\n## Supervisor Analysis\n")
            f.write(f"{analysis}\n")
            
    def _log_final_summary(self, all_results: List[ExecutionResult]):
            """Log final execution summary"""
            total_turns = sum(r.turns_used for r in all_results)
            total_time = sum(r.metadata.get('execution_time', 0) for r in all_results)
            any_complete = any(r.task_complete for r in all_results)
            
            # Python logging
            self.logger.info("="*60)
            self.logger.info("FINAL EXECUTION SUMMARY")
            self.logger.info(f"Total executions: {len(all_results)}")
            self.logger.info(f"Total turns used: {total_turns}")
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Final status: {'✅ Complete' if any_complete else '❌ Incomplete'}")
            self.logger.info("="*60)
            
            # Markdown logging
            if not self.current_log_path:
                return
                
            with open(self.current_log_path, 'a') as f:
                f.write("\n## Final Summary\n")
                f.write(f"- Total Executions: {len(all_results)}\n")
                f.write(f"- Total Turns Used: {total_turns}\n")
                f.write(f"- Total Time: {total_time:.2f}s\n")
                f.write(f"- Final Status: {'✅ Complete' if any_complete else '❌ Incomplete'}\n")
                f.write(f"- Completed: {datetime.now().isoformat()}\n")
