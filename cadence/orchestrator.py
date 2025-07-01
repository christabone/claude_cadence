"""
Orchestrator for Claude Cadence Supervisor-Agent Architecture

This module manages the coordination between supervisor and agent,
ensuring they operate in separate directories and maintain proper state.
"""

import os
import json
import time
import re
import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from .config import ConfigLoader
from .utils import generate_session_id
from .prompts import PromptGenerator, ExecutionContext
from .prompt_loader import PromptLoader
from .log_utils import Colors, setup_file_logging
from .json_stream_monitor import SimpleJSONStreamMonitor
from .retry_utils import run_claude_with_realtime_retry, RetryError

# Import unified agent
from .unified_agent import UnifiedAgent, AgentResult

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SupervisorDecision:
    """Decision made by supervisor analysis"""
    action: str  # "execute", "skip", "complete", "code_review"
    todos: List[str] = None  # For backward compatibility
    task_id: str = ""
    task_title: str = ""
    subtasks: List[Dict] = None  # List of subtask dicts with id, title, description
    project_path: str = ""  # Unified project path
    guidance: str = ""
    session_id: str = ""
    reason: str = ""
    zen_needed: Optional[Dict] = None
    execution_time: float = 0.0  # Time taken to make decision
    quit_too_quickly: bool = False  # True if supervisor quit in < configured seconds
    # Fields for code review action
    review_scope: str = ""  # "task" or "project"
    files_to_review: List[str] = None  # List of files to review
    supervisor_findings: str = ""  # Supervisor's own code review findings
    # Code review validation fields
    code_review_has_critical_or_high_issues: bool = False  # True if code review found CRITICAL or HIGH issues





class SupervisorOrchestrator:
    """Orchestrates between supervisor and agent in separate directories"""

    def __init__(self, project_root: Path, task_file: Optional[Path] = None, config: Optional[Dict] = None):
        self.project_root = Path(project_root).resolve()

        # Load configuration
        config_loader = ConfigLoader()
        self.cadence_config = config_loader.config

        # Default to .taskmaster/tasks/tasks.json if not specified
        if task_file is None:
            task_file = self.project_root / ".taskmaster" / "tasks" / "tasks.json"

        self.task_file = Path(task_file).resolve()
        self.supervisor_dir = self.project_root / ".cadence" / "supervisor"
        self.agent_dir = self.project_root / ".cadence" / "agent"
        self.state_file = self.project_root / ".cadence" / "orchestrator_state.json"
        # Config parameter is deprecated - using cadence_config from YAML instead

        # Load prompts from YAML template
        prompts_file = Path(__file__).parent / "prompts.yml"
        self.prompt_loader = PromptLoader(prompts_file)

        # Create directories
        self.supervisor_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state = self.load_state()

        # Session tracking
        self.current_session_id = None  # Will be created when orchestration starts

        # Log level configuration
        self.log_level_str = self._get_config_value('execution.log_level', 'DEBUG')
        self.log_level = getattr(logging, self.log_level_str.upper(), logging.DEBUG)

        # No dispatch system needed with unified agent approach

    def cleanup_completion_marker(self):
        """Remove any existing completion marker from previous runs"""
        completion_marker = self.project_root / ".cadence" / "project_complete.marker"
        if completion_marker.exists():
            try:
                completion_marker.unlink()
                logger.info("Removed previous completion marker")
            except Exception as e:
                logger.warning(f"Failed to remove completion marker: {e}")


    def _run_async_safely(self, coro):
        """
        Run async coroutine safely from synchronous code without deadlocking.

        Behaviour:
        - No running loop in this thread -> asyncio.run(coro)
        - Running loop in another thread -> run_coroutine_threadsafe()
        - Running loop in this thread -> raise error (anti-pattern)
        """
        try:
            # 1. Attempt to get the loop for the current thread.
            #    This is the modern, preferred way (Python 3.7+).
            #    It raises RuntimeError if no loop is running.
            asyncio.get_running_loop()

            # 2. If the above line succeeds, a loop is active in this thread.
            #    This is the anti-pattern we want to prevent.
            logger.error(
                "Detected a call to _run_async_safely from within an active event loop. "
                "This is an anti-pattern that can cause deadlocks. Please refactor to use 'await'."
            )
            raise RuntimeError(
                "Cannot call blocking async wrapper from a running event loop. "
                "Use 'await' directly instead of calling _run_async_safely."
            )

        except RuntimeError:
            # 3. This block now *only* executes if get_running_loop() failed,
            #    which reliably tells us we are in a synchronous context.
            try:
                # get_event_loop() is more lenient: it gets the current loop
                # or creates one if none exists.
                main_loop = asyncio.get_event_loop()
                if main_loop.is_running():
                    # A loop is running, but in a different thread.
                    # This is a common scenario in threaded applications (e.g., a web server worker).
                    # We must use a thread-safe method to schedule the coroutine.
                    future = asyncio.run_coroutine_threadsafe(coro, main_loop)
                    return future.result()  # Block this sync thread until the future is done.
                else:
                    # The loop exists but is not running, or we just created a new one.
                    # It's safe to use asyncio.run() to manage the coroutine's lifecycle.
                    return asyncio.run(coro)
            except RuntimeError:
                # This is a fallback for rare cases, e.g., on a non-main thread where
                # no event loop policy is set. asyncio.run() is self-contained
                # and will create a new loop, run the task, and close it.
                return asyncio.run(coro)

    def _get_config_value(self, path: str, default: Any = None) -> Any:
        """Safely get nested config value with dot notation"""
        _sentinel = object()  # A unique object to detect missing values
        try:
            obj = self.cadence_config
            for part in path.split('.'):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict):
                    obj = obj.get(part, _sentinel)
                    if obj is _sentinel:
                        return default
                else:
                    return default

            return obj
        except AttributeError:  # KeyError is unlikely with .get()
            return default

    async def run_claude_with_realtime_output(self, cmd: List[str], cwd: str, process_name: str, session_id: str = None) -> tuple[int, List[str]]:
        """Run claude command with real-time output display"""
        # Set up environment
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Add logging environment variables
        log_dir = self.project_root / ".cadence" / "logs"
        # Use passed session_id or fall back to current_session_id
        env["CADENCE_LOG_SESSION"] = session_id or self.current_session_id or "unknown"
        env["CADENCE_LOG_DIR"] = str(log_dir.resolve())
        env["CADENCE_LOG_LEVEL"] = self.log_level_str

        # Choose color based on process name
        if process_name == "SUPERVISOR":
            color = Colors.BOLD_BLUE
        elif process_name == "AGENT":
            color = Colors.BOLD_MAGENTA
        else:
            color = Colors.BOLD_WHITE

        print(f"\n{color}{process_name} working...{Colors.RESET}")
        print(f"{Colors.WHITE}{'-' * 50}{Colors.RESET}")

        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env
        )

        # Store all output lines (no memory bounds for now)
        all_output = []
        line_count = 0
        json_monitor = SimpleJSONStreamMonitor()

        while True:
            try:
                # Read line by line
                line = await process.stdout.readline()
                if not line:  # EOF
                    break

                line_count += 1
                line_str = line.decode('utf-8', errors='replace').strip()

                if not line_str:
                    continue

                all_output.append(line_str)

                # Try to parse as JSON using our monitor
                json_data = json_monitor.process_line(line_str)

                if json_data:
                    # Successfully parsed JSON
                    msg_type = json_data.get('type', 'unknown')

                    if msg_type == 'system':
                        subtype = json_data.get('subtype', '')
                        if subtype == 'init':
                            # Extract and display model info from init message
                            model = json_data.get('model', 'unknown')
                            print(f"{color}[{process_name}]{Colors.RESET} {Colors.BOLD}Model: {model}{Colors.RESET}")

                    elif msg_type == 'assistant':
                        message = json_data.get('message', {})
                        content = message.get('content', [])

                        for item in content:
                            if item.get('type') == 'text':
                                text = item.get('text', '').strip()
                                if text:
                                    print(f"{color}[{process_name}]{Colors.RESET} {text}")
                            elif item.get('type') == 'tool_use':
                                tool_name = item.get('name', 'unknown')
                                tool_input = item.get('input', {})
                                command = tool_input.get('command', '')
                                print(f"{color}[{process_name}]{Colors.RESET} 🛠️  {tool_name}: {command}")

                    elif msg_type == 'user':
                        # Tool results
                        message = json_data.get('message', {})
                        content = message.get('content', [])
                        for item in content:
                            if item.get('type') == 'tool_result':
                                result_content = str(item.get('content', ''))
                                if len(result_content) > 200:
                                    result_content = result_content[:200] + "..."
                                print(f"{color}[{process_name}]{Colors.RESET} 📋 {Colors.CYAN}Result:{Colors.RESET} {result_content}")

                    elif msg_type == 'result':
                        # Final result
                        duration = json_data.get('duration_ms', 0)
                        print(f"{color}[{process_name}]{Colors.RESET} ✅ {Colors.BOLD_GREEN}Completed{Colors.RESET} in {duration}ms")
                else:
                    # Not JSON or incomplete JSON, display as plain text if not buffering
                    if not json_monitor.in_json:
                        print(f"{color}[{process_name}]{Colors.RESET} {line_str}")

            except Exception as e:
                print(f"{Colors.RED}[{process_name}] Error reading output: {e}{Colors.RESET}")
                break

        # Wait for process to complete
        await process.wait()

        print(f"{Colors.WHITE}{'-' * 50}{Colors.RESET}")
        print(f"{color}{process_name} completed with return code {process.returncode}{Colors.RESET}")

        # Save supervisor output to log file
        logger.debug(f"Process name: {process_name}, Output lines: {len(all_output)}, session_id: {session_id}")
        if process_name == "SUPERVISOR" and all_output:
            try:
                log_dir = self.project_root / ".cadence" / "logs"
                # Use the session_id that was passed or fall back
                log_session_id = session_id or self.current_session_id or "unknown"
                supervisor_log_file = log_dir / log_session_id / "supervisor.log"
                supervisor_log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(supervisor_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Supervisor run at {datetime.now().isoformat()}\n")
                    f.write(f"{'='*60}\n")
                    f.write('\n'.join(all_output))
                    f.write('\n')
                logger.info(f"Saved supervisor output to {supervisor_log_file}")
            except Exception as e:
                logger.warning(f"Failed to save supervisor output: {e}")

        return process.returncode, list(all_output)

    def format_stream_json_line(self, line_str: str) -> str:
        """Parse and format a stream-json line for human-readable display"""
        try:
            data = json.loads(line_str)
            msg_type = data.get('type', 'unknown')

            # Handle different message types
            if msg_type == 'system':
                subtype = data.get('subtype', '')
                if subtype == 'init':
                    # Extract model info from init message
                    model = data.get('model', 'unknown')
                    return f"[SYSTEM] Initialized with model: {model}"
                return f"[SYSTEM] {data.get('message', '')}"
            elif msg_type == 'assistant':
                message = data.get('message', {})
                content = message.get('content', '')
                # Handle content that might be a list or string
                if isinstance(content, list):
                    # Extract text from content items
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'tool_use':
                                text_parts.append(f"[Using tool: {item.get('name', 'unknown')}]")
                        else:
                            text_parts.append(str(item))
                    content = ' '.join(text_parts)
                elif isinstance(content, str):
                    # Strip excessive whitespace
                    content = ' '.join(content.split())
                # Don't show empty messages
                if not content.strip():
                    return None
                return f"[CLAUDE] {content}"
            elif msg_type == 'tool_use':
                tool_name = data.get('tool_name', 'unknown')
                return f"[TOOL: {tool_name}] Running..."
            elif msg_type == 'tool_result':
                return f"[RESULT] Tool completed"
            elif msg_type == 'error':
                return f"[ERROR] {data.get('message', '')}"
            else:
                # For other types, just show the type
                return f"[{msg_type.upper()}]"
        except json.JSONDecodeError:
            # If it's not JSON, just return the line as-is
            return line_str

    def load_state(self) -> Dict:
        """Load orchestrator state to track if this is first run"""
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "session_count": 0,
                "last_session_id": None,
                "created_at": datetime.now().isoformat()
            }

    def save_state(self):
        """Save orchestrator state"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return generate_session_id()







    def _run_claude_with_json_retry(self,
                                    build_command_func: Callable[[], List[str]],
                                    parse_output_func: Callable[[List[str]], Any],
                                    working_dir: Path,
                                    process_name: str,
                                    session_id: str) -> Any:
        """
        Run a Claude CLI command with JSON retry logic using unified retry system.

        This helper method centralizes the common pattern of:
        1. Running a subprocess that outputs JSON
        2. Parsing the JSON output
        3. Retrying with modified command on JSON parse failures

        Args:
            build_command_func: Function that builds the command list
            parse_output_func: Function that parses output and returns result
            working_dir: Directory to run the command in
            process_name: Name for logging (e.g., "SUPERVISOR", "AGENT")
            session_id: Current session ID

        Returns:
            Parsed result from parse_output_func

        Raises:
            RuntimeError: If command fails or JSON parsing fails after retries
        """
        max_retries = self._get_config_value("retry_behavior.max_json_retries", 3)

        def realtime_runner(cmd: List[str], cwd: Any, process_name: str) -> Tuple[int, List[str]]:
            """Wrapper for the async realtime output function"""
            start_time = time.time()
            try:
                # Use safe async runner to handle existing event loops
                returncode, all_output = self._run_async_safely(
                    self.run_claude_with_realtime_output(cmd, cwd, process_name, session_id)
                )
                duration = (time.time() - start_time) * 1000
                logger.info(f"{process_name} process completed in {duration:.0f}ms with return code {returncode}")
                return returncode, all_output
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"{process_name} execution error after {duration:.0f}ms: {e}")
                raise

        try:
            return run_claude_with_realtime_retry(
                build_command_func=build_command_func,
                parse_output_func=parse_output_func,
                realtime_runner_func=realtime_runner,
                working_dir=working_dir,
                process_name=process_name,
                max_retries=max_retries,
                session_id=session_id,
                base_delay=self._get_config_value("retry_behavior.base_delay", 2.0)
            )
        except RetryError as e:
            # Convert RetryError to RuntimeError for backward compatibility
            raise RuntimeError(str(e))





    def validate_path(self, path: Path, base_dir: Path) -> Path:
        """Ensure path is within base directory to prevent path traversal attacks"""
        try:
            resolved_path = path.resolve()
            resolved_base = base_dir.resolve()

            # Check if the resolved path starts with the base directory
            if not str(resolved_path).startswith(str(resolved_base)):
                raise ValueError(f"Path '{path}' is outside allowed directory '{base_dir}'")

            return resolved_path
        except (TypeError, ValueError) as e:
            # Catch expected errors from path operations
            raise ValueError(f"Invalid path format '{path}': {e}") from e
        except PermissionError as e:
            raise ValueError(f"Permission denied accessing path '{path}': {e}") from e
        except OSError as e:
            # Catch other OS-level errors
            raise ValueError(f"OS error accessing path '{path}': {e}") from e

    def run_orchestration_loop(self) -> bool:
        """
        Main orchestration loop

        Returns:
            bool: True if all tasks completed successfully
        """
        self.current_session_id = self.generate_session_id()

        # Set up file logging for orchestrator with validation
        log_dir = self.project_root / ".cadence" / "logs"

        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level_str.upper() not in valid_levels:
            logger.warning(f"Invalid log_level '{self.log_level_str}' in config. Using DEBUG. Valid options: {valid_levels}")
            self.log_level_str = 'DEBUG'
            self.log_level = logging.DEBUG

        setup_file_logging(self.current_session_id, "orchestrator", log_dir, level=self.log_level)

        logger.info("="*60)
        logger.info("Starting Claude Cadence Orchestration")
        logger.info(f"Task file: {self.task_file}")
        logger.info(f"Session ID: {self.current_session_id}")
        logger.info("="*60)

        # Display key configuration values
        logger.info("Configuration:")
        logger.info(f"  Supervisor model: {self.cadence_config.supervisor.model}")
        logger.info(f"  Agent model: {self.cadence_config.agent.defaults.get('model', 'NOT SET')}")

        # Group turn configurations together (orchestrator → supervisor → agent)
        logger.info(f"  Max orchestrator iterations: {self.cadence_config.orchestration.max_iterations}")
        logger.info(f"  Max supervisor turns: {self.cadence_config.execution.max_supervisor_turns}")
        logger.info(f"  Max agent turns: {self.cadence_config.execution.max_agent_turns}")

        logger.info(f"  Agent timeout: {self.cadence_config.execution.timeout}s")
        logger.info(f"  Code review frequency: {self.cadence_config.supervisor.zen_integration.code_review_frequency}")

        # Display MCP servers if configured
        mcp_servers = self.cadence_config.integrations.get('mcp', {})
        if mcp_servers:
            supervisor_servers = mcp_servers.get('supervisor_servers', [])
            agent_servers = mcp_servers.get('agent_servers', [])
            logger.info(f"  Supervisor MCP servers: {', '.join(supervisor_servers) if supervisor_servers else 'None'}")
            logger.info(f"  Agent MCP servers: {', '.join(agent_servers) if agent_servers else 'None'}")

        # Display zen integration configuration details
        zen_config = self._get_config_value('zen_processing_config', {})
        logger.info(f"  Code review model: {zen_config.get('primary_review_model', 'NOT SET')}")
        logger.info(f"  Secondary review model: {zen_config.get('secondary_review_model', 'NOT SET')}")
        logger.info(f"  Debug model: {zen_config.get('debug_model', 'NOT SET')}")
        logger.info(f"  Analyze model: {zen_config.get('analyze_model', 'NOT SET')}")

        logger.info("="*60)

        # CRITICAL: Clean up old session files from previous runs (keep recent ones)
        # This MUST happen before any other operations to prevent confusion
        logger.info("Cleaning up old session files...")
        keep_last_n = self._get_config_value("orchestration.cleanup_keep_sessions", 5)
        self.cleanup_old_sessions(keep_last_n=keep_last_n)

        # Also clean up any previous completion marker
        self.cleanup_completion_marker()

        # Force cleanup of any remaining old agent result files (they shouldn't persist across runs)
        old_agent_results = list(self.supervisor_dir.glob("agent_result_*.json"))
        if old_agent_results:
            logger.warning(f"Force cleaning {len(old_agent_results)} old agent result files...")
            for f in old_agent_results:
                try:
                    f.unlink()
                    logger.debug(f"Removed old agent result: {f.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {f}: {e}")

            # Check again after force cleanup
            remaining_files = list(self.supervisor_dir.glob("agent_result_*.json"))
            if remaining_files:
                logger.error(f"ERROR: {len(remaining_files)} agent result files still remain after force cleanup!")
            else:
                logger.info("Successfully cleaned all old agent result files")
        else:
            logger.debug("Confirmed: No old agent result files exist")

        # Update state
        self.state["session_count"] += 1
        self.state["last_session_id"] = self.current_session_id
        self.save_state()

        max_iterations = self.cadence_config.orchestration.max_iterations
        iteration = 0
        previous_agent_needed_help = False  # Track if previous agent needed help/errored
        code_review_pending = False  # Track if code review is needed after task completion
        self._previous_task_id = None  # Track previous task ID for code review validation

        # Define completion marker file
        completion_marker = self.project_root / ".cadence" / "project_complete.marker"

        while iteration < max_iterations:
            iteration += 1
            logger.info("="*60)
            logger.info(f"SUPERVISOR ITERATION {iteration}")
            logger.info("="*60)

            # Check if completion marker exists
            if completion_marker.exists():
                logger.info("Found project completion marker!")
                try:
                    with open(completion_marker, 'r') as f:
                        marker_content = f.read()
                    logger.info(f"Completion marker content:\n{marker_content}")


                    logger.info("="*60)
                    logger.info("PROJECT COMPLETED SUCCESSFULLY!")
                    logger.info("="*60)
                    return True
                except Exception as e:
                    logger.error(f"Error reading completion marker: {e}")

            # 1. Run supervisor to analyze and decide
            # Determine if supervisor should use continue based on config and previous agent status
            supervisor_use_continue = self._should_use_continue_for_supervisor(iteration, previous_agent_needed_help)
            decision = self.run_supervisor_analysis(
                self.current_session_id,
                use_continue=supervisor_use_continue,
                iteration=iteration,
                code_review_pending=code_review_pending
            )

            # Check if supervisor quit too quickly (likely an error)
            if decision.quit_too_quickly:
                logger.error(f"Supervisor quit too quickly ({decision.execution_time:.2f}s) - likely an error condition")
                logger.error("Exiting program due to quick supervisor exit")
                return False

            # 2. Check supervisor decision
            if decision.action == "complete":
                logger.info("Supervisor signaled completion")
                # Double-check for completion marker
                if completion_marker.exists():
                    logger.info("All tasks complete!")
                    return True
                else:
                    logger.warning("Supervisor said complete but no marker file found - continuing...")
                    # Reset state as supervisor is handling the situation
                    previous_agent_needed_help = False
                    # Continue to let supervisor create the marker
                    continue
            elif decision.action == "skip":
                logger.info(f"Skipping: {decision.reason}")
                # Reset state as supervisor is moving to new task
                previous_agent_needed_help = False
                continue
            elif decision.action == "execute":
                # Post-decision validation for code review critical issues
                if code_review_pending and hasattr(self, '_previous_task_id'):
                    logger.debug(f"Post-decision validation: code_review_pending=True, previous_task_id={self._previous_task_id}, current_task_id={decision.task_id}")

                    # Check if supervisor reported critical/high issues
                    if decision.code_review_has_critical_or_high_issues:
                        # Supervisor found critical/high issues
                        is_moving_to_different_task = (decision.task_id != self._previous_task_id)

                        if is_moving_to_different_task:
                            # This is an error - supervisor should stay on same task to fix issues
                            logger.error(
                                f"Code review found CRITICAL/HIGH issues but supervisor is moving "
                                f"to a different task (from {self._previous_task_id} to {decision.task_id}). "
                                f"This should not happen - supervisor prompt may need adjustment."
                            )
                            # For now, trust the supervisor's decision but log the error
                        else:
                            logger.info(
                                f"Code review found CRITICAL/HIGH issues. Supervisor correctly "
                                f"staying on task {decision.task_id} to fix them."
                            )
                    else:
                        # No critical/high issues found
                        logger.info(f"Code review passed with no CRITICAL/HIGH issues. Supervisor can proceed.")
                        code_review_pending = False  # Clear the flag

                # Track current task ID for next iteration.
                # The task_id is validated and normalized within run_supervisor_analysis.
                self._previous_task_id = decision.task_id

                # Save the decision as a snapshot for later review
                snapshot_file = self.validate_path(
                    self.supervisor_dir / f"decision_snapshot_{decision.session_id}.json",
                    self.supervisor_dir
                )
                with open(snapshot_file, 'w') as f:
                    json.dump({
                        "task_id": decision.task_id,
                        "task_title": decision.task_title,
                        "subtasks": decision.subtasks,
                        "project_root": decision.project_path,  # Unified as project_path
                        "guidance": decision.guidance,
                        "session_id": decision.session_id,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)

                # Extract todos from subtasks or use legacy todos
                if decision.subtasks:
                    todos = [f"{st['title']}: {st['description']}" for st in decision.subtasks]
                    logger.info(f"Executing agent with {len(decision.subtasks)} subtasks")
                else:
                    todos = decision.todos
                    logger.info(f"Executing agent with {len(decision.todos)} TODOs")

                # 3. Run agent with supervisor's TODOs (with retry logic)
                # Get retry configuration from agent defaults
                max_agent_retries = self.cadence_config.agent.defaults.get('retry_count', 1)
                agent_retry_count = 0
                agent_result = None
                last_agent_error = None

                # Determine initial continue flag
                agent_use_continue = self._should_use_continue_for_agent(iteration, previous_agent_needed_help)

                while agent_retry_count < max_agent_retries:
                    agent_retry_count += 1

                    # For retries after the first attempt, always use --continue
                    if agent_retry_count > 1:
                        retry_use_continue = True
                        logger.info(f"Agent retry {agent_retry_count}/{max_agent_retries} with --continue flag")
                    else:
                        retry_use_continue = agent_use_continue

                    try:
                        agent_result = self.run_agent(
                            todos=todos,
                            guidance=decision.guidance,
                            session_id=self.current_session_id,
                            use_continue=retry_use_continue,
                            task_id=decision.task_id,
                            subtasks=decision.subtasks,
                            project_root=decision.project_path  # Pass as project_path
                        )

                        # If agent succeeded, break out of retry loop
                        if agent_result.success:
                            logger.info(f"Agent succeeded on attempt {agent_retry_count}")
                            break

                        # Agent failed, check if we should retry
                        last_agent_error = '; '.join(agent_result.errors or ['Unknown error'])
                        if agent_retry_count < max_agent_retries:
                            logger.warning(f"Agent failed with: {last_agent_error}")
                            logger.info(f"Will retry agent (attempt {agent_retry_count + 1}/{max_agent_retries})")

                    except Exception as e:
                        last_agent_error = str(e)
                        logger.error(f"Agent execution error on attempt {agent_retry_count}: {e}")
                        if agent_retry_count >= max_agent_retries:
                            raise

                # If we exhausted retries and still failed, log it
                if not agent_result or not agent_result.success:
                    logger.error(f"Agent failed after {max_agent_retries} attempts. Last error: {last_agent_error}")

                # Check if agent quit too quickly (likely an error)
                if agent_result.quit_too_quickly:
                    logger.error(f"Agent quit too quickly ({agent_result.execution_time:.2f}s) - likely an error condition")
                    logger.error("Exiting program due to quick agent exit")
                    return False

                # 3.5. Validate agent created required scratchpad (with retry logic)
                max_scratchpad_retries = self._get_config_value("scratchpad_retry.max_retries", 5)
                scratchpad_retry_count = 0

                while scratchpad_retry_count < max_scratchpad_retries:
                    if self.validate_agent_scratchpad(self.current_session_id, agent_result):
                        # Scratchpad exists, we're good
                        break

                    scratchpad_retry_count += 1
                    if scratchpad_retry_count >= max_scratchpad_retries:
                        logger.error(f"Agent failed to create scratchpad after {max_scratchpad_retries} attempts")
                        logger.error("This indicates a serious agent prompt following issue")
                        return False

                    logger.warning(f"Scratchpad missing, attempting retry {scratchpad_retry_count}/{max_scratchpad_retries}")

                    # Retry agent with focused scratchpad creation prompt
                    retry_result = self.retry_agent_for_scratchpad(
                        session_id=self.current_session_id,
                        task_id=decision.task_id,
                        project_path=str(self.project_root)
                    )

                    if not retry_result.success:
                        logger.error(f"Scratchpad retry {scratchpad_retry_count} failed")
                        continue

                    # Check if retry actually created the scratchpad
                    if self.validate_agent_scratchpad(self.current_session_id, agent_result):
                        logger.info(f"Scratchpad successfully created on retry {scratchpad_retry_count}")
                        break
                    else:
                        logger.warning(f"Scratchpad retry {scratchpad_retry_count} completed but scratchpad still missing")

                # 4. Save agent results for supervisor
                self.save_agent_results(agent_result, self.current_session_id, decision.todos, decision.task_id)

                # Check if agent requested help
                if agent_result.requested_help:
                    logger.warning("Agent requested help - supervisor will provide assistance")

                # Check if code review is needed after successful task completion
                code_review_pending = False
                if agent_result.success and agent_result.completed_normally:
                    code_review_frequency = self.cadence_config.supervisor.zen_integration.code_review_frequency
                    if code_review_frequency == "task":
                        code_review_pending = True
                        logger.info("Code review pending after task completion (frequency=task)")

                # Update tracking for next iteration's continue decision
                previous_agent_needed_help = agent_result.requested_help or not agent_result.success

                # Note: The actual code review validation happens in the next supervisor iteration
                # We don't know yet if there are critical issues - the supervisor will determine that
                # and set code_review_has_critical_or_high_issues appropriately

                # 6. Continue to next iteration
                continue
            elif decision.action == "code_review":
                # Note: This action should no longer be used since supervisor does code review directly
                # If we get here, treat it as a skip and continue
                logger.warning("Received deprecated 'code_review' action - supervisor should do code review directly now")
                logger.info("Treating as 'skip' and continuing to next iteration")
                continue
            else:
                logger.error(f"Unknown supervisor action: {decision.action}")
                return False

        logger.warning(f"Reached maximum iterations ({max_iterations})")
        return False


    def run_supervisor_analysis(self, session_id: str, use_continue: bool, iteration: int,
                               code_review_pending: bool = False, additional_context: dict = None) -> SupervisorDecision:
            """Run supervisor in its directory to analyze state"""
            original_dir = os.getcwd()
            supervisor_start_time = time.time()

            try:
                # Change to supervisor directory
                self._validate_directory_exists(self.supervisor_dir)
                os.chdir(self.supervisor_dir)

                # Prepare data that doesn't change between retries
                # Get code review config
                code_review_frequency = self.cadence_config.supervisor.zen_integration.code_review_frequency

                # Check if we have previous agent results (only after first iteration)
                previous_agent_result = None
                if iteration > 1:
                    agent_results_file = self.supervisor_dir / self.cadence_config.file_patterns.agent_result_file.format(session_id=session_id)
                    logger.debug(f"Checking for agent results file: {agent_results_file}")
                    if agent_results_file.exists():
                        logger.debug(f"Agent results file exists, loading...")
                        try:
                            with open(agent_results_file, 'r') as f:
                                previous_agent_result = json.load(f)
                            logger.debug(f"Successfully loaded agent results: task_id={previous_agent_result.get('task_id', 'unknown')}")
                        except Exception as e:
                            logger.warning(f"Failed to load previous agent results: {e}")
                    else:
                        logger.debug(f"No agent results file found at {agent_results_file}")
                else:
                    logger.debug(f"First iteration - not checking for agent results")

                # Log agent result status
                if previous_agent_result is not None:
                    logger.info(f"Previous agent result: Found (task_id={previous_agent_result.get('task_id', 'unknown')})")
                else:
                    logger.info("Previous agent result: None")

                # Build prompt context
                context = {
                    "project_path": str(self.project_root),  # Unified project path
                    "session_id": session_id,
                    "iteration": iteration,
                    "is_first_iteration": iteration == 1,
                    "has_previous_agent_result": previous_agent_result is not None,
                    "agent_success": previous_agent_result.get("success", False) if previous_agent_result else False,
                    "agent_completed_normally": previous_agent_result.get("completed_normally", False) if previous_agent_result else False,
                    "agent_todos": previous_agent_result.get("todos", []) if previous_agent_result else [],
                    "agent_task_id": previous_agent_result.get("task_id", "") if previous_agent_result else "",
                    "max_turns": self.cadence_config.execution.max_agent_turns,  # Agent's turn limit for supervisor context
                    "code_review_pending": code_review_pending,
                    # Add zen_processing_config values for prompt templates
                    "primary_review_model": self.cadence_config.zen_processing_config.get("primary_review_model", "gemini-2.5-pro"),
                    "secondary_review_model": self.cadence_config.zen_processing_config.get("secondary_review_model", "o3"),
                    "zen_processing_config": self.cadence_config.zen_processing_config  # Also pass full config for other templates
                }

                # Merge additional context if provided
                if additional_context:
                    context.update(additional_context)

                # Get supervisor config
                supervisor_model = self.cadence_config.supervisor.model
                if not supervisor_model:
                    raise ValueError("Supervisor model not specified in config")

                supervisor_use_continue = self.cadence_config.supervisor.use_continue

                # Build allowed tools from config
                basic_tools = self.cadence_config.supervisor.tools
                mcp_servers = self.cadence_config.integrations.get("mcp", {}).get("supervisor_servers", [
                    "taskmaster-ai", "zen", "serena", "Context7"
                ])
                # Add mcp__ prefix and * suffix to each MCP server
                mcp_tools = [f"mcp__{server}__*" for server in mcp_servers]
                all_tools = basic_tools + mcp_tools

                # Keep track of retry count for command building
                json_retry_count = 0
                build_command_call_count = 0

                # Define command builder function that can be called for retries
                def build_command():
                    nonlocal json_retry_count, build_command_call_count

                    # Increment json_retry_count on subsequent calls (retries)
                    if build_command_call_count > 0:
                        json_retry_count += 1
                    build_command_call_count += 1

                    # Build prompt based on retry status
                    if json_retry_count > 0:
                        # Create a minimal retry prompt to avoid token limits
                        supervisor_prompt = f"""You are the Task Supervisor. Project root: {context['project_path']}

CRITICAL: Your previous output had invalid JSON formatting.
Please analyze the current task state and output ONLY a valid JSON object.

REQUIRED OUTPUT FORMAT:
For "execute" action:
{{
    "action": "execute",
    "task_id": "X",
    "task_title": "Title",
    "subtasks": [{{"id": "X.Y", "title": "...", "description": "..."}}],
    "project_root": "{context['project_path']}",
    "guidance": "Brief guidance",
    "session_id": "{context['session_id']}",
    "reason": "Why this action"
}}

For "skip" or "complete" actions:
{{
    "action": "skip|complete",
    "session_id": "{context['session_id']}",
    "reason": "Why this action"
}}

DO NOT include any explanatory text. Start with {{ and end with }}.
Retry attempt {json_retry_count + 1} of {self.cadence_config.retry_behavior.get("max_json_retries", 3)}."""
                    else:
                        # Build full prompt using YAML templates
                        logger.debug(f"Getting supervisor prompt template with context: has_previous_agent_result={context['has_previous_agent_result']}")
                        base_prompt = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.base_prompt")
                        base_prompt = self.prompt_loader.format_template(base_prompt, context)

                        # Log prompt type based on context variables instead of string matching
                        if context.get('has_previous_agent_result', False):
                            logger.debug("Supervisor prompt includes agent work processing (iteration 2+)")
                        else:
                            logger.debug("Supervisor prompt is for first iteration (no agent work)")

                        # Get code review pending section (if applicable)
                        code_review_pending_section = ""
                        if code_review_pending:
                            code_review_pending_key = "supervisor_prompts.orchestrator_taskmaster.code_review_pending_section"
                            code_review_pending_section = self.prompt_loader.get_template(code_review_pending_key)
                            code_review_pending_section = self.prompt_loader.format_template(code_review_pending_section, context)

                        # Note: code_review_completed_section removed - supervisor now does code review directly

                        # Get code review section based on config
                        if code_review_frequency == "task":
                            # Include BOTH task-level and project-level review instructions
                            task_review_key = "supervisor_prompts.orchestrator_taskmaster.code_review_sections.task"
                            task_review_section = self.prompt_loader.get_template(task_review_key)
                            task_review_section = self.prompt_loader.format_template(task_review_section, context)

                            project_review_key = "supervisor_prompts.orchestrator_taskmaster.code_review_sections.project"
                            project_review_section = self.prompt_loader.get_template(project_review_key)
                            project_review_section = self.prompt_loader.format_template(project_review_section, context)

                            code_review_section = task_review_section + project_review_section
                        else:
                            # For "project" or "none" frequency, use only the specified section
                            code_review_key = f"supervisor_prompts.orchestrator_taskmaster.code_review_sections.{code_review_frequency}"
                            code_review_section = self.prompt_loader.get_template(code_review_key)
                            code_review_section = self.prompt_loader.format_template(code_review_section, context)

                        # Get zen guidance
                        zen_guidance = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.zen_guidance")

                        # Get output format
                        output_format = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.output_format")
                        output_format = self.prompt_loader.format_template(output_format, context)

                        # Combine all sections
                        supervisor_prompt = f"{base_prompt}{code_review_pending_section}{code_review_section}{zen_guidance}{output_format}"

                    # Save supervisor prompt to file for debugging
                    prompt_debug_file = self.supervisor_dir / f"supervisor_prompt_{session_id}.txt"
                    try:
                        with open(prompt_debug_file, 'w') as f:
                            f.write(supervisor_prompt)
                        logger.debug(f"Saved supervisor prompt to {prompt_debug_file}")
                        logger.debug(f"Supervisor prompt length: {len(supervisor_prompt)} characters")
                    except Exception as e:
                        logger.warning(f"Failed to save supervisor prompt: {e}")

                    # Build command
                    cmd = [
                        "claude",
                        "-p", supervisor_prompt,
                        "--model", supervisor_model,
                        "--allowedTools", ",".join(all_tools),
                        "--max-turns", str(self.cadence_config.execution.max_supervisor_turns),
                        "--output-format", "stream-json",
                        "--verbose",
                        "--dangerously-skip-permissions"  # Skip permission prompts
                    ]

                    # Add --continue flag based on config and conditions
                    if supervisor_use_continue and (use_continue or json_retry_count > 0):
                        cmd.append("--continue")
                        if json_retry_count > 0:
                            logger.debug("Adding --continue flag for retry")
                        else:
                            logger.debug("Running supervisor with --continue flag")
                    else:
                        if not supervisor_use_continue:
                            logger.debug("Running supervisor without --continue flag (disabled in config)")
                        else:
                            logger.debug("Running supervisor (first run)")

                    logger.debug(f"Command: {' '.join(cmd[:3])}...")  # Don't log full prompt
                    return cmd

                # Define output parser function
                def parse_output(all_output):
                    nonlocal json_retry_count

                    logger.debug(f"Parsing supervisor output from {len(all_output)} lines...")
                    logger.debug(f"First 3 output lines: {all_output[:3] if all_output else 'None'}")
                    logger.debug(f"Last 3 output lines: {all_output[-3:] if all_output else 'None'}")

                    # Single-pass parsing: look for decision JSON and keep the parsed objects
                    found_decisions = []

                    # Create JSON stream monitor - do NOT reset between text blocks
                    json_monitor = SimpleJSONStreamMonitor()

                    # Process all output lines to find JSON objects (single pass)
                    for line_str in all_output:
                        try:
                            # Parse the stream-json line
                            data = json.loads(line_str)
                            if data.get('type') == 'assistant':
                                # Extract content from assistant message
                                content = data.get('message', {}).get('content', [])
                                # Handle content as a list of items
                                if isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            text = item.get('text', '')
                                            # Process each line of the text with the JSON monitor
                                            for text_line in text.split('\n'):
                                                result = json_monitor.process_line(text_line)
                                                if result and isinstance(result, dict) and 'action' in result:
                                                    # Store the parsed object directly (no double parsing)
                                                    found_decisions.append(result)
                                            # DO NOT reset monitor here - JSON might span multiple text blocks
                        except json.JSONDecodeError as e:
                            # This line wasn't JSON, skip it but log for debugging
                            logger.debug(f"Failed to parse JSON line: {e} - line: {line_str[:100]}...")
                            continue

                    # Take the LAST valid decision found
                    if not found_decisions:
                        logger.error("No decision JSON found in supervisor output")
                        logger.error(f"Total output lines processed: {len(all_output)}")
                        # Log a sample of assistant messages for debugging
                        assistant_lines = []
                        for line_str in all_output[:10]:  # Just first 10 to avoid spam
                            try:
                                data = json.loads(line_str)
                                if data.get('type') == 'assistant':
                                    assistant_lines.append(line_str[:200])  # Truncate for readability
                            except json.JSONDecodeError as e:
                                logger.debug(f"Sample line parse error: {e}")
                        logger.error(f"Sample assistant message lines: {assistant_lines}")
                        raise ValueError("No decision JSON found in supervisor output")

                    decision_data = found_decisions[-1]  # Take the last decision (no re-parsing needed)
                    logger.debug(f"Found decision: {decision_data.get('action', 'unknown')} action")

                    # Validate required fields based on action
                    if decision_data.get('action') == 'execute':
                        # For execute action, check for new subtasks structure
                        if 'subtasks' in decision_data:
                            required_fields = ['action', 'task_id', 'task_title', 'subtasks', 'project_root', 'session_id', 'reason']  # Note: project_root is kept for backward compatibility in JSON
                        else:
                            # Backward compatibility with todos
                            required_fields = ['action', 'todos', 'guidance', 'task_id', 'session_id', 'reason']
                    else:
                        # For skip/complete actions
                        required_fields = ['action', 'session_id', 'reason']

                    missing_fields = [field for field in required_fields if field not in decision_data]
                    if missing_fields:
                        raise ValueError(f"Decision JSON missing required fields: {missing_fields}")

                    return decision_data

                # Define error return function
                def create_error_decision(error_msg):
                    supervisor_execution_time = time.time() - supervisor_start_time
                    quick_quit_threshold = self.cadence_config.orchestration.quick_quit_seconds
                    return SupervisorDecision(
                        action="skip",
                        session_id=session_id,
                        reason=f"Error: {error_msg}",
                        execution_time=supervisor_execution_time,
                        quit_too_quickly=supervisor_execution_time < quick_quit_threshold
                    )

                # Use the helper to run with retry
                decision_data = self._run_claude_with_json_retry(
                    build_command_func=build_command,
                    parse_output_func=parse_output,
                    working_dir=self.supervisor_dir,
                    process_name="SUPERVISOR",
                    session_id=session_id
                )

                # Calculate execution time
                supervisor_execution_time = time.time() - supervisor_start_time
                quick_quit_threshold = self.cadence_config.orchestration.quick_quit_seconds

                # Create decision object
                # Map project_root from JSON to project_path for internal use
                if 'project_root' in decision_data and 'project_path' not in decision_data:
                    decision_data['project_path'] = decision_data.pop('project_root')

                # Validate task_id format - must be a single number without decimal notation
                if 'task_id' in decision_data and decision_data['task_id']:
                    task_id = str(decision_data['task_id'])

                    # First, correct for decimal notation if present
                    if '.' in task_id:
                        main_task_id = task_id.split('.')[0]
                        logger.warning(
                            f"Supervisor provided subtask ID '{task_id}' in task_id field. "
                            f"Correcting to main task ID '{main_task_id}'."
                        )
                        task_id = main_task_id

                    # Then, validate that the result is a number
                    if not task_id.isdigit():
                        logger.error(
                            f"Invalid task_id format: '{task_id}' is not a valid number. "
                            "This may cause issues with task tracking."
                        )

                    decision_data['task_id'] = task_id

                decision = SupervisorDecision(**decision_data)
                decision.execution_time = supervisor_execution_time
                decision.quit_too_quickly = supervisor_execution_time < quick_quit_threshold

                logger.info(f"Supervisor decision: {decision.action}")
                if decision.reason:
                    logger.info(f"   Reason: {decision.reason}")

                return decision

            finally:
                # Always return to original directory
                os.chdir(original_dir)






    def _determine_agent_help_status(self, status: str, agent_data: Dict) -> bool:
        """
        Determine if agent requested help based on status and data.

        Args:
            status: Agent status ("success", "help_needed", "error")
            agent_data: Full agent JSON response data

        Returns:
            True if agent needs help, False if agent completed successfully
        """
        # Explicit status-based determination
        if status == "success":
            return False
        elif status in ["help_needed", "error"]:
            return True
        else:
            # For unknown status, check help_needed field or default to needing help
            return agent_data.get("help_needed", True)

    def _validate_directory_exists(self, directory: Path) -> None:
        """
        Validate that a directory exists before changing to it.

        Args:
            directory: Path to validate

        Raises:
            FileNotFoundError: If directory does not exist
        """
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

    def _should_use_continue_for_supervisor(self, iteration: int, previous_agent_needed_help: bool) -> bool:
        """
        Determine if supervisor should use --continue flag based on config and previous agent status.

        Args:
            iteration: Current iteration number (1-based)
            previous_agent_needed_help: True if previous agent needed help/errored

        Returns:
            True if supervisor should use --continue flag
        """
        if iteration == 1:
            return False  # Never continue on first iteration

        supervisor_config = self.cadence_config.supervisor
        supervisor_use_continue_config = supervisor_config.use_continue

        if supervisor_use_continue_config:
            # Config says always continue after first iteration
            return True
        else:
            # Config says only continue on errors/help needed
            return previous_agent_needed_help

    def _should_use_continue_for_agent(self, iteration: int, previous_agent_needed_help: bool) -> bool:
        """
        Determine if agent should use --continue flag based on config and previous agent status.

        Args:
            iteration: Current iteration number (1-based)
            previous_agent_needed_help: True if previous agent needed help/errored

        Returns:
            True if agent should use --continue flag
        """
        if iteration == 1:
            return False  # Never continue on first iteration

        agent_config = self.cadence_config.agent
        agent_use_continue_config = agent_config.defaults.get('use_continue', False)

        if agent_use_continue_config:
            # Config says always continue after first iteration
            return True
        else:
            # Config says only continue on errors/help needed
            return previous_agent_needed_help

    def run_agent(self, todos: List[str], guidance: str,
                      session_id: str, use_continue: bool,
                      task_id: str = None, subtasks: List[Dict] = None,
                      project_root: str = None) -> AgentResult:
        """Run agent using simplified UnifiedAgent approach"""
        # Create prompt with TODOs
        prompt = self.build_agent_prompt(todos, guidance, task_id, subtasks, project_root, use_continue)

        # Agent will be started by UnifiedAgent with consistent formatting

        # Convert config to dictionary format for UnifiedAgent
        config_dict = {
            'agent': {
                'defaults': {
                    'model': self.cadence_config.agent.defaults.get('model'),
                    'tools': self.cadence_config.agent.defaults.get('tools'),
                    'extra_flags': self.cadence_config.agent.defaults.get('extra_flags'),
                    'retry_count': self.cadence_config.agent.defaults.get('retry_count'),
                    'use_continue': self.cadence_config.agent.defaults.get('use_continue'),
                    'timeout_seconds': self.cadence_config.agent.defaults.get('timeout_seconds'),
                    'temperature': self.cadence_config.agent.defaults.get('temperature'),
                    'max_turns': self.cadence_config.execution.max_agent_turns  # Add max_turns from execution config
                }
            },
            'retry_behavior': self.cadence_config.retry_behavior
        }

        # Add MCP tools to the configuration
        basic_tools = config_dict['agent']['defaults']['tools'] or []
        mcp_servers = self.cadence_config.integrations.get("mcp", {}).get("agent_servers", [])
        mcp_tools = [f"mcp__{server}__*" for server in mcp_servers]
        all_tools = basic_tools + mcp_tools
        config_dict['agent']['defaults']['tools'] = all_tools

        # Set environment variables for logging before creating agent
        log_dir = self.project_root / ".cadence" / "logs"
        os.environ["CADENCE_LOG_SESSION"] = session_id
        os.environ["CADENCE_LOG_DIR"] = str(log_dir.resolve())
        os.environ["CADENCE_LOG_LEVEL"] = self.log_level_str

        # Create and execute unified agent
        agent = UnifiedAgent(
            config=config_dict,
            working_dir=self.agent_dir,
            session_id=session_id
        )

        # Execute the agent
        result = agent.execute(
            prompt=prompt,
            context={'task_id': task_id, 'subtasks': subtasks, 'project_root': project_root},
            continue_session=use_continue
        )

        logger.info("=" * 50)
        logger.info(f"AGENT COMPLETED in {result.execution_time:.2f}s")
        logger.info("=" * 50)

        logger.info(f"Agent execution complete in {result.execution_time:.2f}s")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Completed normally: {result.completed_normally}")
        logger.info(f"   Requested help: {result.requested_help}")

        return result

    def build_agent_prompt(self, todos: List[str], guidance: str,
                              task_id: str = None, subtasks: List[Dict] = None,
                              project_root: str = None, use_continue: bool = False) -> str:
        """Build prompt for agent with TODOs and guidance"""
        # Create a PromptGenerator instance
        prompt_generator = PromptGenerator(self.prompt_loader)

        session_id = self.current_session_id if hasattr(self, 'current_session_id') else "unknown"
        max_turns = self.cadence_config.execution.max_agent_turns

        # Create ExecutionContext properly for both paths
        context = ExecutionContext(
            todos=todos,
            max_turns=max_turns,
            project_path=project_root or str(self.project_root)  # For Serena activation
        )

        if use_continue:
            # Generate continuation prompt for resumed execution
            # Create supervisor analysis for continuation context
            supervisor_analysis = {
                'session_id': session_id,
                'previous_session_id': getattr(self, 'previous_session_id', 'unknown'),
                'completed_normally': False,  # Will be updated based on actual analysis
                'has_issues': False  # Will be updated based on actual analysis
            }

            base_prompt = prompt_generator.generate_continuation_prompt(
                context=context,
                analysis_guidance=guidance or "Continue where you left off. Focus on completing the remaining TODOs.",
                supervisor_analysis=supervisor_analysis
            )
        else:
            # Generate the initial prompt using ExecutionContext
            base_prompt = prompt_generator.generate_initial_todo_prompt(
                context=context,
                session_id=session_id,
                task_numbers=str(task_id) if task_id else "",
                project_root=project_root or str(self.project_root)
            )

        # Add supervisor guidance if provided (only for initial prompts, continuation prompts handle guidance differently)
        if guidance and not use_continue:
            guidance_section = f"\n=== SUPERVISOR GUIDANCE ===\n{guidance}\n"
            # Check if guidance includes Zen assistance
            if "zen assistance" in guidance.lower() or "expert assistance" in guidance.lower():
                zen_reminder = self.prompt_loader.get_template("agent_zen_reminder")
                guidance_section += zen_reminder

            # Insert guidance after the context but before the TODOs
            # Find the TODO list section
            todo_marker = "=== YOUR TODOS ==="
            if todo_marker in base_prompt:
                parts = base_prompt.split(todo_marker)
                base_prompt = parts[0] + guidance_section + todo_marker + parts[1]
            else:
                # Fallback: append at end
                base_prompt += guidance_section

        # Add Task Master information if provided
        if task_id and subtasks and project_root:
            task_master_info = f"\n=== TASK MASTER INFORMATION ===\nYou are working on Task {task_id} with the following subtasks:\n\n"
            for subtask in subtasks:
                task_master_info += f"- Subtask {subtask['id']}: {subtask['title']}\n"
                if subtask.get('description'):
                    task_master_info += f"  Description: {subtask['description']}\n"

            task_master_info += f"\nProject Root: {project_root}\n\n"
            task_master_info += f"IMPORTANT: You have access to Task Master MCP tools. You should:\n"
            task_master_info += f"1. Update subtask status as you complete each one using:\n"
            task_master_info += f"   mcp__taskmaster-ai__set_task_status --id=<subtask_id> --status=in-progress --projectRoot={project_root}\n"
            task_master_info += f"   mcp__taskmaster-ai__set_task_status --id=<subtask_id> --status=done --projectRoot={project_root}\n"
            task_master_info += f"2. Add implementation notes to subtasks as needed:\n"
            task_master_info += f"   mcp__taskmaster-ai__update_subtask --id=<subtask_id> --prompt=\"implementation notes...\" --projectRoot={project_root}\n\n"

            # Insert Task Master info at the beginning after supervised context
            context_marker = "=== SUPERVISED AGENT CONTEXT ==="
            if context_marker in base_prompt:
                parts = base_prompt.split(context_marker)
                # Find the end of the supervised context section
                end_of_context = parts[1].find("\n===")
                if end_of_context > 0:
                    base_prompt = parts[0] + context_marker + parts[1][:end_of_context] + task_master_info + parts[1][end_of_context:]
                else:
                    base_prompt = parts[0] + context_marker + parts[1] + task_master_info
            else:
                # Fallback: prepend
                base_prompt = task_master_info + base_prompt

        # Add agent JSON output format instructions
        agent_output_format = self.prompt_loader.get_template("agent_output_format")
        context_for_format = {
            "session_id": session_id
        }
        agent_output_format = self.prompt_loader.format_template(agent_output_format, context_for_format)

        # Insert at the end before any final reminders
        base_prompt += f"\n\n{agent_output_format}"

        return base_prompt




    def validate_agent_scratchpad(self, session_id: str, agent_result: AgentResult) -> bool:
        """Validate that agent created required scratchpad file"""
        # Check the expected location only - no fallback search
        scratchpad_file = self.project_root / ".cadence" / "scratchpad" / f"session_{session_id}.md"

        if scratchpad_file.exists():
            logger.debug(f"Agent scratchpad found at expected location: {scratchpad_file}")
            return True

        # No fallback - enforce strict file placement
        logger.warning(f"Agent scratchpad not found at expected location: {scratchpad_file}")
        logger.warning(f"Agent must create scratchpad at exact path: {scratchpad_file}")
        logger.warning(f"Agent claimed success but failed to create required scratchpad")
        return False

    def retry_agent_for_scratchpad(self, session_id: str, task_id: str, project_path: str) -> AgentResult:
        """Retry agent with focused prompt to create missing scratchpad"""
        logger.info("Running focused agent retry to create missing scratchpad")

        # Get scratchpad retry prompt from prompts template
        context = {
        "session_id": session_id,
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "project_root": str(self.project_root)
        }

        scratchpad_prompt = self.prompt_loader.get_template("todo_templates.scratchpad_retry")
        scratchpad_prompt = self.prompt_loader.format_template(scratchpad_prompt, context)
        logger.info(f"Using scratchpad retry prompt, length: {len(scratchpad_prompt)} characters")

        # Debug: Check if prompt is empty
        if not scratchpad_prompt or not scratchpad_prompt.strip():
            logger.error("Scratchpad retry prompt is empty!")
            logger.error(f"Template returned: '{scratchpad_prompt}'")
            logger.error(f"Context: {context}")
            return AgentResult(
                success=False,
                session_id=session_id,
                output_file="",
                error_file="",
                execution_time=0.0,
                completed_normally=False,
                requested_help=False,
                errors=["Empty scratchpad retry prompt"]
            )

        # Save prompt for debugging (same pattern as main agent)
        prompt_debug_file = self.agent_dir / f"scratchpad_retry_prompt_{session_id}.txt"
        try:
            with open(prompt_debug_file, 'w') as f:
                f.write(scratchpad_prompt)
            logger.debug(f"Saved scratchpad retry prompt to {prompt_debug_file}")
        except Exception as e:
            logger.warning(f"Failed to save scratchpad retry prompt: {e}")

        original_dir = os.getcwd()
        try:
            self._validate_directory_exists(self.agent_dir)
            os.chdir(self.agent_dir)

            # Build command using same pattern as main agent
            # ALWAYS use --continue for scratchpad retry to maintain context
            cmd = ["claude", "--continue", "-p", scratchpad_prompt]
            cmd.extend([
                "--allowedTools", self._get_config_value("scratchpad_retry.allowed_tools", "Write,Read,Bash,LS"),
                "--max-turns", str(self._get_config_value("scratchpad_retry.max_turns", 5)),
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions"
            ])

            # Debug command
            logger.info(f"Scratchpad retry command: claude --continue -p [PROMPT] {' '.join(cmd[3:])}")
            logger.info(f"Working directory: {os.getcwd()}")
            logger.info("Using --continue flag for scratchpad retry to maintain context")

            start_time = time.time()

            # Run with minimal timeout
            try:
                # Use safe async runner to handle existing event loops
                returncode, all_output = self._run_async_safely(
                    self.run_claude_with_realtime_output(cmd, self.agent_dir, "SCRATCHPAD_AGENT", session_id)
                )

                execution_time = time.time() - start_time

                # Parse JSON output to check for completion
                completed_normally = False
                requested_help = False
                errors = []

                if returncode == 0:
                    # Try to extract JSON status using SimpleJSONStreamMonitor
                    monitor = SimpleJSONStreamMonitor()
                    for line in all_output:
                        monitor.process_line(line)

                    last_json_obj = monitor.get_last_json_object()
                    if last_json_obj and isinstance(last_json_obj, dict):
                        status = last_json_obj.get('status', '')
                        if status == 'success':
                            completed_normally = True
                        elif status == 'help_needed':
                            requested_help = True
                        elif status == 'error':
                            errors = [last_json_obj.get('error_message', 'Unknown error')]
                    else:
                        errors = ["No valid JSON output from scratchpad retry"]
                else:
                    errors = ["Scratchpad retry failed"]

                return AgentResult(
                    success=returncode == 0 and completed_normally,
                    session_id=session_id,
                    output_file="scratchpad_retry_output.log",
                    error_file="scratchpad_retry_error.log",
                    execution_time=execution_time,
                    completed_normally=completed_normally,
                    requested_help=requested_help,
                    errors=errors
                )

            except Exception as e:
                logger.error(f"Scratchpad retry error: {e}")
                return AgentResult(
                    success=False,
                    session_id=session_id,
                    output_file="",
                    error_file="",
                    execution_time=time.time() - start_time,
                    completed_normally=False,
                    requested_help=False,
                    errors=[f"Scratchpad retry failed: {e}"]
                )

        finally:
            os.chdir(original_dir)

    def save_agent_results(self, agent_result: AgentResult, session_id: str, todos: List[str] = None, task_id: str = None):
        """Save agent results for supervisor to analyze"""
        results_file = self.validate_path(
            self.supervisor_dir / self.cadence_config.file_patterns.agent_result_file.format(session_id=session_id),
            self.supervisor_dir
        )

        with open(results_file, 'w') as f:
            json.dump({
                "success": agent_result.success,
                "session_id": agent_result.session_id,
                "output_file": agent_result.output_file,
                "error_file": agent_result.error_file,
                "execution_time": agent_result.execution_time,
                "completed_normally": agent_result.completed_normally,
                "requested_help": agent_result.requested_help,
                "errors": agent_result.errors,
                "todos": todos or [],  # The TODOs that were given to the agent
                "task_id": task_id,  # The task ID being worked on
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        logger.debug(f"Saved agent results to {results_file}")



    def cleanup_all_session_files(self):
        """Clean up ALL session files from previous runs"""
        removed_count = 0

        # Clean supervisor directory
        for pattern in ["decision_*.json", "agent_result_*.json", "decision_snapshot_*.json", "session_*.md"]:
            for file in self.supervisor_dir.glob(pattern):
                try:
                    file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old session file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")

        # Clean agent directory
        for pattern in ["prompt_*.txt", "output_*.log", "error_*.log"]:
            for file in self.agent_dir.glob(pattern):
                try:
                    file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old session file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")

        # Clean scratchpad directory
        scratchpad_dir = self.project_root / ".cadence" / "scratchpad"
        if scratchpad_dir.exists():
            for file in scratchpad_dir.glob("session_*.md"):
                try:
                    file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old scratchpad: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old session files")


    def get_directory_size(self, path: Path) -> float:
        """Calculate total size of a directory in MB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        except (OSError, PermissionError) as e:
            logger.warning(f"Error calculating size of {path}: {e}")
        return total_size / (1024 * 1024)  # Convert to MB  # Convert to MB

    def cleanup_old_sessions(self, keep_last_n: int = 5):
        """Clean up old session files and log directories to save space"""
        # Check if we should clean all logs on startup
        clean_all = self._get_config_value('execution.clean_logs_on_startup', False)
        max_size_mb = self._get_config_value('execution.max_log_size_mb', 20)

        # Get all session files from both directories
        session_files = []

        # Collect files from supervisor directory
        for pattern in ["decision_*.json", "agent_result_*.json", "decision_snapshot_*.json", "session_*.md", "supervisor_prompt_*.txt"]:
            for file in self.supervisor_dir.glob(pattern):
                session_files.append(file)

        # Collect files from agent directory
        for pattern in ["prompt_*.txt", "output_*.log", "error_*.log", "debug_*.log"]:
            for file in self.agent_dir.glob(pattern):
                session_files.append(file)

        # Collect files from scratchpad directory
        scratchpad_dir = self.project_root / ".cadence" / "scratchpad"
        if scratchpad_dir.exists():
            for file in scratchpad_dir.glob("session_*.md"):
                session_files.append(file)

        # Extract session IDs and group by session
        sessions = {}
        # Regex to match the session ID format (e.g., 20231027_103000_a1b2c3d4)
        session_id_pattern = re.compile(r"(\d{8}_\d{6}_[a-f0-9]{8})")

        for file in session_files:
            match = session_id_pattern.search(file.name)
            if match:
                session_id = match.group(1)
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(file)

        # Add log directories to sessions
        log_dir = self.project_root / ".cadence" / "logs"
        if log_dir.exists():
            for session_dir in log_dir.iterdir():
                if session_dir.is_dir():
                    match = session_id_pattern.search(session_dir.name)
                    if match:
                        session_id = match.group(1)
                        if session_id not in sessions:
                            sessions[session_id] = []
                        sessions[session_id].append(session_dir)

        # Sort sessions by timestamp (newest first)
        sorted_sessions = sorted(sessions.keys(), reverse=True)

        # Determine which sessions to remove
        if clean_all:
            # Clean all logs on startup
            logger.info("Cleaning all logs on startup (clean_logs_on_startup=true)")
            sessions_to_remove = sorted_sessions
        else:
            # Check total log directory size
            total_log_size = self.get_directory_size(log_dir) if log_dir.exists() else 0

            if total_log_size > max_size_mb:
                logger.info(f"Log directory size ({total_log_size:.1f} MB) exceeds limit ({max_size_mb} MB)")
                # Remove oldest sessions until we're under the limit
                sessions_to_remove = []
                current_size = total_log_size

                # Start from oldest sessions
                for session_id in reversed(sorted_sessions):
                    if current_size <= max_size_mb and len(sorted_sessions) - len(sessions_to_remove) >= keep_last_n:
                        break

                    # Calculate size of this session's logs
                    session_size = 0
                    for item in sessions[session_id]:
                        if item.is_dir():
                            session_size += self.get_directory_size(item)

                    sessions_to_remove.append(session_id)
                    current_size -= session_size
            else:
                # Just keep the most recent sessions
                sessions_to_remove = sorted_sessions[keep_last_n:]

        # Remove old session files and directories
        removed_count = 0
        removed_dirs = 0

        for session_id in sessions_to_remove:
            for item in sessions[session_id]:
                try:
                    if item.exists():
                        if item.is_dir():
                            # Remove entire directory
                            shutil.rmtree(item)
                            removed_dirs += 1
                            logger.debug(f"Removed log directory from session {session_id}: {item}")
                        else:
                            # Remove file
                            item.unlink()
                            removed_count += 1
                            logger.debug(f"Removed old file from session {session_id}: {item}")
                    else:
                        logger.debug(f"Item already removed: {item}")
                except Exception as e:
                    logger.warning(f"Failed to remove {item}: {e}")

        if removed_count > 0 or removed_dirs > 0:
            logger.info(f"Cleaned up {removed_count} files and {removed_dirs} directories from {len(sessions_to_remove)} old sessions")

    def cleanup_supervisor_logs(self, keep_last_n: int = 10):
        """Clean up old supervisor log files (session_*.md files)

        Args:
        keep_last_n: Number of most recent log files to keep
        """
        # Get all supervisor log files
        log_files = list(self.supervisor_dir.glob("session_*.md"))

        # Sort by modification time (newest first)
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Keep only the most recent files
        files_to_remove = log_files[keep_last_n:]

        removed_count = 0
        for file in files_to_remove:
            try:
                file.unlink()
                removed_count += 1
                logger.debug(f"Removed old supervisor log: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove {file}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old supervisor log files")


    def read_supervisor_output(self, session_id: str) -> List[str]:
        """
        Read supervisor output from log file.

        Args:
            session_id: The session ID to read logs for

        Returns:
            List of output lines from supervisor
        """
        log_file = self.project_root / ".cadence" / "logs" / session_id / "supervisor.log"

        if not log_file.exists():
            logger.warning(f"Supervisor log file not found: {log_file}")
            return []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [line.rstrip('\n') for line in lines]
        except Exception as e:
            logger.error(f"Failed to read supervisor log: {e}")
            return []
