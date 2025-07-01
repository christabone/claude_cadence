"""
Unified Agent - A single, configurable agent class for all agent behaviors
"""

import os
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .utils import generate_session_id
from .json_stream_monitor import SimpleJSONStreamMonitor
from .log_utils import Colors

logger = logging.getLogger(__name__)

# Security allowlists for command construction
ALLOWED_TOOLS = {
    "bash", "read", "write", "edit", "grep", "glob", "search", "mcp", "WebFetch",
    "LS", "Glob", "Grep", "Task", "exit_plan_mode", "Write", "Edit", "MultiEdit",
    "NotebookRead", "NotebookEdit", "TodoRead", "TodoWrite", "WebSearch"
}

ALLOWED_FLAGS = {
    "--dangerously-skip-permissions", "--continue", "--verbose", "--debug",
    "--temperature", "--model", "--max-turns", "--allowedTools",
    "--timeout", "--output", "--error", "--print"
}


@dataclass
class AgentResult:
    """Standardized result from agent execution"""
    success: bool
    session_id: str
    output_file: str
    error_file: str
    execution_time: float
    completed_normally: bool = False
    requested_help: bool = False
    errors: List[str] = None
    quit_too_quickly: bool = False
    retry_count: int = 0


class UnifiedAgent:
    """
    A single, configurable agent for all agent behaviors.
    """

    def __init__(self, config: dict, working_dir: Path,
                 session_id: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            config: Full configuration dictionary
            working_dir: Working directory for the agent
            session_id: Optional session ID (will generate if not provided)
        """
        self.config = config
        self.working_dir = Path(working_dir)
        self.session_id = session_id or generate_session_id()

        # Load settings from defaults
        agent_config = self.config.get('agent', {})
        self.settings = agent_config.get('defaults', {})

        # Core settings
        self.retry_count = self.settings.get('retry_count', 1)
        self.use_continue = self.settings.get('use_continue', False)
        self.timeout_seconds = self.settings.get('timeout_seconds', 300)
        self.temperature = self.settings.get('temperature', 0.2)
        self.model = self.settings.get('model', 'claude-3-5-sonnet-20241022')
        self.tools = self.settings.get('tools', [])
        self.extra_flags = self.settings.get('extra_flags', [])

        # Set up working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging if environment variables are present
        session_id_from_env = os.environ.get("CADENCE_LOG_SESSION", self.session_id)
        log_dir = os.environ.get("CADENCE_LOG_DIR")

        if log_dir:
            try:
                from .log_utils import setup_file_logging
                log_level_str = os.environ.get("CADENCE_LOG_LEVEL", "DEBUG")
                log_level = getattr(logging, log_level_str.upper(), logging.DEBUG)
                setup_file_logging(session_id_from_env, "agent", Path(log_dir), level=log_level)

                # Note: We do NOT redirect stdout/stderr for the agent because:
                # 1. The agent runs as a subprocess (claude CLI), not a Python process
                # 2. The orchestrator already captures the agent's stdout/stderr
                # 3. Redirecting here would interfere with the orchestrator's capture
                logger.debug(f"Set up file logging for agent with session {session_id_from_env}")
            except Exception as e:
                logger.warning(f"Failed to set up file logging for agent: {e}")

        logger.info(f"Initialized UnifiedAgent "
                   f"(retry={self.retry_count}, continue={self.use_continue})")

    async def _run_claude_with_realtime_output(self, cmd: List[str], cwd: str, process_name: str = "AGENT") -> tuple[int, List[str]]:
        """Run claude command with real-time output display (same as orchestrator)"""
        # Set up environment
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Add logging environment variables
        log_dir = Path(cwd).parent / "logs"
        env["CADENCE_LOG_SESSION"] = self.session_id
        env["CADENCE_LOG_DIR"] = str(log_dir.resolve())
        env["CADENCE_LOG_LEVEL"] = os.environ.get("CADENCE_LOG_LEVEL", "DEBUG")

        # Choose color for agent
        color = Colors.BOLD_MAGENTA

        print(f"\n{color}{process_name} working...{Colors.RESET}")
        print(f"{Colors.WHITE}{'-' * 50}{Colors.RESET}")
        print(f"{color}[{process_name}]{Colors.RESET} {Colors.BOLD}Model: {self.model}{Colors.RESET}")

        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env
        )

        # Store all output lines
        all_output = []
        line_count = 0
        json_monitor = SimpleJSONStreamMonitor()

        while True:
            try:
                # Read line by line
                line = await process.stdout.readline()
                if not line:  # EOF
                    break

                line_str = line.decode('utf-8', errors='replace').rstrip('\n\r')
                all_output.append(line_str)
                line_count += 1

                # Process line with JSON monitor
                json_monitor.process_line(line_str)

                # Parse and display if JSON
                try:
                    json_data = json.loads(line_str)
                    msg_type = json_data.get('type', '')

                    if msg_type == 'assistant':
                        # Assistant messages
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
                except json.JSONDecodeError:
                    # Not JSON or incomplete JSON, display as plain text if not buffering
                    if not json_monitor.in_json:
                        print(f"{color}[{process_name}]{Colors.RESET} {line_str}")

            except Exception as e:
                logger.warning(f"Error processing output line: {e}")
                # Continue processing other lines

        # Wait for process completion
        await process.wait()
        return process.returncode, all_output

    def _run_async_safely(self, coroutine):
        """Run async coroutine safely, handling existing event loops"""
        try:
            # Try to get the running event loop
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, we can use asyncio.run()
            return asyncio.run(coroutine)
        else:
            # There is a running event loop, we need to run in a thread
            import threading

            result = [None]
            exception = [None]

            def run_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result[0] = new_loop.run_until_complete(coroutine)
                    new_loop.close()
                except Exception as e:
                    exception[0] = e
                finally:
                    asyncio.set_event_loop(None)

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception[0]:
                raise exception[0]
            return result[0]

    def _validate_security_settings(self) -> None:
        """
        Validate tools and flags against security allowlists to prevent command injection.

        Raises:
            ValueError: If any disallowed tools or flags are configured
        """
        # Validate tools
        for tool in self.tools:
            # Allow MCP tools (they start with mcp__)
            if tool.startswith("mcp__"):
                continue
            if tool not in ALLOWED_TOOLS:
                raise ValueError(f"Security violation: Disallowed tool configured: '{tool}'. "
                               f"Allowed tools: {sorted(ALLOWED_TOOLS)}")

        # Validate extra flags
        for flag in self.extra_flags:
            # Extract flag name (remove values like "--temperature 0.5" -> "--temperature")
            flag_name = flag.split('=')[0].split()[0] if flag else ""
            if flag_name not in ALLOWED_FLAGS:
                raise ValueError(f"Security violation: Disallowed flag configured: '{flag}'. "
                               f"Allowed flags: {sorted(ALLOWED_FLAGS)}")

        logger.debug(f"Security validation passed for tools: {self.tools}, flags: {self.extra_flags}")

    def execute(self, prompt: str, context: Optional[Dict[str, Any]] = None,
                continue_session: bool = False) -> AgentResult:
        """
        Execute the agent with the given prompt and context.

        Args:
            prompt: The main prompt/task for the agent
            context: Optional context dictionary
            continue_session: Whether to continue from a previous session

        Returns:
            AgentResult with execution details
        """
        start_time = time.time()
        context = context or {}

        # Use the prompt as-is (no custom prefix needed)
        full_prompt = prompt

        # Log start with consistent formatting (matching supervisor style)
        continue_status = "with --continue" if continue_session else "without --continue"
        logger.info("=" * 60)
        logger.info(f"AGENT STARTING... [{continue_status}]")
        logger.info("=" * 60)

        try:
            # Execute the agent (no retry loop)
            result = self._run_agent(
                full_prompt,
                continue_session=continue_session
            )

            return result

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            execution_time = time.time() - start_time

            # Create error result
            return AgentResult(
                success=False,
                session_id=self.session_id,
                output_file="",
                error_file="",
                execution_time=execution_time,
                errors=[f"Agent execution failed: {str(e)}"],
                retry_count=0,
            )

    def _run_agent(self, prompt: str, continue_session: bool = False) -> AgentResult:
        """
        Run the actual agent command using async streaming (same as supervisor).

        Args:
            prompt: The prompt to execute
            continue_session: Whether to use --continue flag

        Returns:
            AgentResult with execution details
        """
        # Set up file paths
        prompt_file = self.working_dir / f"prompt_{self.session_id}.txt"
        output_file = self.working_dir / f"output_{self.session_id}.log"
        error_file = self.working_dir / f"error_{self.session_id}.log"
        debug_file = self.working_dir / f"debug_{self.session_id}.log"

        # Write prompt to file
        prompt_file.write_text(prompt)
        logger.debug(f"Wrote prompt to {prompt_file} ({len(prompt)} chars)")

        # Validate security settings before building command
        self._validate_security_settings()

        # Build command
        cmd = self._build_command(prompt_file, continue_session)

        # Execute using async streaming (same as supervisor)
        try:
            start_time = time.time()

            # Save debug info to file
            debug_content = f"Command executed: {' '.join(cmd)}\n"
            debug_content += f"Working directory: {str(self.working_dir)}\n"
            debug_content += f"Session ID: {self.session_id}\n"
            debug_content += f"Model: {self.model}\n"
            debug_content += f"Temperature: {self.temperature}\n"
            debug_content += f"Tools: {self.tools}\n"
            debug_content += f"Extra flags: {self.extra_flags}\n"
            debug_content += f"Continue session: {continue_session}\n"
            debug_content += f"Prompt file: {prompt_file}\n"
            debug_content += f"Prompt length: {len(prompt)} characters\n"
            debug_content += f"--- COMMAND DETAILS ---\n"
            for i, arg in enumerate(cmd):
                debug_content += f"  [{i}]: {arg}\n"
            debug_file.write_text(debug_content)

            # Run async subprocess with realtime output (same as supervisor)
            logger.debug(f"Executing command: {' '.join(cmd[:10])}..." if len(cmd) > 10 else "Executing command: " + ' '.join(cmd))
            logger.debug(f"Working directory: {self.working_dir}")
            logger.debug(f"Command length: {len(cmd)} args, total chars: {sum(len(arg) for arg in cmd)}")

            # Use async streaming execution
            returncode, all_output = self._run_async_safely(
                self._run_claude_with_realtime_output(cmd, str(self.working_dir), "AGENT")
            )

            execution_time = time.time() - start_time

            # Convert output to string for file writing
            final_output = '\n'.join(all_output) if all_output else ""

            # Save output
            output_file.write_text(final_output)

            # Save agent output to log file
            log_dir = os.environ.get("CADENCE_LOG_DIR")
            session_id = os.environ.get("CADENCE_LOG_SESSION", self.session_id)
            if log_dir and final_output:
                try:
                    agent_log_file = Path(log_dir) / session_id / "agent.log"
                    agent_log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(agent_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"Agent run at {datetime.now().isoformat()}\n")
                        f.write(f"{'='*60}\n")
                        f.write(final_output)
                        f.write('\n')
                    logger.debug(f"Saved agent output to {agent_log_file}")
                except Exception as e:
                    logger.warning(f"Failed to save agent output: {e}")

            # Append execution results to debug file
            with open(debug_file, 'a') as f:
                f.write(f"\n--- EXECUTION RESULTS ---\n")
                f.write(f"Exit code: {returncode}\n")
                f.write(f"Execution time: {execution_time:.2f}s\n")
                f.write(f"Output size: {len(final_output)} bytes\n")
                f.write(f"Output lines: {len(all_output)}\n")

            # Parse output for structured results
            json_monitor = SimpleJSONStreamMonitor()
            for line in all_output:
                json_monitor.process_line(line)

            json_result = json_monitor.get_last_json_object()

            # Check for quick quit
            quit_too_quickly = execution_time < 10.0

            # Determine success and completion status
            if json_result and isinstance(json_result, dict):
                # Check for final result object (type="result")
                if json_result.get('type') == 'result':
                    subtype = json_result.get('subtype', '')
                    is_error = json_result.get('is_error', False)
                    success = subtype == 'success' and not is_error
                    completed_normally = (subtype == 'success')
                    requested_help = False  # Final result objects don't indicate help requests
                else:
                    # Check for agent status response (with status field)
                    status = json_result.get('status', '')
                    success = status == 'success'
                    requested_help = status == 'help_needed'
                    completed_normally = success
            else:
                success = returncode == 0
                requested_help = False
                completed_normally = success  # If no JSON, assume normal completion if successful

            # Log file paths if there's an error
            if not success:
                logger.error(f"Agent execution failed")
                logger.error(f"Output log: {output_file.absolute()}")
                logger.error(f"Error log: {error_file.absolute()}")
                if all_output:
                    logger.error(f"Last few output lines: {all_output[-5:]}")

            return AgentResult(
                success=success,
                session_id=self.session_id,
                output_file=str(output_file),
                error_file=str(error_file),
                execution_time=execution_time,
                completed_normally=completed_normally,
                requested_help=requested_help,
                errors=[] if success else ["Agent reported failure"],
                quit_too_quickly=quit_too_quickly
            )

        except Exception as e:
            logger.error(f"Unexpected error during agent execution: {e}")
            execution_time = time.time() - start_time

            # Write error info
            error_file.write_text(str(e))

            # Append to debug file
            with open(debug_file, 'a') as f:
                f.write(f"\n--- EXECUTION ERROR ---\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Execution time: {execution_time:.2f}s\n")

            logger.error(f"Output log: {output_file.absolute()}")
            logger.error(f"Error log: {error_file.absolute()}")

            return AgentResult(
                success=False,
                session_id=self.session_id,
                output_file=str(output_file),
                error_file=str(error_file),
                execution_time=execution_time,
                completed_normally=False,
                errors=[str(e)]
            )

    def _build_command(self, prompt_file: Path, continue_session: bool) -> List[str]:
        """Build the claude command with appropriate flags."""
        cmd = ["claude"]

        # Read prompt from file to pass as argument
        try:
            prompt_content = prompt_file.read_text(encoding='utf-8').strip()
            if not prompt_content:
                raise ValueError(f"Prompt file {prompt_file} is empty")
            logger.debug(f"Read prompt file {prompt_file}, size: {len(prompt_content)} chars")
        except Exception as e:
            logger.error(f"Failed to read prompt file {prompt_file}: {e}")
            raise

        # Add prompt using -p flag (like supervisor does)
        cmd.append("-p")
        cmd.append(prompt_content)

        # Add model
        cmd.extend(["--model", self.model])

        # Add temperature if not default
        if self.temperature != 0.2:
            cmd.extend(["--temperature", str(self.temperature)])

        # Add tools using --allowedTools flag
        if self.tools:
            cmd.extend(["--allowedTools", ",".join(self.tools)])

        # Add critical flags to match supervisor behavior
        cmd.extend([
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--verbose"
        ])

        # Add max-turns flag - get from settings without fallback
        max_turns = self.settings.get("max_turns")
        if max_turns is not None:
            cmd.extend(["--max-turns", str(max_turns)])

        # Add extra flags
        cmd.extend(self.extra_flags)

        # Add continue flag if needed
        if continue_session:
            cmd.append("--continue")

        logger.debug(f"Built command with {len(cmd)} args, prompt size: {len(prompt_content)} chars")
        return cmd
