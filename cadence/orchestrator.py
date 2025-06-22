"""
Orchestrator for Claude Cadence Supervisor-Agent Architecture

This module manages the coordination between supervisor and agent,
ensuring they operate in separate directories and maintain proper state.
"""

import os
import json
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
import shutil
import logging
import threading
import asyncio

from .constants import OrchestratorDefaults, FilePatterns, AgentPromptDefaults
from .prompt_utils import PromptBuilder
from .utils import generate_session_id
from .config import ZenIntegrationConfig
from .prompts import YAMLPromptLoader

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SupervisorDecision:
    """Decision made by supervisor analysis"""
    action: str  # "execute", "skip", "complete"
    todos: List[str] = None  # For backward compatibility
    task_id: str = ""
    task_title: str = ""
    subtasks: List[Dict] = None  # List of subtask dicts with id, title, description
    project_root: str = ""
    guidance: str = ""
    session_id: str = ""
    reason: str = ""
    zen_needed: Optional[Dict] = None
    execution_time: float = 0.0  # Time taken to make decision
    quit_too_quickly: bool = False  # True if supervisor quit in < configured seconds



@dataclass
class AgentResult:
    """Result from agent execution"""
    success: bool
    session_id: str
    output_file: str
    error_file: str
    execution_time: float
    completed_normally: bool = False
    requested_help: bool = False
    errors: List[str] = None
    quit_too_quickly: bool = False  # True if agent quit in < 10 seconds



class SupervisorOrchestrator:
    """Orchestrates between supervisor and agent in separate directories"""
    
    def __init__(self, project_root: Path, task_file: Optional[Path] = None, config: Optional[Dict] = None):
        self.project_root = Path(project_root).resolve()
        
        # Default to .taskmaster/tasks/tasks.json if not specified
        if task_file is None:
            task_file = self.project_root / ".taskmaster" / "tasks" / "tasks.json"
        
        self.task_file = Path(task_file).resolve()
        self.supervisor_dir = self.project_root / ".cadence" / "supervisor"
        self.agent_dir = self.project_root / ".cadence" / "agent"
        self.state_file = self.project_root / ".cadence" / "orchestrator_state.json"
        self.config = config or {}
        
        # Load prompts from YAML
        prompts_file = Path(__file__).parent / "prompts.yaml"
        self.prompt_loader = YAMLPromptLoader(prompts_file)
        
        # Create directories
        self.supervisor_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize state
        self.state = self.load_state()
        
        # Session tracking
        self.current_session_id = None
    
    def cleanup_completion_marker(self):
        """Remove any existing completion marker from previous runs"""
        completion_marker = self.project_root / ".cadence" / "project_complete.marker"
        if completion_marker.exists():
            try:
                completion_marker.unlink()
                logger.info("Removed previous completion marker")
            except Exception as e:
                logger.warning(f"Failed to remove completion marker: {e}")

    async def run_claude_with_realtime_output(self, cmd: List[str], cwd: str, process_name: str) -> tuple[int, List[str]]:
        """Run claude command with real-time output display"""
        # Set up environment 
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        print(f"\n{process_name} working...")
        print("-" * 50)
        
        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env
        )
        
        all_output = []
        line_count = 0
        
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
                
                # Parse and display JSON
                try:
                    json_data = json.loads(line_str)
                    msg_type = json_data.get('type', 'unknown')
                    
                    if msg_type == 'assistant':
                        message = json_data.get('message', {})
                        content = message.get('content', [])
                        
                        for item in content:
                            if item.get('type') == 'text':
                                text = item.get('text', '').strip()
                                if text:
                                    print(f"[{process_name}] {text}")
                            elif item.get('type') == 'tool_use':
                                tool_name = item.get('name', 'unknown')
                                tool_input = item.get('input', {})
                                command = tool_input.get('command', '')
                                print(f"[{process_name}] 🛠️  {tool_name}: {command}")
                    
                    elif msg_type == 'user':
                        # Tool results
                        message = json_data.get('message', {})
                        content = message.get('content', [])
                        for item in content:
                            if item.get('type') == 'tool_result':
                                result_content = str(item.get('content', ''))
                                if len(result_content) > 200:
                                    result_content = result_content[:200] + "..."
                                print(f"[{process_name}] 📋 Result: {result_content}")
                    
                    elif msg_type == 'result':
                        # Final result
                        duration = json_data.get('duration_ms', 0)
                        cost = json_data.get('total_cost_usd', 0)
                        print(f"[{process_name}] ✅ Completed in {duration}ms, cost: ${cost:.4f}")
                        
                except json.JSONDecodeError:
                    # Not JSON, display as plain text
                    print(f"[{process_name}] {line_str}")
                    
            except Exception as e:
                print(f"[{process_name}] Error reading output: {e}")
                break
        
        # Wait for process to complete
        await process.wait()
        
        print("-" * 50)
        print(f"{process_name} completed with return code {process.returncode}")
        
        return process.returncode, all_output
    
    def format_stream_json_line(self, line_str: str) -> str:
        """Parse and format a stream-json line for human-readable display"""
        try:
            data = json.loads(line_str)
            msg_type = data.get('type', 'unknown')
            
            # Handle different message types
            if msg_type == 'system':
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
                "first_run": True,
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
        is_first_run = self.state["first_run"]
        
        logger.info("="*60)
        logger.info("Starting Claude Cadence Orchestration")
        logger.info(f"Task file: {self.task_file}")
        logger.info(f"Session ID: {self.current_session_id}")
        logger.info(f"First run: {is_first_run}")
        logger.info("="*60)
        
        # Clean up any previous completion marker
        self.cleanup_completion_marker()
        
        # Update state
        self.state["first_run"] = False
        self.state["session_count"] += 1
        self.state["last_session_id"] = self.current_session_id
        self.save_state()
        
        # Track if this is first iteration of current session
        first_iteration = True
        max_iterations = self.config.get("max_iterations", OrchestratorDefaults.MAX_ITERATIONS)
        iteration = 0
        
        # Define completion marker file
        completion_marker = self.project_root / ".cadence" / "project_complete.marker"
        
        while iteration < max_iterations:
            iteration += 1
            logger.info("-"*50)
            logger.info(f"Iteration {iteration}")
            logger.info("-"*50)
            
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
            decision = self.run_supervisor_analysis(
                self.current_session_id, 
                use_continue=not is_first_run and not first_iteration
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
                    # Continue to let supervisor create the marker
                    continue
            elif decision.action == "skip":
                logger.info(f"Skipping: {decision.reason}")
                continue
            elif decision.action == "execute":
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
                        "project_root": decision.project_root,
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
                
                # 3. Run agent with supervisor's TODOs
                agent_result = self.run_agent(
                    todos=todos,
                    guidance=decision.guidance,
                    session_id=self.current_session_id,
                    use_continue=not is_first_run and not first_iteration,
                    task_id=decision.task_id,
                    subtasks=decision.subtasks,
                    project_root=decision.project_root
                )
                
                # Check if agent quit too quickly (likely an error)
                if agent_result.quit_too_quickly:
                    logger.error(f"Agent quit too quickly ({agent_result.execution_time:.2f}s) - likely an error condition")
                    logger.error("Exiting program due to quick agent exit")
                    return False
                
                # 4. Save agent results for supervisor
                self.save_agent_results(agent_result, self.current_session_id, decision.todos, decision.task_id)
                
                # Check if agent requested help
                if agent_result.requested_help:
                    logger.warning("Agent requested help - supervisor will provide assistance")
                
                # No longer first iteration
                first_iteration = False
                
                # 5. Continue to next iteration
                continue
            elif decision.action == "use_mcp":
                # This is the old analyze_and_decide response - supervisor should have used MCP
                logger.error("Supervisor returned 'use_mcp' action - it should have used MCP tools to analyze and decide")
                logger.error("The supervisor prompt needs to output a proper JSON decision")
                return False
            else:
                logger.error(f"Unknown supervisor action: {decision.action}")
                return False
        
        logger.warning(f"Reached maximum iterations ({max_iterations})")
        return False

    
    def run_supervisor_analysis(self, session_id: str, use_continue: bool) -> SupervisorDecision:
            """Run supervisor in its directory to analyze state"""
            original_dir = os.getcwd()
            max_json_retries = 5
            json_retry_count = 0
            
            try:
                # Change to supervisor directory
                os.chdir(self.supervisor_dir)
                
                while json_retry_count < max_json_retries:
                    # Build supervisor command using Claude CLI for MCP access
                    supervisor_script = self.project_root / "cadence" / "supervisor_cli.py"
                    config_file = self.project_root / "config.yaml"
                    
                    # Get code review config
                    zen_config = self.config.get("zen_integration", {})
                    code_review_frequency = zen_config.get("code_review_frequency", "task")
                    
                    # Check if we have previous agent results
                    agent_results_file = self.supervisor_dir / FilePatterns.AGENT_RESULT_FILE.format(session_id=session_id)
                    previous_agent_result = None
                    if agent_results_file.exists():
                        try:
                            with open(agent_results_file, 'r') as f:
                                previous_agent_result = json.load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load previous agent results: {e}")
                    
                    # Build prompt using YAML templates
                    context = {
                        "project_root": str(self.project_root),
                        "session_id": session_id,
                        "has_previous_agent_result": previous_agent_result is not None,
                        "agent_success": previous_agent_result.get("success", False) if previous_agent_result else False,
                        "agent_completed_normally": previous_agent_result.get("completed_normally", False) if previous_agent_result else False,
                        "agent_todos": previous_agent_result.get("todos", []) if previous_agent_result else [],
                        "agent_task_id": previous_agent_result.get("task_id", "") if previous_agent_result else ""
                    }
                    
                    # Get base prompt
                    base_prompt = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.base_prompt")
                    base_prompt = self.prompt_loader.format_template(base_prompt, context)
                    
                    # Get code review section based on config
                    code_review_key = f"supervisor_prompts.orchestrator_taskmaster.code_review_sections.{code_review_frequency}"
                    code_review_section = self.prompt_loader.get_template(code_review_key)
                    code_review_section = self.prompt_loader.format_template(code_review_section, context)
                    
                    # Get zen guidance
                    zen_guidance = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.zen_guidance")
                    
                    # Get output format
                    output_format = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.output_format")
                    output_format = self.prompt_loader.format_template(output_format, context)
                    
                    # If this is a retry, add JSON formatting reminder
                    if json_retry_count > 0:
                        json_retry_prompt = f"""
                        
                        CRITICAL: Your previous output had invalid JSON formatting. 
                        Please output ONLY a valid JSON object with NO other text before or after.
                        The JSON must have these exact fields: action, todos, guidance, task_id, session_id, reason
                        
                        Retry attempt {json_retry_count + 1} of {max_json_retries}.
                        """
                        supervisor_prompt = f"{base_prompt}{code_review_section}{zen_guidance}{output_format}{json_retry_prompt}"
                    else:
                        # Combine all sections normally
                        supervisor_prompt = f"{base_prompt}{code_review_section}{zen_guidance}{output_format}"
                    
                    # Build allowed tools from config
                    basic_tools = self.config.get("supervisor", {}).get("tools", [
                        "bash", "read", "write", "edit", "grep", "glob", "search", "WebFetch"
                    ])
                    mcp_servers = self.config.get("integrations", {}).get("mcp", {}).get("supervisor_servers", [
                        "taskmaster-ai", "zen", "serena", "Context7"
                    ])
                    # Add mcp__ prefix and * suffix to each MCP server
                    mcp_tools = [f"mcp__{server}__*" for server in mcp_servers]
                    all_tools = basic_tools + mcp_tools
                    
                    cmd = [
                        "claude",
                        "-p", supervisor_prompt,
                        "--allowedTools", ",".join(all_tools),
                        "--max-turns", "10",
                        "--output-format", "stream-json",
                        "--verbose",
                        "--dangerously-skip-permissions"  # Skip permission prompts
                    ]
                    
                    # Add --continue flag if not first run OR if this is a retry
                    if use_continue or json_retry_count > 0:
                        cmd.append("--continue")
                        if json_retry_count > 0:
                            logger.warning(f"Retrying supervisor with --continue due to JSON parsing error (attempt {json_retry_count + 1})")
                        else:
                            logger.debug("Running supervisor with --continue flag")
                    else:
                        logger.debug("Running supervisor (first run)")
                    
                    
                    # Run supervisor
                    logger.debug(f"Command: {' '.join(cmd)}")
                    logger.info("=" * 50)
                    logger.info("SUPERVISOR STARTING...")
                    if json_retry_count > 0:
                        logger.info(f"JSON RETRY ATTEMPT {json_retry_count + 1} of {max_json_retries}")
                    logger.info("=" * 50)
                    
                    # Run supervisor with real-time output
                    supervisor_start_time = time.time()
                    try:
                        # Run supervisor async for real-time output
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            returncode, all_output = loop.run_until_complete(
                                self.run_claude_with_realtime_output(cmd, self.supervisor_dir, "SUPERVISOR")
                            )
                        finally:
                            loop.close()
                        
                        if returncode != 0:
                            logger.error(f"Supervisor failed with code {returncode}")
                            raise RuntimeError(f"Supervisor failed with exit code {returncode}")
                        
                    except Exception as e:
                        logger.error(f"Supervisor execution error: {e}")
                        raise RuntimeError(f"Supervisor execution failed: {e}")
                    
                    logger.info("=" * 50)
                    logger.info("SUPERVISOR COMPLETED")
                    logger.info("=" * 50)
                    
                    # Parse supervisor JSON output
                    try:
                        logger.debug(f"Parsing supervisor output...")
                        
                        # Look for the decision JSON in assistant messages
                        json_str = None
                        
                        # Try improved JSON extraction using regex
                        import re
                        json_pattern = re.compile(r'\{[^{}]*"action"[^{}]*\}', re.DOTALL)
                        
                        # Process each line looking for our decision JSON
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
                                                # Look for JSON using regex
                                                matches = json_pattern.findall(text)
                                                for match in matches:
                                                    try:
                                                        test_data = json.loads(match)
                                                        # Verify it has action field at minimum
                                                        if 'action' in test_data:
                                                            json_str = match
                                                            break
                                                    except json.JSONDecodeError:
                                                        continue
                            except json.JSONDecodeError:
                                # This line wasn't JSON, skip it
                                continue
                            
                            if json_str:
                                break
                        
                        if not json_str:
                            # Fallback to original method if regex fails
                            for line_str in all_output:
                                try:
                                    data = json.loads(line_str)
                                    if data.get('type') == 'assistant':
                                        content = data.get('message', {}).get('content', [])
                                        if isinstance(content, list):
                                            for item in content:
                                                if isinstance(item, dict) and item.get('type') == 'text':
                                                    text = item.get('text', '')
                                                    if '{' in text and '}' in text:
                                                        json_start = text.find('{')
                                                        json_end = text.rfind('}') + 1
                                                        potential_json = text[json_start:json_end]
                                                        try:
                                                            test_data = json.loads(potential_json)
                                                            if 'action' in test_data:
                                                                json_str = potential_json
                                                                break
                                                        except json.JSONDecodeError:
                                                            continue
                                except json.JSONDecodeError:
                                    continue
                                
                                if json_str:
                                    break
                        
                        if not json_str:
                            raise ValueError("No decision JSON found in supervisor output")
                        
                        decision_data = json.loads(json_str)
                        
                        # Validate required fields based on action
                        if decision_data.get('action') == 'execute':
                            # For execute action, check for new subtasks structure
                            if 'subtasks' in decision_data:
                                required_fields = ['action', 'task_id', 'task_title', 'subtasks', 'project_root', 'session_id', 'reason']
                            else:
                                # Backward compatibility with todos
                                required_fields = ['action', 'todos', 'guidance', 'task_id', 'session_id', 'reason']
                        else:
                            # For skip/complete actions
                            required_fields = ['action', 'session_id', 'reason']
                        
                        missing_fields = [field for field in required_fields if field not in decision_data]
                        if missing_fields:
                            raise ValueError(f"Decision JSON missing required fields: {missing_fields}")
                        
                        # Success! Reset retry counter and break out of retry loop
                        json_retry_count = 0
                        break
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        json_retry_count += 1
                        logger.error(f"Failed to parse supervisor output as JSON: {e}")
                        
                        if json_retry_count >= max_json_retries:
                            logger.error(f"Failed to get valid JSON after {max_json_retries} attempts. Giving up.")
                            # Show last few lines of output for debugging
                            if all_output:
                                logger.error("Last 5 lines of output:")
                                for line in all_output[-5:]:
                                    logger.error(f"  {line.strip()}")
                            raise RuntimeError(f"Supervisor failed to produce valid JSON after {max_json_retries} attempts")
                        
                        logger.warning(f"Will retry supervisor to get valid JSON output...")
                        # Continue to next iteration of retry loop
                        continue
                
                # Calculate execution time
                supervisor_execution_time = time.time() - supervisor_start_time
                quick_quit_threshold = self.config.get("orchestration", {}).get("quick_quit_seconds", OrchestratorDefaults.QUICK_QUIT_SECONDS)
                
                # Create decision object
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





    
    def run_agent(self, todos: List[str], guidance: str, 
                      session_id: str, use_continue: bool,
                      task_id: str = None, subtasks: List[Dict] = None,
                      project_root: str = None) -> AgentResult:
            """Run agent in dedicated agent directory"""
            original_dir = os.getcwd()
            
            try:
                # Change to agent directory
                os.chdir(self.agent_dir)
                
                # Create prompt with TODOs
                prompt = self.build_agent_prompt(todos, guidance, task_id, subtasks, project_root)
                
                # Save prompt for debugging
                prompt_file = self.validate_path(
                    self.agent_dir / FilePatterns.PROMPT_FILE.format(session_id=session_id), 
                    self.agent_dir
                )
                with open(prompt_file, 'w') as f:
                    f.write(prompt)
                
                # Build claude command
                cmd = ["claude"]
                
                # Add --continue flag if not first run
                if use_continue:
                    cmd.extend(["-c", "-p", prompt])
                    logger.debug("Running agent with --continue flag")
                else:
                    cmd.extend(["-p", prompt])
                    logger.debug("Running agent (first run)")
                    
                cmd.extend([
                    "--max-turns", str(self.config.get("max_turns", OrchestratorDefaults.MAX_AGENT_TURNS)),
                    "--output-format", "stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions"  # Skip permission prompts
                ])
                
                # Build allowed tools from config
                basic_tools = self.config.get("agent", {}).get("tools", [
                    "bash", "read", "write", "edit", "grep", "glob", "search",
                    "todo_read", "todo_write", "WebFetch"
                ])
                
                # Get MCP servers from config
                mcp_servers = self.config.get("integrations", {}).get("mcp", {}).get("agent_servers", [
                    "serena", "Context7"
                ])
                # Add mcp__ prefix and * suffix to each MCP server
                mcp_tools = [f"mcp__{server}__*" for server in mcp_servers]
                all_tools = basic_tools + mcp_tools
                
                cmd.extend(["--allowedTools", ",".join(all_tools)])
                
                # Output files
                output_file = self.validate_path(
                    self.agent_dir / FilePatterns.OUTPUT_FILE.format(session_id=session_id),
                    self.agent_dir
                )
                error_file = self.validate_path(
                    self.agent_dir / FilePatterns.ERROR_FILE.format(session_id=session_id),
                    self.agent_dir
                )
                
                logger.debug(f"Command: {' '.join(cmd[:3])}...")  # Don't log full prompt
                logger.debug(f"Working directory: {os.getcwd()}")
                
                logger.info("=" * 50)
                logger.info("AGENT STARTING...")
                logger.info(f"Working on {len(todos)} TODOs")
                logger.info("=" * 50)
                
                # Run agent with real-time output
                start_time = time.time()
                try:
                    # Run agent async for real-time output
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        returncode, all_output = loop.run_until_complete(
                            self.run_claude_with_realtime_output(cmd, self.agent_dir, "AGENT")
                        )
                    finally:
                        loop.close()
                    
                    # Save output to file
                    with open(output_file, 'w') as out:
                        for line in all_output:
                            out.write(line + '\n')
                    
                    # Write empty error file
                    with open(error_file, 'w') as err:
                        err.write("")
                        
                except Exception as e:
                    logger.error(f"Agent execution error: {e}")
                    return AgentResult(
                        success=False,
                        session_id=session_id,
                        output_file=str(output_file),
                        error_file=str(error_file),
                        execution_time=time.time() - start_time,
                        completed_normally=False,
                        requested_help=False,
                        errors=[f"Agent execution failed: {e}"]
                    )
                
                execution_time = time.time() - start_time
                
                logger.info("=" * 50)
                logger.info(f"AGENT COMPLETED in {execution_time:.2f}s")
                logger.info("=" * 50)
                
                # Analyze results
                completed_normally = False
                requested_help = False
                errors = []
                
                # Check output for completion signals
                output_text = '\n'.join(all_output)
                completed_normally = AgentPromptDefaults.COMPLETION_SIGNAL.upper() in output_text.upper()
                requested_help = AgentPromptDefaults.HELP_SIGNAL.upper() in output_text.upper()
                
                agent_result = AgentResult(
                    success=returncode == 0,
                    session_id=session_id,
                    output_file=str(output_file),
                    error_file=str(error_file),
                    execution_time=execution_time,
                    completed_normally=completed_normally,
                    requested_help=requested_help,
                    errors=errors,
                    quit_too_quickly=execution_time < self.config.get("orchestration", {}).get("quick_quit_seconds", OrchestratorDefaults.QUICK_QUIT_SECONDS)
                )
                
                logger.info(f"Agent execution complete in {execution_time:.2f}s")
                logger.info(f"   Success: {agent_result.success}")
                logger.info(f"   Completed normally: {agent_result.completed_normally}")
                logger.info(f"   Requested help: {agent_result.requested_help}")
                
                return agent_result
                
            finally:
                # Always return to original directory
                os.chdir(original_dir)

    
    def build_agent_prompt(self, todos: List[str], guidance: str, 
                          task_id: str = None, subtasks: List[Dict] = None, 
                          project_root: str = None) -> str:
        """Build prompt for agent with TODOs and guidance"""
        # Check if guidance includes Zen assistance
        if "zen assistance" in guidance.lower() or "expert assistance" in guidance.lower():
            zen_reminder = self.prompt_loader.get_template("agent_zen_reminder")
            guidance = guidance + zen_reminder
            
        return PromptBuilder.build_agent_prompt(
            todos=todos,
            guidance=guidance,
            max_turns=self.config.get("max_turns", OrchestratorDefaults.MAX_AGENT_TURNS),
            task_id=task_id,
            subtasks=subtasks,
            project_root=project_root
        )


    
    def save_agent_results(self, agent_result: AgentResult, session_id: str, todos: List[str] = None, task_id: str = None):
            """Save agent results for supervisor to analyze"""
            results_file = self.validate_path(
                self.supervisor_dir / FilePatterns.AGENT_RESULT_FILE.format(session_id=session_id),
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


    
    def cleanup_old_sessions(self, keep_last_n: int = 5):
        """Clean up old session files to save space"""
        # Get all session files from both directories
        session_files = []
        
        # Collect files from supervisor directory
        for pattern in ["decision_*.json", "agent_result_*.json"]:
            for file in self.supervisor_dir.glob(pattern):
                session_files.append(file)
                
        # Collect files from agent directory  
        for pattern in ["prompt_*.txt", "output_*.log", "error_*.log"]:
            for file in self.agent_dir.glob(pattern):
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
                
        # Sort sessions by timestamp (newest first)
        sorted_sessions = sorted(sessions.keys(), reverse=True)
        
        # Keep only the most recent sessions
        sessions_to_remove = sorted_sessions[keep_last_n:]
        
        # Remove old session files
        removed_count = 0
        for session_id in sessions_to_remove:
            for file in sessions[session_id]:
                try:
                    file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old session file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")
                    
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old session files")