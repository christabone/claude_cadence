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


@dataclass
class ExecutionResult:
    """Results from an agent execution"""
    success: bool
    turns_used: int
    output_lines: List[str]
    errors: List[str]
    metadata: Dict[str, Any]
    task_complete: bool


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
        
        if not scratchpad_path.exists():
            self.logger.warning(f"Scratchpad not found: {scratchpad_path}")
            return {"exists": False}
            
        try:
            with open(scratchpad_path, 'r') as f:
                content = f.read()
                
            # Check for key indicators
            has_help_needed = "HELP NEEDED" in content and "Status: STUCK" in content
            has_all_complete = "ALL TASKS COMPLETE" in content
            
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
            f.write(f"\n## Final Summary\n")
            f.write(f"- Total Executions: {len(all_results)}\n")
            f.write(f"- Total Turns Used: {total_turns}\n")
            f.write(f"- Total Time: {total_time:.2f}s\n")
            f.write(f"- Final Status: {'✅ Complete' if any_complete else '❌ Incomplete'}\n")
            f.write(f"- Completed: {datetime.now().isoformat()}\n")
        
    def execute_with_todos(self, todos: List[str], task_list: Optional[List[Dict]] = None, 
                          continuation_context: Optional[str] = None, session_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute agent with a list of TODOs
        
        Args:
            todos: List of TODO items for the agent to complete
            task_list: Optional list of tasks from Task Master for tracking
            continuation_context: Optional context from previous execution
            
        Returns:
            ExecutionResult with execution details
        """
        # Create output log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"execution_{timestamp}.log"
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
            
        # Initialize supervisor log for this session
        self._init_supervisor_log(session_id)
        self._log_supervisor(f"Starting execution with {len(todos)} TODOs")
        
        # Extract task numbers from task_list if available
        task_numbers = ""
        if task_list:
            task_ids = [str(task.get('id', '?')) for task in task_list if not task.get('status') == 'done']
            task_numbers = ", ".join(task_ids)
        
        # Initialize prompt manager if not already done
        if not hasattr(self, 'prompt_manager'):
            self.prompt_manager = TodoPromptManager(todos, self.max_turns)
            self.prompt_manager.session_id = session_id
            self.prompt_manager.task_numbers = task_numbers
            
        # Log task analysis
        self._log_task_analysis(todos, task_list)
        
        # Build prompt with TODOs
        prompt = self._build_todo_prompt(todos, continuation_context)
        
        # Build claude command
        cmd = ["claude"]
        
        if continuation_context:
            cmd.extend(["-c", "-p", prompt])
        else:
            cmd.extend(["-p", prompt])
            
        cmd.extend([
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--output-format=stream-json"
        ])
        
        # Add tool flags from config
        for tool in self.allowed_tools:
            cmd.extend(["--tool", tool])
            
        # Add extra flags from config
        if self.config.agent.extra_flags:
            cmd.extend(self.config.agent.extra_flags)
        
        if self.verbose:
            print(f"\n🚀 Starting task execution with max {self.max_turns} turns...")
            print(f"📋 TODOs: {len(todos)} items")
            
        # Log execution start
        self._log_execution_start(cmd, continuation=bool(continuation_context))
            
        # Track execution details
        start_time = time.time()
        output_lines = []
        errors = []
        turns_used = 0
        
        try:
            # Run the agent with Popen for stream processing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Open log file
            with open(log_file, 'w') as log:
                log.write(f"Command: {' '.join(cmd)}\n")
                log.write("=== STDOUT ===\n")
                
                # Process stdout stream
                stdout_lines = []
                stderr_lines = []
                
                try:
                    # Read stdout line by line with timeout using thread
                    start_time = time.time()
                    
                    # Create queue for thread-safe line reading
                    stdout_queue = queue.Queue()
                    stderr_queue = queue.Queue()
                    
                    # Thread function to read stdout
                    def enqueue_output(out, queue):
                        for line in iter(out.readline, ''):
                            queue.put(line)
                        out.close()
                    
                    # Start threads for reading stdout and stderr
                    stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
                    stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    # Main reading loop
                    last_status_check = time.time()
                    status_check_interval = 30  # Check every 30 seconds
                    
                    while True:
                        # Check timeout
                        if time.time() - start_time > self.timeout:
                            self.logger.error(f"Timeout reached after {self.timeout}s")
                            process.terminate()
                            raise subprocess.TimeoutExpired(cmd, self.timeout)
                            
                        # Periodic status check
                        if time.time() - last_status_check > status_check_interval:
                            elapsed = time.time() - start_time
                            self.logger.debug(f"Execution in progress: {elapsed:.1f}s elapsed")
                            last_status_check = time.time()
                            
                        # Try to read from stdout queue
                        try:
                            line = stdout_queue.get_nowait()
                        except queue.Empty:
                            # No new stdout line, check stderr
                            try:
                                err_line = stderr_queue.get_nowait()
                                stderr_lines.append(err_line.rstrip())
                                if err_line.strip():
                                    self.logger.debug(f"STDERR: {err_line.rstrip()}")
                            except queue.Empty:
                                # Check if process is done
                                if process.poll() is not None:
                                    break
                                time.sleep(0.05)  # Small sleep to prevent busy-waiting
                                continue
                        else:
                            # Process the stdout line
                            stdout_lines.append(line.rstrip())
                            log.write(line)
                            
                            # Try to parse JSON stream output
                            if line.strip().startswith('{'):
                                try:
                                    data = json.loads(line.strip())
                                    # Extract content from JSON stream
                                    if 'content' in data:
                                        output_lines.append(data['content'])
                                    # Note: We can't track turns in real-time with Claude CLI
                                except json.JSONDecodeError:
                                    # Not JSON, treat as regular output
                                    output_lines.append(line.rstrip())
                            else:
                                output_lines.append(line.rstrip())
                    
                    # Wait for threads to finish and get any remaining output
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    
                    # Drain any remaining items from queues
                    while not stdout_queue.empty():
                        line = stdout_queue.get_nowait()
                        stdout_lines.append(line.rstrip())
                        output_lines.append(line.rstrip())
                        log.write(line)
                        
                    while not stderr_queue.empty():
                        err_line = stderr_queue.get_nowait()
                        stderr_lines.append(err_line.rstrip())
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    raise
                    
                # Write stderr to log
                log.write("\n=== STDERR ===\n")
                for line in stderr_lines:
                    log.write(line + '\n')
                    
                log.write(f"\nExit code: {process.returncode}\n")
                
            # Process any errors
            if stderr_lines:
                errors.extend(stderr_lines)
                
            # We can only estimate turns after execution completes
            # Claude CLI doesn't provide real-time turn info in stream-json format
            if turns_used == 0:
                # Simple estimation based on output - this is not accurate
                # but gives us something to work with
                # Count substantive output lines (more than 50 chars)
                substantive_lines = [line for line in output_lines if len(line) > 50]
                # Estimate based on substantive output
                turns_used = min(self.max_turns, max(1, len(substantive_lines)))
                self.logger.warning(f"Turn count is estimated: {turns_used}")
                    
            success = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            success = False
            errors.append(f"Agent execution timed out after {self.timeout} seconds")
            turns_used = self.max_turns
            self.logger.error(f"Execution timeout after {self.timeout}s")
            
        except Exception as e:
            success = False
            errors.append(f"Unexpected error: {str(e)}")
            self.logger.exception("Unexpected error during agent execution")
            
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Analyze if tasks are complete
        output_text = "\n".join(output_lines)
        task_complete = self._analyze_task_completion(output_text, task_list)
        
        metadata = {
            "execution_time": execution_time,
            "log_file": str(log_file),
            "timestamp": timestamp,
            "model": self.model,
            "todos_count": len(todos),
            "session_id": session_id,
            "scratchpad_path": f".cadence/scratchpad/session_{session_id}.md"
        }
        
        result = ExecutionResult(
            success=success,
            turns_used=turns_used,
            output_lines=output_lines,
            errors=errors,
            metadata=metadata,
            task_complete=task_complete
        )
        
        # Log execution result
        self._log_execution_result(result)
        
        # Check scratchpad status
        scratchpad_status = self._check_scratchpad_status(session_id)
        if scratchpad_status.get("exists"):
            self._log_supervisor(f"Scratchpad status: help_needed={scratchpad_status.get('has_help_needed')}, complete={scratchpad_status.get('has_all_complete')}")
            if scratchpad_status.get('has_help_needed'):
                self._log_supervisor("Agent has requested HELP - marking as stuck", "WARNING")
        
        # Check if zen assistance is needed
        zen_check = self.zen.should_call_zen(result, self.prompt_manager.context, session_id)
        if zen_check:
            tool, reason = zen_check
            if self.verbose:
                print(f"\n🔮 Zen assistance needed: {tool} - {reason}")
            
            # Log zen recommendation
            self._log_zen_recommendation(tool, reason)
            
            # Store zen recommendation in metadata
            result.metadata['zen_needed'] = {
                'tool': tool,
                'reason': reason
            }
        
        return result
        
    def _build_todo_prompt(self, todos: List[str], continuation_context: Optional[str] = None) -> str:
        """Build prompt with TODO list for agent"""
        
        # Use prompt manager if available
        if hasattr(self, 'prompt_manager'):
            if continuation_context:
                # Create supervisor analysis dict
                supervisor_analysis = {
                    'session_id': self.prompt_manager.session_id,
                    'previous_session_id': getattr(self, 'previous_session_id', 'unknown')
                }
                return self.prompt_manager.get_continuation_prompt(
                    analysis_guidance="Continue where you left off. Focus on completing the remaining TODOs.",
                    continuation_context=continuation_context,
                    supervisor_analysis=supervisor_analysis
                )
            else:
                return self.prompt_manager.get_initial_prompt()
        
        # Fallback to simple prompt
        prompt_parts = []
        
        # Add continuation context if provided
        if continuation_context:
            prompt_parts.append(f"=== CONTINUATION CONTEXT ===\n{continuation_context}\n")
        
        # Add execution guidelines
        prompt_parts.append("""=== TASK EXECUTION GUIDELINES ===

You have been given specific TODOs to complete. Focus ONLY on these tasks.

IMPORTANT:
- Work naturally and efficiently to complete all TODOs
- The moment ALL TODOs are complete, declare "ALL TASKS COMPLETE" and exit
- You have up to """ + str(self.max_turns) + """ turns as a safety limit (not a target)
- Quality matters more than speed

""")
        
        # Add TODO list
        prompt_parts.append("=== YOUR TODOS ===")
        for i, todo in enumerate(todos, 1):
            prompt_parts.append(f"{i}. {todo}")
        
        prompt_parts.append("\nBegin working on these TODOs now.")
        
        return "\n".join(prompt_parts)
        
    def _analyze_task_completion(self, output: str, task_list: Optional[List[Dict]]) -> bool:
        """Analyze if tasks are complete"""
        
        # Check for explicit completion declaration
        if "all tasks complete" in output.lower():
            return True
            
        # Use task manager if available
        if task_list:
            try:
                task_manager = TaskManager()
                task_manager.tasks = [Task.from_dict(t) for t in task_list]
                progress = task_manager.analyze_progress(output)
                completed = progress.get('completed', [])
                incomplete = progress.get('incomplete', [])
                
                # All complete if no incomplete tasks
                return len(completed) > 0 and len(incomplete) == 0
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Task analysis failed: {e}")
                    
        return False
        
    def run_with_taskmaster(self, task_file: Optional[str] = None) -> bool:
        """
        Run supervisor with Task Master integration
        
        Args:
            task_file: Path to Task Master tasks.json file
            
        Returns:
            bool: True if all tasks completed successfully
        """
        # Load tasks from Task Master
        task_file = task_file or self.config.integrations["taskmaster"]["default_task_file"]
        
        try:
            task_manager = TaskManager()
            if not task_manager.load_tasks(task_file):
                print(f"❌ Failed to load tasks from {task_file}")
                return False
                
            print(f"📋 Loaded {len(task_manager.tasks)} tasks from Task Master")
            
            # Convert tasks to TODOs
            todos = []
            for task in task_manager.tasks:
                if not task.is_complete():
                    todo = f"Task {task.id}: {task.title}"
                    if task.description:
                        todo += f" - {task.description}"
                    todos.append(todo)
                    
            if not todos:
                print("✅ All tasks are already complete!")
                return True
                
            print(f"🎯 {len(todos)} tasks to complete")
            
            # Initialize prompt manager with todos
            self.prompt_manager = TodoPromptManager(todos, self.max_turns)
            
            # Generate session ID for this execution
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_taskmaster"
            
            # Execute with TODOs
            result = self.execute_with_todos(
                todos=todos,
                task_list=[t.__dict__ for t in task_manager.tasks],
                session_id=session_id
            )
            
            # Save execution history
            self.execution_history.append(result)
            
            # Print summary
            if self.verbose:
                print(f"\n📈 Execution Summary:")
                print(f"   - Success: {result.success}")
                print(f"   - Turns used: {result.turns_used}/{self.max_turns}")
                print(f"   - Tasks complete: {result.task_complete}")
                if result.errors:
                    print(f"   - Errors: {len(result.errors)}")
                    
            # Update task status if configured
            if self.config.integrations["taskmaster"]["auto_update_status"] and result.task_complete:
                # Update all tasks to complete
                for task in task_manager.tasks:
                    task_manager.update_task_status(task.id, "done")
                task_manager.save_tasks(task_file)
                print("✅ Updated task status in Task Master")
                self._log_supervisor("All tasks marked as complete in Task Master")
                
            # Log final summary
            self._log_final_summary(self.execution_history)
            
            # Log supervisor log location
            if self.current_log_path and self.verbose:
                print(f"\n📝 Supervisor log: {self.current_log_path}")
                
            return result.task_complete
            
        except Exception as e:
            print(f"❌ Error running with Task Master: {e}")
            return False
            
    def handle_zen_assistance(self, execution_result: ExecutionResult, 
                            session_id: str) -> Dict[str, Any]:
        """
        Handle zen assistance request from execution result
        
        Args:
            execution_result: The execution result with zen_needed metadata
            session_id: Current session ID
            
        Returns:
            Dict with zen response and continuation guidance
        """
        zen_needed = execution_result.metadata.get('zen_needed')
        if not zen_needed:
            return {}
            
        tool = zen_needed['tool']
        reason = zen_needed['reason']
        
        # Call zen for assistance
        zen_response = self.zen.call_zen_support(
            tool=tool,
            reason=reason,
            execution_result=execution_result,
            context=self.prompt_manager.context,
            session_id=session_id
        )
        
        # Generate continuation guidance
        continuation_guidance = self.zen.generate_continuation_guidance(zen_response)
        
        # Log zen assistance details
        self._log_supervisor(f"Zen {tool} assistance provided", "ZEN")
        if 'guidance' in zen_response:
            self._log_supervisor_analysis(f"### Zen {tool.upper()} Guidance\n{zen_response['guidance']}")
        
        return {
            'zen_response': zen_response,
            'continuation_guidance': continuation_guidance,
            'zen_tool_used': tool,
            'zen_reason': reason
        }