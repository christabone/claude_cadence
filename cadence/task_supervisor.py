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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
from .prompts import TodoPromptManager
from .task_manager import TaskManager, Task
from .config import ConfigLoader, CadenceConfig
from .zen_integration import ZenIntegration


@dataclass
class ExecutionResult:
    """Results from an agent execution"""
    success: bool
    turns_used: int
    output_lines: List[str]
    errors: List[str]
    metadata: Dict[str, any]
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
                    while True:
                        # Check timeout
                        if time.time() - start_time > self.timeout:
                            process.terminate()
                            raise subprocess.TimeoutExpired(cmd, self.timeout)
                            
                        # Try to read from stdout queue
                        try:
                            line = stdout_queue.get_nowait()
                        except queue.Empty:
                            # No new stdout line, check stderr
                            try:
                                err_line = stderr_queue.get_nowait()
                                stderr_lines.append(err_line.rstrip())
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
                                    # Count turns from metadata
                                    if 'metadata' in data and 'turn' in data['metadata']:
                                        turns_used = max(turns_used, data['metadata']['turn'])
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
                
            # Basic turn estimation if not found in JSON
            if turns_used == 0:
                # Look for turn indicators in output
                for line in output_lines:
                    if "Turn " in line or "turn " in line:
                        turns_used += 1
                        
                # Final fallback estimation
                if turns_used == 0:
                    turns_used = min(self.max_turns, max(1, len(output_lines) // 50))
                    
            success = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            success = False
            errors.append(f"Agent execution timed out after {self.timeout} seconds")
            turns_used = self.max_turns
            
        except Exception as e:
            success = False
            errors.append(f"Unexpected error: {str(e)}")
            
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
        
        # Check if zen assistance is needed
        zen_check = self.zen.should_call_zen(result, self.prompt_manager.context, session_id)
        if zen_check:
            tool, reason = zen_check
            if self.verbose:
                print(f"\n🔮 Zen assistance needed: {tool} - {reason}")
            
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
        
        return {
            'zen_response': zen_response,
            'continuation_guidance': continuation_guidance,
            'zen_tool_used': tool,
            'zen_reason': reason
        }