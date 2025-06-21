"""
Core checkpoint supervisor implementation
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
from .prompts import ContextAwarePromptManager
from .task_manager import TaskManager, Task


# Cost estimation constants (per million tokens)
SONNET_3_5_INPUT_COST_PER_MILLION = 3.00
SONNET_3_5_OUTPUT_COST_PER_MILLION = 15.00
# Blended rate assuming 30% input, 70% output
BLENDED_COST_PER_1K_TOKENS = (SONNET_3_5_INPUT_COST_PER_MILLION * 0.3 + 
                              SONNET_3_5_OUTPUT_COST_PER_MILLION * 0.7) / 1000
AVG_TOKENS_PER_TURN = 1000


@dataclass
class CheckpointResult:
    """Results from a checkpoint execution"""
    success: bool
    turns_used: int
    output_lines: List[str]
    cost: float
    errors: List[str]
    metadata: Dict[str, any]


@dataclass
class SupervisorAnalysis:
    """Supervisor's analysis of checkpoint results"""
    on_track: bool
    task_complete: bool
    issues_detected: List[str]
    guidance: str
    confidence: float


class CheckpointSupervisor:
    """
    Manages checkpoint-based supervised agent execution
    
    This class implements the core cadence pattern: spawn, wait, analyze, guide, repeat.
    """
    
    def __init__(self, 
                 checkpoint_turns: int = 15,
                 max_checkpoints: int = 3,
                 output_dir: str = "cadence_output",
                 model: str = "claude-3-5-sonnet-20241022",
                 verbose: bool = False,
                 allowed_tools: str = None,
                 checkpoint_timeout: int = 600):
        """
        Initialize the checkpoint supervisor
        
        Args:
            checkpoint_turns: Number of turns per checkpoint
            max_checkpoints: Maximum checkpoints before stopping
            output_dir: Directory for output logs
            model: Claude model to use for execution
            verbose: Enable verbose logging
            allowed_tools: Comma-separated list of allowed tools (default: basic file/bash tools)
            checkpoint_timeout: Timeout in seconds for each checkpoint (default: 600)
        """
        self.checkpoint_turns = checkpoint_turns
        self.max_checkpoints = max_checkpoints
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model = model
        self.verbose = verbose
        self.checkpoint_history = []
        # Default allowed tools for basic file operations
        self.allowed_tools = allowed_tools or "Write,Read,Edit,Bash,Glob,Grep,LS"
        self.checkpoint_timeout = checkpoint_timeout
        
    def spawn_agent(self, prompt: str, checkpoint_num: int, is_continuation: bool = False) -> CheckpointResult:
        """
        Spawn an agent for one checkpoint period
        
        Args:
            prompt: The prompt to send to the agent
            checkpoint_num: Current checkpoint number
            is_continuation: Whether this is a continuation of previous conversation
            
        Returns:
            CheckpointResult with execution details
        """
        # Create checkpoint log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"checkpoint_{checkpoint_num}_{timestamp}.log"
        
        # Build claude command
        cmd = ["claude"]
        
        if is_continuation:
            cmd.extend(["-c", "-p", prompt])
        else:
            cmd.extend(["-p", prompt])
            
        cmd.extend([
            "--model", self.model,
            "--max-turns", str(self.checkpoint_turns),
            "--allowedTools", self.allowed_tools,
            "--output-format=stream-json",
            "--dangerously-skip-permissions"
        ])
        
        if self.verbose:
            print(f"\n🚀 Starting checkpoint {checkpoint_num} with {self.checkpoint_turns} turns...")
            
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
                        if time.time() - start_time > self.checkpoint_timeout:
                            process.terminate()
                            raise subprocess.TimeoutExpired(cmd, self.checkpoint_timeout)
                            
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
                    turns_used = min(self.checkpoint_turns, max(1, len(output_lines) // 50))
                    
            success = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            success = False
            errors.append(f"Agent execution timed out after {self.checkpoint_timeout} seconds")
            turns_used = self.checkpoint_turns
            
        except Exception as e:
            success = False
            errors.append(f"Unexpected error: {str(e)}")
            
        # Calculate execution time and cost
        execution_time = time.time() - start_time
        
        # Cost estimation based on turns and token usage
        estimated_tokens = turns_used * AVG_TOKENS_PER_TURN
        cost = (estimated_tokens / 1000) * BLENDED_COST_PER_1K_TOKENS
        
        metadata = {
            "execution_time": execution_time,
            "log_file": str(log_file),
            "timestamp": timestamp,
            "model": self.model
        }
        
        return CheckpointResult(
            success=success,
            turns_used=turns_used,
            output_lines=output_lines,
            cost=cost,
            errors=errors,
            metadata=metadata
        )
        
    def analyze_checkpoint(self, result: CheckpointResult, checkpoint_num: int) -> SupervisorAnalysis:
        """
        Analyze checkpoint results and determine next action
        
        Args:
            result: The checkpoint execution result
            checkpoint_num: Current checkpoint number
            
        Returns:
            SupervisorAnalysis with guidance for next checkpoint
        """
        if self.verbose:
            print(f"\n🔍 Analyzing checkpoint {checkpoint_num} results...")
            
        # Initialize analysis components
        on_track = True
        task_complete = False
        issues_detected = []
        confidence = 0.8
        
        # Check for execution errors
        if not result.success:
            on_track = False
            issues_detected.append("Agent execution failed")
            confidence = 0.3
            
        if result.errors:
            on_track = False
            issues_detected.extend([f"Error: {err}" for err in result.errors[:3]])
            confidence *= 0.7
            
        # Analyze output for completion signals
        output_text = "\n".join(result.output_lines)
        
        # Check with Task Manager if available
        if hasattr(self, 'task_manager') and self.task_manager:
            try:
                task_progress = self.task_manager.analyze_progress(output_text)
                completed_tasks = task_progress.get('completed', [])
                incomplete_tasks = task_progress.get('incomplete', [])
                
                # Update prompt manager if available
                if hasattr(self, 'prompt_manager'):
                    self.prompt_manager.update_task_progress(
                        completed=[t.title for t in completed_tasks],
                        remaining=[t.title for t in incomplete_tasks]
                    )
                
                # If all tasks are complete, mark as complete
                if completed_tasks and not incomplete_tasks:
                    task_complete = True
                    confidence = 0.95
                elif len(completed_tasks) > 0:
                    # Partial completion
                    confidence = min(0.8, 0.5 + (len(completed_tasks) / (len(completed_tasks) + len(incomplete_tasks))) * 0.4)
                    
            except (AttributeError, TypeError, KeyError) as e:
                if self.verbose:
                    print(f"⚠️  Task manager analysis failed: {e}. Falling back to text-based detection.")
                # Fall back to text-based detection
        
        # Fall back to phrase-based detection if no task manager or on error
        if not task_complete:
            completion_phrases = [
                "task completed",
                "task complete",
                "finished implementing",
                "implementation complete",
                "all tests passing",
                "successfully implemented",
                "all tasks complete",
                "all tasks completed",
                "tasks are complete",
                "everything is complete"
            ]
            
            for phrase in completion_phrases:
                if phrase.lower() in output_text.lower():
                    task_complete = True
                    confidence = min(0.9, confidence + 0.1)
                    break
                
        # Check for common issues
        issue_indicators = {
            "error": "Errors encountered during execution",
            "failed": "Failures detected",
            "unable to": "Agent unable to complete certain actions",
            "not found": "Missing resources or files",
            "permission denied": "Permission issues encountered",
            "timeout": "Operations timing out"
        }
        
        for indicator, issue_desc in issue_indicators.items():
            if indicator in output_text.lower():
                issues_detected.append(issue_desc)
                on_track = False
                confidence *= 0.8
                
        # Check turn usage patterns
        if result.turns_used >= self.checkpoint_turns:
            # Hit turn limit - check if tasks were still in progress
            if not task_complete:
                issues_detected.append(f"Hit {self.checkpoint_turns} turn safety limit while tasks still in progress")
                on_track = True  # This is expected behavior, not an error
                confidence *= 0.95
            else:
                # Completed despite using all turns - that's fine
                pass
        elif result.turns_used < self.checkpoint_turns:
            # Finished before turn limit
            if "all tasks complete" in output_text.lower():
                task_complete = True
                confidence = min(0.95, confidence + 0.2)
                # This is ideal - task completion drove the exit
                
        # Generate guidance based on analysis
        guidance = self._generate_guidance(
            checkpoint_num=checkpoint_num,
            on_track=on_track,
            task_complete=task_complete,
            issues_detected=issues_detected,
            result=result
        )
        
        return SupervisorAnalysis(
            on_track=on_track,
            task_complete=task_complete,
            issues_detected=issues_detected,
            guidance=guidance,
            confidence=confidence
        )
        
    def _generate_guidance(self, checkpoint_num: int, on_track: bool, 
                          task_complete: bool, issues_detected: List[str],
                          result: CheckpointResult) -> str:
        """Generate guidance for next checkpoint"""
        
        if task_complete:
            return "Task appears to be complete. Verify the implementation meets all requirements."
            
        if not on_track and issues_detected:
            guidance_parts = ["The agent encountered issues:"]
            guidance_parts.extend([f"- {issue}" for issue in issues_detected[:3]])
            guidance_parts.append("\nFocus on resolving these issues in the next checkpoint.")
            return "\n".join(guidance_parts)
            
        # Check if we hit turn limit while working
        if result.turns_used >= self.checkpoint_turns:
            return "Continue where you left off. Focus on completing the remaining tasks. The turn limit interrupted your work, but that's OK - just pick up where you stopped."
            
        # Standard guidance for on-track execution
        remaining_checkpoints = self.max_checkpoints - checkpoint_num
        return f"Continue with task completion. You have {remaining_checkpoints} checkpoints remaining if needed. Focus on finishing the work, not managing turns."
        
    def run_supervised_task(self, initial_prompt: str, task_list: Optional[List[Dict]] = None) -> Tuple[bool, float]:
        """
        Run a task with checkpoint supervision
        
        Args:
            initial_prompt: The initial task prompt
            task_list: Optional list of tasks from Task Master
            
        Returns:
            Tuple of (success, total_cost)
        """
        total_cost = 0.0
        task_complete = False
        
        print(f"\n🎯 Starting supervised task with {self.max_checkpoints} checkpoints")
        print(f"📋 Task: {initial_prompt[:100]}{'...' if len(initial_prompt) > 100 else ''}")
        
        # Initialize prompt manager
        self.prompt_manager = ContextAwarePromptManager(
            original_task=initial_prompt,
            checkpoint_turns=self.checkpoint_turns,
            max_checkpoints=self.max_checkpoints
        )
        
        # Initialize task manager if tasks provided
        self.task_manager = None
        if task_list:
            self.task_manager = TaskManager()
            # Convert task list to Task objects
            self.task_manager.tasks = [Task.from_dict(t) for t in task_list]
        
        # Generate initial prompt with full context
        current_prompt = self.prompt_manager.get_initial_prompt(task_list)
        
        for checkpoint_num in range(1, self.max_checkpoints + 1):
            # Spawn agent for this checkpoint
            is_continuation = checkpoint_num > 1
            result = self.spawn_agent(current_prompt, checkpoint_num, is_continuation)
            
            # Track cost
            total_cost += result.cost
            
            # Store checkpoint in history
            self.checkpoint_history.append({
                "checkpoint": checkpoint_num,
                "result": result,
                "prompt": current_prompt
            })
            
            # Analyze the checkpoint
            analysis = self.analyze_checkpoint(result, checkpoint_num)
            
            if self.verbose:
                print(f"\n📊 Checkpoint {checkpoint_num} Analysis:")
                print(f"   - On track: {analysis.on_track}")
                print(f"   - Task complete: {analysis.task_complete}")
                print(f"   - Confidence: {analysis.confidence:.2f}")
                if analysis.issues_detected:
                    print(f"   - Issues: {', '.join(analysis.issues_detected[:2])}")
                    
            # Check if task is complete
            if analysis.task_complete:
                task_complete = True
                print(f"\n✅ Task completed at checkpoint {checkpoint_num}!")
                break
                
            # Prepare next checkpoint prompt if needed
            if checkpoint_num < self.max_checkpoints:
                # Update prompt manager context with progress
                # Update task progress if available
                self.prompt_manager.add_guidance(analysis.guidance)
                if analysis.issues_detected:
                    for issue in analysis.issues_detected:
                        self.prompt_manager.add_issue(issue)
                
                # Advance to next checkpoint
                self.prompt_manager.advance_to_next_checkpoint()
                
                # Use the prompt manager to generate continuation prompt
                current_prompt = self.prompt_manager.get_continuation_prompt(
                    analysis_guidance=analysis.guidance,
                    checkpoint_summary={
                        'summary': f"Checkpoint {checkpoint_num}: {'On track' if analysis.on_track else 'Issues detected'}",
                        'success': analysis.on_track,
                        'task_complete': analysis.task_complete
                    }
                )
                    
        # Final summary
        if self.verbose:
            print(f"\n📈 Supervision Summary:")
            print(f"   - Checkpoints used: {len(self.checkpoint_history)}/{self.max_checkpoints}")
            print(f"   - Total cost: ${total_cost:.4f}")
            print(f"   - Task complete: {task_complete}")
            
        # Save session summary
        self._save_session_summary(task_complete, total_cost, initial_prompt)
        
        return task_complete, total_cost
        
    def _save_session_summary(self, success: bool, total_cost: float, initial_prompt: str):
        """Save a summary of the supervision session"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "total_cost": total_cost,
            "checkpoints_used": len(self.checkpoint_history),
            "max_checkpoints": self.max_checkpoints,
            "checkpoint_turns": self.checkpoint_turns,
            "model": self.model,
            "initial_prompt": initial_prompt,
            "checkpoint_details": [
                {
                    "checkpoint": h["checkpoint"],
                    "success": h["result"].success,
                    "turns_used": h["result"].turns_used,
                    "cost": h["result"].cost,
                    "errors": h["result"].errors
                }
                for h in self.checkpoint_history
            ]
        }
        
        summary_file = self.output_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        if self.verbose:
            print(f"\n💾 Session summary saved to: {summary_file}")