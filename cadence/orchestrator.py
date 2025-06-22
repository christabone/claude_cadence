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

from .constants import OrchestratorDefaults, FilePatterns, AgentPromptDefaults
from .prompt_utils import PromptBuilder
from .utils import generate_session_id

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SupervisorDecision:
    """Decision made by supervisor analysis"""
    action: str  # "execute", "skip", "complete"
    todos: List[str] = None
    guidance: str = ""
    task_id: str = ""
    session_id: str = ""
    reason: str = ""
    zen_needed: Optional[Dict] = None


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


class SupervisorOrchestrator:
    """Orchestrates between supervisor and agent in separate directories"""
    
    def __init__(self, project_root: Path, task_file: Path, config: Optional[Dict] = None):
        self.project_root = Path(project_root).resolve()
        self.task_file = Path(task_file).resolve()
        self.supervisor_dir = self.project_root / ".cadence" / "supervisor"
        self.agent_dir = self.project_root / ".cadence" / "agent"
        self.state_file = self.project_root / ".cadence" / "orchestrator_state.json"
        self.config = config or {}
        
        # Create directories
        self.supervisor_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize state
        self.state = self.load_state()
        
        # Session tracking
        self.current_session_id = None
        
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
        
        # Update state
        self.state["first_run"] = False
        self.state["session_count"] += 1
        self.state["last_session_id"] = self.current_session_id
        self.save_state()
        
        # Track if this is first iteration of current session
        first_iteration = True
        max_iterations = self.config.get("max_iterations", OrchestratorDefaults.MAX_ITERATIONS)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info("-"*50)
            logger.info(f"Iteration {iteration}")
            logger.info("-"*50)
            
            # 1. Run supervisor to analyze and decide
            decision = self.run_supervisor_analysis(
                self.current_session_id, 
                use_continue=not is_first_run and not first_iteration
            )
            
            # 2. Check supervisor decision
            if decision.action == "complete":
                logger.info("All tasks complete!")
                return True
            elif decision.action == "skip":
                logger.info(f"Skipping: {decision.reason}")
                continue
            elif decision.action == "execute":
                logger.info(f"Executing agent with {len(decision.todos)} TODOs")
                
                # 3. Run agent with supervisor's TODOs
                agent_result = self.run_agent(
                    todos=decision.todos,
                    guidance=decision.guidance,
                    session_id=self.current_session_id,
                    use_continue=not is_first_run and not first_iteration
                )
                
                # 4. Save agent results for supervisor
                self.save_agent_results(agent_result, self.current_session_id)
                
                # Check if agent requested help
                if agent_result.requested_help:
                    logger.warning("Agent requested help - supervisor will provide assistance")
                
                # No longer first iteration
                first_iteration = False
                
                # 5. Continue to next iteration
                continue
            else:
                logger.error(f"Unknown supervisor action: {decision.action}")
                return False
        
        logger.warning(f"Reached maximum iterations ({max_iterations})")
        return False
    
    def run_supervisor_analysis(self, session_id: str, use_continue: bool) -> SupervisorDecision:
        """Run supervisor in its directory to analyze state"""
        original_dir = os.getcwd()
        
        try:
            # Change to supervisor directory
            os.chdir(self.supervisor_dir)
            
            # Build supervisor command
            cmd = [
                "python", "-m", "cadence.supervisor",
                "--analyze",
                "--task-file", str(self.task_file),
                "--session-id", session_id,
                "--output-decision"  # Outputs JSON decision
            ]
            
            # Add --continue flag if not first run
            if use_continue:
                cmd.append("--continue")
                logger.debug("Running supervisor with --continue flag")
            else:
                logger.debug("Running supervisor (first run)")
            
            # Run supervisor
            logger.debug(f"Command: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=self.config.get("subprocess_timeout", OrchestratorDefaults.SUBPROCESS_TIMEOUT)
                )
            except subprocess.TimeoutExpired as e:
                logger.error(f"Supervisor timeout after {e.timeout}s")
                raise RuntimeError(f"Supervisor timed out after {e.timeout} seconds")
            
            if result.returncode != 0:
                logger.error(f"Supervisor failed with code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"Supervisor failed: {result.stderr}")
            
            # Parse supervisor decision
            decision_file = self.validate_path(
                self.supervisor_dir / FilePatterns.DECISION_FILE.format(session_id=session_id),
                self.supervisor_dir
            )
            
            try:
                with open(decision_file) as f:
                    decision_data = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Supervisor decision file not found: {decision_file}") from None
            
            # Create decision object
            decision = SupervisorDecision(**decision_data)
            
            logger.info(f"Supervisor decision: {decision.action}")
            if decision.reason:
                logger.info(f"   Reason: {decision.reason}")
            
            return decision
            
        finally:
            # Always return to original directory
            os.chdir(original_dir)
    
    def run_agent(self, todos: List[str], guidance: str, 
                  session_id: str, use_continue: bool) -> AgentResult:
        """Run agent in dedicated agent directory"""
        original_dir = os.getcwd()
        
        try:
            # Change to agent directory
            os.chdir(self.agent_dir)
            
            # Create prompt with TODOs
            prompt = self.build_agent_prompt(todos, guidance)
            
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
                "--output-format", "stream-json",
                "--max-turns", str(self.config.get("max_turns", OrchestratorDefaults.MAX_AGENT_TURNS))
            ])
            
            # Add allowed tools
            for tool in self.config.get("allowed_tools", ["bash", "read", "write", "edit"]):
                cmd.extend(["--tool", tool])
            
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
            
            # Run agent
            start_time = time.time()
            try:
                with open(output_file, 'w') as out, open(error_file, 'w') as err:
                    result = subprocess.run(
                        cmd, 
                        stdout=out, 
                        stderr=err,
                        timeout=self.config.get("subprocess_timeout", OrchestratorDefaults.SUBPROCESS_TIMEOUT)
                    )
            except subprocess.TimeoutExpired as e:
                logger.error(f"Agent execution timeout after {e.timeout}s")
                return AgentResult(
                    success=False,
                    session_id=session_id,
                    output_file=str(output_file),
                    error_file=str(error_file),
                    execution_time=time.time() - start_time,
                    completed_normally=False,
                    requested_help=False,
                    errors=[f"Agent execution timed out after {e.timeout} seconds"]
                )
            
            execution_time = time.time() - start_time
            
            # Analyze results
            completed_normally = False
            requested_help = False
            errors = []
            
            # Check output for completion signals
            try:
                with open(output_file) as f:
                    content = f.read()
                    completed_normally = AgentPromptDefaults.COMPLETION_SIGNAL.upper() in content.upper()
                    requested_help = AgentPromptDefaults.HELP_SIGNAL.upper() in content.upper()
            except FileNotFoundError:
                # Output file may not exist if process failed early
                pass
            
            # Check for errors
            try:
                with open(error_file) as f:
                    errors = f.readlines()
            except FileNotFoundError:
                # Error file may not exist if no errors occurred
                pass
            
            agent_result = AgentResult(
                success=result.returncode == 0,
                session_id=session_id,
                output_file=str(output_file),
                error_file=str(error_file),
                execution_time=execution_time,
                completed_normally=completed_normally,
                requested_help=requested_help,
                errors=errors
            )
            
            logger.info(f"Agent execution complete in {execution_time:.2f}s")
            logger.info(f"   Success: {agent_result.success}")
            logger.info(f"   Completed normally: {agent_result.completed_normally}")
            logger.info(f"   Requested help: {agent_result.requested_help}")
            
            return agent_result
            
        finally:
            # Always return to original directory
            os.chdir(original_dir)
    
    def build_agent_prompt(self, todos: List[str], guidance: str) -> str:
        """Build prompt for agent with TODOs and guidance"""
        return PromptBuilder.build_agent_prompt(
            todos=todos,
            guidance=guidance,
            max_turns=self.config.get("max_turns", OrchestratorDefaults.MAX_AGENT_TURNS)
        )

    
    def save_agent_results(self, agent_result: AgentResult, session_id: str):
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