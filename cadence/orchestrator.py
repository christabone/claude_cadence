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

from .config import ConfigLoader
from .utils import generate_session_id
from .config import ZenIntegrationConfig
from .prompts import PromptGenerator, ExecutionContext
from .prompt_loader import PromptLoader
from .log_utils import Colors
from .json_stream_monitor import SimpleJSONStreamMonitor
from .workflow_state_machine import WorkflowStateManager, WorkflowState, WorkflowContext

# Import dispatch system components
from .enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from .fix_iteration_tracker import FixIterationTracker, FixAttemptLimitEnforcer, EscalationHandler
from .code_review_agent import CodeReviewAgent, ReviewConfig
from .review_result_parser import ReviewResultProcessor, ProcessingConfig
from .scope_validator import ScopeValidator
from .fix_verification_workflow import FixVerificationWorkflow, VerificationConfig
from .agent_messages import AgentMessage, MessageType, AgentType, Priority, MessageContext, SuccessCriteria, CallbackInfo

# Import comprehensive logging
from .dispatch_logging import (
    DispatchLogger, OperationType, DispatchContext, PerformanceMetrics,
    get_dispatch_logger, setup_dispatch_logging
)

# Set up logging
logger = logging.getLogger(__name__)
dispatch_logger = get_dispatch_logger("orchestrator")


@dataclass
class SupervisorDecision:
    """Decision made by supervisor analysis"""
    action: str  # "execute", "skip", "complete"
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
        self.config = config or {}

        # Load prompts from YAML template
        prompts_file = Path(__file__).parent / "prompts.yml.tmpl"
        self.prompt_loader = PromptLoader(prompts_file)

        # Create directories
        self.supervisor_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state = self.load_state()

        # Session tracking
        self.current_session_id = None

        # Initialize workflow state management
        workflow_state_dir = self.project_root / ".cadence" / "workflow_states"
        self.workflow_manager = WorkflowStateManager(persistence_dir=workflow_state_dir)

        # Initialize a default workflow instance for the orchestrator
        self.workflow = None  # Will be created when orchestration starts

        # Initialize dispatch system components
        self._init_dispatch_system()

    def cleanup_completion_marker(self):
        """Remove any existing completion marker from previous runs"""
        completion_marker = self.project_root / ".cadence" / "project_complete.marker"
        if completion_marker.exists():
            try:
                completion_marker.unlink()
                logger.info("Removed previous completion marker")
            except Exception as e:
                logger.warning(f"Failed to remove completion marker: {e}")

    def _init_dispatch_system(self):
        """Initialize the dispatch system components"""
        try:
            # Get dispatch configuration from main config
            dispatch_config_dict = self.config.get("dispatch", {})
            enabled = dispatch_config_dict.get("enabled", False)

            self.dispatch_enabled = enabled

            if not enabled:
                logger.info("Dispatch system disabled in configuration")
                self.agent_dispatcher = None
                self.fix_tracker = None
                self.code_review_agent = None
                self.result_processor = None
                self.scope_validator = None
                self.fix_verifier = None
                return

            logger.info("Initializing dispatch system components...")

            # Create dispatch system directories
            dispatch_dir = self.project_root / ".cadence" / "dispatch"
            dispatch_dir.mkdir(parents=True, exist_ok=True)

            # Initialize fix iteration tracker
            tracker_config = dispatch_config_dict.get("fix_tracking", {})
            persistence_file = dispatch_dir / "fix_iterations.json"

            self.fix_tracker = FixIterationTracker(
                max_attempts=tracker_config.get("max_attempts", 3),
                persistence_file=str(persistence_file) if tracker_config.get("enable_persistence", True) else None
            )

            # Initialize code review agent
            review_config_dict = dispatch_config_dict.get("code_review", {})
            review_config = ReviewConfig(
                primary_model=review_config_dict.get("primary_model", "gemini-2.5-pro"),
                fallback_models=review_config_dict.get("fallback_models", ["gemini-2.5-flash"]),
                max_file_size_mb=review_config_dict.get("max_file_size_mb", 5),
                chunk_size_lines=review_config_dict.get("chunk_size_lines", 500),
                timeout_seconds=review_config_dict.get("timeout_seconds", 300)
            )

            self.code_review_agent = CodeReviewAgent(config=review_config)

            # Initialize result processor
            processing_config_dict = dispatch_config_dict.get("result_processing", {})
            processing_config = ProcessingConfig(
                confidence_threshold=processing_config_dict.get("confidence_threshold", 0.5),
                max_description_length=processing_config_dict.get("max_description_length", 500)
            )

            self.result_processor = ReviewResultProcessor(config=processing_config)

            # Initialize scope validator
            self.scope_validator = ScopeValidator()

            # Initialize fix verification workflow
            verification_config_dict = dispatch_config_dict.get("verification", {})
            verification_config = VerificationConfig(
                minimum_resolution_rate=verification_config_dict.get("minimum_resolution_rate", 0.8),
                max_acceptable_regressions=verification_config_dict.get("max_acceptable_regressions", 0),
                verification_timeout_ms=verification_config_dict.get("timeout_ms", 300000)
            )

            self.fix_verifier = FixVerificationWorkflow(
                config=verification_config,
                code_review_agent=self.code_review_agent,
                result_processor=self.result_processor,
                scope_validator=self.scope_validator
            )

            # Initialize enhanced agent dispatcher
            dispatch_config_obj = DispatchConfig(
                max_concurrent_agents=dispatch_config_dict.get("max_concurrent", 2),
                default_timeout_ms=dispatch_config_dict.get("default_timeout_ms", 600000),
                enable_fix_tracking=True,
                enable_escalation=dispatch_config_dict.get("enable_escalation", True)
            )

            self.agent_dispatcher = EnhancedAgentDispatcher(
                config=dispatch_config_obj.to_dict()
            )

            logger.info("Dispatch system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize dispatch system: {e}")
            self.dispatch_enabled = False
            self.agent_dispatcher = None
            self.fix_tracker = None
            self.code_review_agent = None
            self.result_processor = None
            self.scope_validator = None
            self.fix_verifier = None

    async def run_claude_with_realtime_output(self, cmd: List[str], cwd: str, process_name: str) -> tuple[int, List[str]]:
        """Run claude command with real-time output display"""
        # Set up environment
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

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
                                print(f"{color}[{process_name}]{Colors.RESET} 🛠️  {Colors.YELLOW}{tool_name}{Colors.RESET}: {command}")

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

        return process.returncode, all_output

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


    def handle_workflow_transition(self, trigger: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle workflow state transitions based on triggers.

        Args:
            trigger: What triggered the potential transition
            metadata: Additional context for the transition
        """
        try:
            if self.workflow is None:
                logger.warning("Workflow transition attempted but workflow not initialized")
                return

            current_state = self.workflow.current_state
            metadata = metadata or {}

            # Log workflow transition with comprehensive context
            old_state = current_state.value

            # Define transition logic based on triggers
            if trigger == "code_review_triggered":
                if current_state == WorkflowState.WORKING:
                    self.workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, trigger, metadata)
                elif current_state == WorkflowState.FIXING:
                    # Re-review after fixes
                    self.workflow.transition_to(WorkflowState.VERIFICATION, trigger, metadata)

            elif trigger == "code_review_started":
                if current_state == WorkflowState.REVIEW_TRIGGERED:
                    self.workflow.transition_to(WorkflowState.REVIEWING, trigger, metadata)

            elif trigger == "issues_found":
                if current_state == WorkflowState.REVIEWING:
                    self.workflow.transition_to(WorkflowState.FIX_REQUIRED, trigger, metadata)
                    # Add issues to context
                    if "issues" in metadata:
                        for issue in metadata["issues"]:
                            self.workflow.add_issue(issue)

            elif trigger == "fix_started":
                if current_state == WorkflowState.FIX_REQUIRED:
                    self.workflow.transition_to(WorkflowState.FIXING, trigger, metadata)

            elif trigger == "fix_applied":
                if current_state == WorkflowState.FIXING:
                    self.workflow.transition_to(WorkflowState.VERIFICATION, trigger, metadata)
                    # Add fix to context
                    if "fix" in metadata:
                        self.workflow.add_fix(metadata["fix"])

            elif trigger == "verification_passed":
                if current_state == WorkflowState.VERIFICATION:
                    self.workflow.transition_to(WorkflowState.COMPLETE, trigger, metadata)

            elif trigger == "verification_failed":
                if current_state == WorkflowState.VERIFICATION:
                    self.workflow.transition_to(WorkflowState.FIX_REQUIRED, trigger, metadata)

            elif trigger == "error_occurred":
                # Any state can transition to error
                self.workflow.transition_to(WorkflowState.ERROR, trigger, metadata)

            elif trigger == "task_completed":
                if current_state in [WorkflowState.WORKING, WorkflowState.REVIEWING]:
                    self.workflow.transition_to(WorkflowState.COMPLETE, trigger, metadata)

            # Dispatch system triggers
            elif trigger == "dispatch_code_review_triggered":
                if self.dispatch_enabled and current_state in [WorkflowState.WORKING, WorkflowState.FIXING]:
                    self.workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, trigger, metadata)
                    self._handle_dispatch_code_review(metadata)

            elif trigger == "dispatch_fix_required":
                if self.dispatch_enabled and current_state in [WorkflowState.REVIEWING, WorkflowState.VERIFICATION]:
                    self.workflow.transition_to(WorkflowState.FIX_REQUIRED, trigger, metadata)
                    self._handle_dispatch_fix_required(metadata)

            elif trigger == "dispatch_escalation_required":
                if self.dispatch_enabled:
                    self.workflow.transition_to(WorkflowState.ERROR, trigger, metadata)
                    self._handle_dispatch_escalation(metadata)

            # Log the workflow transition
            new_state = self.workflow.current_state.value
            if old_state != new_state:
                dispatch_logger.log_workflow_transition(
                    from_state=old_state,
                    to_state=new_state,
                    trigger=trigger
                )

            logger.debug(f"Workflow transition: {trigger} -> {new_state}")

        except Exception as e:
            logger.error(f"Error handling workflow transition '{trigger}': {e}")
            # Create error context for dispatch logging
            error_context = DispatchContext(
                operation_type=OperationType.WORKFLOW_TRANSITION,
                session_id=self.current_session_id,
                metadata={"trigger": trigger, "metadata": metadata}
            )
            dispatch_logger.set_context(error_context)
            dispatch_logger.log_error(e, error_context)
            dispatch_logger.clear_context()

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get current workflow state summary for logging"""
        if hasattr(self, 'workflow') and self.workflow is not None:
            return self.workflow.get_state_summary()
        return {"current_state": "not_initialized", "error": "Workflow not initialized"}

    def _handle_dispatch_code_review(self, metadata: Dict[str, Any]) -> None:
        """Handle dispatch-triggered code review"""
        if not self.dispatch_enabled or not self.code_review_agent:
            logger.warning("Dispatch code review triggered but dispatch system not available")
            return

        # Extract files to review from metadata
        files_to_review = metadata.get("files", [])
        if not files_to_review:
            # Default to reviewing recently modified files
            files_to_review = self._get_recently_modified_files()

        if not files_to_review:
            logger.warning("No files specified for code review")
            return

        # Start comprehensive logging context
        with dispatch_logger.operation_context(
            operation_type=OperationType.REVIEW_PARSING,
            session_id=self.current_session_id,
            task_id=metadata.get("task_id"),
            file_paths=files_to_review,
            review_trigger=metadata.get("trigger", "manual"),
            file_count=len(files_to_review)
        ) as context:

            try:
                dispatch_logger.log_trigger_detection(
                    trigger_type="code_review",
                    file_paths=files_to_review,
                    metadata={"source": "orchestrator", "trigger": metadata.get("trigger", "manual")}
                )

                # Perform code review with timing
                start_time = time.time()
                review_result = self.code_review_agent.review_files(
                    file_paths=files_to_review,
                    severity_filter="all"
                )
                review_duration_ms = (time.time() - start_time) * 1000

                if review_result and review_result.success:
                    # Process review results with timing
                    processing_start = time.time()
                    category_result = self.result_processor.process_review_result(
                        review_result.review_output
                    )
                    processing_duration_ms = (time.time() - processing_start) * 1000

                    # Log review parsing metrics
                    dispatch_logger.log_review_parsing(
                        issues_found=category_result.total_issues,
                        processing_time_ms=processing_duration_ms
                    )

                    # Check if fixes are required
                    actionable_issues = category_result.actionable_issues
                    if actionable_issues:
                        logger.info(f"Code review found {len(actionable_issues)} actionable issues")

                        # Log performance metrics
                        perf_metrics = PerformanceMetrics(
                            operation_type=OperationType.REVIEW_PARSING,
                            duration_ms=review_duration_ms + processing_duration_ms,
                            files_processed=len(files_to_review),
                            issues_found=category_result.total_issues,
                            correlation_id=context.correlation_id
                        )
                        dispatch_logger.log_performance_metrics(perf_metrics)

                        # Trigger fix dispatch workflow
                        fix_metadata = {
                            "issues": [issue.to_dict() for issue in actionable_issues],
                            "files": files_to_review,
                            "review_result": category_result,
                            "correlation_id": context.correlation_id
                        }
                        self.handle_workflow_transition("dispatch_fix_required", fix_metadata)
                    else:
                        logger.info("Code review completed - no actionable issues found")
                        self.handle_workflow_transition("verification_passed", metadata)
                else:
                    error_msg = review_result.error_message if review_result else "Unknown error"
                    logger.error(f"Code review failed: {error_msg}")
                    dispatch_logger.log_error(Exception(f"Code review failed: {error_msg}"), context)
                    self.handle_workflow_transition("error_occurred", {"error": error_msg})

            except Exception as e:
                logger.error(f"Error handling dispatch code review: {e}")
                dispatch_logger.log_error(e, context)
                self.handle_workflow_transition("error_occurred", {"error": str(e)})

    def _handle_dispatch_fix_required(self, metadata: Dict[str, Any]) -> None:
        """Handle dispatch-triggered fix dispatch"""
        if not self.dispatch_enabled or not self.agent_dispatcher:
            logger.warning("Dispatch fix required but dispatch system not available")
            return

        # Extract fix information from metadata
        issues = metadata.get("issues", [])
        files = metadata.get("files", [])
        correlation_id = metadata.get("correlation_id")

        if not issues:
            logger.warning("No issues specified for fix dispatch")
            return

        # Create agent message for fix dispatch
        task_id = metadata.get("task_id", f"fix-{uuid.uuid4().hex[:8]}")
        agent_id = f"fix-agent-{uuid.uuid4().hex[:8]}"

        # Start comprehensive logging context
        with dispatch_logger.operation_context(
            operation_type=OperationType.AGENT_DISPATCH,
            session_id=self.current_session_id,
            task_id=task_id,
            agent_id=agent_id,
            file_paths=files,
            parent_correlation_id=correlation_id,
            issue_count=len(issues),
            severity_distribution={
                issue.get("severity", "unknown"): sum(1 for i in issues if i.get("severity") == issue.get("severity"))
                for issue in issues
            }
        ) as context:

            try:
                dispatch_logger.log_agent_dispatch(
                    agent_type="fix",
                    agent_id=agent_id,
                    message_type="FIX_REQUIRED"
                )

                message_context = MessageContext(
                    task_id=task_id,
                    parent_session=self.current_session_id or "unknown",
                    files_modified=files,
                    project_path=str(self.project_root)
                )

                success_criteria = SuccessCriteria(
                    expected_outcomes=[f"Fix {len(issues)} identified issues"],
                    validation_steps=["Verify issues are resolved", "Check for regressions"]
                )

                callback_info = CallbackInfo(
                    handler="orchestrator_fix_callback",
                    timeout_ms=300000  # 5 minutes
                )

                agent_message = AgentMessage(
                    message_type=MessageType.FIX_REQUIRED,
                    agent_type=AgentType.FIX,
                    context=message_context,
                    success_criteria=success_criteria,
                    callback=callback_info,
                    priority=Priority.HIGH,
                    payload={"issues": issues, "files": files}
                )

                # Dispatch the fix agent with timing
                dispatch_start = time.time()
                dispatch_result = self.agent_dispatcher.dispatch_agent(agent_message)
                dispatch_duration_ms = (time.time() - dispatch_start) * 1000

                if dispatch_result and dispatch_result.get("success"):
                    logger.info("Fix agent dispatched successfully")

                    # Log performance metrics
                    perf_metrics = PerformanceMetrics(
                        operation_type=OperationType.AGENT_DISPATCH,
                        duration_ms=dispatch_duration_ms,
                        files_processed=len(files),
                        issues_found=len(issues),
                        correlation_id=context.correlation_id
                    )
                    dispatch_logger.log_performance_metrics(perf_metrics)

                    # Store dispatch information for tracking
                    metadata["dispatch_result"] = dispatch_result
                    metadata["agent_message"] = agent_message.to_dict()
                    metadata["correlation_id"] = context.correlation_id

                    self.handle_workflow_transition("fix_started", metadata)
                else:
                    error_msg = dispatch_result.get("error", "Unknown dispatch error")
                    logger.error(f"Fix agent dispatch failed: {error_msg}")
                    dispatch_logger.log_error(Exception(f"Fix agent dispatch failed: {error_msg}"), context)
                    self.handle_workflow_transition("error_occurred", {"error": error_msg})

            except Exception as e:
                logger.error(f"Error handling dispatch fix required: {e}")
                dispatch_logger.log_error(e, context)
                self.handle_workflow_transition("error_occurred", {"error": str(e)})

    def _handle_dispatch_escalation(self, metadata: Dict[str, Any]) -> None:
        """Handle dispatch-triggered escalation"""
        if not self.dispatch_enabled:
            logger.warning("Dispatch escalation triggered but dispatch system not available")
            return

        escalation_reason = metadata.get("reason", "Unknown escalation")
        task_id = metadata.get("task_id", "unknown")
        correlation_id = metadata.get("correlation_id")

        # Start comprehensive logging context
        with dispatch_logger.operation_context(
            operation_type=OperationType.ESCALATION,
            session_id=self.current_session_id,
            task_id=task_id,
            parent_correlation_id=correlation_id,
            escalation_reason=escalation_reason
        ) as context:

            try:
                logger.error("DISPATCH ESCALATION TRIGGERED")

                # Get escalation details
                attempt_count = 0
                if self.fix_tracker:
                    attempt_count = self.fix_tracker.get_attempt_count(task_id)

                # Log comprehensive escalation information
                escalation_metadata = {
                    "task_id": task_id,
                    "attempt_count": attempt_count,
                    "max_attempts": getattr(self.fix_tracker, 'max_attempts', 'unknown') if self.fix_tracker else 'unknown',
                    "escalation_source": metadata.get("source", "dispatch_system"),
                    "session_id": self.current_session_id,
                    "project_root": str(self.project_root)
                }

                # Add any additional metadata from the escalation
                escalation_metadata.update({k: v for k, v in metadata.items()
                                          if k not in ['reason', 'task_id', 'correlation_id']})

                dispatch_logger.log_escalation(
                    reason=escalation_reason,
                    attempt_count=attempt_count,
                    metadata=escalation_metadata
                )

                logger.error(f"Task {task_id} escalated: {escalation_reason}")
                logger.error(f"Fix attempts for task {task_id}: {attempt_count}")

                # Debug state for troubleshooting
                if self.fix_tracker:
                    fix_state = {
                        "task_id": task_id,
                        "attempt_count": attempt_count,
                        "max_attempts": self.fix_tracker.max_attempts,
                        "has_persistence": hasattr(self.fix_tracker, 'persistence_file'),
                    }
                    dispatch_logger.debug_state("fix_tracker", fix_state)

                if self.agent_dispatcher:
                    dispatch_state = {
                        "active_agents": len(getattr(self.agent_dispatcher, 'active_agents', {})),
                        "max_concurrent": getattr(self.agent_dispatcher, 'max_concurrent_agents', 'unknown'),
                        "enabled": self.dispatch_enabled
                    }
                    dispatch_logger.debug_state("agent_dispatcher", dispatch_state)

                # In a production system, this would trigger alerts, notifications, etc.
                logger.error("Manual intervention required - stopping orchestration")

                # Log performance metrics for escalation handling
                perf_metrics = PerformanceMetrics(
                    operation_type=OperationType.ESCALATION,
                    duration_ms=0,  # Escalation is immediate
                    files_processed=0,
                    issues_found=0,
                    correlation_id=context.correlation_id
                )
                dispatch_logger.log_performance_metrics(perf_metrics)

            except Exception as e:
                logger.error(f"Error handling dispatch escalation: {e}")
                dispatch_logger.log_error(e, context)

    def _get_recently_modified_files(self, max_files: int = 10) -> List[str]:
        """Get list of recently modified files in the project"""
        try:
            # Use git to find recently modified files
            import subprocess
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                # Filter to code files and limit count
                code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.php', '.rb', '.go', '.rs', '.cs'}
                code_files = [
                    f for f in files[:max_files]
                    if any(f.endswith(ext) for ext in code_extensions)
                ]
                return code_files[:max_files]
        except Exception as e:
            logger.warning(f"Failed to get recently modified files: {e}")

        return []

    def trigger_dispatch_code_review(self, files: Optional[List[str]] = None) -> None:
        """Trigger a dispatch code review for specified files"""
        if not self.dispatch_enabled:
            logger.info("Dispatch system not enabled - skipping code review trigger")
            return

        metadata = {"files": files or []}
        self.handle_workflow_transition("dispatch_code_review_triggered", metadata)

    def _should_trigger_code_review(self, iteration: int, decision: 'SupervisorDecision') -> bool:
        """Determine if code review should be triggered based on configuration and context"""
        if not self.dispatch_enabled:
            return False

        # Get code review configuration
        dispatch_config = self.config.get("dispatch", {})
        review_config = dispatch_config.get("code_review", {})

        # Check if code review is enabled
        if not review_config.get("enabled", False):
            return False

        # Check frequency setting
        frequency = review_config.get("frequency", "never")

        if frequency == "never":
            return False
        elif frequency == "always":
            return True
        elif frequency == "on_completion":
            # Trigger on task completion
            return decision.action == "complete"
        elif frequency == "periodic":
            # Trigger every N iterations
            period = review_config.get("period", 5)
            return iteration % period == 0
        elif frequency == "on_significant_changes":
            # Trigger if agent made significant changes (heuristic)
            return self._agent_made_significant_changes(decision)

        # Default to never
        return False

    def _agent_made_significant_changes(self, decision: 'SupervisorDecision') -> bool:
        """Heuristic to determine if agent made significant changes"""
        # Check if there are multiple subtasks (likely significant work)
        if decision.subtasks and len(decision.subtasks) >= 3:
            return True

        # Check for keywords indicating significant changes
        if decision.guidance:
            significant_keywords = [
                "refactor", "implement", "create", "build", "develop",
                "redesign", "rewrite", "major", "substantial", "comprehensive"
            ]
            guidance_lower = decision.guidance.lower()
            if any(keyword in guidance_lower for keyword in significant_keywords):
                return True

        return False

    def _extract_modified_files(self, agent_result: 'AgentResult', decision: 'SupervisorDecision') -> List[str]:
        """Extract list of files that were likely modified by the agent"""
        modified_files = []

        try:
            # Try to parse agent output for file modifications
            # This is a best-effort extraction based on common patterns

            # Look for recently modified files using git
            recent_files = self._get_recently_modified_files()
            if recent_files:
                modified_files.extend(recent_files)

            # If we have task context, look for relevant files
            if decision.subtasks:
                for subtask in decision.subtasks:
                    description = subtask.get("description", "")
                    # Look for file paths mentioned in subtask descriptions
                    import re
                    file_patterns = re.findall(r'(\w+/[\w/]+\.\w+)', description)
                    modified_files.extend(file_patterns)

            # Remove duplicates and limit to reasonable number
            modified_files = list(set(modified_files))[:20]

        except Exception as e:
            logger.warning(f"Failed to extract modified files: {e}")

        return modified_files

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

        # Initialize workflow state for this session
        workflow_context = WorkflowContext(
            session_id=self.current_session_id,
            project_path=str(self.project_root)
        )
        self.workflow = self.workflow_manager.get_or_create_workflow(
            workflow_id=self.current_session_id,
            initial_state=WorkflowState.WORKING,
            context=workflow_context
        )

        logger.info("="*60)
        logger.info("Starting Claude Cadence Orchestration")
        logger.info(f"Task file: {self.task_file}")
        logger.info(f"Session ID: {self.current_session_id}")
        logger.info(f"Workflow state: {self.workflow.current_state.value}")
        logger.info("="*60)

        # Display key configuration values
        logger.info("Configuration:")
        logger.info(f"  Supervisor model: {self.config.get('supervisor', {}).get('model', 'NOT SET')}")
        logger.info(f"  Agent model: {self.config.get('agent', {}).get('model', 'NOT SET')}")

        # Group turn configurations together (orchestrator → supervisor → agent)
        logger.info(f"  Max orchestrator iterations: {self.config.get('orchestration', {}).get('max_iterations', 100)}")
        logger.info(f"  Max supervisor turns: {self.config.get('execution', {}).get('max_supervisor_turns', 'NOT SET')}")
        logger.info(f"  Max agent turns: {self.config.get('execution', {}).get('max_agent_turns', 'NOT SET')}")

        logger.info(f"  Agent timeout: {self.config.get('execution', {}).get('timeout', 'NOT SET')}s")
        logger.info(f"  Code review frequency: {self.config.get('zen_integration', {}).get('code_review_frequency', 'NOT SET')}")

        # Display MCP servers if configured
        mcp_servers = self.config.get('integrations', {}).get('mcp', {})
        if mcp_servers:
            supervisor_servers = mcp_servers.get('supervisor_servers', [])
            agent_servers = mcp_servers.get('agent_servers', [])
            logger.info(f"  Supervisor MCP servers: {', '.join(supervisor_servers) if supervisor_servers else 'None'}")
            logger.info(f"  Agent MCP servers: {', '.join(agent_servers) if agent_servers else 'None'}")

        # Display dispatch system configuration
        logger.info(f"  Dispatch system: {'ENABLED' if self.dispatch_enabled else 'DISABLED'}")
        if self.dispatch_enabled:
            dispatch_config = self.config.get('dispatch', {})
            logger.info(f"  Code review model: {dispatch_config.get('code_review', {}).get('primary_model', 'NOT SET')}")
            logger.info(f"  Max fix attempts: {dispatch_config.get('fix_tracking', {}).get('max_attempts', 3)}")
            logger.info(f"  Escalation enabled: {dispatch_config.get('enable_escalation', True)}")
            logger.info(f"  Max concurrent agents: {dispatch_config.get('max_concurrent', 2)}")

        logger.info("="*60)

        # CRITICAL: Clean up all old session files from previous runs
        # This MUST happen before any other operations to prevent confusion
        logger.info("Cleaning up old session files...")
        self.cleanup_all_session_files()

        # Also clean up any previous completion marker
        self.cleanup_completion_marker()

        # Double-check that no old agent result files exist
        old_agent_results = list(self.supervisor_dir.glob("agent_result_*.json"))
        if old_agent_results:
            logger.error(f"WARNING: Found {len(old_agent_results)} old agent result files after cleanup!")
            for f in old_agent_results:
                logger.error(f"  - {f.name}")
        else:
            logger.debug("Confirmed: No old agent result files exist")

        # Update state
        self.state["session_count"] += 1
        self.state["last_session_id"] = self.current_session_id
        self.save_state()

        max_iterations = self.config.get("orchestration", {}).get("max_iterations", 100)
        iteration = 0

        # Define completion marker file
        completion_marker = self.project_root / ".cadence" / "project_complete.marker"

        while iteration < max_iterations:
            iteration += 1
            logger.info("="*60)
            logger.info(f"ITERATION {iteration}")
            logger.info("="*60)

            # Check if completion marker exists
            if completion_marker.exists():
                logger.info("Found project completion marker!")
                try:
                    with open(completion_marker, 'r') as f:
                        marker_content = f.read()
                    logger.info(f"Completion marker content:\n{marker_content}")

                    # Transition workflow to complete state
                    self.workflow.transition_to(
                        WorkflowState.COMPLETE,
                        "project_completion_marker_found",
                        metadata={"marker_content": marker_content}
                    )

                    logger.info("="*60)
                    logger.info("PROJECT COMPLETED SUCCESSFULLY!")
                    logger.info("="*60)
                    return True
                except Exception as e:
                    logger.error(f"Error reading completion marker: {e}")

            # 1. Run supervisor to analyze and decide
            decision = self.run_supervisor_analysis(
                self.current_session_id,
                use_continue=(iteration > 1),
                iteration=iteration
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

                # Update workflow context with task information
                self.workflow.update_context(
                    task_id=decision.task_id,
                    task_title=decision.task_title
                )

                # 3. Run agent with supervisor's TODOs
                agent_result = self.run_agent(
                    todos=todos,
                    guidance=decision.guidance,
                    session_id=self.current_session_id,
                    use_continue=(iteration > 1),
                    task_id=decision.task_id,
                    subtasks=decision.subtasks,
                    project_root=decision.project_path  # Pass as project_path
                )

                # Check if agent quit too quickly (likely an error)
                if agent_result.quit_too_quickly:
                    logger.error(f"Agent quit too quickly ({agent_result.execution_time:.2f}s) - likely an error condition")
                    logger.error("Exiting program due to quick agent exit")
                    return False

                # 3.5. Validate agent created required scratchpad (with retry logic)
                max_scratchpad_retries = 5
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

                # 4. Check if dispatch code review should be triggered
                if self.dispatch_enabled and self._should_trigger_code_review(iteration, decision):
                    logger.info("Triggering dispatch code review after agent execution...")

                    # Get files modified by the agent (if available)
                    modified_files = self._extract_modified_files(agent_result, decision)

                    # Trigger code review
                    self.trigger_dispatch_code_review(files=modified_files)

                    # Code review will handle the workflow transitions
                    # If issues are found, fix agents will be dispatched
                    # If no issues, workflow will continue normally

                # 5. Save agent results for supervisor
                self.save_agent_results(agent_result, self.current_session_id, decision.todos, decision.task_id)

                # Check if agent requested help
                if agent_result.requested_help:
                    logger.warning("Agent requested help - supervisor will provide assistance")

                # No longer first iteration

                # 6. Continue to next iteration
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


    def run_supervisor_analysis(self, session_id: str, use_continue: bool, iteration: int) -> SupervisorDecision:
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

                    # Build prompt using YAML templates
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
                        "max_turns": self.config.get("execution", {}).get("max_agent_turns", 120)  # Agent's turn limit for supervisor context
                    }

                    # Get base prompt
                    logger.debug(f"Getting supervisor prompt template with context: has_previous_agent_result={context['has_previous_agent_result']}")
                    base_prompt = self.prompt_loader.get_template("supervisor_prompts.orchestrator_taskmaster.base_prompt")
                    base_prompt = self.prompt_loader.format_template(base_prompt, context)

                    # Log a preview of the TASK section to verify correct template
                    if "Process the agent's completed work" in base_prompt:
                        logger.debug("Supervisor prompt includes agent work processing (iteration 2+)")
                    elif "Analyze the current task state and decide what the agent should work on first" in base_prompt:
                        logger.debug("Supervisor prompt is for first iteration (no agent work)")
                    else:
                        logger.warning("Supervisor prompt TASK section unclear - check template rendering")

                    # Get code review section based on config
                    # When code_review_frequency is "task", we should ALSO include project review
                    # instructions so the supervisor knows to run final review before completion
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

                    # If this is a retry, use a shorter prompt to avoid token limits
                    if json_retry_count > 0:
                        # Create a minimal retry prompt to avoid token limits
                        minimal_retry_prompt = f"""You are the Task Supervisor. Project root: {context['project_path']}

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
Retry attempt {json_retry_count + 1} of {max_json_retries}."""
                        supervisor_prompt = minimal_retry_prompt
                    else:
                        # Combine all sections normally
                        supervisor_prompt = f"{base_prompt}{code_review_section}{zen_guidance}{output_format}"

                    # Save supervisor prompt to file for debugging
                    prompt_debug_file = self.supervisor_dir / f"supervisor_prompt_{session_id}.txt"
                    try:
                        with open(prompt_debug_file, 'w') as f:
                            f.write(supervisor_prompt)
                        logger.debug(f"Saved supervisor prompt to {prompt_debug_file}")
                        logger.debug(f"Supervisor prompt length: {len(supervisor_prompt)} characters")
                    except Exception as e:
                        logger.warning(f"Failed to save supervisor prompt: {e}")

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

                    # Get supervisor model from config
                    supervisor_config = self.config.get("supervisor")
                    if not supervisor_config:
                        raise ValueError("Supervisor configuration not found in config")

                    supervisor_model = supervisor_config.get("model")
                    if not supervisor_model:
                        raise ValueError("Supervisor model not specified in config")


                    cmd = [
                        "claude",
                        "-p", supervisor_prompt,
                        "--model", supervisor_model,
                        "--allowedTools", ",".join(all_tools),
                        "--max-turns", "80",
                        "--output-format", "stream-json",
                        "--verbose",
                        "--dangerously-skip-permissions"  # Skip permission prompts
                    ]

                    # Check config for supervisor continuation setting
                    supervisor_use_continue = supervisor_config.get("use_continue", True)  # Default to True for backward compatibility

                    # Add --continue flag based on config and conditions
                    if supervisor_use_continue and (use_continue or json_retry_count > 0):
                        cmd.append("--continue")
                        if json_retry_count > 0:
                            logger.warning(f"Retrying supervisor with --continue due to JSON parsing error (attempt {json_retry_count + 1})")
                        else:
                            logger.debug("Running supervisor with --continue flag")
                    else:
                        if not supervisor_use_continue:
                            logger.debug("Running supervisor without --continue flag (disabled in config)")
                        else:
                            logger.debug("Running supervisor (first run)")


                    # Run supervisor
                    logger.debug(f"Command: {' '.join(cmd)}")
                    logger.info("-" * 60)
                    if json_retry_count > 0:
                        logger.info(f"SUPERVISOR STARTING... (JSON RETRY {json_retry_count + 1}/{max_json_retries})")
                    else:
                        logger.info("SUPERVISOR STARTING...")
                    logger.info("-" * 60)

                    # Run supervisor with real-time output
                    supervisor_start_time = time.time()
                    supervisor_end_time = None
                    supervisor_duration = None
                    try:
                        logger.debug(f"Starting supervisor subprocess with {len(cmd)} args")
                        # Run supervisor async for real-time output
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            returncode, all_output = loop.run_until_complete(
                                self.run_claude_with_realtime_output(cmd, self.supervisor_dir, "SUPERVISOR")
                            )
                            supervisor_end_time = time.time()
                            supervisor_duration = (supervisor_end_time - supervisor_start_time) * 1000
                            logger.info(f"Supervisor process completed in {supervisor_duration:.0f}ms with return code {returncode}")
                        finally:
                            loop.close()

                        if returncode != 0:
                            logger.error(f"Supervisor failed with code {returncode}")
                            logger.error(f"Output lines collected: {len(all_output)}")
                            if all_output:
                                logger.error(f"Last few output lines: {all_output[-5:]}")
                            raise RuntimeError(f"Supervisor failed with exit code {returncode}")

                    except asyncio.TimeoutError as e:
                        supervisor_end_time = time.time()
                        supervisor_duration = (supervisor_end_time - supervisor_start_time) * 1000
                        logger.error(f"Supervisor timed out after {supervisor_duration:.0f}ms: {e}")
                        raise RuntimeError(f"Supervisor execution timed out after {supervisor_duration:.0f}ms")
                    except Exception as e:
                        supervisor_end_time = time.time()
                        supervisor_duration = (supervisor_end_time - supervisor_start_time) * 1000
                        logger.error(f"Supervisor execution error after {supervisor_duration:.0f}ms: {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise RuntimeError(f"Supervisor execution failed: {e}")

                    logger.info("-" * 60)
                    logger.info("SUPERVISOR COMPLETED")
                    logger.info("-" * 60)

                    # Parse supervisor JSON output using SimpleJSONStreamMonitor
                    try:
                        logger.debug(f"Parsing supervisor output from {len(all_output)} lines...")
                        logger.debug(f"First 3 output lines: {all_output[:3] if all_output else 'None'}")
                        logger.debug(f"Last 3 output lines: {all_output[-3:] if all_output else 'None'}")

                        # Look for the decision JSON in assistant messages
                        json_str = None
                        found_jsons = []

                        # Create JSON stream monitor
                        from .json_stream_monitor import SimpleJSONStreamMonitor
                        json_monitor = SimpleJSONStreamMonitor()

                        # Process all output lines to find JSON objects
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
                                                        # Convert back to string for compatibility
                                                        found_jsons.append(json.dumps(result))
                                                # Reset monitor after each text block
                                                json_monitor.reset()
                            except json.JSONDecodeError:
                                # This line wasn't JSON, skip it
                                continue

                        # Take the LAST valid JSON found (the final decision)
                        if found_jsons:
                            json_str = found_jsons[-1]

                        if not json_str:
                            logger.error("No decision JSON found in supervisor output")
                            logger.error(f"Found {len(found_jsons)} candidate JSONs using SimpleJSONStreamMonitor")
                            logger.error(f"Total output lines processed: {len(all_output)}")
                            # Log a sample of assistant messages for debugging
                            assistant_lines = []
                            for line_str in all_output[:10]:  # Just first 10 to avoid spam
                                try:
                                    data = json.loads(line_str)
                                    if data.get('type') == 'assistant':
                                        assistant_lines.append(line_str[:200])  # Truncate for readability
                                except:
                                    pass
                            logger.error(f"Sample assistant message lines: {assistant_lines}")
                            raise ValueError("No decision JSON found in supervisor output")

                        logger.debug(f"Found decision JSON: {json_str[:100]}...")
                        decision_data = json.loads(json_str)

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
                quick_quit_threshold = self.config.get("orchestration", {}).get("quick_quit_seconds", 10.0)

                # Create decision object
                # Map project_root from JSON to project_path for internal use
                if 'project_root' in decision_data and 'project_path' not in decision_data:
                    decision_data['project_path'] = decision_data.pop('project_root')

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
                prompt = self.build_agent_prompt(todos, guidance, task_id, subtasks, project_root, use_continue)

                # Save prompt for debugging
                prompt_file = self.validate_path(
                    self.agent_dir / self.cadence_config.file_patterns.prompt_file.format(session_id=session_id),
                    self.agent_dir
                )
                with open(prompt_file, 'w') as f:
                    f.write(prompt)

                # Get agent model from config
                agent_config = self.config.get("agent")
                if not agent_config:
                    raise ValueError("Agent configuration not found in config")

                agent_model = agent_config.get("model")
                if not agent_model:
                    raise ValueError("Agent model not specified in config")


                # Build claude command
                cmd = ["claude"]

                # Check config for agent continuation setting
                agent_use_continue = agent_config.get("use_continue", True)  # Default to True for backward compatibility

                # Add --continue flag based on config and conditions
                if agent_use_continue and use_continue:
                    cmd.extend(["-c", "-p", prompt])
                    logger.debug("Running agent with --continue flag")
                else:
                    cmd.extend(["-p", prompt])
                    if not agent_use_continue:
                        logger.debug("Running agent without --continue flag (disabled in config)")
                    else:
                        logger.debug("Running agent (first run)")

                cmd.extend([
                    "--model", agent_model,
                    "--max-turns", str(self.config.get("execution", {}).get("max_agent_turns", 120)),
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
                    self.agent_dir / self.cadence_config.file_patterns.output_file.format(session_id=session_id),
                    self.agent_dir
                )
                error_file = self.validate_path(
                    self.agent_dir / self.cadence_config.file_patterns.error_file.format(session_id=session_id),
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
                completed_normally = self.cadence_config.task_detection.completion_phrase.upper() in output_text.upper()
                requested_help = self.cadence_config.task_detection.help_needed_phrase.upper() in output_text.upper()

                agent_result = AgentResult(
                    success=returncode == 0,
                    session_id=session_id,
                    output_file=str(output_file),
                    error_file=str(error_file),
                    execution_time=execution_time,
                    completed_normally=completed_normally,
                    requested_help=requested_help,
                    errors=errors,
                    quit_too_quickly=execution_time < self.config.get("orchestration", {}).get("quick_quit_seconds", 10.0)
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
                              project_root: str = None, use_continue: bool = False) -> str:
            """Build prompt for agent with TODOs and guidance"""
            # Create a PromptGenerator instance
            prompt_generator = PromptGenerator(self.prompt_loader)

            session_id = self.current_session_id if hasattr(self, 'current_session_id') else "unknown"
            max_turns = self.config.get("execution", {}).get("max_agent_turns", 120)

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

            return base_prompt




    def validate_agent_scratchpad(self, session_id: str, agent_result: AgentResult) -> bool:
        """Validate that agent created required scratchpad file"""
        # First check the expected location
        scratchpad_file = self.project_root / ".cadence" / "scratchpad" / f"session_{session_id}.md"

        if scratchpad_file.exists():
            logger.debug(f"Agent scratchpad found at expected location: {scratchpad_file}")
            return True

        # Fallback: search recursively for the scratchpad file
        logger.warning(f"Agent scratchpad not found at expected location: {scratchpad_file}")
        logger.info("Searching recursively for scratchpad file...")

        # Search pattern for the specific session file
        search_pattern = f"**/session_{session_id}.md"

        # Search under the project root
        for found_file in self.project_root.glob(search_pattern):
            # Verify it's in a .cadence/scratchpad directory structure
            if ".cadence" in found_file.parts and "scratchpad" in found_file.parts:
                logger.info(f"Found scratchpad at alternate location: {found_file}")
                logger.warning(f"Agent created scratchpad at {found_file} instead of {scratchpad_file}")
                # TODO: Consider copying the file to the expected location
                return True

        logger.warning(f"Agent scratchpad missing: could not find session_{session_id}.md anywhere")
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
            os.chdir(self.agent_dir)

            # Build command using same pattern as main agent
            # ALWAYS use --continue for scratchpad retry to maintain context
            cmd = ["claude", "-c", "-p", scratchpad_prompt]
            cmd.extend([
                "--allowedTools", "Write,Read,Bash,LS",
                "--max-turns", "5",
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions"
            ])

            # Debug command
            logger.info(f"Scratchpad retry command: claude -c -p [PROMPT] {' '.join(cmd[3:])}")
            logger.info(f"Working directory: {os.getcwd()}")
            logger.info("Using --continue flag for scratchpad retry to maintain context")

            start_time = time.time()

            # Run with minimal timeout
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    returncode, all_output = loop.run_until_complete(
                        self.run_claude_with_realtime_output(cmd, self.agent_dir, "SCRATCHPAD_AGENT")
                    )
                finally:
                    loop.close()

                execution_time = time.time() - start_time

                # Check for completion signal
                output_text = '\n'.join(all_output)
                completed_normally = "SCRATCHPAD CREATION COMPLETE" in output_text.upper()

                return AgentResult(
                    success=returncode == 0,
                    session_id=session_id,
                    output_file="scratchpad_retry_output.log",
                    error_file="scratchpad_retry_error.log",
                    execution_time=execution_time,
                    completed_normally=completed_normally,
                    requested_help=False,
                    errors=[] if returncode == 0 else ["Scratchpad retry failed"]
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


    def cleanup_old_sessions(self, keep_last_n: int = 5):
        """Clean up old session files to save space"""
        # Get all session files from both directories
        session_files = []

        # Collect files from supervisor directory
        for pattern in ["decision_*.json", "agent_result_*.json", "decision_snapshot_*.json", "session_*.md"]:
            for file in self.supervisor_dir.glob(pattern):
                session_files.append(file)

        # Collect files from agent directory
        for pattern in ["prompt_*.txt", "output_*.log", "error_*.log"]:
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
                    logger.debug(f"Removed old file from session {session_id}: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} files from {len(sessions_to_remove)} old sessions")

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
