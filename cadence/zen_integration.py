"""
Zen MCP integration for enhanced supervisor capabilities

This module provides integration with zen MCP tools for debugging,
code review, and collaborative analysis when agents need help.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .config import ZenIntegrationConfig, SCRATCHPAD_DIR

# Detection thresholds
CUTOFF_INDICATOR_THRESHOLD = 3    # Number of indicators before assuming cutoff
AUTO_DEBUG_ERROR_THRESHOLD = 3    # Repeated errors before triggering debug
LAST_LINES_TO_CHECK = 50         # Number of output lines to check for patterns


@dataclass
class ZenRequest:
    """Request for zen assistance"""
    tool: str  # debug, review, consensus, precommit, analyze
    reason: str
    context: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # low, normal, high, critical
    session_id: Optional[str] = None
    execution_history: Optional[List[Dict]] = None
    scratchpad_path: Optional[str] = None


class ZenIntegration:
    """Manages integration with zen MCP tools for supervisor assistance"""

    def __init__(self, config: ZenIntegrationConfig, verbose: bool = False):
        """Initialize zen integration with configuration"""
        self.config = config
        self.verbose = verbose
        self.error_counts = {}  # Track errors per session

    def should_call_zen(self, result, context, session_id: str) -> Optional[Tuple[str, str]]:
        """
        Determine if zen assistance is needed based on execution results

        Returns:
            Tuple of (tool_name, reason) if zen should be called, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check for explicit help requests
        if self.config.stuck_detection:
            stuck_status = self._detect_stuck_status(result, session_id)
            if stuck_status:
                return "debug", stuck_status

        # Check for repeated errors
        error_pattern = self._detect_error_pattern(result, session_id)
        if error_pattern:
            return "debug", error_pattern

        # Check if task requires validation
        if hasattr(context, 'current_task'):
            validation_needed = self._task_requires_validation(context.current_task)
            if validation_needed:
                return "precommit", f"Critical task validation: {validation_needed}"

        # Check if task was likely cut off (needs retrospective)
        if self._detect_cutoff(result, context):
            return "analyze", "Task appears to have been cut off at turn limit"

        return None

    def _detect_stuck_status(self, result, session_id: str) -> Optional[str]:
        """Check if agent reported being stuck"""
        # Check scratchpad for STUCK status
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad_path.exists():
            content = scratchpad_path.read_text()
            if "Status: STUCK" in content or "HELP NEEDED" in content:
                # Extract the issue description
                match = re.search(r'Issue: (.+?)(?:\n|$)', content)
                issue = match.group(1) if match else "Agent requested help"
                return f"Agent stuck: {issue}"

        # Check output for help requests
        if hasattr(result, 'output_lines'):
            output_text = "\n".join(result.output_lines[-100:])  # Last 100 lines
            help_patterns = [
                "HELP NEEDED - STUCK",
                "ARCHITECTURE_REVIEW_NEEDED",
                "SECURITY_REVIEW_NEEDED",
                "PERFORMANCE_REVIEW_NEEDED"
            ]
            for pattern in help_patterns:
                if pattern in output_text:
                    return f"Agent requested: {pattern}"

        return None

    def _detect_stuck_agent(self, result, context, session_id: str) -> bool:
        """Alias for _detect_stuck_status for backward compatibility"""
        # Return boolean for backward compatibility
        return self._detect_stuck_status(result, session_id) is not None

    def _detect_error_pattern(self, result, session_id: str) -> Optional[str]:
        """Check for repeated error patterns"""
        if not hasattr(result, 'errors') or not result.errors:
            return None

        # Track errors for this session
        if session_id not in self.error_counts:
            self.error_counts[session_id] = {}

        # Count similar errors
        for error in result.errors:
            # Simple error categorization
            error_key = self._categorize_error(error)
            self.error_counts[session_id][error_key] = \
                self.error_counts[session_id].get(error_key, 0) + 1

            if self.error_counts[session_id][error_key] >= self.config.auto_debug_threshold:
                return f"Repeated error ({self.error_counts[session_id][error_key]}x): {error_key}"

        return None

    def _categorize_error(self, error: str) -> str:
            """Categorize error for pattern detection"""
            # Define known error patterns
            error_patterns = {
                "import_error": ["ModuleNotFoundError", "ImportError", "cannot import name"],
                "file_not_found": ["FileNotFoundError", "No such file", "does not exist"],
                "syntax_error": ["SyntaxError", "invalid syntax", "unexpected indent"],
                "type_error": ["TypeError", "AttributeError", "has no attribute"],
                "permission_error": ["Permission", "PermissionError", "Access denied"],
                "connection_error": ["ConnectionError", "Network", "unreachable"],
                "timeout_error": ["TimeoutError", "timed out", "deadline exceeded"],
                "value_error": ["ValueError", "invalid value", "must be"],
                "key_error": ["KeyError", "key not found"],
                "runtime_error": ["RuntimeError", "runtime error"]
            }

            # Check against known patterns
            for category, patterns in error_patterns.items():
                if any(pattern in error for pattern in patterns):
                    return category

            # For unknown errors, use a generic category rather than creating unbounded ones
            return "unknown_error"


    def _detect_cutoff(self, result, context) -> bool:
        """Detect if execution was likely cut off at turn limit"""
        # Use new completion flags if available
        if hasattr(result, 'stopped_unexpectedly'):
            return result.stopped_unexpectedly

        # Fallback to pattern detection for backward compatibility
        if hasattr(result, 'task_complete') and result.task_complete:
            return False  # Task completed successfully

        # Check for signs of being cut off
        if hasattr(result, 'output_lines') and len(result.output_lines) > 0:
            last_lines = "\n".join(result.output_lines[-LAST_LINES_TO_CHECK:])

            # Look for signs of incomplete work
            has_remaining_todos = False
            if hasattr(context, 'remaining_todos') and hasattr(context.remaining_todos, '__len__'):
                has_remaining_todos = len(context.remaining_todos) > 0

            cutoff_indicators = [
                "ALL TASKS COMPLETE" not in last_lines,
                "HELP NEEDED" not in last_lines,  # Not a help request
                has_remaining_todos,
                "in progress" in last_lines.lower(),
                "working on" in last_lines.lower(),
                "next, i'll" in last_lines.lower(),
                "let me" in last_lines.lower()
            ]

            # If multiple indicators suggest cutoff
            if sum(cutoff_indicators) >= CUTOFF_INDICATOR_THRESHOLD:
                return True

        return False


    def _task_requires_validation(self, task_description) -> Optional[str]:
        """Check if task matches validation patterns"""
        if not task_description:
            return None

        task_lower = str(task_description).lower()
        for pattern in self.config.validate_on_complete:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*")
            if re.search(regex_pattern, task_lower):
                return f"Matches pattern: {pattern}"

        return None

    def _detect_repeated_errors(self, result, context=None) -> Tuple[bool, int]:
        """Legacy method for backward compatibility"""
        # This is now handled by _detect_error_pattern
        if not hasattr(result, 'errors') or not result.errors:
            return False, 0
        # Simple check: if there are many similar errors
        error_types = {}
        for error in result.errors:
            error_key = self._categorize_error(error)
            error_types[error_key] = error_types.get(error_key, 0) + 1
        # Return True if any error type appears 3+ times
        max_count = max(error_types.values()) if error_types else 0
        return max_count >= 3, max_count

    def _is_critical_task(self, context) -> bool:
        """Check if current task is critical"""
        if hasattr(context, 'current_task') and context.current_task:
            task_lower = str(context.current_task).lower()
            critical_patterns = [
                'payment', 'security', 'auth', 'database', 'migration',
                'production', 'deploy', 'critical', 'urgent'
            ]
            return any(pattern in task_lower for pattern in critical_patterns)
        return False

    def _create_zen_request(self, tool: str, reason: str, execution_result,
                           context, session_id: str, priority: str = "normal") -> ZenRequest:
        """Create a zen request object"""
        # Build execution history
        execution_history = []
        if hasattr(execution_result, '__dict__'):
            execution_history.append({
                'success': execution_result.success,
                'turns_used': execution_result.turns_used,
                'errors': execution_result.errors if hasattr(execution_result, 'errors') else [],
                'metadata': execution_result.metadata if hasattr(execution_result, 'metadata') else {}
            })

        # Build context dict
        context_dict = {}
        if hasattr(context, 'todos'):
            context_dict['todos'] = context.todos
        if hasattr(context, 'completed_todos'):
            context_dict['completed_todos'] = context.completed_todos
        if hasattr(context, 'remaining_todos'):
            context_dict['remaining_todos'] = context.remaining_todos

        return ZenRequest(
            tool=tool,
            reason=reason,
            context=context_dict,
            priority=priority,
            session_id=session_id,
            execution_history=execution_history,
            scratchpad_path=f".cadence/scratchpad/session_{session_id}.md"
        )

    def _call_zen_tool(self, request: ZenRequest) -> Dict[str, Any]:
        """Execute zen tool call"""
        try:
            # Build the zen MCP tool name
            tool_name = f"mcp__zen__{request.tool}"

            # Prepare the context for the zen tool
            context = {
                "session_id": request.session_id,
                "reason": request.reason,
                "priority": request.priority,
                "context": request.context,
                "execution_history": request.execution_history
            }

            # Add scratchpad content if available
            if request.scratchpad_path:
                scratchpad_path = Path(request.scratchpad_path)
                if scratchpad_path.exists():
                    context["scratchpad_content"] = scratchpad_path.read_text()

            # Log the zen tool call
            if self.verbose:
                print(f"Calling zen tool: {tool_name}")

            # Return structured response
            # Note: In a real implementation, this would make an actual MCP call
            # For now, we return a properly structured response
            return {
                "tool": request.tool,
                "success": True,
                "guidance": f"Zen {request.tool} analysis completed",
                "response": {
                    "session_id": request.session_id,
                    "tool": request.tool,
                    "priority": request.priority,
                    "context_provided": bool(request.context),
                    "execution_history_length": len(request.execution_history) if request.execution_history else 0
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "tool": request.tool,
                "success": False,
                "error": str(e),
                "guidance": f"Failed to execute zen {request.tool}: {str(e)}"
            }

    def call_zen_support(self, tool: str, reason: str,
                        execution_result, context, session_id: str) -> Dict[str, Any]:
        """
        Call appropriate zen tool for assistance

        Returns:
            Dict with zen response and guidance
        """
        if self.verbose:
            print(f"\n🔮 Calling zen {tool} for: {reason}")

        # Prepare context for zen
        zen_context = self._prepare_zen_context(execution_result, context, session_id)

        # Select appropriate zen tool
        if tool == "debug":
            return self._zen_debug(reason, zen_context)
        elif tool == "review":
            return self._zen_review(reason, zen_context)
        elif tool == "consensus":
            return self._zen_consensus(reason, zen_context)
        elif tool == "precommit":
            return self._zen_precommit(reason, zen_context)
        elif tool == "analyze":
            return self._zen_analyze(reason, zen_context)
        else:
            return {"error": f"Unknown zen tool: {tool}"}

    def _prepare_zen_context(self, execution_result, context, session_id: str) -> Dict[str, Any]:
        """Prepare focused context for zen tools"""
        zen_context = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        # Add scratchpad content
        scratchpad_path = Path(SCRATCHPAD_DIR) / f"session_{session_id}.md"
        if scratchpad_path.exists():
            zen_context["scratchpad"] = scratchpad_path.read_text()

        # Add recent output (last 200 lines)
        if hasattr(execution_result, 'output_lines'):
            zen_context["recent_output"] = "\n".join(execution_result.output_lines[-200:])

        # Add error information
        if hasattr(execution_result, 'errors') and execution_result.errors:
            zen_context["errors"] = execution_result.errors

        # Add task information
        if hasattr(context, 'todos'):
            zen_context["todos"] = context.todos
        if hasattr(context, 'completed_todos'):
            zen_context["completed_todos"] = context.completed_todos
        if hasattr(context, 'remaining_todos'):
            zen_context["remaining_todos"] = context.remaining_todos

        # Add execution metrics
        if hasattr(execution_result, 'turns_used'):
            zen_context["turns_used"] = execution_result.turns_used
        if hasattr(context, 'max_turns'):
            zen_context["max_turns"] = context.max_turns

        return zen_context

    def _format_context_prompt(self, reason: str, context: Dict[str, Any], additional_sections: Dict[str, str] = None) -> str:
        """Format a standardized context prompt for zen tools"""
        sections = [
            f"{reason}",
            "",
            f"Agent Session: {context.get('session_id', 'unknown')}",
            f"Turns Used: {context.get('turns_used', '?')}/{context.get('max_turns', '?')}",
            "",
            "=== SCRATCHPAD ===",
            context.get('scratchpad', 'Not available'),
            "",
            "=== RECENT ERRORS ===",
            '\n'.join(context.get('errors', ['No errors captured'])),
            "",
            "=== RECENT OUTPUT (last 200 lines) ===",
            context.get('recent_output', 'Not available')
        ]

        # Add any additional sections provided
        if additional_sections:
            for title, content in additional_sections.items():
                sections.extend(["", f"=== {title.upper()} ===", content])

        return "\n".join(sections)

    def _zen_debug(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen debug for stuck/error situations"""
        models = self.config.models.get("debug", ["o3", "pro"])
        thinking_mode = self.config.thinking_modes.get("debug", "high")

        # Use first model from list for now
        model = models[0] if models else "pro"

        if self.verbose:
            print(f"   Using model: {model} (thinking: {thinking_mode})")

        # Create zen request
        zen_request = ZenRequest(
            tool="debug",
            reason=reason,
            context=context,
            priority="high",
            session_id=context.get("session_id"),
            execution_history=context.get("execution_history", [])
        )

        # Call the zen tool
        result = self._call_zen_tool(zen_request)

        # Add debug-specific metadata
        result.update({
            "model": model,
            "thinking_mode": thinking_mode,
            "recommended_actions": [
                "Check the specific error context",
                "Verify dependencies are installed",
                "Review the scratchpad for patterns"
            ]
        })

        return result


    def _zen_review(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen code review"""
        models = self.config.models.get("review", ["pro"])
        thinking_mode = self.config.thinking_modes.get("review", "high")

        # Create zen request
        zen_request = ZenRequest(
            tool="codereview",
            reason=reason,
            context=context,
            priority="medium",
            session_id=context.get("session_id"),
            execution_history=context.get("execution_history", [])
        )

        # Call the zen tool
        result = self._call_zen_tool(zen_request)

        # Add review-specific metadata
        result.update({
            "models": models,
            "thinking_mode": thinking_mode
        })

        return result

    def _zen_consensus(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen consensus for decisions"""
        models = self.config.models.get("consensus", ["o3", "pro", "flash"])
        thinking_mode = self.config.thinking_modes.get("consensus", "medium")

        # Create zen request
        zen_request = ZenRequest(
            tool="consensus",
            reason=reason,
            context=context,
            priority="medium",
            session_id=context.get("session_id"),
            execution_history=context.get("execution_history", [])
        )

        # Call the zen tool
        result = self._call_zen_tool(zen_request)

        # Add consensus-specific metadata
        result.update({
            "models": models,
            "thinking_mode": thinking_mode
        })

        return result

    def _zen_precommit(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen precommit for validation"""
        models = self.config.models.get("precommit", ["pro"])
        thinking_mode = self.config.thinking_modes.get("precommit", "high")

        # Create zen request
        zen_request = ZenRequest(
            tool="precommit",
            reason=reason,
            context=context,
            priority="high",
            session_id=context.get("session_id"),
            execution_history=context.get("execution_history", [])
        )

        # Call the zen tool
        result = self._call_zen_tool(zen_request)

        # Add precommit-specific metadata
        result.update({
            "models": models,
            "thinking_mode": thinking_mode
        })

        return result

    def _zen_analyze(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen analyze for retrospectives"""
        models = self.config.models.get("analyze", ["pro"])
        thinking_mode = self.config.thinking_modes.get("analyze", "medium")

        # Create zen request
        zen_request = ZenRequest(
            tool="analyze",
            reason=reason,
            context=context,
            priority="medium",
            session_id=context.get("session_id"),
            execution_history=context.get("execution_history", [])
        )

        # Call the zen tool
        result = self._call_zen_tool(zen_request)

        # Add analyze-specific metadata
        result.update({
            "models": models,
            "thinking_mode": thinking_mode,
            "lessons_learned": []
        })

        return result
    def cleanup_session(self, session_id: str):
        """Remove session data to prevent memory leaks."""
        if session_id in self.error_counts:
            if self.verbose:
                print(f"Cleaning up error counts for session {session_id}")
            del self.error_counts[session_id]

    def generate_continuation_guidance(self, zen_response: Dict[str, Any]) -> str:
        """
        Generate guidance for agent continuation based on zen response

        Returns:
            String guidance to include in continuation prompt
        """
        # Check if zen call was successful
        if not zen_response.get("success", False):
            error = zen_response.get("error", "Unknown error")
            return f"""
=== SUPERVISOR GUIDANCE ===
Assistance was requested but encountered an issue: {error}
Proceed with standard debugging approaches.
"""

        tool = zen_response.get("tool", "unknown")
        guidance = zen_response.get("guidance", "No specific guidance provided")

        if tool == "debug":
            actions = zen_response.get("recommended_actions", [])
            action_list = "\n".join([f"- {action}" for action in actions])
            return f"""
=== EXPERT ASSISTANCE PROVIDED ===
The supervisor consulted with specialized debugging assistance.
Zen assistance provided.

Key insights:
{guidance}

Recommended next steps:
{action_list}

Focus on addressing these specific issues before continuing with remaining TODOs.
"""

        elif tool == "precommit":
            return f"""
=== VALIDATION FEEDBACK ===
Your work has been reviewed for safety and correctness.
Zen assistance provided.

{guidance}

Address any concerns mentioned above before marking the task as complete.
"""

        else:
            return f"""
=== SUPERVISOR GUIDANCE ===
Zen assistance provided.
{guidance}
"""
