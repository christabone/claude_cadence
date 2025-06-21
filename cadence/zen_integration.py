"""
Zen MCP integration for enhanced supervisor capabilities

This module provides integration with zen MCP tools for debugging,
code review, and collaborative analysis when agents need help.
"""

import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .config import ZenIntegrationConfig, SCRATCHPAD_DIR


@dataclass
class ZenRequest:
    """Request for zen assistance"""
    tool: str  # debug, review, consensus, precommit, analyze
    reason: str
    context: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, critical


class ZenIntegration:
    """Manages integration with zen MCP tools for supervisor assistance"""
    
    def __init__(self, config: ZenIntegrationConfig, verbose: bool = True):
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
        # Simple categorization - can be enhanced
        if "ModuleNotFoundError" in error or "ImportError" in error:
            return "import_error"
        elif "FileNotFoundError" in error:
            return "file_not_found"
        elif "SyntaxError" in error:
            return "syntax_error"
        elif "TypeError" in error or "AttributeError" in error:
            return "type_error"
        elif "Permission" in error:
            return "permission_error"
        else:
            # Use first few words as category
            words = error.split()[:5]
            return "_".join(words).lower()
            
    def _detect_cutoff(self, result, context) -> bool:
        """Detect if execution was likely cut off at turn limit"""
        # Check if task is incomplete
        if hasattr(result, 'task_complete') and result.task_complete:
            return False  # Task completed successfully
            
        # Check for signs of being cut off
        if hasattr(result, 'output_lines') and len(result.output_lines) > 0:
            last_lines = "\n".join(result.output_lines[-50:])
            
            # Look for signs of incomplete work
            cutoff_indicators = [
                "ALL TASKS COMPLETE" not in last_lines,
                "HELP NEEDED" not in last_lines,  # Not a help request
                hasattr(context, 'remaining_todos') and len(context.remaining_todos) > 0,
                "in progress" in last_lines.lower(),
                "working on" in last_lines.lower(),
                "next, i'll" in last_lines.lower(),
                "let me" in last_lines.lower()
            ]
            
            # If multiple indicators suggest cutoff
            if sum(cutoff_indicators) >= 3:
                return True
                
        return False
            
    def _task_requires_validation(self, task_description: str) -> Optional[str]:
        """Check if task matches validation patterns"""
        if not task_description:
            return None
            
        task_lower = task_description.lower()
        for pattern in self.config.validate_on_complete:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*")
            if re.search(regex_pattern, task_lower):
                return f"Matches pattern: {pattern}"
                
        return None
        
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
        
    def _zen_debug(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen debug for stuck/error situations"""
        models = self.config.models.get("debug", ["o3", "pro"])
        thinking_mode = self.config.thinking_modes.get("debug", "high")
        
        prompt = f"""
{reason}

Agent Session: {context.get('session_id', 'unknown')}
Turns Used: {context.get('turns_used', '?')}/{context.get('max_turns', '?')}

=== SCRATCHPAD ===
{context.get('scratchpad', 'Not available')}

=== RECENT ERRORS ===
{chr(10).join(context.get('errors', ['No errors captured']))}

=== RECENT OUTPUT (last 200 lines) ===
{context.get('recent_output', 'Not available')}

Please analyze what's going wrong and provide specific guidance to help the agent get unstuck.
Focus on actionable next steps.
"""
        
        # Use first model from list for now
        model = models[0] if models else "pro"
        
        if self.verbose:
            print(f"   Using model: {model} (thinking: {thinking_mode})")
            
        # Build zen command
        cmd = [
            "claude", "-p", prompt,
            "--tool", "mcp",
            "--mcp-tool", f"zen__debug",
            "--model", model
        ]
        
        # For now, return mock response - real implementation would call zen
        return {
            "tool": "debug",
            "model": model,
            "thinking_mode": thinking_mode,
            "guidance": f"[Zen debug analysis would appear here for: {reason}]",
            "recommended_actions": [
                "Check the specific error context",
                "Verify dependencies are installed",
                "Review the scratchpad for patterns"
            ]
        }
        
    def _zen_review(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen code review"""
        # Similar pattern to debug
        return {
            "tool": "review",
            "guidance": f"[Zen review would appear here for: {reason}]"
        }
        
    def _zen_consensus(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen consensus for decisions"""
        models = self.config.models.get("consensus", ["o3", "pro", "flash"])
        return {
            "tool": "consensus",
            "models": models,
            "guidance": f"[Zen consensus would appear here for: {reason}]"
        }
        
    def _zen_precommit(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen precommit for validation"""
        return {
            "tool": "precommit",
            "guidance": f"[Zen precommit validation would appear here for: {reason}]"
        }
        
    def _zen_analyze(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call zen analyze for retrospectives"""
        return {
            "tool": "analyze",
            "guidance": f"[Zen analysis would appear here for: {reason}]",
            "lessons_learned": []
        }
        
    def generate_continuation_guidance(self, zen_response: Dict[str, Any]) -> str:
        """
        Generate guidance for agent continuation based on zen response
        
        Returns:
            String guidance to include in continuation prompt
        """
        tool = zen_response.get("tool", "unknown")
        guidance = zen_response.get("guidance", "No specific guidance provided")
        
        if tool == "debug":
            actions = zen_response.get("recommended_actions", [])
            action_list = "\n".join([f"- {action}" for action in actions])
            return f"""
=== EXPERT ASSISTANCE PROVIDED ===
The supervisor consulted with specialized debugging assistance.

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

{guidance}

Address any concerns mentioned above before marking the task as complete.
"""
        
        else:
            return f"""
=== SUPERVISOR GUIDANCE ===
{guidance}
"""