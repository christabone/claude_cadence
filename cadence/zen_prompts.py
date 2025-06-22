"""
Specialized prompts for Zen MCP tool usage

This module provides focused, scenario-specific prompts for each Zen tool
to ensure efficient assistance while preventing scope creep.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime


class ZenPrompts:
    """Specialized prompts for different Zen MCP scenarios"""
    
    @staticmethod
    def debug_prompt(reason: str, context: Dict[str, Any], zen_context: str) -> str:
        """
        Generate specialized prompt for Zen debug assistance
        
        Focus: Unblock the agent with minimal, targeted fixes
        """
        return f"""You are assisting a Claude agent that is stuck during task execution.

CRITICAL CONSTRAINTS:
1. Stay STRICTLY focused on unblocking the immediate issue
2. Provide MINIMAL fixes - just enough to get unstuck
3. DO NOT suggest architectural improvements or refactoring
4. Focus on the CURRENT error/blocker only
5. If you see broader issues, note them briefly but don't expand scope

STUCK REASON: {reason}

CURRENT TASK CONTEXT:
TODOs: {context.get('todos', [])}
Completed: {context.get('completed_todos', [])}
Remaining: {context.get('remaining_todos', [])}

AGENT SESSION INFO:
{zen_context}

REQUIRED OUTPUT:
1. Root cause of the current blocker (1-2 sentences)
2. MINIMAL fix to unblock (specific steps)
3. One-line summary of any broader issues to document

Remember: The goal is task completion, not perfection. Provide the smallest intervention needed."""

    @staticmethod
    def review_prompt(task_type: str, context: Dict[str, Any], code_context: str) -> str:
        """
        Generate specialized prompt for Zen code review
        
        Focus: Critical issues only, no style nitpicks
        """
        return f"""You are reviewing code for a specific task completion.

CRITICAL CONSTRAINTS:
1. Focus ONLY on functionality and critical bugs
2. IGNORE style issues, minor optimizations, or preferences
3. Only flag issues that would prevent the task from working
4. Keep feedback minimal and actionable
5. Document nice-to-have improvements separately

TASK TYPE: {task_type}

TASK CONTEXT:
Current Task: {context.get('current_task', 'Unknown')}
Implementation Goal: {context.get('task_description', 'Complete the assigned task')}

CODE TO REVIEW:
{code_context}

REQUIRED OUTPUT:
1. CRITICAL issues that must be fixed (if any)
2. PASS/FAIL verdict for task completion
3. Optional: One-line note about future improvements

Remember: We're validating task completion, not seeking perfection."""

    @staticmethod
    def consensus_prompt(decision_type: str, options: List[str], context: Dict[str, Any]) -> str:
        """
        Generate specialized prompt for Zen consensus decisions
        
        Focus: Quick, practical decisions to keep moving forward
        """
        return f"""Help make a quick implementation decision to keep the task moving.

CRITICAL CONSTRAINTS:
1. Choose the SIMPLEST option that meets requirements
2. Prioritize what can be implemented NOW
3. Avoid over-engineering or future-proofing
4. Make a clear decision - no "it depends"
5. Brief reasoning only (2-3 sentences max)

DECISION TYPE: {decision_type}

OPTIONS TO CONSIDER:
{chr(10).join(f"- {opt}" for opt in options)}

CURRENT CONTEXT:
Task: {context.get('current_task', 'Unknown')}
Time Remaining: {context.get('turns_remaining', 'Unknown')} turns

REQUIRED OUTPUT:
1. DECISION: [Choose one option]
2. REASON: [1-2 sentences why]
3. RISK: [Any critical risk to note in 1 sentence]

Make the practical choice that unblocks progress."""

    @staticmethod
    def precommit_prompt(task_description: str, changes_summary: str, context: Dict[str, Any]) -> str:
        """
        Generate specialized prompt for Zen precommit validation
        
        Focus: Safety and correctness checks only
        """
        return f"""Validate that a critical task has been completed safely.

CRITICAL CONSTRAINTS:
1. Check ONLY for safety, security, and correctness issues
2. Assume the implementation approach is acceptable if it works
3. Don't suggest alternative implementations
4. Focus on "Does it work?" not "Could it be better?"
5. Be decisive - clear PASS or FAIL

TASK COMPLETED: {task_description}

CHANGES MADE:
{changes_summary}

VALIDATION CONTEXT:
Task Type: {context.get('task_type', 'Unknown')}
Critical Patterns: {context.get('critical_patterns', [])}

REQUIRED CHECKS:
1. Security vulnerabilities (auth, injection, exposure)
2. Data integrity risks (corruption, loss)
3. Runtime errors (null refs, type errors)
4. Resource leaks (memory, connections)
5. Breaking changes (API compatibility)

REQUIRED OUTPUT:
1. VERDICT: PASS or FAIL
2. If FAIL: Specific issue(s) that must be fixed
3. If PASS with concerns: One-line note for documentation

Focus on critical issues only. Minor improvements are not failures."""

    @staticmethod
    def analyze_prompt(cutoff_reason: str, progress_summary: str, context: Dict[str, Any]) -> str:
        """
        Generate specialized prompt for Zen retrospective analysis
        
        Focus: Learn what happened and plan minimal next steps
        """
        return f"""Analyze why an agent task was cut off and plan minimal recovery.

CRITICAL CONSTRAINTS:
1. Focus on understanding what was completed vs. remaining
2. Suggest MINIMAL steps to complete the task
3. Don't redesign or rearchitect - work with what exists
4. Keep analysis brief and actionable
5. Document broader insights separately

CUTOFF REASON: {cutoff_reason}

PROGRESS MADE:
{progress_summary}

TASK CONTEXT:
Original TODOs: {context.get('original_todos', [])}
Completed: {context.get('completed_todos', [])}
Remaining: {context.get('remaining_todos', [])}
Turns Used: {context.get('turns_used', 'Unknown')}/{context.get('max_turns', 'Unknown')}

REQUIRED OUTPUT:
1. COMPLETED: What actually got done (bullet points)
2. REMAINING: What's left to finish the task (bullet points)
3. NEXT STEPS: Minimal actions to complete (1-3 items)
4. LESSON: One-line insight for future task planning

Keep focus on task completion, not optimization."""

    @staticmethod
    def format_zen_guidance(tool: str, zen_response: Dict[str, Any], stay_focused: bool = True) -> str:
        """
        Format Zen response into guidance for the agent
        
        Args:
            tool: The Zen tool that was called
            zen_response: Response from Zen tool
            stay_focused: Whether to include focus reminder
            
        Returns:
            Formatted guidance string
        """
        base_guidance = zen_response.get('guidance', 'No specific guidance provided')
        
        if not stay_focused:
            return base_guidance
            
        focus_reminder = """
IMPORTANT: While this analysis provides useful insights, remember to:
- Stay focused on completing your current TODOs
- Implement only the minimal fixes needed
- Document broader suggestions for later review
- Avoid scope creep or architectural changes
"""
        
        if tool == 'debug':
            return f"""{base_guidance}

{focus_reminder}

Focus on the specific fix suggested and continue with your TODOs."""
        
        elif tool == 'precommit':
            return f"""{base_guidance}

{focus_reminder}

Address only the critical issues mentioned, then proceed."""
            
        elif tool == 'analyze':
            return f"""{base_guidance}

{focus_reminder}

Complete the minimal remaining steps identified above."""
            
        else:
            return f"""{base_guidance}

{focus_reminder}"""

    @staticmethod
    def generate_zen_documentation(session_id: str, zen_requests: List[Dict[str, Any]]) -> str:
        """
        Generate markdown documentation of Zen suggestions for later review
        
        Args:
            session_id: Current session ID
            zen_requests: List of Zen interactions from this session
            
        Returns:
            Markdown formatted documentation
        """
        doc_parts = [
            f"# Zen MCP Suggestions - Session {session_id}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Overview",
            f"This session involved {len(zen_requests)} Zen assistance requests.",
            "",
            "## Suggestions for Future Consideration",
            ""
        ]
        
        for i, request in enumerate(zen_requests, 1):
            tool = request.get('tool', 'unknown')
            reason = request.get('reason', 'No reason provided')
            response = request.get('response', {})
            
            doc_parts.extend([
                f"### {i}. {tool.capitalize()} Assistance",
                f"**Reason**: {reason}",
                "",
                "**Key Insights**:",
                response.get('guidance', 'No guidance provided'),
                ""
            ])
            
            # Add tool-specific sections
            if tool == 'debug' and 'recommended_actions' in response:
                doc_parts.extend([
                    "**Recommended Future Improvements**:",
                    *[f"- {action}" for action in response['recommended_actions']],
                    ""
                ])
            
            elif tool == 'analyze' and 'lessons_learned' in response:
                doc_parts.extend([
                    "**Lessons for Future Tasks**:",
                    *[f"- {lesson}" for lesson in response['lessons_learned']],
                    ""
                ])
            
            elif tool == 'review' and 'future_improvements' in response:
                doc_parts.extend([
                    "**Code Quality Improvements**:",
                    response['future_improvements'],
                    ""
                ])
            
            doc_parts.append("---")
            doc_parts.append("")
        
        # Add summary section
        doc_parts.extend([
            "## Summary",
            "",
            "These suggestions were noted during task execution but deferred to maintain focus.",
            "Consider reviewing and implementing these improvements in future iterations.",
            "",
            "### Priority Areas:",
            "1. Performance optimizations noted during debugging",
            "2. Architectural improvements identified during analysis", 
            "3. Code quality enhancements from reviews",
            "",
            "_Remember: Task completion takes precedence over perfection._"
        ])
        
        return "\n".join(doc_parts)


# Convenience functions for common scenarios
def get_debug_prompt(reason: str, context: Dict[str, Any], zen_context: str) -> str:
    """Get specialized debug prompt"""
    return ZenPrompts.debug_prompt(reason, context, zen_context)

def get_review_prompt(task_type: str, context: Dict[str, Any], code_context: str) -> str:
    """Get specialized code review prompt"""
    return ZenPrompts.review_prompt(task_type, context, code_context)

def get_consensus_prompt(decision_type: str, options: List[str], context: Dict[str, Any]) -> str:
    """Get specialized consensus prompt"""
    return ZenPrompts.consensus_prompt(decision_type, options, context)

def get_precommit_prompt(task_description: str, changes_summary: str, context: Dict[str, Any]) -> str:
    """Get specialized precommit validation prompt"""
    return ZenPrompts.precommit_prompt(task_description, changes_summary, context)

def get_analyze_prompt(cutoff_reason: str, progress_summary: str, context: Dict[str, Any]) -> str:
    """Get specialized analysis prompt"""
    return ZenPrompts.analyze_prompt(cutoff_reason, progress_summary, context)