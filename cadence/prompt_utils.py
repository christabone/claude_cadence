"""
Shared prompt builder utilities for Claude Cadence

This module provides centralized prompt building functionality
to eliminate duplication across orchestrator and supervisor.
"""

from typing import List, Optional
from .constants import AgentPromptDefaults


class PromptBuilder:
    """Centralized prompt building for agent execution"""
    
    TASK_GUIDELINES = """=== TASK EXECUTION GUIDELINES ===

You have been given specific TODOs to complete. Focus ONLY on these tasks.

IMPORTANT:
- Work naturally and efficiently to complete all TODOs
- The moment ALL TODOs are complete, declare "{completion_signal}" and exit
- If you get stuck or need help, declare "{help_signal}" with explanation
- {safety_limit_message}
- Quality matters more than speed

"""
    
    @staticmethod
    def build_agent_prompt(todos: List[str], 
                          guidance: str = "", 
                          max_turns: int = AgentPromptDefaults.DEFAULT_MAX_TURNS,
                          continuation_context: Optional[str] = None) -> str:
        """
        Build prompt for agent with TODOs and guidance
        
        Args:
            todos: List of TODO items for the agent
            guidance: Optional supervisor guidance
            max_turns: Maximum turns allowed
            continuation_context: Optional context from previous execution
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add continuation context if provided
        if continuation_context:
            prompt_parts.append(f"=== CONTINUATION CONTEXT ===\n{continuation_context}\n")
        
        # Add guidance if provided
        if guidance:
            prompt_parts.append(f"{AgentPromptDefaults.SUPERVISOR_GUIDANCE_HEADER}\n{guidance}\n")
        
        # Add execution guidelines
        guidelines = PromptBuilder.TASK_GUIDELINES.format(
            completion_signal=AgentPromptDefaults.COMPLETION_SIGNAL,
            help_signal=AgentPromptDefaults.HELP_SIGNAL,
            safety_limit_message=AgentPromptDefaults.SAFETY_LIMIT_MESSAGE.format(max_turns=max_turns)
        )
        prompt_parts.append(guidelines)
        
        # Add TODO list
        prompt_parts.append(AgentPromptDefaults.TODO_LIST_HEADER)
        for i, todo in enumerate(todos, 1):
            prompt_parts.append(f"{i}. {todo}")
        
        prompt_parts.append("\nBegin working on these TODOs now.")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_supervisor_analysis_prompt(context: str, 
                                       include_json_format: bool = True) -> str:
        """
        Build prompt for supervisor AI analysis
        
        Args:
            context: Context about current task state
            include_json_format: Whether to request JSON response
            
        Returns:
            Formatted analysis prompt
        """
        prompt = context + """

Analyze the situation and provide:
1. Whether the agent should execute these TODOs (should_execute: true/false)
2. Specific guidance for the agent to complete these subtasks successfully
3. Your reasoning for this decision
4. Whether zen assistance might be helpful (needs_assistance: true/false)

Consider:
- The complexity and nature of the subtasks
- Any previous execution results and errors
- The best approach for the agent to succeed
- Whether the subtasks are clear and well-defined
"""
        
        if include_json_format:
            prompt += "\nRespond in JSON format with keys: should_execute, guidance, reasoning, needs_assistance"
            
        return prompt
    
    @staticmethod
    def build_task_context(task_id: str, title: str, 
                          completed_subtasks: int, total_subtasks: int,
                          todos: List[str]) -> str:
        """
        Build context description for a task
        
        Args:
            task_id: Task identifier
            title: Task title
            completed_subtasks: Number of completed subtasks
            total_subtasks: Total number of subtasks
            todos: List of remaining TODOs
            
        Returns:
            Formatted context string
        """
        context = f"""
You are a supervisor analyzing the current state of task execution.

Current Task: {task_id} - {title}
Status: {completed_subtasks}/{total_subtasks} subtasks complete

Remaining TODOs:
{chr(10).join(f'- {todo}' for todo in todos)}
"""
        return context
    
    @staticmethod
    def format_execution_results(success: bool, execution_time: float,
                               completed_normally: bool, requested_help: bool,
                               error_count: int) -> str:
        """
        Format execution results for logging or analysis
        
        Args:
            success: Whether execution succeeded
            execution_time: Time taken in seconds
            completed_normally: Whether agent declared completion
            requested_help: Whether agent requested help
            error_count: Number of errors encountered
            
        Returns:
            Formatted results string
        """
        return f"""
Previous Execution Results:
- Success: {success}
- Execution Time: {execution_time:.2f}s
- Completed Normally: {completed_normally}
- Requested Help: {requested_help}
- Errors: {error_count}
"""