"""
Prompt management for Claude Cadence

This module handles the generation and management of prompts for both
supervisors and agents, maintaining context across executions.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from jinja2 import Template

# Output processing constants
MAX_OUTPUT_TRUNCATE_LENGTH = 3000  # Max chars before truncating supervisor analysis
SECONDS_PER_TURN_ESTIMATE = 30    # Rough estimate for duration calculations


@dataclass
class ExecutionContext:
    """Maintains context across execution boundaries"""
    todos: List[str]
    max_turns: int
    completed_todos: List[str] = field(default_factory=list)
    remaining_todos: List[str] = field(default_factory=list)
    issues_encountered: List[str] = field(default_factory=list)
    previous_guidance: List[str] = field(default_factory=list)
    continuation_context: Optional[str] = None
    project_path: Optional[str] = None  # For Serena activation and general use




class PromptGenerator:
    """Generates context-aware prompts for agents and supervisors"""

    def __init__(self, loader_or_config_path=None):
        """Initialize with a PromptLoader instance or config path"""
        # Import here to avoid circular imports
        from .prompt_loader import PromptLoader

        if isinstance(loader_or_config_path, PromptLoader):
            # It's already a PromptLoader instance
            self.loader = loader_or_config_path
        else:
            # It's a config path, create a new PromptLoader
            self.loader = PromptLoader(loader_or_config_path)

    def get_initial_prompt(self, *args, **kwargs):
        """Alias for generate_initial_todo_prompt"""
        return self.generate_initial_todo_prompt(*args, **kwargs)

    def get_continuation_prompt(self, *args, **kwargs):
        """Alias for generate_continuation_prompt"""
        return self.generate_continuation_prompt(*args, **kwargs)

    def generate_initial_todo_prompt(
            self,
            context: ExecutionContext,
            session_id: str = "unknown",
            task_numbers: str = "",
            project_root: str = None
        ) -> str:
            """Generate the initial prompt for the agent with TODOs"""

            prompt_context = {
                'max_turns': context.max_turns,
                'session_id': session_id,
                'task_numbers': task_numbers if task_numbers else "N/A",
                'project_path': context.project_path   # Unified project path
            }

            # Generate TODO list
            todo_items = []
            for i, todo in enumerate(context.todos, 1):
                item = self.loader.format_template(
                    self.loader.config['todo_templates']['todo_item'],
                    {
                        'number': i,
                        'todo_text': todo
                    }
                )
                todo_items.append(item)

            todo_list_str = "\n".join(todo_items)
            prompt_context['todo_list'] = self.loader.format_template(
                self.loader.config['todo_templates']['todo_list'],
                {'todo_items': todo_list_str}
            )

            # Build initial prompt from sections
            sections = self.loader.config['agent_prompts']['initial']['sections']
            return self.loader.build_prompt_from_sections(sections, prompt_context)


    def _determine_continuation_type(self, supervisor_analysis: Dict[str, Any]) -> str:
        """Determine the type of continuation based on supervisor analysis"""
        if supervisor_analysis.get('all_complete', False):
            return self.loader.config['todo_templates']['continuation_types']['complete_new_tasks']
        elif supervisor_analysis.get('has_issues', False):
            return self.loader.config['todo_templates']['continuation_types']['fixing_issues']
        else:
            return self.loader.config['todo_templates']['continuation_types']['incomplete']

    def _generate_supervisor_analysis_section(self, supervisor_analysis: Dict[str, Any], analysis_guidance: str) -> str:
        """Generate the supervisor analysis section of the prompt"""
        if supervisor_analysis.get('all_complete', False):
            return self.loader.format_template(
                self.loader.config['todo_templates']['supervisor_complete_analysis'],
                {
                    'previous_work_summary': supervisor_analysis.get('work_summary', 'See previous scratchpad'),
                    'new_objectives': supervisor_analysis.get('new_objectives', 'Complete the new TODOs below')
                }
            )
        else:
            return self.loader.format_template(
                self.loader.config['todo_templates']['supervisor_incomplete_analysis'],
                {
                    'previous_work_summary': supervisor_analysis.get('work_summary', 'See previous scratchpad'),
                    'issues_found': supervisor_analysis.get('issues', 'None identified'),
                    'specific_guidance': analysis_guidance
                }
            )

    def _generate_task_status_section(self, context: ExecutionContext) -> str:
        """Generate the task status section"""
        if not (context.completed_todos or context.remaining_todos):
            return ""

        completed_summary = f"{len(context.completed_todos)} TODOs completed" if context.completed_todos else "No TODOs completed yet"
        remaining_summary = f"{len(context.remaining_todos)} TODOs remaining" if context.remaining_todos else "All TODOs complete"

        return f"""=== TASK STATUS ===
{completed_summary}
{remaining_summary}
"""

    def _generate_remaining_todos_section(self, context: ExecutionContext, supervisor_analysis: Dict[str, Any]) -> str:
        """Generate the remaining TODOs section"""
        if not context.remaining_todos:
            return "=== NO REMAINING TODOS ===\nAll previous TODOs have been completed."

        remaining_items = []
        for i, todo in enumerate(context.remaining_todos[:10], 1):
            item = self.loader.format_template(
                self.loader.config['todo_templates']['todo_item'],
                {
                    'number': i,
                    'todo_text': todo
                }
            )
            remaining_items.append(item)

        todo_list_str = "\n".join(remaining_items)
        return self.loader.format_template(
            self.loader.config['todo_templates']['todo_list'],
            {
                'todo_items': todo_list_str,
                'session_id': supervisor_analysis.get('session_id', 'unknown'),
                'task_numbers': supervisor_analysis.get('task_numbers', 'N/A')
            }
        )

    def _generate_issues_section(self, context: ExecutionContext) -> str:
        """Generate the issues section"""
        if not context.issues_encountered:
            return ""

        issue_list = "\n".join([f"⚠️  {issue}" for issue in context.issues_encountered[-3:]])
        return self.loader.format_template(
            self.loader.config['todo_templates']['issues_section'],
            {'issue_list': issue_list}
        )

    def generate_continuation_prompt(
            self,
            context: ExecutionContext,
            analysis_guidance: str,
            supervisor_analysis: Dict[str, Any]
        ) -> str:
            """Generate continuation prompt for resumed execution"""

            # Build prompt context with all sections
            prompt_context = {
                'max_turns': context.max_turns,
                'continuation_type': self._determine_continuation_type(supervisor_analysis),
                'session_id': supervisor_analysis.get('session_id', 'unknown'),
                'previous_session_id': supervisor_analysis.get('previous_session_id', 'unknown'),
                'next_steps_guidance': analysis_guidance,
                'supervisor_analysis': self._generate_supervisor_analysis_section(supervisor_analysis, analysis_guidance),
                'task_status_section': self._generate_task_status_section(context),
                'remaining_todos': self._generate_remaining_todos_section(context, supervisor_analysis),
                'issues_section': self._generate_issues_section(context),
                'project_path': context.project_path  # Use context.project_path directly
            }

            # Build continuation prompt from sections
            sections = self.loader.config['agent_prompts']['continuation']['sections']
            return self.loader.build_prompt_from_sections(sections, prompt_context)


    def generate_supervisor_analysis_prompt(
        self,
        execution_output: str,
        context: ExecutionContext,
        previous_executions: List[Dict]
    ) -> str:
        """Generate prompt for supervisor analysis"""

        # Truncate output if too long
        max_output_chars = MAX_OUTPUT_TRUNCATE_LENGTH
        if len(execution_output) > max_output_chars:
            execution_output = (
                execution_output[:max_output_chars//2] +
                "\n\n[... OUTPUT TRUNCATED ...]\n\n" +
                execution_output[-max_output_chars//2:]
            )

        prompt_context = {
            'max_turns': context.max_turns,
            'turns_used': 0,  # Will be set by supervisor
            'execution_output': execution_output
        }

        # Add task progress if available
        if context.completed_todos or context.remaining_todos:
            prompt_context['task_progress'] = self.loader.format_template(
                self.loader.config['supervisor_prompts']['task_progress_template'],
                {
                    'completed_count': len(context.completed_todos),
                    'remaining_count': len(context.remaining_todos)
                }
            )
        else:
            prompt_context['task_progress'] = ""

        # Add execution history
        if previous_executions:
            history_items = []
            for i, exec in enumerate(previous_executions[-2:], 1):  # Last 2 executions
                item = self.loader.format_template(
                    self.loader.config['supervisor_prompts']['history_item'],
                    {
                        'num': i,
                        'summary': exec.get('summary', 'No summary available')
                    }
                )
                history_items.append(item)

            prompt_context['execution_history'] = self.loader.format_template(
                self.loader.config['supervisor_prompts']['execution_history_template'],
                {'history_items': "\n".join(history_items)}
            )
        else:
            prompt_context['execution_history'] = ""

        # Build supervisor prompt from sections
        sections = self.loader.config['supervisor_prompts']['analysis']['sections']
        return self.loader.build_prompt_from_sections(sections, prompt_context)

    def generate_final_summary(
        self,
        executions: List[Dict],
        context: ExecutionContext,
        total_turns: int
    ) -> str:
        """Generate a final summary for the user"""

        duration_estimate = total_turns * SECONDS_PER_TURN_ESTIMATE

        prompt_context = {
            'executions_count': len(executions),
            'total_turns': total_turns,
            'duration_minutes': duration_estimate // 60
        }

        # Add completed tasks section
        if context.completed_todos:
            completed_list = "\n".join([f"✅ {todo}" for todo in context.completed_todos])
            prompt_context['completed_section'] = self.loader.format_template(
                self.loader.config['final_summary']['completed_section'],
                {'completed_list': completed_list}
            )
        else:
            prompt_context['completed_section'] = ""

        # Add incomplete tasks section
        if context.remaining_todos:
            incomplete_list = "\n".join([f"❌ {todo}" for todo in context.remaining_todos])
            prompt_context['incomplete_section'] = self.loader.format_template(
                self.loader.config['final_summary']['incomplete_section'],
                {'incomplete_list': incomplete_list}
            )
        else:
            prompt_context['incomplete_section'] = ""

        # Add execution progression
        progression_lines = []
        for i, exec in enumerate(executions, 1):
            status = "✅" if exec.get('success') else "⚠️"
            progression_lines.append(
                f"{status} Execution {i}: {exec.get('summary', 'No summary')}"
            )
        prompt_context['execution_progression'] = "\n".join(progression_lines)

        # Add recommendations if tasks remain
        if context.remaining_todos:
            focus_items = "\n".join([f"- {todo}" for todo in context.remaining_todos[:3]])
            prompt_context['recommendations'] = self.loader.format_template(
                self.loader.config['final_summary']['recommendations'],
                {'focus_items': focus_items}
            )
        else:
            prompt_context['recommendations'] = ""

        # Format the complete summary
        return self.loader.format_template(
            self.loader.config['final_summary']['template'],
            prompt_context
        )


class TodoPromptManager:
    """Manages prompts for TODO-based execution"""

    def __init__(self, todos: List[str], max_turns: int,
                 config_path: Optional[str] = None):
        self.context = ExecutionContext(
            todos=todos,
            max_turns=max_turns,
            remaining_todos=todos.copy()
        )
        self.generator = PromptGenerator(config_path)
        self.session_id = "unknown"
        self.task_numbers = ""

    def update_progress(self, completed_todos: List[str], remaining_todos: List[str]):
        """Update execution progress"""
        self.context.completed_todos = completed_todos
        self.context.remaining_todos = remaining_todos

    def add_issue(self, issue: str):
        """Add an issue encountered during execution"""
        self.context.issues_encountered.append(issue)

    def get_initial_prompt(self) -> str:
            """Get the initial agent prompt with TODOs"""
            return self.generator.generate_initial_todo_prompt(
                context=self.context,
                session_id=self.session_id,
                task_numbers=self.task_numbers
            )


    def get_continuation_prompt(self, analysis_guidance: str, continuation_context: str,
                               supervisor_analysis: Dict[str, Any]) -> str:
        """Get continuation prompt for resumed execution"""
        self.context.continuation_context = continuation_context
        return self.generator.generate_continuation_prompt(
            context=self.context,
            analysis_guidance=analysis_guidance,
            supervisor_analysis=supervisor_analysis
        )

    def get_supervisor_prompt(self, execution_output: str, previous_executions: List[Dict]) -> str:
        """Get supervisor analysis prompt"""
        return self.generator.generate_supervisor_analysis_prompt(
            execution_output=execution_output,
            context=self.context,
            previous_executions=previous_executions
        )



    def add_guidance(self, guidance: str):
        """Add guidance to history"""
        self.context.previous_guidance.append(guidance)

    def get_final_summary(self, executions: List[Dict], total_turns: int) -> str:
        """Get final summary for the user"""
        return self.generator.generate_final_summary(
            executions=executions,
            context=self.context,
            total_turns=total_turns
        )
