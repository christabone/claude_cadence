"""
Prompt management for Claude Cadence

This module handles the generation and management of prompts for both
supervisors and agents, maintaining context across executions.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

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

    def _get_common_agent_context(self) -> Dict[str, Any]:
        """Get common agent context configuration sections"""
        return {
            'core_agent_context': self.loader.config.get('core_agent_context', {}),
            'shared_agent_context': self.loader.config.get('shared_agent_context', {}),
            'safety_notice_section': self.loader.config.get('safety_notice_section', ''),
            'serena_setup': self.loader.config.get('serena_setup', '')
        }

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
                'project_path': context.project_path,   # Unified project path
                **self._get_common_agent_context()  # Include common agent context
            }

            # Generate TODO list
            todo_items = []
            for i, todo in enumerate(context.todos, 1):
                todo_templates = self.loader.config.get('todo_templates', {})
                item = self.loader.format_template(
                    todo_templates.get('todo_item', 'TODO {number}: {todo_text}'),
                    {
                        'number': i,
                        'todo_text': todo
                    }
                )
                todo_items.append(item)

            todo_list_str = "\n".join(todo_items)
            todo_templates = self.loader.config.get('todo_templates', {})
            prompt_context['todo_list'] = self.loader.format_template(
                todo_templates.get('todo_list', '{todo_items}'),
                {'todo_items': todo_list_str}
            )

            # Build initial prompt from sections
            agent_prompts = self.loader.config.get('agent_prompts', {})
            initial_prompt = agent_prompts.get('initial', {})
            sections = initial_prompt.get('sections', [])
            return self.loader.build_prompt_from_sections(sections, prompt_context)


    def _determine_continuation_type(self, supervisor_analysis: Dict[str, Any]) -> str:
        """Determine the type of continuation based on supervisor analysis"""
        todo_templates = self.loader.config.get('todo_templates', {})
        continuation_types = todo_templates.get('continuation_types', {})

        if supervisor_analysis.get('all_complete', False):
            return continuation_types.get('complete_new_tasks', 'complete_new_tasks')
        elif supervisor_analysis.get('has_issues', False):
            return continuation_types.get('fixing_issues', 'fixing_issues')
        else:
            return continuation_types.get('incomplete', 'incomplete')

    def _generate_supervisor_analysis_section(self, supervisor_analysis: Dict[str, Any], analysis_guidance: str) -> str:
        """Generate the supervisor analysis section of the prompt"""
        todo_templates = self.loader.config.get('todo_templates', {})

        if supervisor_analysis.get('all_complete', False):
            return self.loader.format_template(
                todo_templates.get('supervisor_complete_analysis', 'Previous work completed. {previous_work_summary}\nNew objectives: {new_objectives}'),
                {
                    'previous_work_summary': supervisor_analysis.get('work_summary', 'See previous scratchpad'),
                    'new_objectives': supervisor_analysis.get('new_objectives', 'Complete the new TODOs below')
                }
            )
        else:
            return self.loader.format_template(
                todo_templates.get('supervisor_incomplete_analysis', 'Previous work summary: {previous_work_summary}\nIssues found: {issues_found}\nGuidance: {specific_guidance}'),
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
        todo_templates = self.loader.config.get('todo_templates', {})

        for i, todo in enumerate(context.remaining_todos[:10], 1):
            item = self.loader.format_template(
                todo_templates.get('todo_item', 'TODO {number}: {todo_text}'),
                {
                    'number': i,
                    'todo_text': todo
                }
            )
            remaining_items.append(item)

        todo_list_str = "\n".join(remaining_items)
        return self.loader.format_template(
            todo_templates.get('todo_list', '{todo_items}'),
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
        todo_templates = self.loader.config.get('todo_templates', {})
        return self.loader.format_template(
            todo_templates.get('issues_section', '=== ISSUES ENCOUNTERED ===\n{issue_list}'),
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
                'project_path': context.project_path,  # Use context.project_path directly
                **self._get_common_agent_context()  # Include common agent context
            }

            # Build continuation prompt from sections
            agent_prompts = self.loader.config.get('agent_prompts', {})
            continuation_prompt = agent_prompts.get('continuation', {})
            sections = continuation_prompt.get('sections', [])
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
        supervisor_prompts = self.loader.config.get('supervisor_prompts', {})

        if context.completed_todos or context.remaining_todos:
            prompt_context['task_progress'] = self.loader.format_template(
                supervisor_prompts.get('task_progress_template', 'Completed: {completed_count}, Remaining: {remaining_count}'),
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
                    supervisor_prompts.get('history_item', 'Execution {num}: {summary}'),
                    {
                        'num': i,
                        'summary': exec.get('summary', 'No summary available')
                    }
                )
                history_items.append(item)

            prompt_context['execution_history'] = self.loader.format_template(
                supervisor_prompts.get('execution_history_template', 'Previous executions:\n{history_items}'),
                {'history_items': "\n".join(history_items)}
            )
        else:
            prompt_context['execution_history'] = ""

        # Build supervisor prompt from sections
        analysis_prompt = supervisor_prompts.get('analysis', {})
        sections = analysis_prompt.get('sections', [])
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
        final_summary = self.loader.config.get('final_summary', {})

        if context.completed_todos:
            completed_list = "\n".join([f"✅ {todo}" for todo in context.completed_todos])
            prompt_context['completed_section'] = self.loader.format_template(
                final_summary.get('completed_section', 'Completed tasks:\n{completed_list}'),
                {'completed_list': completed_list}
            )
        else:
            prompt_context['completed_section'] = ""

        # Add incomplete tasks section
        if context.remaining_todos:
            incomplete_list = "\n".join([f"❌ {todo}" for todo in context.remaining_todos])
            prompt_context['incomplete_section'] = self.loader.format_template(
                final_summary.get('incomplete_section', 'Incomplete tasks:\n{incomplete_list}'),
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
                final_summary.get('recommendations', 'Next focus areas:\n{focus_items}'),
                {'focus_items': focus_items}
            )
        else:
            prompt_context['recommendations'] = ""

        # Format the complete summary
        return self.loader.format_template(
            final_summary.get('template', 'Summary:\nExecutions: {executions_count}\nTotal turns: {total_turns}\n{completed_section}\n{incomplete_section}'),
            prompt_context
        )
