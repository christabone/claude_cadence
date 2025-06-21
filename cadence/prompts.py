"""
Prompt management for Claude Cadence

This module handles the generation and management of prompts for both
supervisors and agents, maintaining context across checkpoints.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml
import re


@dataclass
class PromptContext:
    """Maintains context across checkpoint boundaries"""
    original_task: str
    current_checkpoint: int
    max_checkpoints: int
    checkpoint_turns: int
    completed_tasks: List[str] = field(default_factory=list)
    remaining_tasks: List[str] = field(default_factory=list)
    issues_encountered: List[str] = field(default_factory=list)
    previous_guidance: List[str] = field(default_factory=list)


class YAMLPromptLoader:
    """Loads and manages prompts from YAML configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with YAML config file"""
        if config_path is None:
            # Default to prompts.yaml in same directory
            config_path = Path(__file__).parent / "prompts.yaml"
        else:
            config_path = Path(config_path)
            
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise IOError(f"Prompt configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
            
    def format_template(self, template: str, context: Dict[str, Any], visited: Optional[set] = None) -> str:
        """Format a template string with context variables"""
        if visited is None:
            visited = set()
            
        # Handle nested references like {shared_agent_context.supervision_explanation}
        def replace_ref(match):
            ref = match.group(1)
            
            # Check for cycles
            if ref in visited:
                return f"{{CYCLE_DETECTED: {ref}}}"
                
            parts = ref.split('.')
            
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    # Not a config reference, check context
                    return match.group(0) if ref not in context else str(context[ref])
                    
            # If we found a config value, format it recursively
            if isinstance(value, str):
                visited.add(ref)
                result = self.format_template(value, context, visited)
                visited.remove(ref)
                return result
            return str(value)
            
        # First pass: replace config references
        result = re.sub(r'\{([^}]+)\}', replace_ref, template)
        
        # Second pass: simple format with context
        try:
            result = result.format(**context)
        except KeyError:
            # Some keys might be missing, that's OK for optional sections
            pass
            
        return result
        
    def build_prompt_from_sections(self, sections: List[str], context: Dict[str, Any]) -> str:
        """Build a prompt from a list of sections"""
        formatted_sections = []
        
        for section in sections:
            # Skip empty sections or those with missing required context
            if section.strip():
                try:
                    formatted = self.format_template(section, context)
                    if formatted.strip() and '{' not in formatted:  # No unresolved vars
                        formatted_sections.append(formatted)
                except KeyError:
                    continue
                    
        return "\n".join(formatted_sections)


class PromptGenerator:
    """Generates context-aware prompts for agents and supervisors"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path"""
        self.loader = YAMLPromptLoader(config_path)
    
    def generate_initial_agent_prompt(
        self,
        task_description: str,
        checkpoint_turns: int,
        max_checkpoints: int,
        task_list: Optional[List[Dict]] = None
    ) -> str:
        """Generate the initial prompt for the agent"""
        
        context = {
            'checkpoint_turns': checkpoint_turns,
            'max_checkpoints': max_checkpoints,
            'task_description': task_description
        }
        
        # Generate task breakdown if tasks provided
        if task_list:
            task_items = []
            for task in task_list:
                status_icon = "✅" if task.get('status') == 'done' else "⏳"
                item = self.loader.format_template(
                    self.loader.config['task_templates']['task_item'],
                    {
                        'status_icon': status_icon,
                        'task_id': task.get('id', '?'),
                        'task_title': task.get('title', 'Untitled')
                    }
                )
                task_items.append(item)
                
                if task.get('description'):
                    details = self.loader.format_template(
                        self.loader.config['task_templates']['task_details'],
                        {'task_description': task['description']}
                    )
                    task_items.append(details)
                    
            task_list_str = "\n".join(task_items)
            context['task_breakdown'] = self.loader.format_template(
                self.loader.config['task_templates']['task_breakdown'],
                {'task_list': task_list_str}
            )
        else:
            context['task_breakdown'] = ""
            
        # Build initial prompt from sections
        sections = self.loader.config['agent_prompts']['initial']['sections']
        return self.loader.build_prompt_from_sections(sections, context)
    
    def generate_continuation_prompt(
        self,
        context: PromptContext,
        analysis_guidance: str,
        checkpoint_summary: Dict
    ) -> str:
        """Generate continuation prompt for subsequent checkpoints"""
        
        prompt_context = {
            'current_checkpoint': context.current_checkpoint,
            'max_checkpoints': context.max_checkpoints,
            'checkpoint_turns': context.checkpoint_turns,
            'supervisor_guidance': analysis_guidance,
            'original_task': context.original_task
        }
        
        # Generate progress summary if needed
        if context.completed_tasks:
            completed_items = "\n".join([f"✅ {task}" for task in context.completed_tasks[-5:]])
            prompt_context['progress_summary'] = self.loader.format_template(
                self.loader.config['task_templates']['progress_summary'],
                {'completed_items': completed_items}
            )
        else:
            prompt_context['progress_summary'] = ""
            
        # Generate remaining work section
        if context.remaining_tasks:
            remaining_items = "\n".join([f"⏳ {task}" for task in context.remaining_tasks[:5]])
            prompt_context['remaining_work'] = self.loader.format_template(
                self.loader.config['task_templates']['remaining_work'],
                {'remaining_items': remaining_items}
            )
        else:
            prompt_context['remaining_work'] = ""
            
        # Generate issues section
        if context.issues_encountered:
            issue_list = "\n".join([f"⚠️  {issue}" for issue in context.issues_encountered[-3:]])
            prompt_context['issues_section'] = self.loader.format_template(
                self.loader.config['task_templates']['issues_section'],
                {'issue_list': issue_list}
            )
        else:
            prompt_context['issues_section'] = ""
            
        # Add checkpoint-specific warnings
        remaining_checkpoints = context.max_checkpoints - context.current_checkpoint
        if remaining_checkpoints == 0:
            prompt_context['checkpoint_warnings'] = self.loader.config['checkpoint_warnings']['final']
        elif remaining_checkpoints == 1:
            prompt_context['checkpoint_warnings'] = self.loader.config['checkpoint_warnings']['penultimate']
        else:
            prompt_context['checkpoint_warnings'] = ""
            
        # Build continuation prompt from sections
        sections = self.loader.config['agent_prompts']['continuation']['sections']
        return self.loader.build_prompt_from_sections(sections, prompt_context)
    
    def generate_supervisor_analysis_prompt(
        self,
        checkpoint_output: str,
        context: PromptContext,
        previous_checkpoints: List[Dict]
    ) -> str:
        """Generate prompt for supervisor analysis"""
        
        # Truncate output if too long
        max_output_chars = 3000
        if len(checkpoint_output) > max_output_chars:
            checkpoint_output = (
                checkpoint_output[:max_output_chars//2] + 
                "\n\n[... OUTPUT TRUNCATED ...]\n\n" + 
                checkpoint_output[-max_output_chars//2:]
            )
            
        prompt_context = {
            'current_checkpoint': context.current_checkpoint,
            'max_checkpoints': context.max_checkpoints,
            'turns_used': context.checkpoint_turns,
            'original_task': context.original_task,
            'checkpoint_output': checkpoint_output
        }
        
        # Add task progress if available
        if context.completed_tasks or context.remaining_tasks:
            prompt_context['task_progress'] = self.loader.format_template(
                self.loader.config['supervisor_prompts']['task_progress_template'],
                {
                    'completed_count': len(context.completed_tasks),
                    'remaining_count': len(context.remaining_tasks)
                }
            )
        else:
            prompt_context['task_progress'] = ""
            
        # Add checkpoint history
        if previous_checkpoints:
            history_items = []
            for cp in previous_checkpoints[-2:]:  # Last 2 checkpoints
                item = self.loader.format_template(
                    self.loader.config['supervisor_prompts']['history_item'],
                    {
                        'num': cp['num'],
                        'summary': cp.get('summary', 'No summary available')
                    }
                )
                history_items.append(item)
                
            prompt_context['checkpoint_history'] = self.loader.format_template(
                self.loader.config['supervisor_prompts']['checkpoint_history_template'],
                {'history_items': "\n".join(history_items)}
            )
        else:
            prompt_context['checkpoint_history'] = ""
            
        # Build supervisor prompt from sections
        sections = self.loader.config['supervisor_prompts']['analysis']['sections']
        return self.loader.build_prompt_from_sections(sections, prompt_context)
    
    def generate_final_summary_prompt(
        self,
        all_checkpoints: List[Dict],
        context: PromptContext,
        total_cost: float
    ) -> str:
        """Generate a final summary for the user"""
        
        duration_estimate = len(all_checkpoints) * context.checkpoint_turns * 30  # ~30 sec per turn
        
        prompt_context = {
            'checkpoints_used': len(all_checkpoints),
            'max_checkpoints': context.max_checkpoints,
            'total_cost': total_cost,
            'duration_minutes': duration_estimate // 60,
            'original_task': context.original_task
        }
        
        # Add completed tasks section
        if context.completed_tasks:
            completed_list = "\n".join([f"✅ {task}" for task in context.completed_tasks])
            prompt_context['completed_section'] = self.loader.format_template(
                self.loader.config['final_summary']['completed_section'],
                {'completed_list': completed_list}
            )
        else:
            prompt_context['completed_section'] = ""
            
        # Add incomplete tasks section
        if context.remaining_tasks:
            incomplete_list = "\n".join([f"❌ {task}" for task in context.remaining_tasks])
            prompt_context['incomplete_section'] = self.loader.format_template(
                self.loader.config['final_summary']['incomplete_section'],
                {'incomplete_list': incomplete_list}
            )
        else:
            prompt_context['incomplete_section'] = ""
            
        # Add checkpoint progression
        progression_lines = []
        for i, cp in enumerate(all_checkpoints, 1):
            status = "✅" if cp.get('success') else "⚠️"
            progression_lines.append(
                f"{status} Checkpoint {i}: {cp.get('summary', 'No summary')}"
            )
        prompt_context['checkpoint_progression'] = "\n".join(progression_lines)
        
        # Add recommendations if tasks remain
        if context.remaining_tasks:
            focus_items = "\n".join([f"- {task}" for task in context.remaining_tasks[:3]])
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


class ContextAwarePromptManager:
    """Manages prompts with full context awareness across checkpoints"""
    
    def __init__(self, original_task: str, checkpoint_turns: int, max_checkpoints: int, 
                 config_path: Optional[str] = None):
        self.context = PromptContext(
            original_task=original_task,
            current_checkpoint=1,
            max_checkpoints=max_checkpoints,
            checkpoint_turns=checkpoint_turns
        )
        self.generator = PromptGenerator(config_path)
        
    def get_initial_prompt(self, task_list: Optional[List[Dict]] = None) -> str:
        """Get the initial agent prompt"""
        return self.generator.generate_initial_agent_prompt(
            task_description=self.context.original_task,
            checkpoint_turns=self.context.checkpoint_turns,
            max_checkpoints=self.context.max_checkpoints,
            task_list=task_list
        )
        
    def get_continuation_prompt(self, analysis_guidance: str, checkpoint_summary: Dict) -> str:
        """Get continuation prompt for next checkpoint"""
        return self.generator.generate_continuation_prompt(
            context=self.context,
            analysis_guidance=analysis_guidance,
            checkpoint_summary=checkpoint_summary
        )
    
    def advance_to_next_checkpoint(self):
        """Increment the checkpoint counter"""
        self.context.current_checkpoint += 1
        
    def get_supervisor_prompt(self, checkpoint_output: str, previous_checkpoints: List[Dict]) -> str:
        """Get supervisor analysis prompt"""
        return self.generator.generate_supervisor_analysis_prompt(
            checkpoint_output=checkpoint_output,
            context=self.context,
            previous_checkpoints=previous_checkpoints
        )
        
    def update_task_progress(self, completed: List[str], remaining: List[str]):
        """Update task progress in context"""
        self.context.completed_tasks = completed
        self.context.remaining_tasks = remaining
        
    def add_issue(self, issue: str):
        """Add an issue to context"""
        self.context.issues_encountered.append(issue)
        
    def add_guidance(self, guidance: str):
        """Add guidance to history"""
        self.context.previous_guidance.append(guidance)
        
    def get_final_summary(self, all_checkpoints: List[Dict], total_cost: float) -> str:
        """Get final summary for the user"""
        return self.generator.generate_final_summary_prompt(
            all_checkpoints=all_checkpoints,
            context=self.context,
            total_cost=total_cost
        )