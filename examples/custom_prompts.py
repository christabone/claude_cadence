#!/usr/bin/env python3
"""
Custom Prompts Example

This example shows how to customize prompts using YAML configuration
for specific use cases or domains.
"""

import yaml
from pathlib import Path
from cadence import CheckpointSupervisor, PromptGenerator


def create_custom_prompt_config():
    """Create a custom prompt configuration for code review tasks"""
    
    custom_config = """
# Custom prompt configuration for code review tasks

shared_agent_context:
  supervision_explanation: |
    === CODE REVIEW CHECKPOINT SYSTEM ===
    You are performing a systematic code review under supervision.
    You have {checkpoint_turns} turns per review checkpoint.
    Total checkpoints: {max_checkpoints}
    Your review will be evaluated after each checkpoint.
  
  work_guidelines: |
    REVIEW FOCUS AREAS:
    1. Code quality and maintainability
    2. Security vulnerabilities
    3. Performance issues
    4. Best practices compliance
    5. Test coverage adequacy
    
    Exit early if you complete the review of all specified files.
  
  early_exit_protocol: |
    COMPLETION PROTOCOL:
    - When all files are reviewed, state 'CODE REVIEW COMPLETE' and exit
    - Provide a summary of findings before exiting
    - Do not continue beyond the requested scope

agent_prompts:
  initial:
    sections:
      - "{shared_agent_context.supervision_explanation}"
      - "{shared_agent_context.work_guidelines}"
      - "{shared_agent_context.early_exit_protocol}"
      - |
        === CODE REVIEW TASK ===
        {task_description}
        
        Review systematically and provide actionable feedback.
        Start your review now.

supervisor_prompts:
  analysis:
    sections:
      - |
        === CODE REVIEW SUPERVISOR ===
        You are reviewing a code reviewer's checkpoint.
        
        Checkpoint: {current_checkpoint} of {max_checkpoints}
        Review focus: Code quality analysis
      - |
        === REVIEW SCOPE ===
        {original_task}
      - |
        === REVIEWER OUTPUT ===
        {checkpoint_output}
      - |
        === EVALUATION CRITERIA ===
        Assess the reviewer's work:
        1. Are critical issues being identified?
        2. Is the feedback actionable and specific?
        3. Are security concerns adequately addressed?
        4. Should the review strategy be adjusted?
        
        Provide guidance for the next checkpoint.
"""
    
    # Save custom config
    config_path = Path("code_review_prompts.yaml")
    with open(config_path, 'w') as f:
        f.write(custom_config)
        
    return config_path


def main():
    """Demonstrate custom prompt usage"""
    
    print("🎨 Custom Prompts Demonstration")
    print("=" * 50)
    
    # Create custom prompt config
    config_path = create_custom_prompt_config()
    print(f"✅ Created custom config: {config_path}")
    
    # Create supervisor with custom prompts
    class CustomPromptSupervisor(CheckpointSupervisor):
        """Supervisor that uses custom prompt configuration"""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Override prompt manager with custom config
            from cadence.prompts import ContextAwarePromptManager
            self.prompt_manager = ContextAwarePromptManager(
                original_task=kwargs.get('task', ''),
                checkpoint_turns=self.checkpoint_turns,
                max_checkpoints=self.max_checkpoints,
                config_path=str(config_path)
            )
    
    # Create code review task
    review_task = """Review the Python files in the 'src/' directory for:
    1. Security vulnerabilities
    2. Performance bottlenecks
    3. Code style violations
    4. Missing error handling
    5. Inadequate documentation
    
    Focus on actionable improvements and critical issues."""
    
    # Run with custom prompts
    supervisor = CustomPromptSupervisor(
        checkpoint_turns=10,
        max_checkpoints=2,
        verbose=True,
        task=review_task
    )
    
    # Note: In a real scenario, you'd need actual code files to review
    print("\n📝 Task:", review_task)
    print("\n💡 This example shows how to create and use custom prompts.")
    print("   In practice, you would run: supervisor.run_supervised_task(review_task)")
    
    # Show how to load and modify existing prompts
    print("\n" + "=" * 50)
    print("📖 Loading and Modifying Prompts")
    
    generator = PromptGenerator(str(config_path))
    
    # Access specific templates
    print("\nWork Guidelines:")
    print(generator.loader.config['shared_agent_context']['work_guidelines'])
    
    # Clean up
    config_path.unlink()
    print("\n✅ Cleanup complete")


if __name__ == "__main__":
    main()