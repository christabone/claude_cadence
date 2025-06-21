#!/usr/bin/env python3
"""
Custom Prompts Example

This example shows how to customize prompts using YAML configuration
for specific use cases or domains.
"""

import sys
import yaml
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadence import TaskSupervisor
from cadence.prompts import PromptGenerator


def create_custom_prompt_config():
    """Create a custom prompt configuration for code review tasks"""
    
    custom_config = """
# Custom prompt configuration for code review tasks

shared_agent_context:
  supervision_explanation: |
    === CODE REVIEW EXECUTION SYSTEM ===
    You are performing a systematic code review.
    Safety limit: {max_turns} turns (not a target)
    Your review will be task-driven and thorough.
  
  work_guidelines: |
    REVIEW FOCUS AREAS:
    1. Code quality and maintainability
    2. Security vulnerabilities
    3. Performance issues
    4. Best practices compliance
    5. Test coverage adequacy
    
    Work naturally through the review tasks.
    Quality matters more than speed.
  
  early_exit_protocol: |
    COMPLETION PROTOCOL:
    - When all review TODOs are complete, state 'ALL TASKS COMPLETE' and exit
    - Provide a summary of findings before exiting
    - Do not continue beyond the requested scope

todo_templates:
  todo_list: |
    === CODE REVIEW TODOS ===
    The following review tasks need to be completed:
    
    {todo_items}
    
    Review systematically and provide actionable feedback.
  
  todo_item: "{number}. {todo_text}"

agent_prompts:
  initial:
    sections:
      - "{shared_agent_context.supervision_explanation}"
      - "{shared_agent_context.work_guidelines}"
      - "{shared_agent_context.early_exit_protocol}"
      - "{todo_list}"
      - |
        === BEGIN CODE REVIEW ===
        Start reviewing the code according to the TODOs above.
        Remember: Complete all review tasks then exit immediately.

supervisor_prompts:
  analysis:
    sections:
      - |
        === CODE REVIEW SUPERVISOR ===
        You are reviewing a code reviewer's progress.
        
        Review focus: Code quality analysis
        Turns used: {turns_used} of {max_turns}
      - |
        === REVIEW OUTPUT ===
        {execution_output}
      - |
        === EVALUATION CRITERIA ===
        Assess the reviewer's work:
        1. Are critical issues being identified?
        2. Is the feedback actionable and specific?
        3. Are security concerns adequately addressed?
        4. Is the review progressing effectively?
        
        Provide guidance to help complete remaining TODOs.
"""
    
    # Save custom config
    config_path = Path("code_review_prompts.yaml")
    with open(config_path, 'w') as f:
        f.write(custom_config)
        
    return config_path


def create_testing_prompt_config():
    """Create a custom prompt configuration for test generation"""
    
    test_config = """
# Custom prompt configuration for test generation

shared_agent_context:
  supervision_explanation: |
    === TEST GENERATION SYSTEM ===
    You are generating comprehensive test suites.
    Safety limit: {max_turns} turns (complete tasks naturally)
  
  work_guidelines: |
    TEST GENERATION GUIDELINES:
    - Write clear, descriptive test names
    - Cover edge cases and error conditions
    - Use appropriate assertions
    - Follow AAA pattern (Arrange, Act, Assert)
    - Include both positive and negative test cases
  
  early_exit_protocol: |
    When all test TODOs are complete:
    1. Run the test suite to verify
    2. State 'ALL TASKS COMPLETE'
    3. Exit immediately

todo_templates:
  todo_list: |
    === TEST GENERATION TODOS ===
    Complete these testing tasks:
    
    {todo_items}
    
    Ensure comprehensive test coverage.

agent_prompts:
  initial:
    sections:
      - "{shared_agent_context.supervision_explanation}"
      - "{shared_agent_context.work_guidelines}"
      - "{shared_agent_context.early_exit_protocol}"
      - "{todo_list}"
      - |
        Begin creating the test suite now.
"""
    
    config_path = Path("test_gen_prompts.yaml")
    with open(config_path, 'w') as f:
        f.write(test_config)
        
    return config_path


def main():
    """Demonstrate custom prompt usage"""
    
    print("🎨 Custom Prompts Demonstration")
    print("=" * 50)
    
    # Example 1: Code Review Prompts
    print("\n📝 Example 1: Code Review Configuration")
    review_config = create_custom_prompt_config()
    print(f"✅ Created custom config: {review_config}")
    
    # Create supervisor with custom prompts
    review_supervisor = TaskSupervisor(
        max_turns=30,
        verbose=True,
        config_path=str(review_config)
    )
    
    # Define code review TODOs
    review_todos = [
        "Review src/auth.py for security vulnerabilities",
        "Check src/database.py for SQL injection risks",
        "Analyze src/api.py for performance bottlenecks",
        "Verify error handling in all files",
        "Create summary report of findings"
    ]
    
    print(f"\n📋 Code Review TODOs: {len(review_todos)}")
    print("💡 Custom prompts guide the reviewer to focus on security and quality")
    
    # Note: In a real scenario, you'd run the review
    # result = review_supervisor.execute_with_todos(todos=review_todos)
    
    # Example 2: Test Generation Prompts
    print("\n\n📝 Example 2: Test Generation Configuration")
    test_config = create_testing_prompt_config()
    print(f"✅ Created custom config: {test_config}")
    
    test_supervisor = TaskSupervisor(
        max_turns=40,
        verbose=True,
        config_path=str(test_config)
    )
    
    test_todos = [
        "Create unit tests for Calculator class",
        "Test edge cases (division by zero, overflow)",
        "Add integration tests for API endpoints",
        "Ensure 90% code coverage"
    ]
    
    print(f"\n📋 Test Generation TODOs: {len(test_todos)}")
    print("💡 Custom prompts emphasize test patterns and coverage")
    
    # Example 3: Loading and Inspecting Prompts
    print("\n\n📖 Example 3: Inspecting Prompt Templates")
    
    # Load the code review prompts
    generator = PromptGenerator(str(review_config))
    
    print("\nWork Guidelines from Code Review Config:")
    print("-" * 40)
    print(generator.loader.config['shared_agent_context']['work_guidelines'])
    
    print("\nTODO Template:")
    print("-" * 40)
    print(generator.loader.config['todo_templates']['todo_list'])
    
    # Example 4: Runtime Prompt Customization
    print("\n\n🔧 Example 4: Runtime Prompt Customization")
    
    # You can also modify prompts at runtime
    custom_todos = [
        "PRIORITY: Fix critical security bug in login system",
        "Update dependencies to latest secure versions",
        "Add rate limiting to prevent DoS attacks"
    ]
    
    # The supervisor will use the security-focused prompts
    print("💡 Using security-focused configuration for critical fixes")
    
    # Clean up
    review_config.unlink()
    test_config.unlink()
    print("\n✅ Cleanup complete")
    
    print("\n" + "=" * 50)
    print("📚 Summary:")
    print("1. Create custom YAML configs for different domains")
    print("2. Focus prompts on specific quality attributes")
    print("3. Guide agents with domain-specific instructions")
    print("4. Maintain consistent review/generation patterns")


if __name__ == "__main__":
    main()