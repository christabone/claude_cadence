#!/usr/bin/env python3
"""
Claude Cadence CLI

Simple command-line interface for checkpoint supervision.
"""

import argparse
import sys
from pathlib import Path

# Handle both installed and development usage
try:
    from cadence import CheckpointSupervisor
except ImportError:
    # Development mode - add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from cadence import CheckpointSupervisor


def main():
    parser = argparse.ArgumentParser(
        description="Claude Cadence - Checkpoint supervision for Claude Code agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic supervision with default settings
  cadence "Write a Python web scraper"
  
  # Custom checkpoint settings
  cadence --turns 10 --checkpoints 5 "Complex refactoring task"
  
  # Use specific model
  cadence --model claude-3-opus-20240229 "Critical code review"
  
  # Task Master integration
  cadence --taskmaster "Complete all project tasks"
"""
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task prompt for the agent (optional with --taskmaster)"
    )
    
    parser.add_argument(
        "-t", "--turns",
        type=int,
        default=15,
        help="Number of turns per checkpoint (default: 15)"
    )
    
    parser.add_argument(
        "-c", "--checkpoints",
        type=int,
        default=3,
        help="Maximum number of checkpoints (default: 3)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use (default: claude-3-5-sonnet-20241022)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="cadence_output",
        help="Output directory for logs (default: cadence_output)"
    )
    
    parser.add_argument(
        "--taskmaster",
        action="store_true",
        help="Use Task Master integration if available"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle Task Master mode
    if args.taskmaster:
        try:
            from cadence.task_manager import TaskManager
            from examples.taskmaster_guided import TaskGuidedSupervisor
            
            supervisor = TaskGuidedSupervisor(
                project_dir=".",
                checkpoint_turns=args.turns,
                max_checkpoints=args.checkpoints,
                model=args.model,
                output_dir=args.output,
                verbose=args.verbose
            )
            
            if supervisor.tasks_loaded:
                print("✅ Task Master integration enabled")
                prompt = args.prompt  # Can be None
            else:
                print("⚠️  No Task Master tasks found")
                if not args.prompt:
                    print("❌ Error: No prompt provided and no tasks found")
                    sys.exit(1)
                prompt = args.prompt
                
        except ImportError:
            print("⚠️  Task Master integration not available")
            if not args.prompt:
                print("❌ Error: No prompt provided")
                sys.exit(1)
            prompt = args.prompt
            supervisor = CheckpointSupervisor(
                checkpoint_turns=args.turns,
                max_checkpoints=args.checkpoints,
                model=args.model,
                output_dir=args.output,
                verbose=args.verbose
            )
    else:
        # Standard mode requires prompt
        if not args.prompt:
            parser.print_help()
            sys.exit(1)
            
        prompt = args.prompt
        supervisor = CheckpointSupervisor(
            checkpoint_turns=args.turns,
            max_checkpoints=args.checkpoints,
            model=args.model,
            output_dir=args.output,
            verbose=args.verbose
        )
    
    # Run supervision
    try:
        success, cost = supervisor.run_supervised_task(prompt)
        
        # Exit code based on success
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Supervision interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()