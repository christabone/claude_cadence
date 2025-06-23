#!/usr/bin/env python3
"""
Claude Cadence CLI

Command-line interface for task-driven agent supervision.
"""

import argparse
import sys
from pathlib import Path

# Handle both installed and development usage
try:
    from cadence import TaskSupervisor
except ImportError:
    # Development mode - add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from cadence import TaskSupervisor


def main():
    parser = argparse.ArgumentParser(
        description="Claude Cadence - Task-driven supervision for Claude Code agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Task Master (default)
  cadence

  # Custom max turns
  cadence --max-turns 60

  # Use specific model
  cadence --model claude-3-opus-latest

  # Specify task file
  cadence --task-file .taskmaster/tasks/tasks.json

  # Manual TODO mode (without Task Master)
  cadence --todo "Implement user authentication" --todo "Add tests"
"""
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum turns before forcing completion (safety limit)"
    )

    parser.add_argument(
        "-m", "--model",
        help="Claude model to use"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for logs"
    )

    parser.add_argument(
        "--task-file",
        help="Path to Task Master tasks.json file"
    )

    parser.add_argument(
        "--todo",
        action="append",
        dest="todos",
        help="Manual TODO item (can be used multiple times)"
    )

    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Create supervisor
    supervisor = TaskSupervisor(
        max_turns=args.max_turns,
        model=args.model,
        output_dir=args.output,
        verbose=args.verbose,
        config_path=args.config
    )

    # Run based on mode
    try:
        if args.todos:
            # Manual TODO mode
            print(f"🎯 Running with {len(args.todos)} manual TODOs")
            result = supervisor.execute_with_todos(todos=args.todos)
            success = result.task_complete
        else:
            # Task Master mode (default)
            success = supervisor.run_with_taskmaster(task_file=args.task_file)

        # Exit code based on success
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
