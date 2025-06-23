"""
Command-line interface for TaskSupervisor

Supports both traditional and analysis modes for the new architecture.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# Handle both module and script execution
try:
    from .task_supervisor import TaskSupervisor
    from .config import ConfigLoader
except ImportError:
    # When run as a script, use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cadence.task_supervisor import TaskSupervisor
    from cadence.config import ConfigLoader


def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Claude Cadence Task Supervisor"
    )

    # Mode selection
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run in analysis mode for orchestrator"
    )

    # Common arguments
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Path to Task Master tasks.json file (default: .taskmaster/tasks/tasks.json)"
    )

    parser.add_argument(
        "--session-id",
        type=str,
        help="Session ID (required for analysis mode)"
    )

    parser.add_argument(
        "--continue",
        dest="continue_from_previous",
        action="store_true",
        help="Continue from previous execution"
    )

    parser.add_argument(
        "--output-decision",
        action="store_true",
        help="Output decision in JSON format (analysis mode)"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        help="Override max turns from config"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Override supervisor model from config"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )


    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if args.analyze and not args.session_id:
        parser.error("--session-id is required when using --analyze")

    # Handle task file default
    if args.task_file is None:
        args.task_file = ".taskmaster/tasks/tasks.json"

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading configuration from: {config_path}")
        loader = ConfigLoader(str(config_path))
        config = loader.config
    else:
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    # Apply overrides
    if args.max_turns:
        config.execution.max_turns = args.max_turns
    if args.model:
        config.supervisor.model = args.model

    # Validate supervisor model
    if config.supervisor.model == "heuristic":
        logger.error("ERROR: Supervisor MUST use AI model, not heuristic!")
        logger.error("Please update config to use an AI model")
        return 1

    try:
        # Create supervisor
        supervisor = TaskSupervisor(config=config)

        if args.analyze:
            # Run in analysis mode for orchestrator
            logger.info(f"Running in analysis mode for session {args.session_id}")

            decision = supervisor.analyze_and_decide(
                task_file=args.task_file,
                session_id=args.session_id,
                continue_from_previous=args.continue_from_previous
            )

            if args.output_decision:
                # Output decision as JSON for orchestrator
                print(json.dumps(decision))
            else:
                # Pretty print decision
                print(f"\nDecision: {decision['action']}")
                if decision.get('reason'):
                    print(f"Reason: {decision['reason']}")
                if decision.get('todos'):
                    print(f"TODOs: {len(decision['todos'])} items")

            return 0 if decision['action'] != 'error' else 1

        else:
            # Run in traditional mode (deprecated but kept for compatibility)
            logger.warning("Running in traditional mode - consider using orchestrated mode")
            success = supervisor.run_with_taskmaster(args.task_file)
            return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
