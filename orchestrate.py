#!/usr/bin/env python3
"""
Claude Cadence Orchestration Script

This script orchestrates the supervisor-agent workflow with proper
directory separation and state management.
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cadence.orchestrator import SupervisorOrchestrator
from cadence.config import ConfigLoader
from cadence.log_utils import setup_colored_logging


def setup_logging(verbose: bool = False):
    """Set up logging configuration with color support"""
    level = logging.DEBUG if verbose else logging.INFO
    setup_colored_logging(level)


def main():
    """Main orchestration entry point"""
    parser = argparse.ArgumentParser(
        description="Orchestrate Claude Cadence supervisor-agent workflow"
    )

    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Path to Task Master tasks.json file (default: .taskmaster/tasks/tasks.json)"
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum orchestration iterations (default: 100)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Resolve paths
    project_root = Path(args.project_root).resolve()

    # Validate initial project root exists
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        return 1

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Cannot proceed without configuration file")
        return 1

    logger.info(f"Loading configuration from: {config_path}")

    # Load the YAML directly to get the raw dictionary
    try:
        with open(config_path, 'r') as f:
            orchestrator_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Override project_root with config value if available
    if 'project' in orchestrator_config and 'root_directory' in orchestrator_config['project']:
        # Make path handling more robust
        config_path_str = orchestrator_config['project']['root_directory']
        config_project_path = Path(config_path_str).expanduser()

        # If relative, resolve relative to the config file's directory
        if not config_project_path.is_absolute():
            config_project_path = config_path.parent / config_project_path

        config_project_root = config_project_path.resolve()

        if config_project_root.exists():
            logger.info(f"Using project root from config: {config_project_root}")
            project_root = config_project_root
        else:
            logger.error(f"Project root from config does not exist: {config_project_root}")
            logger.error("Please check your config.yaml and ensure the project.root_directory path exists")
            return 1

    # NOW handle task file after project_root is finalized
    if args.task_file:
        task_file = Path(args.task_file)
        # Make task file absolute if relative
        if not task_file.is_absolute():
            task_file = project_root / task_file
    else:
        # Check if config has a custom taskmaster_file
        taskmaster_file = orchestrator_config.get('project', {}).get('taskmaster_file')
        if taskmaster_file:
            task_file = project_root / taskmaster_file
        else:
            # Use default location with the FINAL project_root
            task_file = project_root / ".taskmaster" / "tasks" / "tasks.json"

    # Validate task file exists
    if not task_file.exists():
        logger.error(f"Task file does not exist: {task_file}")
        logger.error("Please ensure Task Master is initialized and tasks.json exists")
        return 1

    # Add command line max_iterations to the config
    if 'orchestration' not in orchestrator_config:
        orchestrator_config['orchestration'] = {}
    orchestrator_config['orchestration']['max_iterations'] = args.max_iterations

    # Validate supervisor model
    supervisor_model = orchestrator_config.get("supervisor", {}).get("model", "")
    if supervisor_model == "heuristic":
        logger.error("ERROR: Supervisor MUST use AI model, not heuristic!")
        logger.error("Please update config.yaml to set supervisor.model to an AI model")
        return 1

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Project root: {project_root}")
        print(f"Task file: {task_file}")
        print(f"Config: {orchestrator_config}")
        print("\nWould create directories:")
        print(f"  - {project_root / '.cadence' / 'supervisor'}")
        print(f"  - {project_root / '.cadence' / 'agent'}")
        print("\nWould start orchestration loop with:")
        print(f"  - Max iterations: {args.max_iterations}")
        print(f"  - Max turns per agent: {orchestrator_config.get('execution', {}).get('max_turns')}")
        return 0

    try:
        # Create orchestrator
        logger.info("Creating orchestrator...")
        orchestrator = SupervisorOrchestrator(
            project_root=project_root,
            task_file=task_file,
            config=orchestrator_config
        )

        # Run orchestration loop
        logger.info("Starting orchestration loop...")
        success = orchestrator.run_orchestration_loop()

        if success:
            logger.info("🎉 Orchestration completed successfully!")
            return 0
        else:
            logger.error("❌ Orchestration failed or incomplete")
            return 1

    except KeyboardInterrupt:
        logger.info("\n⚠️  Orchestration interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error during orchestration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
