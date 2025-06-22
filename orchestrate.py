#!/usr/bin/env python3
"""
Claude Cadence Orchestration Script

This script orchestrates the supervisor-agent workflow with proper
directory separation and state management.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cadence.orchestrator import SupervisorOrchestrator
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
    """Main orchestration entry point"""
    parser = argparse.ArgumentParser(
        description="Orchestrate Claude Cadence supervisor-agent workflow"
    )
    
    parser.add_argument(
        "--task-file",
        type=str,
        required=True,
        help="Path to Task Master tasks.json file"
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
    task_file = Path(args.task_file)
    
    # Make task file absolute if relative
    if not task_file.is_absolute():
        task_file = project_root / task_file
    
    # Validate paths
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        return 1
        
    if not task_file.exists():
        logger.error(f"Task file does not exist: {task_file}")
        return 1
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
        
    if config_path.exists():
        logger.info(f"Loading configuration from: {config_path}")
        loader = ConfigLoader(str(config_path))
        config = loader.config
        
        # Extract relevant config
        orchestrator_config = {
            "max_turns": config.execution.max_turns,
            "max_iterations": args.max_iterations,
            "allowed_tools": config.agent.tools,
            "supervisor_model": config.supervisor.model,
            "agent_model": config.agent.model,
        }
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        logger.warning("Using default configuration")
        orchestrator_config = {
            "max_turns": 40,
            "max_iterations": args.max_iterations,
            "allowed_tools": ["bash", "read", "write", "edit", "grep", "glob"],
        }
    
    # Validate supervisor model
    if orchestrator_config.get("supervisor_model") == "heuristic":
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
        print(f"  - Max turns per agent: {orchestrator_config.get('max_turns', 40)}")
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