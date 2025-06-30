#!/usr/bin/env python3
"""Test supervisor logging implementation"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cadence.orchestrator import SupervisorOrchestrator

def test_logging():
    """Test if supervisor logging is working"""
    # Create orchestrator instance
    project_root = "/home/ctabone/programming/claude_code/claude_cadence/agr_mcp"

    orchestrator = SupervisorOrchestrator(project_root)

    # Check if current_session_id is set
    print(f"Initial session ID: {orchestrator.current_session_id}")

    # Manually set session ID to test
    orchestrator.current_session_id = "test_session_123"

    # Check log directory
    log_dir = orchestrator.project_root / ".cadence" / "logs"
    print(f"Log directory: {log_dir}")
    print(f"Log directory exists: {log_dir.exists()}")

    # Create test supervisor log
    supervisor_log_file = log_dir / orchestrator.current_session_id / "supervisor.log"
    print(f"Supervisor log file path: {supervisor_log_file}")

    # Try to create the directory and file
    try:
        supervisor_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(supervisor_log_file, 'w') as f:
            f.write("Test supervisor log\n")
        print(f"Successfully created test log file")
        print(f"File exists: {supervisor_log_file.exists()}")
    except Exception as e:
        print(f"Error creating log file: {e}")

if __name__ == "__main__":
    test_logging()
