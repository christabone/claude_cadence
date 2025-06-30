#!/usr/bin/env python3
"""Direct test of supervisor logging implementation"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cadence.orchestrator import SupervisorOrchestrator

def test_direct_logging():
    """Test logging directly without full orchestration"""
    # Create orchestrator instance
    project_root = Path("/home/ctabone/programming/claude_code/claude_cadence/agr_mcp")
    orchestrator = SupervisorOrchestrator(project_root)

    # Generate a session ID
    session_id = orchestrator.generate_session_id()
    print(f"Generated session ID: {session_id}")

    # Create test output
    test_output = [
        "=== SUPERVISOR OUTPUT ===",
        "Analyzing task file...",
        "Found 10 tasks",
        "Selected task 1.1 for execution",
        "=== END SUPERVISOR OUTPUT ==="
    ]

    # Test the logging code directly
    log_dir = orchestrator.project_root / ".cadence" / "logs"
    supervisor_log_file = log_dir / session_id / "supervisor.log"

    try:
        supervisor_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(supervisor_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Supervisor run at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write('\n'.join(test_output))
            f.write('\n')
        print(f"Successfully wrote to: {supervisor_log_file}")

        # Verify file exists and has content
        if supervisor_log_file.exists():
            size = supervisor_log_file.stat().st_size
            print(f"File exists with size: {size} bytes")

            # Read and display content
            with open(supervisor_log_file, 'r') as f:
                content = f.read()
            print(f"\nFile content:\n{content}")
        else:
            print("ERROR: File was not created!")

    except Exception as e:
        print(f"ERROR writing log file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_logging()
