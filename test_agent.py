#!/usr/bin/env python3
"""Quick test of UnifiedAgent execution"""

import sys
import logging
from pathlib import Path
from cadence.unified_agent import UnifiedAgent
from cadence.config import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Load config
config_loader = ConfigLoader()
config = config_loader.config

# Create test directory
test_dir = Path("test_agent_run")
test_dir.mkdir(exist_ok=True)

# Create agent with standard profile
agent = UnifiedAgent(
    profile_name="standard",
    config=config.to_dict(),
    working_dir=test_dir,
    session_id="test_001"
)

# Simple test prompt
prompt = """Please create a file called test.txt with the content "Hello from agent".

Then output your JSON status.
"""

# Execute
print("Running agent test...")
result = agent.execute(prompt)

print(f"\nAgent Result:")
print(f"  Success: {result.success}")
print(f"  Requested Help: {result.requested_help}")
print(f"  Execution Time: {result.execution_time:.2f}s")
print(f"  Profile Used: {result.profile_used}")
print(f"  Errors: {result.errors}")

# Check if test file was created
test_file = test_dir / "test.txt"
if test_file.exists():
    print(f"\nTest file created successfully with content: {test_file.read_text()}")
else:
    print("\nERROR: Test file was not created")

# Cleanup
import shutil
shutil.rmtree(test_dir)
print("\nTest directory cleaned up")
