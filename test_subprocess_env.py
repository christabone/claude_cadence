#!/usr/bin/env python3
"""Test subprocess environment inheritance"""

import os
import subprocess
import sys

# Test 1: Check current environment
print("=== Current Environment Variables ===")
for key, value in os.environ.items():
    if 'CADENCE' in key or 'LOG' in key:
        print(f"{key}={value}")

# Test 2: Run a simple claude command with inherited environment
print("\n=== Testing Claude with Inherited Environment ===")
try:
    result = subprocess.run(
        ["claude", "--print", "What is 2+2?"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
except subprocess.TimeoutExpired:
    print("ERROR: Command timed out!")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Run claude with cleaned environment
print("\n=== Testing Claude with Clean Environment ===")
clean_env = {k: v for k, v in os.environ.items() if 'CADENCE_LOG' not in k}
try:
    result = subprocess.run(
        ["claude", "--print", "What is 2+2?"],
        capture_output=True,
        text=True,
        timeout=10,
        env=clean_env
    )
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
except subprocess.TimeoutExpired:
    print("ERROR: Command timed out!")
except Exception as e:
    print(f"ERROR: {e}")
