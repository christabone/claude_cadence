#!/bin/bash
# Test runner script for Claude Cadence

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Default to running all tests
TEST_TARGET="${1:-tests/}"

echo "Running Claude Cadence tests..."
echo "Target: $TEST_TARGET"
echo "=============================="

# Run tests with appropriate flags
if [[ "$TEST_TARGET" == *"e2e"* ]]; then
    echo "Running E2E tests (may take longer)..."
    pytest "$TEST_TARGET" -v --tb=short
else
    echo "Running tests..."
    pytest "$TEST_TARGET" -v --tb=short
fi
