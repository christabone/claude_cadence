#!/usr/bin/env python3
"""Test script for logging configuration."""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.logging_config import get_logger, setup_logging, with_request_id


def test_basic_logging():
    """Test basic logging functionality."""
    print("\n=== Testing Basic Logging ===")

    # Get a logger
    logger = get_logger('test.basic')

    # Log at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print(" Basic logging test completed")


def test_multiple_loggers():
    """Test multiple logger instances."""
    print("\n=== Testing Multiple Loggers ===")

    logger1 = get_logger('test.module1')
    logger2 = get_logger('test.module2')

    logger1.info("Message from module1")
    logger2.info("Message from module2")

    print(" Multiple loggers test completed")


def test_request_id_context():
    """Test request ID context manager."""
    print("\n=== Testing Request ID Context ===")

    logger = get_logger('test.context')

    # Log without request ID context
    logger.info("Message without request ID context")

    # Log with specific request ID
    with with_request_id('test-request-123') as rid:
        logger.info(f"Processing request with ID: {rid}")
        logger.info("Another message in same request context")

        # Nested logger should also get the request ID
        nested_logger = get_logger('test.context.nested')
        nested_logger.info("Message from nested logger")

    # Log after context (should have different request ID)
    logger.info("Message after request context")

    print(" Request ID context test completed")


def test_error_with_traceback():
    """Test logging exceptions with tracebacks."""
    print("\n=== Testing Error Logging ===")

    logger = get_logger('test.errors')

    try:
        # Intentionally cause an error
        result = 1 / 0
    except Exception as e:
        logger.error("Division by zero error occurred", exc_info=True)

    print(" Error logging test completed")


def test_log_files():
    """Test that log files are created properly."""
    print("\n=== Testing Log File Creation ===")

    # Check if log directory was created
    log_dir = './logs'
    if os.path.exists(log_dir):
        print(f" Log directory created: {log_dir}")

        # Check for log file
        log_file = os.path.join(log_dir, 'agr_mcp.log')
        if os.path.exists(log_file):
            print(f" Log file created: {log_file}")

            # Show last few lines of log file
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f" Log file contains {len(lines)} lines")
                    print("\nLast 3 lines from log file:")
                    for line in lines[-3:]:
                        print(f"  {line.strip()}")
        else:
            print(" Log file not found")
    else:
        print(" Log directory not found")


def main():
    """Run all tests."""
    print("AGR MCP Logging Configuration Test")
    print("==================================")

    # Run tests
    test_basic_logging()
    test_multiple_loggers()
    test_request_id_context()
    test_error_with_traceback()
    test_log_files()

    print("\n All tests completed!")
    print("\nNote: Check ./logs/agr_mcp.log for file output")


if __name__ == '__main__':
    main()
