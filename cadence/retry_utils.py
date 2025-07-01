"""
Generic retry utilities for handling malformed JSON responses from agents/supervisors.

This module provides a simple, hobby-level retry mechanism that can be used
throughout the orchestration system to handle JSON parsing failures and
malformed responses from Claude CLI agents.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Custom exception for retry failures."""
    def __init__(self, message: str, last_output: str = "", last_error: str = ""):
        super().__init__(message)
        self.last_output = last_output
        self.last_error = last_error




def parse_json_with_retry(
    json_string: str,
    parser_func: Callable[[str], Any] = json.loads,
    max_retries: int = 3,
    retry_callback: Optional[Callable[[str, int], str]] = None,
    base_delay: float = 2.0
) -> Any:
    """
    Parse JSON with retry logic for malformed responses.

    This is useful when you already have the JSON string and just need
    to parse it with retry logic (e.g., for MCP responses).

    Args:
        json_string: The JSON string to parse
        parser_func: Function to use for parsing (default: json.loads)
        max_retries: Maximum number of attempts (default: 3)
        retry_callback: Optional callback to modify the string before retry
                       Takes (json_string, attempt_num) and returns modified string
        base_delay: Base delay in seconds for retry backoff (default: 2.0)

    Returns:
        Parsed JSON data

    Raises:
        RetryError: If all parsing attempts fail
    """
    current_string = json_string

    for attempt in range(max_retries):
        # Add delay for retries (simple linear backoff)
        if attempt > 0:
            delay = base_delay * attempt  # Configurable delay with linear backoff
            logger.info(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)

        try:
            parsed_data = parser_func(current_string)
            return parsed_data
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Parse error on attempt {attempt + 1}: {e}")

            if attempt == max_retries - 1:
                # Last attempt failed
                raise RetryError(
                    f"Failed to parse JSON after {max_retries} attempts",
                    last_output=current_string
                )

            # Apply retry callback if provided
            if retry_callback:
                current_string = retry_callback(current_string, attempt + 1)
            # Otherwise just retry with the same string

    # Should never reach here
    raise RetryError(f"Parse loop exited unexpectedly")


def run_claude_with_realtime_retry(
    build_command_func: Callable[[], List[str]],
    parse_output_func: Callable[[List[str]], Any],
    realtime_runner_func: Callable[[List[str], Any, str], Tuple[int, List[str]]],
    working_dir: Union[str, Path],
    process_name: str,
    max_retries: int = 3,
    session_id: Optional[str] = None,
    base_delay: float = 2.0
) -> Any:
    """
    Enhanced retry function for orchestrator's realtime Claude execution needs.

    This function supports the orchestrator's pattern of:
    1. Building dynamic commands (that may change on retry)
    2. Running with realtime output via async subprocess
    3. Parsing complex output (like JSON streams)
    4. Retrying with modified commands on failures

    Args:
        build_command_func: Function that builds the command list (may add --continue on retry)
        parse_output_func: Function that parses output and returns result
        realtime_runner_func: Function that runs subprocess with realtime output
        working_dir: Directory to run the command in
        process_name: Name for logging (e.g., "SUPERVISOR", "AGENT")
        max_retries: Maximum number of retry attempts (default: 3)
        session_id: Optional session ID for logging
        base_delay: Base delay in seconds for retry backoff (default: 2.0)

    Returns:
        Parsed result from parse_output_func

    Raises:
        RetryError: If all retry attempts fail
    """

    for attempt in range(max_retries):
        # Add delay for retries (simple linear backoff)
        if attempt > 0:
            delay = base_delay * attempt  # Configurable delay with linear backoff
            logger.info(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)

        # Build command (may be different on retries)
        cmd = build_command_func()

        # Check if --continue flag is present
        continue_flag = "--continue" in cmd
        continue_status = "with --continue" if continue_flag else "without --continue"

        # Log retry status
        logger.info("=" * 60)
        if attempt > 0:
            logger.warning(f"Retrying {process_name} due to parsing error (attempt {attempt + 1})")
            logger.info(f"{process_name} STARTING... (RETRY {attempt + 1}/{max_retries}) [{continue_status}]")
        else:
            logger.info(f"{process_name} STARTING... [{continue_status}]")
        logger.info("=" * 60)

        try:
            # Run subprocess with realtime output
            returncode, all_output = realtime_runner_func(cmd, working_dir, process_name)

            if returncode != 0:
                logger.error(f"{process_name} failed with code {returncode}")
                raise RetryError(
                    f"{process_name} failed with exit code {returncode}",
                    last_output="\n".join(all_output) if all_output else "",
                    last_error=f"Exit code: {returncode}"
                )

        except Exception as e:
            logger.error(f"{process_name} execution error: {e}")
            raise RetryError(f"{process_name} execution failed: {e}")

        logger.info("-" * 60)
        logger.info(f"{process_name} COMPLETED")
        logger.info("-" * 60)

        # Try to parse output
        try:
            result = parse_output_func(all_output)
            logger.info(f"Successfully parsed {process_name} output")
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse {process_name} output: {e}")

            if attempt == max_retries - 1:
                # Last attempt failed
                logger.error(f"Failed to get valid output after {max_retries} attempts. Giving up.")
                if all_output:
                    logger.error("Last 5 lines of output:")
                    for line in all_output[-5:]:
                        logger.error(f"  {line.strip()}")
                raise RetryError(
                    f"{process_name} failed to produce valid output after {max_retries} attempts",
                    last_output="\n".join(all_output) if all_output else "",
                    last_error=str(e)
                )

            logger.warning(f"Will retry {process_name} to get valid output...")
            # Continue to next iteration

    # Should never reach here
    raise RetryError(f"Unexpected exit from {process_name} retry loop")


def save_json_with_retry(
    data: Any,
    file_path: str,
    max_retries: int = 3,
    indent: int = 2
) -> None:
    """
    Save JSON data to file with retry logic for serialization errors.

    Args:
        data: Python object to serialize to JSON
        file_path: Path to save the JSON file
        max_retries: Maximum number of attempts (default: 3)
        indent: JSON indentation level (default: 2)

    Raises:
        RetryError: If all save attempts fail
        IOError: If file writing fails
    """
    for attempt in range(max_retries):
        try:
            # First try to serialize to string (catches TypeError)
            json_string = json.dumps(data, indent=indent)

            # Then write to file
            with open(file_path, 'w') as f:
                f.write(json_string)

            logger.info(f"Successfully saved JSON to {file_path}")
            return

        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # TypeError means the data contains non-serializable objects
            # Retrying won't help, so fail immediately
            raise RetryError(
                f"Failed to serialize JSON - data contains non-serializable objects: {e}"
            )

        except IOError as e:
            logger.error(f"File write error: {e}")
            raise  # Don't retry IO errors
