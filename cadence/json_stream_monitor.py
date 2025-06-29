"""
Simple JSON Stream Monitor for handling multi-line JSON objects
"""
import json
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class SimpleJSONStreamMonitor:
    """
    Simple JSON buffering for multi-line objects.

    Accumulates lines that look like JSON and attempts to parse
    after each new line. Returns complete JSON objects when found.
    """

    MAX_BUFFER_SIZE = 50000  # Maximum number of lines to buffer (50K lines)
    MAX_LINE_LENGTH = 1000000  # Maximum length of a single line (1MB)

    def __init__(self):
        self.buffer: List[str] = []
        self.in_json = False

    def process_line(self, line: str) -> Optional[dict]:
        """
        Process a line and return JSON if complete.

        Args:
            line: A line of text from the stream

        Returns:
            Parsed JSON dict if complete, None otherwise
        """
        line = line.strip()
        if not line:
            return None

        # Check line length limit
        if len(line) > self.MAX_LINE_LENGTH:
            error_msg = f"Line exceeds maximum length ({len(line)} > {self.MAX_LINE_LENGTH}). This could indicate corrupt data or a security issue."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Start buffering if line starts with JSON object or array
        if not self.in_json and line and line[0] in '{[':
            self.in_json = True
            self.buffer = [line]
        elif self.in_json:
            # Check buffer size limit
            if len(self.buffer) >= self.MAX_BUFFER_SIZE:
                error_msg = f"JSON buffer size exceeded ({self.MAX_BUFFER_SIZE} lines). The JSON object is too large or malformed."
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.buffer.append(line)
        else:
            # Not in JSON mode and doesn't start JSON - plain text
            return None

        # Try to parse accumulated buffer
        if self.in_json and self.buffer:
            json_str = '\n'.join(self.buffer)
            try:
                result = json.loads(json_str)
                # Success - reset and return
                self.buffer = []
                self.in_json = False
                return result
            except json.JSONDecodeError:
                # Not complete yet, keep buffering
                pass

        return None

    def reset(self):
        """Reset the buffer and state"""
        self.buffer = []
        self.in_json = False
