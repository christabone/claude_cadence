"""
Simple JSON Stream Monitor for handling multi-line JSON objects
"""
import json
import logging
import re
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for better performance
# Strategy 1: Markdown code blocks with optional language specifier
JSON_BLOCK_PATTERN = re.compile(
    r'```(?:json)?\s*\n?([\s\S]*?)\n?```',
    re.IGNORECASE | re.DOTALL
)

# Strategy 2: Find potential JSON start positions
JSON_START_PATTERN = re.compile(r'[{\[]')

# Strategy 3: Handle escaped JSON - matches structures with escaped quotes
ESCAPED_JSON_PATTERN = re.compile(
    r'(\{[^{}]*?\\"[^{}]*?\})',
    re.DOTALL
)


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
        self.last_json_object: Optional[dict] = None
        self.decoder = json.JSONDecoder()

    def _extract_json_object(self, text: str, start_index: int) -> Optional[str]:
        """
        Scans forward from a starting index to find a complete, balanced JSON object or array.

        Args:
            text: The string to search within
            start_index: The index of an opening '{' or '['

        Returns:
            The complete JSON string if found, otherwise None
        """
        stack = []
        in_string = False
        is_escaped = False

        if start_index >= len(text) or text[start_index] not in '{[':
            return None

        end_char_map = {'{': '}', '[': ']'}
        stack.append(end_char_map[text[start_index]])

        for i in range(start_index + 1, len(text)):
            char = text[i]

            if in_string:
                if is_escaped:
                    is_escaped = False
                elif char == '\\':
                    is_escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                    is_escaped = False
                elif char == stack[-1] if stack else None:
                    stack.pop()
                elif char in end_char_map:
                    stack.append(end_char_map[char])
                elif char in end_char_map.values() and stack:
                    # Mismatched closer, invalid structure
                    return None

            if not stack:
                return text[start_index:i + 1]

        return None  # Reached end without finding balanced object

    def _extract_json_candidates(self, text: str) -> List[str]:
        """Extract potential JSON strings from text using multiple strategies."""
        candidates = []

        # Strategy 1: Extract from markdown code blocks
        for match in JSON_BLOCK_PATTERN.finditer(text):
            candidate = match.group(1).strip()
            if candidate:
                candidates.append(candidate)

        # Strategy 2: Find and extract JSON objects/arrays starting anywhere
        processed_indices = set()
        for match in JSON_START_PATTERN.finditer(text):
            start_index = match.start()
            if start_index in processed_indices:
                continue

            json_str = self._extract_json_object(text, start_index)
            if json_str:
                candidates.append(json_str)
                # Mark indices as processed
                for i in range(start_index, start_index + len(json_str)):
                    processed_indices.add(i)

        # Strategy 3: Handle escaped JSON
        for match in ESCAPED_JSON_PATTERN.finditer(text):
            escaped = match.group(1)
            try:
                # Try to unescape using unicode_escape
                unescaped = escaped.encode('utf-8').decode('unicode_escape')
                candidates.append(unescaped)
            except (UnicodeDecodeError, ValueError):
                # Fallback to simple replacement
                unescaped = escaped.replace('\\"', '"').replace('\\\\', '\\')
                candidates.append(unescaped)

        return candidates

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

        # Detect JSON starting anywhere in the line
        if not self.in_json:
            # Find the starting position of a JSON object or array
            idx_obj = line.find('{')
            idx_arr = line.find('[')

            # Filter out -1 (not found) and find minimum index
            start_indices = [i for i in (idx_obj, idx_arr) if i != -1]
            if start_indices:
                start_index = min(start_indices)
                self.in_json = True
                # Buffer only from the start of the JSON
                self.buffer = [line[start_index:]]
            else:
                # Not in JSON mode and no JSON start found
                return None
        else:
            # Already in JSON mode, continue buffering
            if len(self.buffer) >= self.MAX_BUFFER_SIZE:
                error_msg = f"JSON buffer size exceeded ({self.MAX_BUFFER_SIZE} lines). The JSON object is too large or malformed."
                logger.error(error_msg)
                self.in_json = False
                self.buffer = []
                raise ValueError(error_msg)
            self.buffer.append(line)

        # Try to parse accumulated buffer
        if self.in_json and self.buffer:
            json_str = '\n'.join(self.buffer)
            try:
                # Use raw_decode to handle trailing text
                result, end_idx = self.decoder.raw_decode(json_str)

                # Check if there's trailing text
                trailing = json_str[end_idx:].strip()

                # Success - reset buffer
                self.buffer = []
                self.in_json = False

                # If there's trailing text, it might contain another JSON object
                if trailing:
                    # Process trailing text as a new line (recursive call)
                    logger.debug(f"Found trailing text after JSON: {trailing[:50]}...")
                    # Don't process it now, let the caller handle it in the next call

                # Handle stream-json format: extract embedded JSON from assistant messages
                if isinstance(result, dict) and result.get('type') == 'assistant':
                    message = result.get('message', {})
                    content = message.get('content', [])

                    # Collect all valid embedded JSON objects
                    valid_candidates = []

                    # Look for text content that might contain embedded JSON
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text = item.get('text', '').strip()

                            # Extract JSON candidates using improved strategies
                            json_candidates = self._extract_json_candidates(text)

                            # Validate each candidate
                            for candidate in json_candidates:
                                candidate = candidate.strip()
                                if candidate:
                                    try:
                                        embedded_json = json.loads(candidate)
                                        # Validate it has expected agent result fields
                                        if isinstance(embedded_json, dict) and \
                                           'status' in embedded_json and \
                                           'session_id' in embedded_json:
                                            valid_candidates.append(embedded_json)
                                            logger.debug(f"Found valid embedded JSON: keys={list(embedded_json.keys())}")
                                    except json.JSONDecodeError:
                                        logger.debug(f"Failed to parse JSON candidate: {candidate[:50]}...")
                                        continue

                    # Choose the best candidate (last one typically represents final state)
                    if valid_candidates:
                        # Simple selection strategy: take the last valid JSON
                        # Could be enhanced with scoring based on status values
                        self.last_json_object = valid_candidates[-1]
                        logger.debug(
                            f"Selected embedded JSON (status={self.last_json_object.get('status')}, "
                            f"session_id={self.last_json_object.get('session_id')})"
                        )
                    else:
                        # No embedded agent JSON found, store the stream wrapper
                        self.last_json_object = result
                else:
                    # Not an assistant message, store as-is
                    self.last_json_object = result

                return result

            except (json.JSONDecodeError, ValueError):
                # Not complete yet, keep buffering
                pass

        return None

    def reset(self):
        """Reset the buffer and state"""
        self.buffer = []
        self.in_json = False
        self.last_json_object = None

    def get_last_json_object(self) -> Optional[dict]:
        """Get the last successfully parsed JSON object"""
        return self.last_json_object
