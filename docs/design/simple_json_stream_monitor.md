# Simple JSON Stream Monitor Design

## Goal
Handle multi-line JSON objects in Claude's streaming output with minimal complexity.

## Problem
Current implementation processes line-by-line and fails when JSON spans multiple lines.

## Simple Solution
A basic class that:
1. Accumulates text in a buffer
2. Tries to parse JSON after each newline
3. Yields complete JSON objects
4. Clears buffer on success

## Implementation (< 50 lines)

```python
import json
from typing import List, Optional, Iterator

class SimpleJSONStreamMonitor:
    """Simple JSON buffering for multi-line objects"""

    def __init__(self):
        self.buffer = []
        self.in_json = False

    def process_line(self, line: str) -> Optional[dict]:
        """Process a line and return JSON if complete"""
        line = line.strip()
        if not line:
            return None

        # Start buffering if line looks like JSON
        if line.startswith('{'):
            self.in_json = True
            self.buffer = [line]
        elif self.in_json:
            self.buffer.append(line)
        else:
            return None

        # Try to parse accumulated buffer
        if self.in_json:
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
```

## Integration

Replace current parsing in `orchestrator.py`:

```python
# Before:
try:
    json_data = json.loads(line_str)
    # ... process
except json.JSONDecodeError:
    # ... fallback

# After:
monitor = SimpleJSONStreamMonitor()
# In loop:
json_data = monitor.process_line(line_str)
if json_data:
    # ... process
else:
    # ... plain text
```

## Testing

1. Single-line JSON: `{"type": "assistant", "content": "hello"}`
2. Multi-line JSON:
   ```
   {
     "type": "assistant",
     "content": "hello"
   }
   ```
3. Buffer reset after success

That's it. Simple, focused, effective.
