# JSON Stream Handling Analysis - Task 1.1

## Overview
This document analyzes the existing JSON stream handling in Claude Cadence's supervisor/agent communication system to inform the design of the JSONStreamMonitor class.

## Current Implementation

### 1. JSON Stream Processing in Orchestrator (`cadence/orchestrator.py`)

#### `run_claude_with_realtime_output` Method (lines 106-207)
- **Purpose**: Processes real-time output from Claude CLI (both agent and supervisor)
- **Key Features**:
  - Reads stdout line by line asynchronously
  - Attempts to parse each line as JSON
  - Extracts and displays different message types:
    - `system` messages (including model info from init)
    - `assistant` messages (text content and tool usage)
    - `user` messages (tool results)
    - `result` messages (completion info)
  - Falls back to plain text display for non-JSON lines
  - Uses color coding for different process types (SUPERVISOR vs AGENT)

**Code Pattern**:
```python
try:
    json_data = json.loads(line_str)
    msg_type = json_data.get('type', 'unknown')
    # Process based on message type
except json.JSONDecodeError:
    # Display as plain text
```

#### `run_supervisor_analysis` Method (lines 517-917)
- **JSON Parsing Strategy**:
  - Collects all output lines from supervisor
  - Uses regex pattern to find JSON objects containing "action" field
  - Processes lines to find the LAST valid JSON (final decision)
  - Implements retry logic (up to 5 attempts) if valid JSON not found
  - Validates required fields based on action type

**Key JSON Extraction Logic**:
```python
json_pattern = re.compile(r'\{[^{}]*"action"[^{}]*\}', re.DOTALL)
# Looks for JSON objects with "action" field
# Takes the LAST valid JSON found
```

### 2. JSON Stream Processing in Task Supervisor (`cadence/task_supervisor.py`)

#### `execute_with_todos` Method (lines 363-637)
- **Stream Processing**:
  - Uses subprocess with real-time line-by-line reading
  - Attempts to parse JSON from lines starting with '{'
  - Extracts 'content' field from JSON if present
  - Falls back to treating as regular output if not JSON

**Code Pattern**:
```python
if line.strip().startswith('{'):
    try:
        data = json.loads(line.strip())
        if 'content' in data:
            output_lines.append(data['content'])
    except json.JSONDecodeError:
        output_lines.append(line.rstrip())
```

#### `supervisor_ai_analysis` Method (lines 882-954)
- Uses `--output-format json` flag for Claude CLI
- Expects complete JSON response in stdout
- Parses entire stdout as single JSON object

### 3. Key Observations

#### Current Strengths:
1. **Async Processing**: Uses asyncio for non-blocking stream reading
2. **Error Handling**: Graceful fallback for non-JSON lines
3. **Message Type Awareness**: Differentiates between various Claude message types
4. **Retry Logic**: Supervisor has retry mechanism for JSON parsing failures

#### Current Gaps and Issues:

1. **No Buffering for Partial JSON**:
   - Current implementation assumes each line is complete JSON
   - No handling for JSON objects split across multiple lines
   - No accumulation of partial JSON data

2. **No Stream State Management**:
   - Each line processed independently
   - No context maintained between lines
   - No recovery from partial reads

3. **Limited Error Recovery**:
   - JSON parsing errors simply fall back to text
   - No attempt to reconstruct partial JSON
   - No tracking of malformed JSON patterns

4. **Inconsistent JSON Detection**:
   - Supervisor uses regex for flexible JSON finding
   - Agent processing uses simple line parsing
   - No unified approach to JSON extraction

5. **No Event System**:
   - Direct processing without event emission
   - No listener/observer pattern for JSON objects
   - Tight coupling between parsing and display

6. **No JSON Validation**:
   - Basic field checking but no schema validation
   - No type checking for JSON fields
   - Manual validation of required fields

7. **Performance Concerns**:
   - Regex parsing on every line (supervisor)
   - Multiple JSON parse attempts per line
   - No caching of parsed results

## Recommendations for JSONStreamMonitor

Based on this analysis, the JSONStreamMonitor should address these gaps:

### 1. Implement Proper Buffering
- Accumulate partial JSON data across lines
- Detect JSON object boundaries (bracket counting)
- Handle nested objects and arrays

### 2. Add Stream State Management
- Maintain parsing state (in_json, in_string, escape_next)
- Track current nesting level
- Remember partial data between reads

### 3. Create Robust Error Recovery
- Attempt to recover from malformed JSON
- Log parsing errors with context
- Continue processing after errors

### 4. Unified JSON Detection
- Consistent approach for finding JSON in mixed output
- Support both line-based and multi-line JSON
- Handle JSON embedded in text output

### 5. Implement Event System
- Event emitter pattern for complete JSON objects
- Different event types for different message types
- Decoupled parsing from consumption

### 6. Add JSON Schema Validation
- Define schemas for expected JSON types
- Validate parsed objects against schemas
- Type checking and field validation

### 7. Optimize Performance
- Efficient bracket counting algorithm
- Minimal regex usage
- Smart buffering with size limits

## Integration Points

The JSONStreamMonitor should integrate with:

1. **`run_claude_with_realtime_output`**: Replace current JSON parsing logic
2. **`run_supervisor_analysis`**: Use for extracting decision JSON
3. **`execute_with_todos`**: Process agent output streams
4. **Future code review detection**: Monitor for code review triggers

## Conclusion

The current implementation provides basic JSON stream handling but lacks robustness for production use. The JSONStreamMonitor will build upon existing patterns while adding proper buffering, state management, error recovery, and event-driven architecture to handle complex streaming scenarios reliably.
