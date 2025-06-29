# JSON Handoff Retry System Design

## Overview
This document outlines a comprehensive retry system for JSON handoffs between Claude Cadence components. The system ensures robust recovery from JSON parsing failures by using the `--continue` flag to maintain conversation context during retries.

## Design Principles

### 1. Common JSON Envelope Structure
All components should use a standard JSON envelope for handoffs:

```json
{
    "component": "agent|supervisor|code_review",
    "session_id": "string",
    "timestamp": "ISO8601",
    "status": "success|failure|partial",
    "data": {
        // Component-specific payload
    },
    "metadata": {
        "retry_count": 0,
        "previous_errors": []
    }
}
```

### 2. Detection Strategy
- **Malformed JSON**: Syntax errors, truncated output, invalid characters
- **Missing Required Fields**: Schema validation failures
- **Semantic Errors**: Valid JSON but missing critical data
- **Partial JSON**: Incomplete objects due to interruption

### 3. Retry Mechanism
Each retry should:
1. Use `--continue` flag to maintain conversation context
2. Include a special retry prompt explaining the failure
3. Track retry attempts to prevent infinite loops
4. Log all attempts for debugging

## Identified JSON Handoff Points

### 1. Agent → Orchestrator Handoff
**File**: `cadence/orchestrator.py:1080-1095`
**Current Implementation**: Agent writes JSON result to `agent_result_${session_id}.json`
**JSON Structure**:
```json
{
    "task_id": "string",
    "success": boolean,
    "completed_normally": boolean,
    "todos": [],
    "scratchpad_path": "path",
    "execution_time": float
}
```

### 2. Supervisor → Orchestrator Handoff
**File**: `cadence/orchestrator.py:1472-1569`
**Current Implementation**: Supervisor outputs JSON decision via stdout
**JSON Structure**:
```json
{
    "action": "execute|skip|complete|error|review",
    "task_id": "string",
    "task_title": "string",
    "subtasks": [],
    "project_root": "path",
    "guidance": "string",
    "session_id": "string",
    "reason": "string"
}
```

### 3. Code Review Agent → Supervisor Handoff
**File**: `cadence/code_review_agent.py:354-388`
**Current Implementation**: Returns ReviewResult object (should be JSON)
**JSON Structure**:
```json
{
    "status": "analyze_complete|pause_for_code_review",
    "issues_found": [],
    "confidence": "string",
    "step_number": int,
    "total_steps": int,
    "findings": "string",
    "code_review_status": {}
}
```

### 4. Scratchpad Validation
**File**: `cadence/orchestrator.py:1142-1175`
**Current Implementation**: Already has retry logic with special prompt
**Enhancement Needed**: Ensure --continue flag is used consistently

## Implementation Requirements

### A. Scratchpad Retry Enhancement
**Current State**: Already implemented with retry logic
**Required Changes**:
1. Verify `--continue` flag usage in `retry_agent_for_scratchpad()`
2. Add structured error tracking
3. Implement exponential backoff

### B. Agent JSON Handoff Retry
**New Implementation Required**:
```python
def retry_agent_for_json_handoff(self, session_id: str, error_details: str, retry_count: int) -> AgentResult:
    """Retry agent execution when JSON handoff fails"""
    retry_prompt = f"""Your previous execution completed but the JSON output was malformed.

Error details: {error_details}

Please complete your task summary again, ensuring you write a valid JSON object to your scratchpad with the following structure:

{{
    "task_id": "the task ID you worked on",
    "success": true/false,
    "completed_normally": true/false,
    "todos": ["list", "of", "remaining", "todos"],
    "scratchpad_path": "path to your scratchpad file"
}}

Remember: You have access to your previous work context. Focus only on producing the correct JSON output."""

    cmd = ["claude", "-c", "-p", retry_prompt]  # -c flag for continue
    # Execute and return result
```

### C. Supervisor JSON Handoff Retry
**Current State**: Basic retry exists (lines 1554-1569)
**Required Enhancement**:
```python
def create_supervisor_json_retry_prompt(self, error: str, attempt: int) -> str:
    """Create targeted retry prompt for supervisor JSON issues"""
    if attempt == 1:
        return f"""CRITICAL: Your previous output had invalid JSON formatting.
Error: {error}

Please analyze the current task state and output ONLY a valid JSON object.
[Minimal format template]"""
    else:
        return f"""JSON PARSING FAILED AGAIN (Attempt {attempt}/5)
Previous error: {error}

OUTPUT ONLY THE JSON OBJECT - NO OTHER TEXT:
{{
    "action": "execute|skip|complete|error|review",
    "session_id": "{self.current_session_id}",
    "reason": "brief explanation"
    // other required fields based on action
}}"""
```

### D. Code Review JSON Handoff Retry
**New Implementation Required**:
```python
def retry_code_review_with_continue(self, files: List[str], previous_error: str) -> ReviewResult:
    """Retry code review with continuation context"""
    retry_params = {
        "step": f"Previous review failed with JSON error: {previous_error}. Please complete the review and ensure valid JSON output.",
        "next_step_required": False,
        "confidence": "high",
        "use_continue": True,  # Critical for context
        "retry_metadata": {
            "previous_error": previous_error,
            "retry_timestamp": datetime.now().isoformat()
        }
    }
    # Execute with MCP tool
```

## Edge Cases and Mitigation

### 1. Retry Loops
- **Issue**: Component repeatedly fails to produce valid JSON
- **Mitigation**: Hard limit of 5 retries, then escalate to fatal error
- **Tracking**: Store retry history in metadata

### 2. Semantic JSON Errors
- **Issue**: Valid JSON but missing critical fields
- **Detection**: Schema validation after parsing
- **Retry Strategy**: Include specific missing fields in retry prompt

### 3. Partial JSON Recovery
- **Issue**: JSON truncated mid-object
- **Detection**: Use SimpleJSONStreamMonitor for buffering
- **Recovery**: Request completion of specific section with --continue

### 4. Context Window Exhaustion
- **Issue**: Retries consume too much context
- **Mitigation**: Progressive prompt reduction on each retry
- **Final Attempt**: Minimal prompt with just JSON template

## Timeout and Stuck State Handling

### Process Timeout Detection
```python
class TimeoutHandler:
    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timeout_seconds
        self.start_time = None

    def check_timeout(self) -> bool:
        if self.start_time and (time.time() - self.start_time) > self.timeout:
            return True
        return False
```

### Stuck State vs Parse Error Distinction
1. **Parse Error**: Process completes but output is malformed
   - Action: Retry with --continue and error details
2. **Stuck State**: Process doesn't complete within timeout
   - Action: Kill process, log state, attempt fresh start
3. **Semantic Error**: Valid JSON but wrong content
   - Action: Retry with specific correction prompt

## Logging and Debugging

### Required Logging
1. **Pre-retry**: Original output, error details, retry count
2. **Retry Attempt**: Command used, prompt given, timestamp
3. **Post-retry**: Success/failure, new output, total attempts
4. **Final State**: Resolution or escalation decision

### Debug Information Structure
```json
{
    "handoff_type": "agent|supervisor|code_review",
    "session_id": "string",
    "original_error": {
        "type": "json_parse|missing_field|semantic",
        "details": "string",
        "output_sample": "first 500 chars"
    },
    "retry_attempts": [
        {
            "attempt": 1,
            "timestamp": "ISO8601",
            "prompt_used": "string",
            "result": "success|failure",
            "error": "string or null"
        }
    ],
    "final_status": "recovered|escalated|fatal"
}
```

## Testing Strategy

### Unit Tests Required
1. Test JSON parsing with various malformed inputs
2. Verify retry prompt generation
3. Validate retry counting and limits
4. Test timeout detection

### Integration Tests Required
1. Simulate JSON failures at each handoff point
2. Verify --continue flag preserves context
3. Test retry exhaustion escalation
4. Validate logging completeness

### Failure Injection Points
1. Agent result file corruption
2. Supervisor stdout truncation
3. Code review MCP response malformation
4. Network timeout simulation

## Implementation Priority

1. **High Priority** (Immediate implementation):
   - Agent JSON handoff retry (most common failure point)
   - Enhance supervisor retry with better prompts
   - Consistent --continue flag usage

2. **Medium Priority** (Next sprint):
   - Code review JSON retry system
   - Comprehensive retry tracking
   - Debug information structure

3. **Low Priority** (Future enhancement):
   - Partial JSON recovery
   - Advanced timeout strategies
   - Retry analytics dashboard

## Configuration Additions

```yaml
# config.yaml additions
retry_settings:
  max_json_retries: 5  # Already exists
  json_retry_delay: 1.0  # seconds between retries
  use_exponential_backoff: true
  max_backoff_delay: 30.0
  enable_partial_recovery: false
  retry_prompt_reduction: true  # Progressively shorter prompts

handoff_validation:
  strict_schema_validation: true
  log_all_handoffs: true
  save_failed_handoffs: true
  failed_handoff_dir: ".cadence/failed_handoffs"
```

## Monitoring and Alerts

### Metrics to Track
1. JSON handoff success rate by component
2. Average retry count before success
3. Retry exhaustion frequency
4. Most common failure types

### Alert Conditions
1. Retry exhaustion rate > 5%
2. Average retries > 2
3. Any component timeout > 300s
4. Circular retry pattern detected

## Code Review Findings

### O3 Review Results (Critical Issues Found: 4)
1. **No retry for agent JSON write** (orchestrator.py:2002) - Agent results written without retry mechanism
2. **No retry for supervisor reading agent JSON** (orchestrator.py:1256) - Supervisor reads agent results without retry
3. **Code review agent lacks retry logic** (code_review_agent.py:280) - MCP tool calls have no retry
4. **Enhanced agent dispatcher missing JSON handling** - No JSON validation or retry in dispatch operations

### Gemini 2.5 Pro Review Results (Confirmed All O3 Findings)
- Validated all 4 critical issues from O3 review
- Confirmed 6 high-priority issues
- Verified positive findings: scratchpad and supervisor retries already use --continue correctly

### Complete List of JSON Handoff Points Requiring Retry Logic

#### 1. Agent Result Write (CRITICAL - No Retry)
**Location**: `orchestrator.py:save_agent_results()` (line 2002)
```python
# Current implementation - no retry on write failure
with open(results_file, 'w') as f:
    json.dump({...}, f, indent=2)
```

#### 2. Agent Result Read (CRITICAL - No Retry)
**Location**: `orchestrator.py:run_supervisor()` (line 1256)
```python
# Current implementation - no retry on read failure
with open(agent_results_file, 'r') as f:
    previous_agent_result = json.load(f)
```

#### 3. Code Review MCP Call (HIGH - No Retry)
**Location**: `code_review_agent.py:_perform_review()` (line 280)
```python
# Current implementation - no retry on MCP failure
response = self.mcp_client.call_tool("mcp__zen__codereview", review_params)
```

#### 4. Enhanced Agent Dispatcher (HIGH - No JSON Handling)
**Location**: `enhanced_agent_dispatcher.py`
- No JSON validation in dispatch operations
- No retry mechanism for JSON communication failures

#### 5. Scratchpad Validation (POSITIVE - Already Implemented)
**Location**: `orchestrator.py:retry_agent_for_scratchpad()` (lines 1925-1947)
- ✅ Correctly uses --continue flag
- ✅ Has retry logic with special prompts
- ✅ Tracks retry attempts

#### 6. Supervisor JSON Output (POSITIVE - Already Implemented)
**Location**: `orchestrator.py:run_supervisor()` (lines 1406-1409)
- ✅ Uses --continue flag for JSON retries
- ✅ Has minimal retry prompt for token limit issues

## HELP Request System Integration

### Overview
The HELP request system allows agents to signal when they're stuck and need expert guidance. This integrates with the JSON handoff retry system to ensure reliable communication of help requests and zen advice responses.

### 1. HELP Signal Detection

**Signal Format** (Consensus from O3 and Gemini):
```
>>>HELP<<<
```
Or for system namespace approach:
```
CLAUDE_CADENCE_SYSTEM_SIGNAL::HELP_REQUEST
```

**Agent Implementation**:
```python
# Agent signals for help when stuck
print(">>>HELP<<<", flush=True)
# Optional: Include context on next line
print(json.dumps({"reason": "stuck_in_loop", "step": 17}), flush=True)
```

### 2. Orchestrator Real-Time Monitoring

**Enhanced run_claude_with_realtime_output()**:
```python
async def run_claude_with_realtime_output(self, cmd, cwd, prefix):
    help_signal_detected = False
    help_events = []

    async for line in process.stdout:
        # Check for HELP signal
        if ">>>HELP<<<" in line or "CLAUDE_CADENCE_SYSTEM_SIGNAL::HELP_REQUEST" in line:
            help_signal_detected = True
            help_events.append({
                "timestamp": datetime.now().isoformat(),
                "raw_line": line.strip()
            })
            # Try to read optional JSON context (non-blocking)
            try:
                next_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=0.1
                )
                if next_line.strip().startswith('{'):
                    help_events[-1]["context"] = json.loads(next_line)
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

        # Forward to live output
        print(f"{prefix}: {line}", end='')

    return returncode, all_output, help_signal_detected, help_events
```

### 3. Updated JSON Structures

#### AgentResult Enhancement
```python
@dataclass
class AgentResult:
    """Result from agent execution"""
    success: bool
    session_id: str
    output_file: str
    error_file: str
    execution_time: float
    completed_normally: bool = False
    requested_help: bool = False  # ← New field
    help_events: List[Dict] = None  # ← New field for detailed events
    errors: List[str] = None
    quit_too_quickly: bool = False
```

#### New ZenAdviceResult Structure
```python
@dataclass
class ZenAdviceResult:
    """Result from zen advice consultation"""
    success: bool
    advice_content: Dict[str, str]  # Model -> advice mapping
    models_consulted: List[str]
    session_id: str
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: str = ""
```

### 4. Discrepancy Handling

**Orchestrator Reconciliation Logic**:
```python
def save_agent_results_with_help_reconciliation(
    self,
    agent_result: AgentResult,
    session_id: str,
    help_detected_in_stream: bool,
    help_events: List[Dict]
):
    """Save agent results with HELP signal reconciliation"""

    # Reconcile help detection discrepancies
    if help_detected_in_stream and not agent_result.requested_help:
        logger.warning(
            f"HELP signal detected in stream but not in agent JSON. "
            f"Overriding requested_help to True for session {session_id}"
        )
        agent_result.requested_help = True
        agent_result.help_events = help_events

    # Save enhanced results
    results_file = self.supervisor_dir / f"agent_result_{session_id}.json"

    # Retry logic for JSON write (addressing critical gap #1)
    for attempt in range(self.config.get("max_json_retries", 3)):
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    "success": agent_result.success,
                    "session_id": agent_result.session_id,
                    "requested_help": agent_result.requested_help,
                    "help_events": agent_result.help_events or [],
                    # ... other fields
                }, f, indent=2)
            break
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Failed to write agent results (attempt {attempt + 1}): {e}")
            time.sleep(1 * (attempt + 1))  # Exponential backoff
```

### 5. Supervisor Zen Advice Integration

**Supervisor Decision with HELP Handling**:
```python
def process_agent_result(self, agent_result: AgentResult) -> SupervisorDecision:
    """Process agent result and decide next action"""

    if agent_result.requested_help:
        logger.info("Agent requested help - consulting zen advisors")

        # Get zen advice with retry logic
        zen_result = self.get_zen_advice_with_retry(
            agent_result=agent_result,
            max_retries=3
        )

        if zen_result.success:
            # Include advice in decision
            return SupervisorDecision(
                action="execute",  # Re-run agent with guidance
                guidance=self.format_zen_guidance(zen_result),
                zen_advice=zen_result.advice_content,
                session_id=agent_result.session_id
            )
        else:
            # Escalate if zen advice failed
            return SupervisorDecision(
                action="error",
                reason=f"Failed to get zen advice: {zen_result.error_message}",
                session_id=agent_result.session_id
            )
```

**Zen Advice with Retry**:
```python
def get_zen_advice_with_retry(
    self,
    agent_result: AgentResult,
    max_retries: int = 3
) -> ZenAdviceResult:
    """Get advice from zen models with retry logic"""

    # Check for resume state
    resume_file = self.supervisor_dir / f"zen_advice_request_{agent_result.session_id}.json"

    if resume_file.exists() and self.use_continue:
        # Resume from previous attempt
        with open(resume_file, 'r') as f:
            resume_state = json.load(f)
            retry_count = resume_state.get("retry_count", 0) + 1
    else:
        retry_count = 0
        # Save initial request state
        with open(resume_file, 'w') as f:
            json.dump({
                "agent_result": agent_result.__dict__,
                "retry_count": 0,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    for attempt in range(retry_count, max_retries):
        try:
            # Prepare zen consultation prompt
            prompt = self.build_zen_help_prompt(agent_result, attempt)

            # Call zen MCP tools with --continue on retry
            advice = {}
            for model in ["o3", "gemini-2.5-pro"]:
                cmd = ["mcp__zen__chat"]
                params = {
                    "prompt": prompt,
                    "model": model,
                    "continuation_id": f"help_{agent_result.session_id}" if attempt > 0 else None
                }

                result = self.call_mcp_tool(cmd, params)
                if result.get("status") == "success":
                    advice[model] = result.get("content", "")

            if advice:
                # Success - clean up state file
                resume_file.unlink(missing_ok=True)
                return ZenAdviceResult(
                    success=True,
                    advice_content=advice,
                    models_consulted=list(advice.keys()),
                    session_id=agent_result.session_id
                )

        except Exception as e:
            logger.error(f"Zen advice attempt {attempt + 1} failed: {e}")
            # Update retry count in state file
            with open(resume_file, 'w') as f:
                json.dump({
                    "agent_result": agent_result.__dict__,
                    "retry_count": attempt + 1,
                    "last_error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    return ZenAdviceResult(
        success=False,
        advice_content={},
        models_consulted=[],
        session_id=agent_result.session_id,
        error_message=f"Failed after {max_retries} attempts",
        retry_count=max_retries
    )
```

### 6. Complete HELP Request Flow

```mermaid
sequenceDiagram
    participant Agent
    participant Orchestrator
    participant Supervisor
    participant ZenAdvisor

    Agent->>Orchestrator: stdout: ">>>HELP<<<"
    Orchestrator->>Orchestrator: Set help_detected=True
    Agent->>Orchestrator: Complete with JSON result

    alt JSON missing requested_help
        Orchestrator->>Orchestrator: Override requested_help=True
        Orchestrator->>Orchestrator: Log discrepancy warning
    end

    Orchestrator->>Supervisor: AgentResult (requested_help=True)

    Supervisor->>Supervisor: Save zen_advice_request.json

    loop Retry up to 3 times
        Supervisor->>ZenAdvisor: Request advice (with --continue if retry)
        alt Success
            ZenAdvisor-->>Supervisor: JSON advice response
            Supervisor->>Supervisor: Delete state file
            break
        else Failure
            ZenAdvisor-->>Supervisor: Error/timeout
            Supervisor->>Supervisor: Update retry count
        end
    end

    Supervisor->>Agent: Re-execute with zen guidance
```

### 7. Key Design Decisions

Based on expert consensus from O3 and Gemini 2.5 Pro:

1. **Signal Format**: Both experts recommend simple, unambiguous signals
   - O3 suggests: `>>>HELP<<<` (simpler, less collision risk)
   - Gemini suggests: `CLAUDE_CADENCE_SYSTEM_SIGNAL::HELP_REQUEST` (namespaced, extensible)
   - Either works, but consistency is key

2. **Orchestrator as Source of Truth**: Both experts agree the orchestrator's stream detection should override agent JSON
   - Handles agent crashes after signaling
   - Ensures help requests are never lost
   - Logs discrepancies for debugging

3. **Synchronous Zen Advice**: Both recommend keeping zen consultation within supervisor flow
   - Simpler than spawning separate agent
   - Reuses existing retry patterns
   - Maintains linear control flow

4. **State File Pattern**: For resumability with --continue
   - Save request state before zen consultation
   - Check for state file on startup
   - Clean up on success

### 8. Implementation Checklist

- [ ] Add HELP signal detection to `run_claude_with_realtime_output()`
- [ ] Update `AgentResult` dataclass with help fields
- [ ] Implement discrepancy reconciliation in orchestrator
- [ ] Add retry logic to `save_agent_results()`
- [ ] Create `ZenAdviceResult` dataclass
- [ ] Implement `get_zen_advice_with_retry()` in supervisor
- [ ] Add state file management for zen advice resumption
- [ ] Update supervisor decision logic for help requests
- [ ] Add unit tests for help signal detection
- [ ] Add integration tests for full help flow
- [ ] Update agent documentation with HELP signal format
- [ ] Add help request rate metrics for monitoring

## Updated Implementation Priority

### Immediate Implementation (Critical)
1. **Agent JSON Write Retry**
   - Wrap save_agent_results() with retry logic
   - Add structured error tracking
   - Implement exponential backoff

2. **Agent JSON Read Retry**
   - Add retry wrapper around supervisor's agent result reading
   - Include validation for required fields
   - Track read failures separately from write failures

3. **Standardize JSON Envelope**
   - Implement common structure across all components
   - Add version field for future compatibility
   - Include retry metadata in all handoffs

### Next Sprint (High Priority)
1. **Code Review JSON Retry**
   - Implement retry_code_review_with_continue()
   - Handle token limit failures specially
   - Add progressive prompt reduction

2. **Enhanced Dispatcher JSON Handling**
   - Add JSON validation to dispatch operations
   - Implement retry for dispatch communications
   - Track dispatch-specific failures

3. **Comprehensive Logging**
   - Structured logging for all retry attempts
   - Metrics collection for monitoring
   - Debug mode for detailed retry traces

## Conclusion

This comprehensive retry system will significantly improve Claude Cadence's resilience to JSON handoff failures. The code review with both O3 and Gemini 2.5 Pro has validated the design and identified all critical gaps. By using the --continue flag strategically and implementing targeted retry prompts, we can maintain conversation context while recovering from transient failures. The system balances robustness with performance, ensuring that retries don't create infinite loops or consume excessive resources.

**Review Status**: Design validated by O3 and Gemini 2.5 Pro with 100% agreement on critical issues.
