# Agent Architecture Simplification Plan

## Current State (Overcomplicated)
- Multiple agent classes:
  - agent_dispatcher.py
  - enhanced_agent_dispatcher.py
  - fix_agent_dispatcher.py
  - code_review_agent.py
  - agent_communication_handler.py
- Multiple config sections:
  - fix_agent_dispatcher (max_attempts, timeouts, verification, etc.)
  - dispatch (code_review, fix_tracking, escalation)
- Complex orchestration with multiple imports and class instantiations

## Target State (Simplified)
- ONE UnifiedAgent class that can be configured via profiles
- ALL config under single 'agent' section
- Different behaviors via simple profile-based settings

## Agreed Architecture (Consensus from o3 and Gemini 2.5 Pro)

### 1. Configuration Structure
```yaml
agent:
  # Default settings inherited by all profiles
  defaults:
    model: "claude-sonnet-4-20250514"
    timeout_seconds: 120
    retry_count: 1
    use_continue: false
    temperature: 0.2
    tools:
      - bash
      - read
      - write
      - edit
      - mcp

  # Specific agent profiles
  profiles:
    standard:
      description: "Standard agent for task execution"
      # Uses all defaults

    review:
      description: "Code review agent"
      retry_count: 1
      custom_prompt: |
        You are a senior code reviewer. Review the changes and provide focused feedback.
        Focus on: security, performance, maintainability, and correctness.
        {task}

    fix:
      description: "Fix agent for addressing issues"
      retry_count: 3
      use_continue: true
      custom_prompt: |
        A previous attempt encountered issues. Your task is to fix the following:
        {task}

        Previous error: {last_error}

        Approach this systematically and ensure the fix is complete.
```

### 2. UnifiedAgent Class Design
```python
class UnifiedAgent:
    def __init__(self, profile_name: str, config: dict, session_context: dict):
        # Load profile settings with defaults
        # Set up retry behavior, prompts, etc.

    def execute(self, task: str, context: dict = None) -> AgentResult:
        # Main execution loop with retry logic
        # Handles both normal and --continue cases
        # Returns standardized result
```

### 3. Migration Steps

#### Phase 1: Config Consolidation
1. Create new `agent` section with profiles
2. Remove `fix_agent_dispatcher` section
3. Remove `dispatch` section
4. Update config.py to handle new structure

#### Phase 2: Create UnifiedAgent
1. Create `cadence/unified_agent.py`
2. Implement profile loading and defaults merging
3. Implement retry logic and --continue handling
4. Add structured result return (AgentResult)

#### Phase 3: Update Orchestrator
1. Replace all dispatcher imports with UnifiedAgent
2. Replace instantiation code:
   ```python
   # Old:
   dispatcher = EnhancedAgentDispatcher(config)
   result = dispatcher.execute(task)

   # New:
   agent = UnifiedAgent("standard", config, session_context)
   result = agent.execute(task)
   ```

#### Phase 4: Remove Old Code
1. Delete all old dispatcher/agent classes
2. Clean up unused imports
3. Update tests to use UnifiedAgent

## Key Benefits
1. **Simplicity**: One class to understand and maintain
2. **Flexibility**: New agent types are just config changes
3. **Consistency**: All agents share same core behavior
4. **Testability**: Easier to test one class with different configs
5. **Maintainability**: Less code, less complexity

## Implementation Notes
- Keep existing AgentResult structure for compatibility
- Preserve session management and logging
- Maintain backward compatibility during migration
- Use dependency injection for claude client

## Success Criteria
- All existing functionality preserved
- Config reduced from ~100 lines to ~40 lines
- Code files reduced from 6 to 1
- No special cases in orchestrator
- All tests passing
