# TODO: Implement Consensus and Review Zen Tools

## Consensus Tool Implementation
**Purpose**: Help agents make architectural decisions between multiple approaches

### When to Trigger (Supervisor Decision):
- Task involves choosing between multiple valid implementations
- Architecture decisions that affect the whole project
- Technology selection (e.g., which library/framework to use)
- Design pattern choices

### Example Scenarios:
```json
// Supervisor detects architecture decision needed
{
    "action": "zen_assistance",
    "zen_needed": {
        "required": true,
        "tool": "consensus",
        "reason": "Need to choose between REST API, GraphQL, or gRPC for service communication",
        "focus_area": "service architecture"
    }
}
```

### Implementation Needs:
1. Actual MCP call to zen__consensus with options
2. Format consensus results into actionable guidance
3. Pass guidance to agent in next iteration

## Review Tool Implementation  
**Purpose**: Proactive code quality review before implementation

### When to Trigger (Supervisor Decision):
- Complex algorithm implementation
- Security-sensitive code sections
- Performance-critical paths
- Refactoring existing code

### Example Scenarios:
```json
// Supervisor wants code review before proceeding
{
    "action": "zen_assistance", 
    "zen_needed": {
        "required": true,
        "tool": "review",
        "reason": "Task involves refactoring authentication system - need review of approach",
        "focus_area": "security implications"
    }
}
```

### Implementation Needs:
1. Actual MCP call to zen__review with code context
2. Extract critical feedback only (not style issues)
3. Incorporate feedback into agent guidance

## Integration Architecture

### Current Flow (Not Working):
1. Supervisor decides zen_assistance needed
2. Orchestrator logs warning "Zen X integration not yet implemented"
3. Continues without actually getting help

### Needed Flow:
1. Supervisor decides zen_assistance needed with specific tool
2. Orchestrator calls appropriate Zen tool via MCP
3. Zen provides focused guidance
4. Orchestrator updates supervisor guidance
5. Agent receives enhanced guidance in next iteration

### Key Design Principle:
Both tools should provide MINIMAL, FOCUSED guidance to prevent scope creep:
- Consensus: Pick the simplest working option
- Review: Only flag critical issues that block task completion