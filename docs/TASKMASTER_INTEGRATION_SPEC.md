# TaskMaster-AI MCP Integration Specification

## Overview

This document defines the expected behavior and integration points for TaskMaster-AI MCP within Claude Cadence. The system should operate entirely through TaskMaster for task management, with no --issue flag or issue-based workflows.

## Core Requirements

### 1. MCP Access Configuration

Both supervisor and agent components MUST have access to taskmaster-ai MCP tools:

**Required MCP Tools:**
- `mcp__taskmaster_ai__get_tasks` - List all tasks
- `mcp__taskmaster_ai__next_task` - Get next pending task
- `mcp__taskmaster_ai__get_task` - Get specific task details
- `mcp__taskmaster_ai__set_task_status` - Update task status
- `mcp__taskmaster_ai__update_subtask` - Update subtask information

**Configuration Requirements:**
- `.mcp.json` file must exist in project root with taskmaster-ai server configuration
- `config.yaml` must list taskmaster MCP tools in both supervisor.tools and agent.tools sections
- TaskMaster directory `.taskmaster/` must exist with initialized `tasks/tasks.json`

### 2. Workflow Architecture

```
┌─────────────────────┐
│   CLI Entry Point   │
│  (no --issue flag)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌────────────────┐
│    Orchestrator     │◄────┤  TaskMaster    │
│                     │     │      MCP       │
└──────────┬──────────┘     └────────────────┘
           │                         ▲
           ▼                         │
┌─────────────────────┐              │
│    Supervisor       │──────────────┘
│ (reads next task)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       Agent         │
│ (executes subtasks) │
│ (updates status)    │
└─────────────────────┘
```

### 3. Task Flow Specifications

#### 3.1 Orchestrator Behavior
- On startup, orchestrator invokes supervisor to check TaskMaster availability
- Orchestrator no longer accepts or processes --issue flag
- Main loop uses supervisor to retrieve next task from TaskMaster MCP
- Orchestrator NEVER directly calls TaskMaster - all access through supervisor/agent

#### 3.2 Supervisor Task Processing
- Supervisor uses `mcp__taskmaster_ai__next_task` to get next pending task
- Supervisor uses `mcp__taskmaster_ai__get_task` to get full task details
- Task details include:
  - Task ID
  - Title
  - Description
  - Status
  - Subtasks (if any)
  - Dependencies
- Supervisor creates execution plan based on task, not on issue description
- Supervisor passes task context to agent in prompt
- Supervisor can return special JSON action `"get_next_task"` to orchestrator

#### 3.3 Agent Task Execution
- Agent receives task ID and subtask details in context
- Agent updates task status through MCP when work begins:
  ```
  mcp__taskmaster_ai__set_task_status --id=<task_id> --status=in-progress
  ```
- Agent marks subtasks complete as they finish
- Agent updates main task status when all subtasks complete:
  ```
  mcp__taskmaster_ai__set_task_status --id=<task_id> --status=done
  ```

### 4. JSON Command System

The supervisor's JSON response must support these actions:

```json
{
  "action": "get_next_task",
  "session_id": "20250628_123456",
  "reason": "Ready to retrieve next pending task from TaskMaster"
}
```

```json
{
  "action": "execute",
  "task_id": "1.2",
  "task_title": "Implement user authentication",
  "subtasks": [
    {"id": "1.2.1", "title": "Set up JWT tokens", "description": "..."},
    {"id": "1.2.2", "title": "Create login endpoint", "description": "..."}
  ],
  "project_path": "/path/to/project",
  "session_id": "20250628_123456",
  "guidance": "Implementation instructions...",
  "reason": "Task ready for execution"
}
```

```json
{
  "action": "zen_mcp_codereview",
  "task_id": "1.2",
  "review_scope": "task",
  "session_id": "20250628_123456",
  "reason": "Task complete, triggering code review"
}
```

```json
{
  "action": "complete",
  "task_id": "1.2",
  "session_id": "20250628_123456",
  "summary": "Task completed successfully",
  "reason": "All subtasks finished"
}
```

### 5. Code Review Integration

#### 5.1 Trigger Conditions
Based on `zen_integration.code_review_frequency` in config.yaml:
- `"task"`: Trigger code review after each task completion
- `"none"`: Never trigger automatic code reviews
- `"project"`: Trigger code review only when all tasks complete

#### 5.2 Review Flow
1. Supervisor completes a task
2. Supervisor returns JSON with `action: "zen_mcp_codereview"`
3. Orchestrator detects this action
4. Orchestrator triggers zen MCP code review
5. Review results are logged but don't block task progression

### 6. Expected File Structure

```
project_root/
├── .mcp.json                    # MCP server configuration
├── config.yaml                  # Cadence configuration
├── .taskmaster/
│   ├── config.json             # TaskMaster config
│   ├── tasks/
│   │   ├── tasks.json          # Main task database
│   │   ├── task-1.md           # Individual task files
│   │   └── task-2.md
│   └── docs/
│       └── prd.txt             # Product requirements
└── .cadence/
    ├── supervisor/             # Supervisor artifacts
    ├── agent/                  # Agent artifacts
    └── dispatch/               # Dispatch system artifacts
```

### 7. Success Criteria

The integration is considered complete when:

1. **No Issue Flag**: Running `cadence run` without any --issue flag successfully starts task processing
2. **Task Retrieval**: Supervisor automatically gets next pending task from TaskMaster
3. **Status Updates**: Tasks progress from pending → in-progress → done in TaskMaster
4. **Subtask Tracking**: Individual subtasks are marked complete as agent progresses
5. **Code Review Triggers**: Code reviews trigger based on config settings after task completion
6. **JSON Commands**: Supervisor can issue zen_mcp_codereview commands that orchestrator executes
7. **MCP Access**: Both supervisor and agent can successfully call TaskMaster MCP tools

### 8. Error Handling

- If TaskMaster is not initialized: Clear error message directing user to run `task-master init`
- If no pending tasks: Orchestrator should gracefully complete with success message
- If MCP connection fails: Clear error indicating MCP configuration issue
- If task not found: Log error and continue to next task

### 9. Testing Verification

To verify the integration works correctly:

1. Initialize a test project with TaskMaster
2. Create at least 3 tasks with subtasks
3. Run `cadence run` (no flags)
4. Verify tasks are processed in order
5. Check task status updates in `.taskmaster/tasks/tasks.json`
6. Confirm code reviews trigger based on config
7. Ensure no --issue references remain in codebase

## Implementation Checklist

- [ ] Remove all --issue flag handling from CLI
- [ ] Remove issue-related arguments from orchestrator
- [ ] Remove direct supervisor mode (`cadence` CLI) entirely
- [ ] Update orchestrator to handle "get_next_task" action from supervisor
- [ ] Create supervisor prompt template for getting next task from TaskMaster
- [ ] Modify supervisor prompt to use task context instead of issue
- [ ] Add support for new JSON actions in orchestrator decision handling
- [ ] Add task ID tracking through workflow
- [ ] Implement zen_mcp_codereview action detection
- [ ] Connect code review triggers to task completion
- [ ] Update agent prompts to include task status update instructions
- [ ] Remove all issue-based code paths
- [ ] Test complete workflow end-to-end

## Non-Goals

- No backward compatibility with --issue flag
- No fallback to issue-based workflow
- No support for running without TaskMaster
- No hybrid mode mixing issues and tasks
- No direct supervisor mode - all execution through orchestrator
- No manual TODO mode bypass of TaskMaster
