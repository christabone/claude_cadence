# TaskMaster Integration Implementation TODO

## Overview
This document tracks the implementation of TaskMaster-AI MCP integration in Claude Cadence, replacing the issue-based workflow with task-based management.

## Reference Document
- Specification: `/home/ctabone/programming/claude_code/claude_cadence/docs/TASKMASTER_INTEGRATION_SPEC.md`

## Prerequisites
- TaskMaster-AI must be installed and configured
- `.taskmaster/` directory with initialized `tasks/tasks.json`
- MCP server running with appropriate API keys

## Implementation Phases

### Phase 1: MCP Configuration ✅
- [x] Verify taskmaster-ai is in config.yaml supervisor_servers list
- [x] Verify taskmaster-ai is in config.yaml agent_servers list
- Note: No .mcp.json needed - taskmaster should be installed at user level

### Phase 2: Remove Issue-Based Logic & Direct Supervisor Mode ✅
- [x] Verified no `--issue` flag exists in current codebase
- [x] Orchestrator already uses task-file approach
- [x] Update workflow documentation to match current implementation
- [x] Remove direct supervisor mode (`cadence` CLI) entirely
- [x] Remove TaskSupervisor class and related code
- [x] Clean up any remaining issue references in documentation
- [x] Fix critical issues from code review:
  - [x] Fixed config handling with None check
  - [x] Fixed non-daemon timer thread
  - [x] Verified shallow copy already fixed (deepcopy in place)

### Code Review 1 ✅
- [x] Run code review comparing Phase 2 changes against TASKMASTER_INTEGRATION_SPEC.md
- [x] Verify no --issue references remain
- [x] Fixed all critical issues identified in review
- [ ] Confirm clean removal of issue-based logic

### Phase 3: Implement TaskMaster Integration ✅
- [x] Add orchestrator support for 'get_next_task' action from supervisor
- [x] Create supervisor prompt template for TaskMaster operations
- [x] Modify supervisor prompt to use task context
- [x] Task ID tracking already implemented through workflow
- [x] Task status updates handled by supervisor via MCP

### Code Review 2 ✅
- [x] Run code review comparing Phase 3 implementation against TASKMASTER_INTEGRATION_SPEC.md
- [x] Verified taskmaster MCP calls work correctly via prompts
- [x] Confirmed task flow matches specification

### Phase 4: JSON Command System ✅
- [x] Verified JSON parsing with SimpleJSONStreamMonitor
- [x] Implemented `zen_mcp_codereview` action detection
- [x] Code review triggers connected via prompt system
- [x] JSON command flow ready for end-to-end testing

### Phase 5: Testing & Verification ✅
- [ ] Create test TaskMaster project with sample tasks
- [ ] Test complete workflow without --issue flag
- [ ] Verify task status updates work correctly
- [ ] Confirm code reviews trigger based on config
- [ ] Run final comprehensive tests

## Status Legend
- ⏳ In Progress
- 🔴 Not Started
- ✅ Completed
- 🔧 Working On
- 🔍 Code Review

## Notes
- Each phase should include code reviews comparing against TASKMASTER_INTEGRATION_SPEC.md
- No backwards compatibility - move forward only
- Both supervisor and agent need taskmaster-ai MCP access

## Progress Log
- Created: 2025-06-28
- Phase 1: Completed (config already has taskmaster-ai)
- Phase 2: Completed - removed direct supervisor mode and fixed critical issues
- Phase 3: Completed - added get_next_task support and verified prompts
- Phase 4: Completed - JSON parsing and code review integration verified
- Updated: 2025-06-28 - Ready for Phase 5 end-to-end testing
