# TaskMaster Integration TODO Backup
Saved: 2025-06-28

## Current TODO Status

### Completed ✅
- Phase 2: Verify no --issue flag exists in codebase
- Phase 2: Update workflow documentation to match implementation
- Code Review 1: Verify Phase 2 changes against TASKMASTER_INTEGRATION_SPEC.md

### In Progress 🔧
- Phase 2: Remove direct supervisor mode (`cadence` CLI) entirely

### Pending 🔴
- Phase 2: Remove TaskSupervisor class and related code
- Phase 2: Clean up any remaining issue references in documentation
- Phase 3: Add orchestrator support for 'get_next_task' action from supervisor
- Phase 3: Create supervisor prompt template for TaskMaster operations
- Phase 3: Modify supervisor prompt to use task context
- Code Review 2: Verify Phase 3 implementation against TASKMASTER_INTEGRATION_SPEC.md
- Phase 4: Verify JSON command parsing for zen_mcp_codereview
- Phase 4: Connect code review triggers to task completion
- Phase 5: Test complete workflow end-to-end

## Critical Fixes to Address

### From Code Review 1
1. **Direct Supervisor Mode Still Exists** - Remove cadence.py CLI entirely
2. **Manual TODO Mode Bypass** - Remove --todo flag from cadence.py
3. **Config Handling Issues** - Fix dataclass corruption in config.py:368-372
4. **Security Issues** - Fix shallow copy in agent_messages.py:346
5. **Threading Issues** - Fix non-daemon timer in fix_agent_dispatcher.py:545-547
