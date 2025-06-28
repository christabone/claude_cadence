# Prose Content Extraction Analysis for prompts.yaml

## Overview
- **File**: `cadence/prompts.yaml`
- **Total Lines**: 931
- **Analysis Date**: 2025-06-26

## Identified Prose Blocks for Extraction

### 1. Serena MCP Activation Instructions
- **Location**: Lines 5-18 (`serena_setup`)
- **Content Type**: Setup instructions
- **Word Count**: ~150 words
- **Characters**: ~650
- **Target File**: `core/setup/serena-activation.md`
- **Usage**: Referenced in core_agent_context

### 2. Supervised Agent Context
- **Location**: Lines 30-41 (`shared_agent_context.supervision_explanation`)
- **Content Type**: Context explanation
- **Word Count**: ~100 words
- **Characters**: ~550
- **Target File**: `core/context/supervised-agent-context.md`
- **Usage**: Referenced via {shared_agent_context.supervision_explanation}

### 3. Work Guidelines
- **Location**: Lines 43-151 (`shared_agent_context.work_guidelines`)
- **Content Type**: Execution guidelines
- **Word Count**: ~900 words
- **Characters**: ~5500
- **Target File**: `core/guidelines/work-guidelines.md`
- **Usage**: Referenced via {shared_agent_context.work_guidelines}
- **Note**: This is the largest prose block, contains 8 numbered sections

### 4. Early Exit Protocol
- **Location**: Lines 151-157 (`shared_agent_context.early_exit_protocol`)
- **Content Type**: Completion protocol
- **Word Count**: ~50 words
- **Characters**: ~350
- **Target File**: `core/guidelines/early-exit-protocol.md`
- **Usage**: Referenced via {shared_agent_context.early_exit_protocol}

### 5. Safety Notice Section
- **Location**: Lines 159-169 (`safety_notice_section`)
- **Content Type**: Safety warning
- **Word Count**: ~100 words
- **Characters**: ~500
- **Target File**: `core/safety/safety-notice.md`
- **Usage**: Referenced via {safety_notice_section}

### 6. Agent Zen Reminder
- **Location**: Lines 171-178 (`agent_zen_reminder`)
- **Content Type**: Implementation guidance
- **Word Count**: ~80 words
- **Characters**: ~400
- **Target File**: `core/guidelines/agent-zen-reminder.md`
- **Usage**: Used in continuation contexts

### 7. Begin Work Instructions (Initial Agent)
- **Location**: Lines 190-198 (inline in agent_prompts.initial)
- **Content Type**: Startup instructions
- **Word Count**: ~70 words
- **Characters**: ~400
- **Target File**: `core/instructions/begin-work.md`
- **Usage**: Part of initial agent prompt

### 8. Continuation Context Template
- **Location**: Lines 208-219 (inline in agent_prompts.continuation)
- **Content Type**: Continuation instructions
- **Word Count**: ~50 words
- **Characters**: ~350
- **Target File**: `core/templates/continuation-context.md`
- **Usage**: Part of continuation agent prompt

## Additional Prose Blocks (Continued analysis)

### 9. TODO List Template
- **Location**: Lines 225-236 (`todo_templates.todo_list`)
- **Content Type**: Dynamic template
- **Word Count**: ~80 words
- **Characters**: ~450
- **Target File**: `core/templates/todo-list.md`
- **Usage**: Referenced via {todo_templates.todo_list}

### 10. Scratchpad Retry Instructions
- **Location**: Lines 241-263 (`todo_templates.scratchpad_retry`)
- **Content Type**: Retry instructions
- **Word Count**: ~120 words
- **Characters**: ~800
- **Target File**: `core/instructions/scratchpad-retry.md`
- **Usage**: Used for agent retry scenarios

### 11. Supervisor Analysis Templates
- **Location**: Lines 276-297 (`todo_templates.supervisor_incomplete_analysis` and `supervisor_complete_analysis`)
- **Content Type**: Analysis templates
- **Word Count**: ~100 words each
- **Characters**: ~600 each
- **Target File**: `core/templates/supervisor-analysis.md`
- **Usage**: Dynamic supervisor feedback

### 12. Issues Section Template
- **Location**: Lines 304-306 (`todo_templates.issues_section`)
- **Content Type**: Issues template
- **Word Count**: ~15 words
- **Characters**: ~80
- **Target File**: `core/templates/issues-section.md`
- **Usage**: For displaying issues to address

### 13. Orchestrator Task Master Prompt (LARGEST BLOCK)
- **Location**: Lines 312-561 (`supervisor_prompts.orchestrator_taskmaster.base_prompt`)
- **Content Type**: Complex supervisor instructions
- **Word Count**: ~2800 words
- **Characters**: ~18000
- **Target File**: `core/supervisor/orchestrator-taskmaster.md`
- **Usage**: Main supervisor prompt
- **Note**: This is the largest single prose block in the file

### 14. Code Review Task Instructions
- **Location**: Lines 564-612 (`supervisor_prompts.orchestrator_taskmaster.code_review_sections.task`)
- **Content Type**: Code review process
- **Word Count**: ~600 words
- **Characters**: ~4000
- **Target File**: `core/guidelines/code-review-task.md`
- **Usage**: Task-level code review guidance

### 15. Code Review Project Instructions
- **Location**: Lines 613-667 (`supervisor_prompts.orchestrator_taskmaster.code_review_sections.project`)
- **Content Type**: Project-level code review process
- **Word Count**: ~700 words
- **Characters**: ~4500
- **Target File**: `core/guidelines/code-review-project.md`
- **Usage**: Project-level code review guidance

### 16. Zen Tool Usage Guidance
- **Location**: Lines 670-719 (`supervisor_prompts.orchestrator_taskmaster.zen_guidance`)
- **Content Type**: Tool usage instructions
- **Word Count**: ~600 words
- **Characters**: ~4000
- **Target File**: `core/guidelines/zen-tool-usage.md`
- **Usage**: Guidance on when and how to use Zen tools

### 17. JSON Output Format Instructions
- **Location**: Lines 727-780 (`supervisor_prompts.orchestrator_taskmaster.output_format`)
- **Content Type**: Technical output requirements
- **Word Count**: ~450 words
- **Characters**: ~3000
- **Target File**: `core/instructions/json-output-format.md`
- **Usage**: Critical output formatting requirements

### 18. Supervisor Analysis Task Instructions
- **Location**: Lines 784-819 (`supervisor_prompts.analysis.sections`)
- **Content Type**: Analysis instructions
- **Word Count**: ~350 words
- **Characters**: ~2200
- **Target File**: `core/supervisor/analysis-instructions.md`
- **Usage**: Supervisor analysis workflow

### 19. Zen Assistance Evaluation Guidelines
- **Location**: Lines 821-849 (`supervisor_prompts.analysis.sections` continued)
- **Content Type**: Decision framework
- **Word Count**: ~300 words
- **Characters**: ~2000
- **Target File**: `core/guidelines/zen-assistance-evaluation.md`
- **Usage**: Framework for determining when to use Zen tools

### 20. Task Supervisor Analysis Prompt
- **Location**: Lines 895-912 (`task_supervisor.analysis_prompt`)
- **Content Type**: Analysis prompt template
- **Word Count**: ~150 words
- **Characters**: ~900
- **Target File**: `core/supervisor/task-analysis-prompt.md`
- **Usage**: Template for task analysis prompts

## Final Analysis Summary

**Total Prose Blocks Identified**: 20
**Estimated Total Characters**: ~52,000
**Estimated Total Words**: ~8,500
**Largest Block**: Orchestrator Task Master Prompt (~18,000 characters)
**Original File Size**: 931 lines

## Extraction Strategy

1. **Phase 1**: Extract standalone blocks (items 1-6, 9-12)
   - These have clear YAML keys and can be directly replaced with _from_file pattern
   - Examples: serena_setup, work_guidelines, safety_notice_section

2. **Phase 2**: Extract supervisor blocks (items 13-20)
   - Large supervisor prompts and templates
   - Require careful handling due to complex nesting and variable usage

3. **Phase 3**: Extract inline blocks (items 7-8)
   - These are embedded in larger structures and need careful extraction
   - Part of agent prompt structures

4. **Directory Structure**:
   ```
   core/
   ├── setup/               # Activation and setup instructions
   ├── context/            # Agent context and explanations
   ├── guidelines/         # Work guidelines and rules
   ├── instructions/       # Specific instructions (begin work, retry, etc.)
   ├── templates/          # Reusable templates
   ├── supervisor/         # Supervisor-specific content
   └── safety/            # Safety notices and warnings
   ```

5. **Naming Convention**:
   - Use kebab-case for filenames
   - Match content purpose in filename
   - Group by functional category in subdirectories

## Variable Preservation

The following variables appear in prose and must be preserved:
- `{project_path}`
- `{session_id}`
- `{previous_session_id}`
- `{max_turns}`
- `{task_numbers}`
- `{continuation_type}`
- `{supervisor_analysis}`
- `{task_status_section}`
- `{remaining_todos}`
- `{next_steps_guidance}`
- `{todo_list}`

## Next Steps

1. Continue analysis for remaining sections of prompts.yaml
2. Create directory structure proposal
3. Plan extraction order to minimize disruption
