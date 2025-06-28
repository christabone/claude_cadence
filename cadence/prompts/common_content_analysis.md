# Common Content Analysis for prompts.yaml

## Analysis Date: 2025-01-26

## Overview
This document analyzes the 932-line prompts.yaml file to identify common content patterns, repeated structures, and shared elements that can be extracted into YAML anchors in _common.yaml.

## 1. Frequently Repeated Content Patterns

### 1.1 Project Path References
- **Pattern**: `{project_path}` appears 45+ times
- **Usage**: File paths, scratchpad locations, task master calls
- **Examples**:
  ```yaml
  - {project_path}/.cadence/scratchpad/
  - --projectRoot={project_path}
  - {project_path}/.cadence/agent/output_{session_id}.log
  ```

### 1.2 Session ID References
- **Pattern**: `{session_id}` appears 30+ times
- **Usage**: File naming, tracking, identification
- **Examples**:
  ```yaml
  - session_{session_id}.md
  - decision_snapshot_{session_id}.json
  - Session ID: {session_id}
  ```

### 1.3 MCP Tool References
- **Pattern**: Tool names repeated across guidance sections
- **Frequency**:
  - Serena MCP: 15+ references
  - Task Master MCP: 20+ references
  - Context7 MCP: 8+ references
  - Zen MCP: 25+ references

### 1.4 Safety and Warning Messages
- **Pattern**: Safety notices about --dangerously-skip-permissions
- **Occurrences**: 3 times (lines 159-169, embedded in guidelines)
- **Content**: Identical warning text about dangerous operations

### 1.5 Scratchpad Templates
- **Pattern**: Scratchpad file structure and content
- **Occurrences**: 5+ times with variations
- **Base template at lines 49-71, retry at 248-261

### 1.6 Code Review Instructions
- **Pattern**: Review workflow with multiple models
- **Occurrences**: 3 times (task review, project review)
- **Lines**: 565-612 (task), 614-667 (project)

## 2. Structural Patterns

### 2.1 TODO List Formatting
- **Pattern**: Numbered list with specific formatting
- **Template**: `{number}. {todo_text}`
- **Used in**: Initial prompts, continuation prompts, supervisor analysis

### 2.2 Section Headers
- **Pattern**: `=== SECTION_NAME ===` format
- **Count**: 30+ different section headers
- **Examples**:
  - `=== SUPERVISED AGENT CONTEXT ===`
  - `=== YOUR TODOS ===`
  - `=== COMPLETION PROTOCOL ===`

### 2.3 Status Messages
- **Pattern**: Consistent status reporting format
- **Templates**:
  - `Status: COMPLETE ✅`
  - `Status: IN_PROGRESS`
  - `Status: STUCK`

### 2.4 File Path Patterns
- **Pattern**: Consistent directory structure references
- **Common paths**:
  - `.cadence/scratchpad/`
  - `.cadence/agent/`
  - `.cadence/supervisor/`
  - `.cadence/logs/`

## 3. Shared Variables and Placeholders

### 3.1 Core Variables (Used Everywhere)
- `{max_turns}` - Turn limit
- `{project_path}` - Project root directory
- `{session_id}` - Current session identifier
- `{task_numbers}` - Task Master task references

### 3.2 Conditional Variables
- `{previous_session_id}` - For continuations
- `{agent_task_id}` - Current task being worked on
- `{todo_list}` - Dynamically generated TODO list
- `{analysis_guidance}` - Supervisor guidance

### 3.3 Status Variables
- `{completed_count}` - Number of completed tasks
- `{remaining_count}` - Number of remaining tasks
- `{turns_used}` - Turns consumed so far

## 4. Common Templates and Blocks

### 4.1 Serena Activation Block (lines 5-18)
- Used in: Multiple supervisor and agent contexts
- Purpose: Standard Serena MCP activation instructions
- Frequency: Referenced 3+ times

### 4.2 Safety Notice Block (lines 159-169)
- Used in: All agent prompts
- Purpose: Warn about dangerous permissions
- Frequency: Embedded in core_agent_context

### 4.3 Work Guidelines Block (lines 44-157)
- Used in: All agent prompts
- Purpose: Standard execution guidelines
- Frequency: Core part of every agent interaction

### 4.4 Completion Protocol Block (lines 151-157)
- Used in: All agent prompts
- Purpose: Standard completion instructions
- Frequency: Critical for every agent run

### 4.5 Code Review Templates (lines 565-667)
- Used in: Supervisor analysis
- Purpose: Standardized review process
- Variations: Task-level and project-level

## 5. Candidate Anchors for _common.yaml

### 5.1 High-Priority Anchors (Most Reused)
1. **&base_paths**: Common directory paths
2. **&serena_activation**: Serena MCP activation instructions
3. **&safety_notice**: Dangerous permissions warning
4. **&completion_protocol**: Standard completion instructions
5. **&scratchpad_template**: Base scratchpad structure
6. **&code_review_task**: Task-level review process
7. **&code_review_project**: Project-level review process

### 5.2 Variable Anchors
1. **&var_project_context**: project_path, session_id, task_numbers
2. **&var_status_tracking**: completed_count, remaining_count, turns_used
3. **&var_file_paths**: Common file path patterns

### 5.3 Template Anchors
1. **&tpl_section_header**: Section header formatting
2. **&tpl_todo_item**: TODO list item format
3. **&tpl_status_message**: Status message formatting
4. **&tpl_help_request**: Help request template

## 6. Duplication Statistics

### Most Duplicated Content:
1. Serena activation instructions: 3 full copies
2. Code review with model specifications: 6 occurrences (o3, gemini-2.5-pro patterns)
3. Scratchpad creation instructions: 5 variations
4. Safety warnings: 3 identical copies
5. Project path patterns: 45+ uses

### Potential Space Savings:
- Extracting top 10 patterns could reduce file by ~200 lines (20%+)
- Better organization and maintainability
- Single source of truth for critical instructions

## 7. Dependencies and References

### Cross-Reference Patterns:
- `{shared_agent_context.supervision_explanation}` → supervision_explanation
- `{serena_setup}` → serena_setup block
- `{safety_notice_section}` → safety_notice_section
- `{core_agent_context.*}` → Various core context elements

### Include Chain:
1. agent_prompts.initial.sections includes:
   - core_agent_context (which includes 5 sub-elements)
   - todo_list (dynamically generated)
   - Additional initial-specific content

2. agent_prompts.continuation.sections includes:
   - Same core_agent_context as initial
   - Continuation-specific sections

## 8. Recommendations for _common.yaml Structure

```yaml
# _common.yaml structure recommendation
system:
  paths: &base_paths
  serena: &serena_activation
  safety: &safety_notice

variables:
  project: &var_project
  status: &var_status

templates:
  formatting: &tpl_formatting
  messages: &tpl_messages

workflows:
  code_review: &workflow_code_review
  completion: &workflow_completion
```

## 9. Migration Impact

### Files Requiring Updates:
- All prompts that reference serena_setup
- All prompts that include safety notices
- All code review sections
- All scratchpad templates

### Risk Areas:
- Circular dependencies between shared blocks
- Variable resolution order
- Maintaining backward compatibility during migration
