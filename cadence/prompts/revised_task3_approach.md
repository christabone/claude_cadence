# Revised Approach for Task 3: YAML Anchor Strategy

## Date: 2025-01-26

## POC Findings Summary

The early proof-of-concept validation revealed a critical limitation with our custom YAML loader:
- **YAML anchors defined in included files are NOT accessible in the including file**
- This is due to parsing order: anchors are resolved at parse time, but !include processing happens after parsing
- The custom loader returns parsed content, not raw YAML with anchors preserved

## Original vs Revised Approach

### Original Plan (Not Feasible)
- Create a centralized `_common.yaml` with shared anchors
- Include this file in other YAML files to reference the anchors
- Use cross-file anchor references to eliminate duplication

### Revised Plan (Aligned with PRD)
- Use !include directive ONLY for long prose content (markdown files)
- Keep YAML anchors within individual YAML files for local reuse
- Focus on patterns for reducing duplication within single files

## Revised Task 3 Strategy

### 1. Document Within-File Anchor Patterns
Instead of cross-file anchors, document best practices for using anchors within individual YAML files:

```yaml
# Example: agents/initial.yaml
# Define local anchors at the top
local_anchors:
  paths: &local_paths
    scratchpad: "{project_path}/.cadence/scratchpad/session_{session_id}.md"
    agent_log: "{project_path}/.cadence/agent/output_{session_id}.log"

  variables: &local_vars
    max_turns: "{max_turns}"
    session_id: "{session_id}"

# Use anchors within the same file
agent_prompts:
  initial:
    context:
      <<: *local_vars
      paths: *local_paths
    sections:
      # Include markdown prose
      - !include ../core/supervised_context.md
      - !include ../core/guidelines.md
```

### 2. Focus on Prose Extraction
The main duplication reduction will come from extracting long prose to markdown files:
- Serena activation instructions → `core/serena_setup.md`
- Safety notices → `core/safety_notice.md`
- Work guidelines → `core/guidelines.md`
- Completion protocols → `core/completion_protocol.md`

### 3. Create Pattern Documentation
Document patterns for organizing YAML files to minimize duplication:
- Group related configuration at the top with anchors
- Use merge keys for extending configurations
- Keep anchor definitions close to their usage

## Benefits of Revised Approach

1. **Simpler Implementation**: No complex cross-file anchor resolution needed
2. **Follows PRD Intent**: Focuses on extracting prose to markdown files
3. **Maintains Loader Simplicity**: The ~20-line custom loader remains simple
4. **Clear Separation**: YAML for structure, Markdown for content
5. **No Magic**: Developers can understand file relationships easily

## Updated Subtask Actions

### Subtask 3.2: Create dependency graph
- **Action**: Skip or repurpose to document within-file anchor patterns

### Subtask 3.3: Design naming conventions
- **Action**: Focus on conventions for local anchors within files

### Subtask 3.4: Create _common.yaml
- **Action**: Convert to creating example patterns documentation

### Subtask 3.6: Document integration guidelines
- **Action**: Focus on !include usage for markdown files

### Subtask 3.7: Migration strategy
- **Action**: Focus on prose extraction strategy

## Next Steps

1. Update Task 3 description to reflect the revised approach
2. Focus efforts on Task 4 (Extract Prose to Markdown Files)
3. Document best practices for within-file anchor usage
4. Create examples showing the split between YAML structure and markdown content

## Conclusion

The POC validation was valuable in discovering this limitation early. The revised approach is actually simpler and more aligned with the original PRD's intent of solving the "giant file" problem through prose extraction rather than complex anchor systems.
