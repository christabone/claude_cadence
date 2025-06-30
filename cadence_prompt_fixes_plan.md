# Claude Cadence Prompt System Fixes Plan

## Background & Findings

### Key Discoveries:
1. **remaining_subtasks**: This field is NOT used anywhere in the code - only appears in agent-output-format.md documentation
2. **Scratchpad Usage**: The orchestrator validates scratchpad exists and supervisor reads it for additional context
3. **JSON Primary, Scratchpad Secondary**: System already prioritizes JSON status but falls back to scratchpad text patterns for backward compatibility
4. **"ALL TASKS COMPLETE" Check**: Line 1799 in orchestrator.py checks for this phrase in output text (not just scratchpad)

### Answers to Questions:
- **Q: How is remaining_subtasks used?** A: It's NOT used anywhere - safe to remove
- **Q: Does supervisor read scratchpads?** A: YES - analysis-context.md explicitly tells supervisor to check scratchpad at lines 5-13
- **Q: How does scratchpad work?** A: Agents create it for tracking, supervisor reads it for context, but JSON is the primary signal

## Complete TODO List

### Phase 1: Clean Up agent-output-format.md

1. **Remove redundant fields from agent-output-format.md**
   - Remove line 82: `help_needed` (redundant with status field)
   - Remove line 81: `remaining_subtasks` (not used anywhere in code)
   - Remove line 87: `blocked_on` (legacy status, only used with old "blocked" status)
   - Remove line 84: `todos_remaining` (supervisor determines next TODOs)
   - Update Field Descriptions section accordingly

2. **Clarify JSON vs Scratchpad relationship**
   - Replace current backward compatibility section (lines 69-76) with:
   ```markdown
   ## Important Notes

   **JSON Output is Primary**: The JSON output is the authoritative signal for the orchestrator.

   **Scratchpad is for Context**: The scratchpad provides additional debugging information and context for the supervisor to review. Always maintain a detailed scratchpad throughout execution.

   **Update Both**: You must:
   1. Maintain your scratchpad with detailed progress notes
   2. End with the JSON status object
   ```

### Phase 2: Remove Backward Compatibility Fallback

3. **Remove text pattern fallback in orchestrator.py**
   - Remove lines 1797-1804 (fallback to text pattern detection)
   - Replace with direct error if no JSON found:
   ```python
   # No valid JSON found
   logger.error("Agent failed to provide required JSON output")
   return {
       "status": "error",
       "help_needed": True,
       "session_id": session_id,
       "error_type": "missing_json_output",
       "error_message": "Agent did not provide required JSON result object"
   }
   ```

### Phase 3: Update Supervisor Instructions

4. **Update analysis-context.md**
   - Change line 9 from checking for "ALL TASKS COMPLETE" to:
   ```markdown
   - Check the agent's JSON status field for completion status
   - Review the scratchpad for additional context and debugging information
   ```
   - Remove lines mentioning specific phrases to look for

### Phase 4: Add Missing Prompts

5. **Add fix_issues prompt to prompts.yml**
   ```yaml
   # Add after line 239 in prompts.yml
   fix_issues:
     sections:
       - "{{ core_agent_context.supervised_context }}"
       - "{{ core_agent_context.safety_notice }}"
       - "{{ core_agent_context.guidelines }}"
       - "{{ core_agent_context.exit_protocol }}"
       - |
         === CODE REVIEW FEEDBACK: ISSUES TO FIX ===
         The following issues were identified in the code review:

         {{ code_review_issues }}

         === YOUR TASK ===
         1. Fix each issue systematically
         2. Test your fixes
         3. Update your scratchpad with what you fixed
         4. Output JSON status when complete
   ```

6. **Add json_fix_retry prompt to prompts.yml**
   ```yaml
   # Add after the fix_issues prompt
   json_fix_retry: |
     URGENT: Your previous response was not valid JSON.

     Your last response ended with:
     {{ invalid_output }}

     Provide ONLY the corrected JSON output following this exact format:
     {
       "status": "success" | "help_needed" | "error",
       "completed_subtasks": ["1.1", "1.2"],
       "session_id": "{{ session_id }}",
       "summary": "Brief summary",
       "execution_notes": "Details"
     }
   ```

### Phase 5: Standardize TODO Terminology

7. **Update work-execution.md**
   - Change all instances of "subtask" to "TODO" for consistency
   - Update sections 7 and 8 to be explicit:
   ```markdown
   7. If stuck → Update scratchpad with details AND output JSON with status: "help_needed"
   8. When complete → Update scratchpad with final notes AND output JSON with status: "success"
   ```

### Phase 6: Enhance Supervisor Scratchpad Usage

8. **Ensure supervisor reads scratchpad for context**
   - The functionality already exists in analysis-context.md
   - No changes needed - supervisor is already instructed to check scratchpad

## Implementation Order

1. Phase 1: Clean up agent-output-format.md (removes confusion)
2. Phase 2: Remove backward compatibility fallback (forces JSON usage)
3. Phase 3: Update supervisor instructions (align with JSON-first approach)
4. Phase 4: Add missing prompts (enables new capabilities)
5. Phase 5: Standardize terminology (improves clarity)
6. Phase 6: Verify supervisor functionality (already working correctly)

## Testing Strategy

After implementation:
1. Test agent completes successfully with JSON output
2. Test agent requests help with JSON output
3. Test agent error handling with JSON output
4. Test supervisor correctly reads both JSON and scratchpad
5. Test fix_issues prompt works correctly
6. Test json_fix_retry prompt recovers from invalid JSON

## Risk Mitigation

- Keep detailed scratchpad generation (valuable for debugging)
- Ensure clear error messages when JSON is missing
- Test thoroughly before removing backward compatibility
- Document the JSON-first approach clearly
