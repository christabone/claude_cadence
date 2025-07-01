# Work Execution Guidelines

## 0. Leverage Sub-Agents for Complex Tasks

### When to Use the Task Tool for Sub-Agents:
- **File Analysis**: When you need to analyze multiple files, spin off sub-agents to read and analyze them in parallel
- **Search Operations**: For complex searches across the codebase, use multiple sub-agents with different search patterns
- **Documentation Review**: When checking multiple documentation sources, parallelize with sub-agents
- **Code Understanding**: For understanding different modules or components, assign each to a sub-agent
- **Multiple Operations**: Any time you have 3+ similar operations, consider parallelizing with sub-agents

### Example Sub-Agent Usage:
```
# Instead of sequentially reading 5 files:
Task 1: "Read and summarize the authentication logic in auth.py"
Task 2: "Read and analyze the user model in models/user.py"
Task 3: "Check the authentication middleware in middleware/auth.py"
# Launch all three concurrently for faster results
```

### Benefits:
- ⚡ Dramatically faster execution through parallelization
- 🎯 More focused analysis from specialized sub-agents
- 📊 Better coverage of complex codebases
- 🔍 Reduced chance of missing important details

## 1. First Action: Create Your Scratchpad File

- Create directory: `{{ project_path }}/.cadence/scratchpad/`
- Create file: `{{ project_path }}/.cadence/scratchpad/session_{{ session_id }}.md`
- Initial content:
  ```markdown
  # Task Execution Scratchpad
  Session ID: {{ session_id }}
  Task Master Tasks: {{ task_numbers }}
  Started: [timestamp]
  Status: IN_PROGRESS

  ## TODOs Overview
  [List all TODOs here at start]

  ## Progress Log
  [Update after EACH TODO]

  ## Issues/Blockers
  [Note any problems immediately]

  ## Help Requests
  [Document if you need assistance]

  ## Completion Summary
  [Fill when all TODOs done]
  ```

## 2. Code Navigation: ALWAYS Use Serena MCP Tools First

They are faster and more accurate.

### MANDATORY: Try Serena Tools BEFORE Any File Reading or Grep Operations

🚀 **FIRST**: Use `mcp__serena__get_symbols_overview` to understand project structure
🚀 **FIRST**: Use `mcp__serena__find_symbol` to locate specific functions/classes/methods
🚀 **FIRST**: Use `mcp__serena__find_referencing_symbols` to trace dependencies
🚀 **FIRST**: Use `mcp__serena__replace_symbol_body` for precise function updates
🚀 **FIRST**: Use `mcp__serena__search_for_pattern` for regex searches in code

### ⛔ AVOID These Slow Alternatives Unless Serena Fails

❌ Reading entire files with Read tool when you need specific symbols
❌ Using grep/Grep tools when Serena can find symbols semantically
❌ Using Glob to find files when you're looking for code symbols
❌ Using Edit tool for large function changes (use replace_symbol_body instead)

### 🎯 Serena Benefits
10x faster, semantic understanding, precise symbol-level edits

## 3. Library Documentation: Use Context7 MCP Tools

When you need library/framework documentation:

✅ Use `mcp__Context7__resolve-library-id` to find the correct library
✅ Use `mcp__Context7__get-library-docs` to get up-to-date documentation
❌ Avoid guessing API usage - always check documentation first
❌ Don't rely on potentially outdated knowledge - use Context7

Context7 provides current, accurate documentation for thousands of libraries.

## 4. Focus: Complete ONLY the Assigned TODOs

Avoid scope creep.

### Minor Improvements Are OK If They Directly Support the TODO

✅ Adding docstrings to functions you create
✅ Basic error handling for code you write
✅ Helpful comments explaining complex logic
❌ Creating additional features not requested
❌ Refactoring existing code beyond the TODO scope
❌ Adding new dependencies or frameworks

## 5. Safety: You Have --dangerously-skip-permissions Enabled

- **NEVER** perform destructive operations without explicit TODO instruction
- **NEVER** delete repositories, drop databases, or remove critical files

### DANGEROUS COMMANDS Requiring Extra Caution

* `rm -rf` (especially with wildcards or root paths)
* `git push --force` or `git reset --hard`
* `DROP TABLE`, `DROP DATABASE`, `TRUNCATE`
* Any command with `sudo` or affecting system files

When in doubt, note the risky operation in your scratchpad and proceed cautiously.

## 6. Progress: Update Your Scratchpad IMMEDIATELY After EACH TODO

- As soon as you complete a TODO, update the scratchpad
- This ensures progress is saved even if execution stops
- Log each completion in your scratchpad:

```markdown
## TODO #1: [original TODO text]
Status: COMPLETE ✅
Summary: [brief description of what was done]
Notes: [any issues or deviations]
```

## 7. Requesting Help: If You Encounter Situations Where You're Genuinely Stuck

Update your scratchpad with:

```markdown
## HELP NEEDED
Status: STUCK
Issue: [Clear description of the problem]
Attempted: [What you've tried so far]
Context: [Relevant files/errors]
Recommendation: [What kind of help would be most useful]
```

Then output JSON with status: "help_needed" to signal the supervisor

### You Can Also Request Specific Reviews

- "ARCHITECTURE_REVIEW_NEEDED" - For design decisions
- "SECURITY_REVIEW_NEEDED" - For security concerns
- "PERFORMANCE_REVIEW_NEEDED" - For optimization questions

## 8. Completion: When ALL TODOs Are Done

- Update the "Completion Summary" section in your scratchpad
- Include: what was completed (✅), issues encountered (⚠️), follow-up suggestions (💡)
- Output JSON with status: "success" to signal completion
