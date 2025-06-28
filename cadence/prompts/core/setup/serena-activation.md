# Serena MCP Activation (Critical)

**BEFORE any code analysis or file operations, you MUST activate Serena MCP:**

## Step 1: Activate Serena project
- Run: `mcp__serena__activate_project --project={{ project_path }}`
- If this fails, try: `mcp__serena__activate_project --project=.`
- If still fails, proceed without Serena but note this in your output

## Step 2: Get initial instructions (optional but recommended)
- Run: `mcp__serena__initial_instructions`
- This provides project-specific context and guidelines

## Why This Matters
Serena provides 10x faster semantic code analysis than basic file tools.
