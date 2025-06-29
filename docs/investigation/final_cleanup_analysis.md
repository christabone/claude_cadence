# Final Cleanup Analysis - Claude Cadence Repository

## Files in Root Directory That Need Organization

### 1. Scripts That Should Move to `scripts/` Directory
- **`run_tests.sh`** - Test runner script, belongs in scripts/
- **`cadence.py`** - CLI entry point, could stay in root or move to scripts/
- **`orchestrate.py`** - Orchestration script, could stay in root or move to scripts/

### 2. Documentation That Should Move to `docs/`
- **`execution_context_refactor_plan.md`** - Refactoring documentation, should be in docs/design/ or docs/refactoring/
- **`TODO.md`** - Could move to docs/development/ to consolidate with existing TODO.md there
- **`CHANGELOG.md`** - Standard to keep in root, but could move to docs/

### 3. Potentially Outdated or Duplicate Directories

#### `analysis/` Directory Structure
The `analysis/` directory contains:
- **`alliance/agr_mcp/`** - Complete copy of agr_mcp project (duplicate of root agr_mcp/)
- **`alliance/kanban_756_mcp/`** - Another MCP project
- **`work-in-progress/`** - Contains test files that should be reviewed

This appears to be a workspace for analyzing other projects and should probably be:
- Moved to a separate analysis workspace outside the main project
- Or cleaned up if no longer needed

#### `agr_mcp/` Directory
- Contains a complete MCP server implementation
- Has its own test files in root: `test_http_client.py`, `test_logging.py`
- Question: Is this actively used by Claude Cadence or should it be a separate project?

### 4. Documentation Consolidation

#### Duplicate/Related Documentation
1. **TODO Files**:
   - `/TODO.md` (root)
   - `/docs/development/TODO.md`
   - Should be consolidated into one location

2. **Investigation/Analysis Docs**:
   - `/docs/analysis/` - Contains json_stream_handling_analysis.md
   - `/docs/investigation/` - Contains code_review_findings.md, dispatcher_analysis.md
   - Should be consolidated into one directory

3. **Design/Implementation Docs**:
   - `/docs/design/` - JSON stream monitor designs
   - `/docs/implementation/` - Implementation summaries
   - `/execution_context_refactor_plan.md` - Should move here

## Recommended Actions

### 1. Move Scripts
```bash
mkdir -p scripts
mv run_tests.sh scripts/
# Keep cadence.py and orchestrate.py in root as entry points
```

### 2. Organize Documentation
```bash
# Move refactor plan to design docs
mv execution_context_refactor_plan.md docs/design/

# Consolidate TODOs
cat TODO.md >> docs/development/TODO.md
rm TODO.md

# Consolidate investigation docs
mv docs/analysis/*.md docs/investigation/
rmdir docs/analysis
```

### 3. Handle the `analysis/` Directory
This directory appears to be a workspace for analyzing other projects. Options:
1. **Move it outside the repository** if it's a personal workspace
2. **Clean it up** if the analysis is complete
3. **Document its purpose** if it's needed for the project

### 4. Review `agr_mcp/` Integration
- If it's a dependency, it should be properly configured as a git submodule or package
- If it's not used, it should be removed
- The test files in its root should move to its tests/ directory

### 5. Small Improvements
- Consider adding a `.gitignore` entry for `venv/` directories
- Review if all test files in `tests/` are still needed
- Check if example files in `examples/` are up to date

## Summary
The main cleanup tasks are:
1. Moving scripts to `scripts/` directory
2. Consolidating documentation in `docs/`
3. Deciding what to do with the `analysis/` workspace
4. Clarifying the role of `agr_mcp/` in the project
5. Removing duplicate files and consolidating related content

This will result in a cleaner, more organized repository structure that's easier to navigate and maintain.
