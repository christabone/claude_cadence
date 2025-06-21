# Changelog

All notable changes to Claude Cadence will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced prompt system with comprehensive agent context
- Dynamic continuation prompts that adapt based on execution status
- Scratchpad system for progress tracking (`.cadence/scratchpad/session_*.md`)
- Session ID generation and tracking for execution continuity
- Safety-first prompt design with full context preservation in every continuation
- Supervisor analysis integration in continuation prompts
- Task Master task number tracking in prompts
- YAML-based prompt configuration with template inheritance
- Constants for system-wide strings (COMPLETION_PHRASE, SCRATCHPAD_DIR)
- **Zen MCP Integration**: Intelligent assistance for stuck agents
  - Automatic detection of stuck/blocked agents via "HELP NEEDED" protocol
  - Error pattern detection with configurable thresholds
  - Task validation for critical operations (security, database, etc.)
  - Retrospective analysis for high turn usage (>80% configurable)
  - Support for multiple zen tools: debug, review, consensus, precommit, analyze
- **MCP Server Checker**: Preprocess script to verify MCP installations
  - Checks Claude CLI availability
  - Validates required MCP servers (zen, taskmaster-ai)
  - Provides installation guidance for missing servers

### Changed
- Refactored prompt generation to use YAML templates
- Improved supervisor-agent communication with structured prompts
- Enhanced TODO presentation with clear formatting and instructions
- Updated README with prompt system documentation

### Security
- Explicit safety warnings about `--dangerously-skip-permissions` in all prompts
- Full context repetition in continuations to prevent agent drift
- Clear exit protocols to prevent runaway execution

## [0.1.0] - TBD (Initial Release)
- Initial implementation of task-driven supervision system
- Task Master integration for task management
- Basic supervisor-agent execution pattern