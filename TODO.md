# Claude Cadence TODO List

## High Priority

### Orchestrator Refactoring
- The orchestrator.py file has grown to 2200+ lines and violates Single Responsibility Principle
- Should be split into smaller, focused modules:
  - Session management
  - Supervisor operations
  - Agent operations
  - State management
  - Dispatch integration
  - File operations

## Medium Priority

### Scratchpad Validation Optimization
- The scratchpad validation does an expensive recursive search
- Could be optimized by tracking scratchpad files in a more efficient way
- Consider caching or indexing approach

## Low Priority

### Memory Management
- Re-implement memory bounds for output collection in orchestrator.py
- Currently removed the 10k line limit for simplicity on modern hardware
- Consider implementing configurable memory management for resource-constrained environments

### General Code Improvements
- Continue improving error handling across the codebase
- Add more comprehensive test coverage
- Document complex workflows
