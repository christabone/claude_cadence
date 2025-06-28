# Claude Cadence TODO Items

## Subscription Limit Handling

### Catch and Parse Claude Usage Limit Messages
- **Issue**: When Claude subscription runs out, agents output "Claude AI usage limit reached|timestamp" but we don't catch this message in our monitoring
- **Current Behavior**: Agent exits with return code 1, but we don't parse the specific error message
- **Needed Enhancement**: Add message parsing to detect subscription limit messages and handle them gracefully
- **Example Error Pattern**: `[AGENT] Claude AI usage limit reached|1750780800`
- **Priority**: Medium
- **Implementation**: Add regex pattern matching in agent output parsing to detect subscription limit messages

### Graceful Degradation on Subscription Limits
- **Issue**: When subscription runs out, the orchestrator continues trying to run supervisor and agents
- **Needed Enhancement**: Detect subscription limit and either:
  - Pause execution until limit resets
  - Gracefully exit with appropriate error message
  - Switch to alternative model if available
- **Priority**: Medium

## General Improvements

### Error Message Standardization
- Standardize error message formats across all components
- Improve error logging and debugging capabilities
- Add structured error codes for different failure types

### Resource Management
- Add memory usage monitoring for large validation tasks
- Implement cleanup procedures for temporary files and processes
- Add disk space checks before starting large operations
