# Centralized File Logging Implementation Plan

## Overview
Implement a comprehensive file logging system for Claude Cadence that captures all log levels (DEBUG, INFO, WARNING, ERROR) for orchestrator, supervisor, and agent components while maintaining the existing console output.

## Architecture

### 1. Environment Variable Communication
Pass logging configuration to subprocesses via environment variables:
- `CADENCE_LOG_SESSION`: Session ID (e.g., "20250630_103110_e6a0af64")
- `CADENCE_LOG_DIR`: Absolute path to logs directory (e.g., "/path/to/.cadence/logs")

### 2. Directory Structure
```
.cadence/logs/
├── 20250630_103110_e6a0af64/
│   ├── orchestrator.log
│   ├── supervisor.log
│   └── agent.log
├── 20250630_104522_b7c9d8e5/
│   ├── orchestrator.log
│   ├── supervisor.log
│   └── agent.log
```

### 3. Log Format
- **Console**: Keep existing ColoredFormatter with ANSI colors
- **File**: Plain text formatter without colors
  ```
  2025-06-30 10:31:10 | INFO | cadence.orchestrator | Starting orchestration
  2025-06-30 10:31:10 | DEBUG | cadence.orchestrator | Session ID: 20250630_103110_e6a0af64
  ```

## Implementation Tasks

### Task 1: Extend log_utils.py
Add the following function to `cadence/log_utils.py`:

```python
def setup_file_logging(session_id: str, component_type: str, log_dir: Path, level=logging.DEBUG):
    """Attach a plain file handler to the component's logger."""
    session_dir = log_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    log_file = session_dir / f"{component_type}.log"
    logger = logging.getLogger(component_type)

    # Avoid duplicate handlers when called twice
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in logger.handlers):
        return logger

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger
```

### Task 2: Update orchestrator.py
1. At the start of `orchestrate()`, set up file logging:
   ```python
   # Set up file logging for orchestrator
   log_dir = self.project_root / ".cadence" / "logs"
   setup_file_logging(self.current_session_id, "orchestrator", log_dir)
   ```

2. Before spawning supervisor subprocess, set environment variables:
   ```python
   env = os.environ.copy()
   env["CADENCE_LOG_SESSION"] = self.current_session_id
   env["CADENCE_LOG_DIR"] = str(log_dir.resolve())
   ```

### Task 3: Update supervisor entry point
Add logging setup at the beginning of supervisor execution:
```python
import os
import sys
from pathlib import Path
from cadence.log_utils import setup_colored_logging, setup_file_logging

# Get logging config from environment
session_id = os.environ.get("CADENCE_LOG_SESSION")
log_dir = os.environ.get("CADENCE_LOG_DIR")

if session_id and log_dir:
    # Set up file logging
    setup_file_logging(session_id, "supervisor", Path(log_dir))

    # Redirect stdout/stderr to capture all output
    log_path = Path(log_dir) / session_id / "supervisor.log"
    with log_path.open("a", buffering=1) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())
```

### Task 4: Update UnifiedAgent
In `unified_agent.py` `__init__` method:
```python
# Get logging config from environment
session_id = os.environ.get("CADENCE_LOG_SESSION", self.session_id)
log_dir = os.environ.get("CADENCE_LOG_DIR")

if log_dir:
    # Set up file logging
    from cadence.log_utils import setup_file_logging
    setup_file_logging(session_id, "agent", Path(log_dir))

    # Redirect stdout/stderr
    log_path = Path(log_dir) / session_id / "agent.log"
    with log_path.open("a", buffering=1) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())
```

### Task 5: Update retry_utils.py
Pass environment variables when spawning supervisor:
```python
# Get env vars if they exist
env = os.environ.copy()
if "CADENCE_LOG_SESSION" in env and "CADENCE_LOG_DIR" in env:
    # Already set by orchestrator
    pass
else:
    # Fallback for standalone supervisor runs
    env["CADENCE_LOG_SESSION"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    env["CADENCE_LOG_DIR"] = str(Path(".cadence/logs").resolve())
```

## Testing Plan

1. Run a complete orchestration cycle
2. Verify log files are created in correct structure
3. Check that all log levels appear in files
4. Confirm console output remains colored and at INFO level
5. Test subprocess stdout/stderr capture
6. Verify no duplicate log entries

## Future Enhancements (Not in scope)

- Log rotation with `RotatingFileHandler`
- JSON structured logging
- Integration with observability platforms
- Compression of old log files

## Success Criteria

1. All Python logging statements captured to files
2. Subprocess stdout/stderr captured to files
3. Console output unchanged (colored, INFO+)
4. No performance impact
5. Clean session-based organization
6. Works for orchestrator, supervisor, and agent
