# Minimal Logging Fix for Supervisor and Agent

## Problem Analysis

The supervisor and agent run as `claude` subprocesses, so their output is captured in memory but never saved to log files. The orchestrator captures the output in `all_output` but doesn't write it to disk.

## Solution: Save Captured Output to Log Files

### Fix 1: Save Supervisor Output

In `orchestrator.py`, find where the supervisor output is captured. After the `run_claude_with_json_retry` returns (around line 1090), add:

```python
# Save supervisor output to log file
if hasattr(self, 'all_output') and self.all_output:
    try:
        supervisor_log_file = log_dir / self.current_session_id / "supervisor.log"
        supervisor_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(supervisor_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Supervisor run at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write('\n'.join(self.all_output))
            f.write('\n')
        logger.debug(f"Saved supervisor output to {supervisor_log_file}")
    except Exception as e:
        logger.warning(f"Failed to save supervisor output: {e}")
```

### Fix 2: Save Agent Output

In `unified_agent.py`, after line 369 where `final_output` is created:

```python
# Save agent output to log file
log_dir = os.environ.get("CADENCE_LOG_DIR")
session_id = os.environ.get("CADENCE_LOG_SESSION", self.session_id)
if log_dir and final_output:
    try:
        agent_log_file = Path(log_dir) / session_id / "agent.log"
        agent_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(agent_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Agent run at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write(final_output)
            f.write('\n')
        logger.debug(f"Saved agent output to {agent_log_file}")
    except Exception as e:
        logger.warning(f"Failed to save agent output: {e}")
```

### Fix 3: Capture Output in run_claude_with_realtime_output

The `all_output` needs to be accessible. In `orchestrator.py`, modify the `run_claude_with_realtime_output` method to save output for supervisor:

```python
# At the end of run_claude_with_realtime_output, before returning:
if process_name == "SUPERVISOR":
    # Save supervisor output
    try:
        log_dir = self.project_root / ".cadence" / "logs"
        supervisor_log_file = log_dir / self.current_session_id / "supervisor.log"
        supervisor_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(supervisor_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Supervisor run at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write('\n'.join(all_output))
            f.write('\n')
    except Exception as e:
        logger.warning(f"Failed to save supervisor output: {e}")

return process.returncode, all_output
```

## Implementation Notes

- Use append mode ('a') to accumulate logs across multiple runs
- Add timestamps and separators for clarity
- Only save if output exists
- Handle errors gracefully without breaking the main flow
- The orchestrator already passes the needed environment variables
