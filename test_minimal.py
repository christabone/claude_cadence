#!/usr/bin/env python3
"""
Minimal test to isolate the hang issue
"""

import tempfile
from pathlib import Path

def test_workflow_only():
    """Test just workflow state machine"""
    print("Testing workflow state machine...")

    from cadence.workflow_state_machine import WorkflowStateMachine, WorkflowState, WorkflowContext

    temp_dir = Path(tempfile.mkdtemp(prefix="minimal_test_"))

    try:
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="test-task",
                session_id="test-session",
                project_path=str(temp_dir)
            )
        )
        print(f"Initial state: {workflow.current_state}")

        workflow.transition_to(WorkflowState.REVIEW_TRIGGERED, "test_trigger")
        print(f"New state: {workflow.current_state}")

        print("Workflow test passed!")

    except Exception as e:
        print(f"Workflow error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_logging_only():
    """Test just logging"""
    print("Testing logging...")

    try:
        from cadence.dispatch_logging import setup_dispatch_logging, get_dispatch_logger

        setup_dispatch_logging(level="DEBUG")
        logger = get_dispatch_logger("test")

        logger.logger.info("Test message")
        print("Logging test passed!")

    except Exception as e:
        print(f"Logging error: {e}")
        import traceback
        traceback.print_exc()

def test_dispatcher_only():
    """Test just dispatcher"""
    print("Testing dispatcher...")

    try:
        from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig

        config = DispatchConfig()
        dispatcher = EnhancedAgentDispatcher(config=config.to_dict())
        print("Dispatcher created")

        dispatcher.cleanup()
        print("Dispatcher test passed!")

    except Exception as e:
        print(f"Dispatcher error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_workflow_only()
    test_logging_only()
    test_dispatcher_only()
    print("All minimal tests completed")
