#!/usr/bin/env python3
"""
Simple integration test to debug issues
"""

import tempfile
from pathlib import Path
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig
from cadence.workflow_state_machine import WorkflowStateMachine, WorkflowState, WorkflowContext
from cadence.dispatch_logging import setup_dispatch_logging, get_dispatch_logger, OperationType

def test_basic_integration():
    """Test basic integration without pytest framework"""
    print("Starting basic integration test...")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
    print(f"Created temp dir: {temp_dir}")

    try:
        # Setup logging
        setup_dispatch_logging(level="DEBUG")
        print("Logging setup complete")

        # Create dispatch config
        dispatch_config = DispatchConfig(
            max_concurrent_agents=2,
            default_timeout_ms=3000,
            enable_fix_tracking=True,
            enable_escalation=True
        )
        print("Config created")

        # Create dispatcher
        dispatcher = EnhancedAgentDispatcher(config=dispatch_config.to_dict())
        print("Dispatcher created")

        # Create workflow
        workflow = WorkflowStateMachine(
            initial_state=WorkflowState.WORKING,
            context=WorkflowContext(
                task_id="test-task",
                session_id="test-session",
                project_path=str(temp_dir)
            )
        )
        print("Workflow created")

        # Get logger
        dispatch_logger = get_dispatch_logger("test")
        print("Logger created")

        # Test logging context
        with dispatch_logger.operation_context(
            operation_type=OperationType.TRIGGER_DETECTION,
            session_id="test-session",
            task_id="test-task"
        ) as context:
            print(f"Context created with correlation_id: {context.correlation_id}")

            # Test workflow transition
            workflow.transition_to(
                WorkflowState.REVIEW_TRIGGERED,
                "test_trigger",
                metadata={"correlation_id": context.correlation_id}
            )
            print(f"Workflow transitioned to: {workflow.current_state}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if 'dispatcher' in locals():
            dispatcher.cleanup()

        # Remove temp dir
        import shutil
        shutil.rmtree(temp_dir)
        print("Cleanup complete")

if __name__ == "__main__":
    test_basic_integration()
