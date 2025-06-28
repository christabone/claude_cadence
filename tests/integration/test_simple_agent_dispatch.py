"""
Simplified Agent Dispatch Integration Tests

Simple integration tests that focus on core agent dispatch functionality
without the complexity of workflow state machines or complex orchestration.

These tests verify:
- Basic agent message creation and dispatch
- Agent type routing
- Callback mechanisms
- Error handling in dispatch
- Message context handling
"""

import pytest
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List

# Core components for basic dispatch testing
from cadence.agent_messages import (
    AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo
)
from cadence.enhanced_agent_dispatcher import EnhancedAgentDispatcher, DispatchConfig


@dataclass
class SimpleTestResult:
    """Simple test result for tracking dispatch outcomes"""
    success: bool
    agent_type: AgentType
    message_id: str
    callback_executed: bool
    result_data: Dict[str, Any]
    error_message: str = ""


class MockSimpleAgent:
    """Simple mock agent for testing basic dispatch functionality"""

    def __init__(self, agent_type: AgentType, should_succeed: bool = True, delay: float = 0.1):
        self.agent_type = agent_type
        self.should_succeed = should_succeed
        self.delay = delay
        self.calls_received = []

    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process a simple agent message"""
        self.calls_received.append(message)

        # Simulate processing time
        time.sleep(self.delay)

        if not self.should_succeed:
            return {
                "success": False,
                "error": f"Mock {self.agent_type.value} agent failed intentionally",
                "agent_type": self.agent_type.value
            }

        # Generate success response based on agent type
        return self._generate_success_response(message)

    def _generate_success_response(self, message: AgentMessage) -> Dict[str, Any]:
        """Generate agent-specific success response"""
        base_response = {
            "success": True,
            "agent_type": self.agent_type.value,
            "message_id": getattr(message, 'message_id', 'test_msg'),
            "processed_files": message.context.files_modified
        }

        if self.agent_type == AgentType.REVIEW:
            base_response.update({
                "issues_found": [
                    {"file": "test.py", "line": 10, "type": "security", "severity": "medium"},
                    {"file": "utils.py", "line": 25, "type": "style", "severity": "low"}
                ],
                "review_score": 8.5,
                "recommendations": ["Add input validation", "Improve variable names"]
            })
        elif self.agent_type == AgentType.FIX:
            base_response.update({
                "fixes_applied": [
                    {"file": "test.py", "line": 10, "change": "Added input validation"},
                    {"file": "utils.py", "line": 25, "change": "Improved variable naming"}
                ],
                "tests_run": True,
                "verification_needed": True
            })
        elif self.agent_type == AgentType.VERIFY:
            base_response.update({
                "verification_result": "passed",
                "tests_passed": 15,
                "tests_failed": 0,
                "confidence_score": 9.2
            })

        return base_response


class TestSimpleAgentDispatch:
    """Test basic agent dispatch functionality without workflow complexity"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="simple_dispatch_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def basic_config(self):
        """Basic dispatch configuration for testing"""
        return DispatchConfig(
            max_concurrent_agents=2,
            default_timeout_ms=5000,
            enable_fix_tracking=False,  # Simplified - no fix tracking
            enable_escalation=False,    # Simplified - no escalation
            persistence_type="memory"   # Use memory for simplicity
        )

    @pytest.fixture
    def mock_agents(self):
        """Create simple mock agents"""
        return {
            AgentType.REVIEW: MockSimpleAgent(AgentType.REVIEW, should_succeed=True),
            AgentType.FIX: MockSimpleAgent(AgentType.FIX, should_succeed=True),
            AgentType.VERIFY: MockSimpleAgent(AgentType.VERIFY, should_succeed=True),
        }

    def test_basic_agent_message_creation(self, temp_dir):
        """Test basic agent message creation and validation"""

        # Create basic message context
        context = MessageContext(
            task_id="basic_test_1",
            parent_session="session_basic_1",
            files_modified=["test.py", "utils.py"],
            project_path=str(temp_dir)
        )

        # Create success criteria
        criteria = SuccessCriteria(
            expected_outcomes=["Code reviewed"],
            validation_steps=["Static analysis"]
        )

        # Create callback info
        callback = CallbackInfo(handler="test_callback")

        # Create agent message
        message = AgentMessage(
            message_type=MessageType.DISPATCH_AGENT,
            agent_type=AgentType.REVIEW,
            context=context,
            success_criteria=criteria,
            callback=callback
        )

        # Verify message properties
        assert message.message_type == MessageType.DISPATCH_AGENT
        assert message.agent_type == AgentType.REVIEW
        assert message.context.task_id == "basic_test_1"
        assert message.context.project_path == str(temp_dir)
        assert len(message.context.files_modified) == 2
        assert message.success_criteria.expected_outcomes == ["Code reviewed"]
        assert message.callback.handler == "test_callback"

    def test_simple_agent_dispatch_success(self, temp_dir, basic_config, mock_agents):
        """Test successful agent dispatch without workflow complexity"""

        dispatcher = EnhancedAgentDispatcher(config=basic_config.to_dict())
        mock_agent = mock_agents[AgentType.REVIEW]

        # Track callback execution
        callback_results = []

        def test_callback(result: Dict[str, Any]):
            callback_results.append(result)

        try:
            # Create simple message
            message = AgentMessage(
                message_type=MessageType.DISPATCH_AGENT,
                agent_type=AgentType.REVIEW,
                context=MessageContext(
                    task_id="simple_success_test",
                    parent_session="session_success",
                    files_modified=["test.py"],
                    project_path=str(temp_dir)
                ),
                success_criteria=SuccessCriteria(expected_outcomes=["Review completed"]),
                callback=CallbackInfo(handler="test_callback")
            )

            # Simulate agent processing
            agent_result = mock_agent.process_message(message)

            # Verify agent processing
            assert agent_result["success"] is True
            assert agent_result["agent_type"] == "review"
            assert "issues_found" in agent_result
            assert len(agent_result["issues_found"]) == 2

            # Verify agent received the message
            assert len(mock_agent.calls_received) == 1
            received_message = mock_agent.calls_received[0]
            assert received_message.context.task_id == "simple_success_test"
            assert received_message.agent_type == AgentType.REVIEW

        finally:
            dispatcher.cleanup()

    def test_multiple_agent_types_dispatch(self, temp_dir, basic_config, mock_agents):
        """Test dispatching to different agent types"""

        dispatcher = EnhancedAgentDispatcher(config=basic_config.to_dict())
        results = {}

        try:
            # Test each agent type
            for agent_type in [AgentType.REVIEW, AgentType.FIX]:
                mock_agent = mock_agents[agent_type]

                message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=agent_type,
                    context=MessageContext(
                        task_id=f"multi_test_{agent_type.value}",
                        parent_session="session_multi",
                        files_modified=["test.py", "utils.py"],
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria(expected_outcomes=[f"{agent_type.value} completed"]),
                    callback=CallbackInfo(handler="test_callback")
                )

                # Process with mock agent
                result = mock_agent.process_message(message)
                results[agent_type.value] = result

            # Verify each agent type responded correctly
            assert results["review"]["success"] is True
            assert "issues_found" in results["review"]
            assert results["review"]["agent_type"] == "review"

            assert results["fix"]["success"] is True
            assert "fixes_applied" in results["fix"]
            assert results["fix"]["agent_type"] == "fix"

            # Verify agents received their respective messages
            for agent_type in [AgentType.REVIEW, AgentType.FIX]:
                agent = mock_agents[agent_type]
                assert len(agent.calls_received) == 1
                assert agent.calls_received[0].agent_type == agent_type

        finally:
            dispatcher.cleanup()

    def test_agent_error_handling(self, temp_dir, basic_config):
        """Test error handling in agent dispatch"""

        dispatcher = EnhancedAgentDispatcher(config=basic_config.to_dict())

        # Create agent that will fail
        failing_agent = MockSimpleAgent(AgentType.REVIEW, should_succeed=False)

        try:
            message = AgentMessage(
                message_type=MessageType.DISPATCH_AGENT,
                agent_type=AgentType.REVIEW,
                context=MessageContext(
                    task_id="error_test",
                    parent_session="session_error",
                    files_modified=["test.py"],
                    project_path=str(temp_dir)
                ),
                success_criteria=SuccessCriteria(expected_outcomes=["Review completed"]),
                callback=CallbackInfo(handler="test_callback")
            )

            # Process with failing agent
            result = failing_agent.process_message(message)

            # Verify error response
            assert result["success"] is False
            assert "error" in result
            assert result["agent_type"] == "review"
            assert "failed intentionally" in result["error"]

            # Verify agent still received the message
            assert len(failing_agent.calls_received) == 1

        finally:
            dispatcher.cleanup()

    def test_concurrent_simple_dispatch(self, temp_dir, basic_config, mock_agents):
        """Test concurrent dispatch of multiple simple agents"""

        dispatcher = EnhancedAgentDispatcher(config=basic_config.to_dict())

        # Use agents with different delays to test concurrency
        review_agent = MockSimpleAgent(AgentType.REVIEW, delay=0.1)
        fix_agent = MockSimpleAgent(AgentType.FIX, delay=0.15)

        agents = [review_agent, fix_agent]

        try:
            import threading
            import time

            start_time = time.time()
            threads = []
            results = {}

            def process_agent(agent, task_suffix):
                message = AgentMessage(
                    message_type=MessageType.DISPATCH_AGENT,
                    agent_type=agent.agent_type,
                    context=MessageContext(
                        task_id=f"concurrent_test_{task_suffix}",
                        parent_session="session_concurrent",
                        files_modified=["test.py"],
                        project_path=str(temp_dir)
                    ),
                    success_criteria=SuccessCriteria(expected_outcomes=["Task completed"]),
                    callback=CallbackInfo(handler="test_callback")
                )

                result = agent.process_message(message)
                results[agent.agent_type.value] = result

            # Start concurrent processing
            for i, agent in enumerate(agents):
                thread = threading.Thread(target=process_agent, args=(agent, i))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            total_time = time.time() - start_time

            # Verify both agents completed successfully
            assert len(results) == 2
            assert results["review"]["success"] is True
            assert results["fix"]["success"] is True

            # Verify concurrent execution (should be faster than sequential)
            sequential_time = sum(agent.delay for agent in agents)
            assert total_time < sequential_time + 0.1  # Add small buffer for threading overhead

            # Verify each agent received exactly one message
            for agent in agents:
                assert len(agent.calls_received) == 1

        finally:
            dispatcher.cleanup()

    def test_message_context_validation(self, temp_dir, basic_config):
        """Test validation of message context data"""

        dispatcher = EnhancedAgentDispatcher(config=basic_config.to_dict())
        mock_agent = MockSimpleAgent(AgentType.REVIEW)

        try:
            # Test with comprehensive context
            context = MessageContext(
                task_id="validation_test",
                parent_session="session_validation",
                files_modified=["src/main.py", "src/utils.py", "tests/test_main.py"],
                project_path=str(temp_dir)
            )

            message = AgentMessage(
                message_type=MessageType.DISPATCH_AGENT,
                agent_type=AgentType.REVIEW,
                context=context,
                success_criteria=SuccessCriteria(
                    expected_outcomes=["All files reviewed", "Issues identified"],
                    validation_steps=["Static analysis", "Security scan", "Style check"]
                ),
                callback=CallbackInfo(handler="validation_callback", timeout_ms=10000),
                payload={"custom_data": "test_value", "priority": "high"}
            )

            # Process message
            result = mock_agent.process_message(message)

            # Verify context was properly passed
            received_message = mock_agent.calls_received[0]
            assert received_message.context.task_id == "validation_test"
            assert received_message.context.parent_session == "session_validation"
            assert len(received_message.context.files_modified) == 3
            assert received_message.context.project_path == str(temp_dir)

            # Verify success criteria
            assert len(received_message.success_criteria.expected_outcomes) == 2
            assert len(received_message.success_criteria.validation_steps) == 3

            # Verify callback info
            assert received_message.callback.handler == "validation_callback"
            assert received_message.callback.timeout_ms == 10000

            # Verify payload
            assert received_message.payload["custom_data"] == "test_value"
            assert received_message.payload["priority"] == "high"

            # Verify agent response includes processed files
            assert result["processed_files"] == ["src/main.py", "src/utils.py", "tests/test_main.py"]

        finally:
            dispatcher.cleanup()

    def test_simple_callback_mechanism(self, temp_dir, basic_config):
        """Test basic callback mechanism without complex orchestration"""

        dispatcher = EnhancedAgentDispatcher(config=basic_config.to_dict())
        mock_agent = MockSimpleAgent(AgentType.REVIEW)

        # Track callback execution
        callback_data = {"executed": False, "result": None}

        def simple_callback(result: Dict[str, Any]):
            callback_data["executed"] = True
            callback_data["result"] = result

        try:
            message = AgentMessage(
                message_type=MessageType.DISPATCH_AGENT,
                agent_type=AgentType.REVIEW,
                context=MessageContext(
                    task_id="callback_test",
                    parent_session="session_callback",
                    files_modified=["test.py"],
                    project_path=str(temp_dir)
                ),
                success_criteria=SuccessCriteria(expected_outcomes=["Callback tested"]),
                callback=CallbackInfo(handler="simple_callback")
            )

            # Process message
            agent_result = mock_agent.process_message(message)

            # Simulate callback execution (in real system this would be handled by dispatcher)
            simple_callback(agent_result)

            # Verify callback was executed
            assert callback_data["executed"] is True
            assert callback_data["result"] is not None
            assert callback_data["result"]["success"] is True
            assert callback_data["result"]["agent_type"] == "review"

        finally:
            dispatcher.cleanup()
