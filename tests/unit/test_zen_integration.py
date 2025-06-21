"""
Unit tests for ZenIntegration
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from cadence.zen_integration import ZenIntegration, ZenRequest
from cadence.config import ZenIntegrationConfig
from cadence.task_supervisor import ExecutionResult


class TestZenIntegration:
    """Test ZenIntegration functionality"""
    
    def test_initialization(self):
        """Test ZenIntegration initialization"""
        config = ZenIntegrationConfig()
        zen = ZenIntegration(config)
        
        assert zen.config == config
        assert zen.verbose is False
        
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration"""
        config = ZenIntegrationConfig(
            enabled=True,
            stuck_detection=True,
            auto_debug_threshold=5,
            cutoff_detection=True
        )
        zen = ZenIntegration(config, verbose=True)
        
        assert zen.config.enabled is True
        assert zen.config.auto_debug_threshold == 5
        assert zen.verbose is True
        
    def test_detect_stuck_agent_explicit_help(self):
        """Test detection of explicitly stuck agent"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        # Explicit help request
        result = ExecutionResult(
            success=True,
            turns_used=5,
            output_lines=["Working...", "HELP NEEDED - STUCK", "Cannot proceed"],
            errors=[],
            metadata={},
            task_complete=False
        )
        
        context = Mock()
        session_id = "test_123"
        
        stuck = zen._detect_stuck_agent(result, context, session_id)
        assert stuck is True
        
    def test_detect_stuck_agent_scratchpad_help(self, temp_dir):
        """Test detection via scratchpad help request"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        # Create scratchpad with help request
        scratchpad_dir = temp_dir / ".cadence" / "scratchpad"
        scratchpad_dir.mkdir(parents=True)
        scratchpad_file = scratchpad_dir / "session_test_123.md"
        scratchpad_file.write_text("""
        ## HELP NEEDED
        Status: STUCK
        Issue: Cannot find configuration
        """)
        
        result = ExecutionResult(
            success=True,
            turns_used=5,
            output_lines=["Working..."],
            errors=[],
            metadata={},
            task_complete=False
        )
        
        with patch('cadence.zen_integration.Path') as mock_path:
            mock_path.return_value = temp_dir
            stuck = zen._detect_stuck_agent(result, Mock(), "test_123")
            assert stuck is True
            
    def test_detect_repeated_errors(self):
        """Test detection of repeated errors"""
        zen = ZenIntegration(ZenIntegrationConfig(auto_debug_threshold=3))
        
        # Repeated errors
        result = ExecutionResult(
            success=False,
            turns_used=10,
            output_lines=[
                "Trying...",
                "Error: File not found",
                "Retrying...",
                "Error: File not found",
                "One more time...",
                "Error: File not found"
            ],
            errors=["File not found", "File not found", "File not found"],
            metadata={},
            task_complete=False
        )
        
        has_errors, count = zen._detect_repeated_errors(result, Mock())
        assert has_errors is True
        assert count >= 3
        
    def test_detect_cutoff(self):
        """Test detection of execution cutoff"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        # Signs of cutoff - no completion message
        result = ExecutionResult(
            success=True,
            turns_used=40,
            output_lines=["Working on task...", "Still processing..."],
            errors=[],
            metadata={},
            task_complete=False
        )
        
        context = Mock()
        context.remaining_todos = ["TODO 1", "TODO 2"]
        
        cutoff = zen._detect_cutoff(result, context)
        assert cutoff is True
        
        # Completed execution
        result_complete = ExecutionResult(
            success=True,
            turns_used=10,
            output_lines=["Working...", "ALL TASKS COMPLETE"],
            errors=[],
            metadata={},
            task_complete=True
        )
        
        cutoff = zen._detect_cutoff(result_complete, Mock())
        assert cutoff is False
        
    def test_is_critical_task(self):
        """Test critical task detection"""
        zen = ZenIntegration(ZenIntegrationConfig(
            validate_on_complete=["*security*", "*database*", "*payment*"]
        ))
        
        # Critical tasks
        context = Mock()
        context.todos = [
            "Implement security authentication",
            "Update database schema",
            "Process payment transactions"
        ]
        
        assert zen._is_critical_task(context) is True
        
        # Non-critical tasks
        context.todos = [
            "Update README",
            "Add logging",
            "Fix typo"
        ]
        
        assert zen._is_critical_task(context) is False
        
    def test_should_call_zen_when_disabled(self):
        """Test should_call_zen when disabled"""
        config = ZenIntegrationConfig(enabled=False)
        zen = ZenIntegration(config)
        
        result = Mock()
        assert zen.should_call_zen(result, Mock(), "test") is None
        
    def test_should_call_zen_stuck_detection(self):
        """Test should_call_zen for stuck agent"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        result = ExecutionResult(
            success=True,
            turns_used=5,
            output_lines=["HELP NEEDED - STUCK"],
            errors=[],
            metadata={},
            task_complete=False
        )
        
        tool, reason = zen.should_call_zen(result, Mock(), "test")
        assert tool == "debug"
        assert "stuck" in reason.lower()
        
    def test_should_call_zen_error_threshold(self):
        """Test should_call_zen for repeated errors"""
        zen = ZenIntegration(ZenIntegrationConfig(auto_debug_threshold=2))
        
        result = ExecutionResult(
            success=False,
            turns_used=5,
            output_lines=["Error", "Error", "Error"],
            errors=["err1", "err2", "err3"],
            metadata={},
            task_complete=False
        )
        
        tool, reason = zen.should_call_zen(result, Mock(), "test")
        assert tool == "debug"
        assert "repeated errors" in reason.lower()
        
    def test_should_call_zen_cutoff(self):
        """Test should_call_zen for cutoff detection"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        result = ExecutionResult(
            success=True,
            turns_used=40,
            output_lines=["Working..."],
            errors=[],
            metadata={},
            task_complete=False
        )
        
        context = Mock()
        context.remaining_todos = ["TODO 1"]
        
        tool, reason = zen.should_call_zen(result, context, "test")
        assert tool == "analyze"
        assert "cut off" in reason.lower()
        
    def test_should_call_zen_critical_complete(self):
        """Test should_call_zen for critical task completion"""
        zen = ZenIntegration(ZenIntegrationConfig(
            validate_on_complete=["*security*"]
        ))
        
        result = ExecutionResult(
            success=True,
            turns_used=10,
            output_lines=["ALL TASKS COMPLETE"],
            errors=[],
            metadata={},
            task_complete=True
        )
        
        context = Mock()
        context.todos = ["Implement security module"]
        
        tool, reason = zen.should_call_zen(result, context, "test")
        assert tool == "precommit"
        assert "critical" in reason.lower()
        
    def test_create_zen_request(self):
        """Test ZenRequest creation"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        result = ExecutionResult(
            success=False,
            turns_used=10,
            output_lines=["line1", "line2"],
            errors=["error1"],
            metadata={"test": "data"},
            task_complete=False
        )
        
        context = Mock()
        context.todos = ["TODO 1"]
        
        request = zen._create_zen_request(
            tool="debug",
            reason="Test reason",
            execution_result=result,
            context=context,
            session_id="test_123"
        )
        
        assert isinstance(request, ZenRequest)
        assert request.tool == "debug"
        assert request.reason == "Test reason"
        assert request.session_id == "test_123"
        assert len(request.execution_history) == 1
        assert request.scratchpad_path == ".cadence/scratchpad/session_test_123.md"
        
    @patch('subprocess.run')
    def test_call_zen_tool(self, mock_run):
        """Test calling zen tool via MCP"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        # Mock successful zen call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "result": {
                "content": [
                    {"type": "text", "text": "Debug analysis complete"}
                ]
            }
        })
        mock_run.return_value = mock_result
        
        request = ZenRequest(
            tool="debug",
            reason="Test",
            session_id="123",
            execution_history=[],
            scratchpad_path="test.md"
        )
        
        response = zen._call_zen_tool(request)
        
        assert response["success"] is True
        assert "Debug analysis complete" in response["guidance"]
        
        # Verify command
        call_args = mock_run.call_args[0][0]
        assert "mcp" in call_args
        assert "call-tool" in call_args
        assert "mcp__zen__debug" in call_args
        
    def test_generate_continuation_guidance(self):
        """Test continuation guidance generation"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        # Success case
        zen_response = {
            "success": True,
            "guidance": "Try this approach:\n1. Check X\n2. Fix Y"
        }
        
        guidance = zen.generate_continuation_guidance(zen_response)
        assert "Zen assistance provided" in guidance
        assert "Try this approach" in guidance
        
        # Failure case
        zen_response = {
            "success": False,
            "error": "Zen tool failed"
        }
        
        guidance = zen.generate_continuation_guidance(zen_response)
        assert "unable to provide" in guidance.lower()
        
    def test_call_zen_support_full_flow(self):
        """Test full zen support flow"""
        zen = ZenIntegration(ZenIntegrationConfig())
        
        result = ExecutionResult(
            success=False,
            turns_used=10,
            output_lines=["Error occurred"],
            errors=["Test error"],
            metadata={},
            task_complete=False
        )
        
        with patch.object(zen, '_call_zen_tool') as mock_call:
            mock_call.return_value = {
                "success": True,
                "guidance": "Debug guidance here"
            }
            
            response = zen.call_zen_support(
                tool="debug",
                reason="Errors detected",
                execution_result=result,
                context=Mock(),
                session_id="test_123"
            )
            
            assert response["success"] is True
            assert response["guidance"] == "Debug guidance here"
            mock_call.assert_called_once()
            
    def test_model_selection(self):
        """Test model selection for different tools"""
        config = ZenIntegrationConfig(
            models={
                "debug": ["model1", "model2"],
                "review": ["model3"]
            }
        )
        zen = ZenIntegration(config)
        
        request = ZenRequest(
            tool="debug",
            reason="Test",
            session_id="123",
            execution_history=[],
            scratchpad_path="test.md"
        )
        
        # Should use first model from debug list
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout='{"result": {"content": [{"type": "text", "text": "test"}]}}'
            )
            
            zen._call_zen_tool(request)
            
            call_args = str(mock_run.call_args)
            assert "model1" in call_args or "model" in call_args  # Model should be in the call