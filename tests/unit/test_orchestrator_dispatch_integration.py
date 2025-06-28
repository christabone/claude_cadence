"""
Unit tests for Orchestrator Dispatch Integration
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cadence.orchestrator import SupervisorOrchestrator, SupervisorDecision, AgentResult
from cadence.workflow_state_machine import WorkflowState
from cadence.agent_messages import AgentMessage, MessageType, AgentType, Priority
from cadence.review_result_parser import ParsedIssue, IssueSeverity, IssueCategory
from cadence.scope_validator import FixProposal


class TestOrchestratorDispatchIntegration:
    """Test orchestrator dispatch system integration"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        self.task_file = self.project_root / "tasks.json"

        # Create minimal task file
        self.task_file.write_text('{"tasks": []}')

        # Base configuration without dispatch
        self.base_config = {
            "supervisor": {"model": "test-model"},
            "agent": {"model": "test-model"},
            "orchestration": {"max_iterations": 10},
            "execution": {"timeout": 30}
        }

    def test_orchestrator_init_dispatch_disabled(self):
        """Test orchestrator initialization with dispatch disabled"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": False}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        assert not orchestrator.dispatch_enabled
        assert orchestrator.agent_dispatcher is None
        assert orchestrator.fix_tracker is None
        assert orchestrator.code_review_agent is None
        assert orchestrator.result_processor is None
        assert orchestrator.scope_validator is None
        assert orchestrator.fix_verifier is None

    def test_orchestrator_init_dispatch_enabled(self):
        """Test orchestrator initialization with dispatch enabled"""
        config = self.base_config.copy()
        config["dispatch"] = {
            "enabled": True,
            "code_review": {
                "primary_model": "gemini-2.5-pro",
                "fallback_models": ["gemini-2.5-flash"]
            },
            "fix_tracking": {
                "max_attempts": 5,
                "enable_persistence": True
            }
        }

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        assert orchestrator.dispatch_enabled
        assert orchestrator.agent_dispatcher is not None
        assert orchestrator.fix_tracker is not None
        assert orchestrator.code_review_agent is not None
        assert orchestrator.result_processor is not None
        assert orchestrator.scope_validator is not None
        assert orchestrator.fix_verifier is not None

    def test_handle_dispatch_code_review_trigger(self):
        """Test dispatch code review trigger handling"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock the workflow
        orchestrator.workflow = Mock()
        orchestrator.workflow.current_state = WorkflowState.WORKING

        # Mock the code review agent
        orchestrator.code_review_agent = Mock()
        review_result = Mock()
        review_result.success = True
        review_result.review_output = "No issues found"
        orchestrator.code_review_agent.review_files.return_value = review_result

        # Mock the result processor
        orchestrator.result_processor = Mock()
        category_result = Mock()
        category_result.actionable_issues = []
        orchestrator.result_processor.process_review_result.return_value = category_result

        # Test code review trigger
        metadata = {"files": ["test.py"]}
        orchestrator._handle_dispatch_code_review(metadata)

        # Verify code review was called
        orchestrator.code_review_agent.review_files.assert_called_once_with(
            file_paths=["test.py"],
            severity_filter="all"
        )

        # Verify result processing was called
        orchestrator.result_processor.process_review_result.assert_called_once_with("No issues found")

    def test_handle_dispatch_fix_required(self):
        """Test dispatch fix required handling"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock the workflow
        orchestrator.workflow = Mock()
        orchestrator.workflow.current_state = WorkflowState.FIX_REQUIRED
        orchestrator.current_session_id = "test-session"

        # Mock the agent dispatcher
        orchestrator.agent_dispatcher = Mock()
        dispatch_result = {"success": True, "agent_id": "fix-agent-123"}
        orchestrator.agent_dispatcher.dispatch_agent.return_value = dispatch_result

        # Test fix dispatch
        issues = [
            {
                "severity": "high",
                "category": "bug",
                "description": "Test issue",
                "file_path": "test.py"
            }
        ]
        metadata = {
            "issues": issues,
            "files": ["test.py"],
            "task_id": "test-task"
        }

        orchestrator._handle_dispatch_fix_required(metadata)

        # Verify agent dispatch was called
        orchestrator.agent_dispatcher.dispatch_agent.assert_called_once()
        call_args = orchestrator.agent_dispatcher.dispatch_agent.call_args[0][0]

        assert isinstance(call_args, AgentMessage)
        assert call_args.message_type == MessageType.FIX_REQUIRED
        assert call_args.agent_type == AgentType.FIX
        assert call_args.priority == Priority.HIGH
        assert call_args.context.task_id == "test-task"
        assert call_args.payload["issues"] == issues

    def test_should_trigger_code_review_frequency_always(self):
        """Test code review trigger logic with frequency always"""
        config = self.base_config.copy()
        config["dispatch"] = {
            "enabled": True,
            "code_review": {
                "enabled": True,
                "frequency": "always"
            }
        }

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        decision = SupervisorDecision(action="execute")

        assert orchestrator._should_trigger_code_review(1, decision)
        assert orchestrator._should_trigger_code_review(5, decision)
        assert orchestrator._should_trigger_code_review(10, decision)

    def test_should_trigger_code_review_frequency_periodic(self):
        """Test code review trigger logic with periodic frequency"""
        config = self.base_config.copy()
        config["dispatch"] = {
            "enabled": True,
            "code_review": {
                "enabled": True,
                "frequency": "periodic",
                "period": 3
            }
        }

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        decision = SupervisorDecision(action="execute")

        assert not orchestrator._should_trigger_code_review(1, decision)
        assert not orchestrator._should_trigger_code_review(2, decision)
        assert orchestrator._should_trigger_code_review(3, decision)
        assert not orchestrator._should_trigger_code_review(4, decision)
        assert not orchestrator._should_trigger_code_review(5, decision)
        assert orchestrator._should_trigger_code_review(6, decision)

    def test_should_trigger_code_review_frequency_on_completion(self):
        """Test code review trigger logic with on_completion frequency"""
        config = self.base_config.copy()
        config["dispatch"] = {
            "enabled": True,
            "code_review": {
                "enabled": True,
                "frequency": "on_completion"
            }
        }

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        execute_decision = SupervisorDecision(action="execute")
        complete_decision = SupervisorDecision(action="complete")

        assert not orchestrator._should_trigger_code_review(1, execute_decision)
        assert orchestrator._should_trigger_code_review(1, complete_decision)

    def test_agent_made_significant_changes_multiple_subtasks(self):
        """Test significant changes detection with multiple subtasks"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Decision with many subtasks
        decision = SupervisorDecision(
            action="execute",
            subtasks=[
                {"title": "Task 1", "description": "Do something"},
                {"title": "Task 2", "description": "Do something else"},
                {"title": "Task 3", "description": "Do more work"},
                {"title": "Task 4", "description": "Even more work"}
            ]
        )

        assert orchestrator._agent_made_significant_changes(decision)

        # Decision with few subtasks
        decision_small = SupervisorDecision(
            action="execute",
            subtasks=[
                {"title": "Task 1", "description": "Do something"}
            ]
        )

        assert not orchestrator._agent_made_significant_changes(decision_small)

    def test_agent_made_significant_changes_keywords(self):
        """Test significant changes detection with keywords"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Decision with significant keyword
        decision = SupervisorDecision(
            action="execute",
            guidance="Implement a comprehensive refactor of the authentication system"
        )

        assert orchestrator._agent_made_significant_changes(decision)

        # Decision without significant keywords
        decision_minor = SupervisorDecision(
            action="execute",
            guidance="Fix a small typo in the comments"
        )

        assert not orchestrator._agent_made_significant_changes(decision_minor)

    @patch('subprocess.run')
    def test_get_recently_modified_files(self, mock_subprocess):
        """Test getting recently modified files"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock git output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "src/main.py\nsrc/utils.py\nREADME.md\ntests/test_main.js"
        mock_subprocess.return_value = mock_result

        files = orchestrator._get_recently_modified_files()

        # Should filter to code files only
        expected_files = ["src/main.py", "src/utils.py", "tests/test_main.js"]
        assert files == expected_files

        # Verify git command was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == ["git", "diff", "--name-only", "HEAD~1"]
        assert call_args[1]["cwd"] == self.project_root

    def test_extract_modified_files(self):
        """Test extracting modified files from agent result and decision"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock get_recently_modified_files
        with patch.object(orchestrator, '_get_recently_modified_files') as mock_recent:
            mock_recent.return_value = ["src/main.py", "src/utils.py"]

            agent_result = AgentResult(
                success=True,
                session_id="test-session",
                output_file="output.txt",
                error_file="error.txt",
                execution_time=10.0
            )

            decision = SupervisorDecision(
                action="execute",
                subtasks=[
                    {
                        "title": "Update config",
                        "description": "Modify the config/settings.py file to add new options"
                    },
                    {
                        "title": "Add tests",
                        "description": "Create tests/test_config.py for the new functionality"
                    }
                ]
            )

            modified_files = orchestrator._extract_modified_files(agent_result, decision)

            # Should include both git-detected files and files from subtask descriptions
            assert "src/main.py" in modified_files
            assert "src/utils.py" in modified_files
            assert "config/settings.py" in modified_files
            assert "tests/test_config.py" in modified_files

    def test_handle_dispatch_escalation(self):
        """Test dispatch escalation handling"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock fix tracker
        orchestrator.fix_tracker = Mock()
        orchestrator.fix_tracker.get_attempt_count.return_value = 5

        metadata = {
            "reason": "Maximum fix attempts exceeded",
            "task_id": "test-task-123"
        }

        # This should not raise an exception
        orchestrator._handle_dispatch_escalation(metadata)

        # Verify attempt count was queried
        orchestrator.fix_tracker.get_attempt_count.assert_called_once_with("test-task-123")

    def test_trigger_dispatch_code_review_disabled(self):
        """Test triggering code review when dispatch is disabled"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": False}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock workflow
        orchestrator.workflow = Mock()

        # Should not trigger anything when disabled
        orchestrator.trigger_dispatch_code_review(files=["test.py"])

        # Workflow transition should not be called
        orchestrator.workflow.transition_to.assert_not_called()

    def test_workflow_transition_dispatch_triggers(self):
        """Test workflow transitions for dispatch-specific triggers"""
        config = self.base_config.copy()
        config["dispatch"] = {"enabled": True}

        orchestrator = SupervisorOrchestrator(
            project_root=self.project_root,
            task_file=self.task_file,
            config=config
        )

        # Mock workflow
        orchestrator.workflow = Mock()
        orchestrator.workflow.current_state = WorkflowState.WORKING

        # Mock dispatch handlers
        with patch.object(orchestrator, '_handle_dispatch_code_review') as mock_review, \
             patch.object(orchestrator, '_handle_dispatch_fix_required') as mock_fix, \
             patch.object(orchestrator, '_handle_dispatch_escalation') as mock_escalation:

            # Test code review trigger
            metadata = {"files": ["test.py"]}
            orchestrator.handle_workflow_transition("dispatch_code_review_triggered", metadata)

            mock_review.assert_called_once_with(metadata)
            orchestrator.workflow.transition_to.assert_called_with(
                WorkflowState.REVIEW_TRIGGERED,
                "dispatch_code_review_triggered",
                metadata
            )

            # Reset mocks
            orchestrator.workflow.reset_mock()
            mock_review.reset_mock()

            # Test fix required trigger
            orchestrator.workflow.current_state = WorkflowState.REVIEWING
            fix_metadata = {"issues": []}
            orchestrator.handle_workflow_transition("dispatch_fix_required", fix_metadata)

            mock_fix.assert_called_once_with(fix_metadata)
            orchestrator.workflow.transition_to.assert_called_with(
                WorkflowState.FIX_REQUIRED,
                "dispatch_fix_required",
                fix_metadata
            )

            # Reset mocks
            orchestrator.workflow.reset_mock()
            mock_fix.reset_mock()

            # Test escalation trigger
            escalation_metadata = {"reason": "Test escalation"}
            orchestrator.handle_workflow_transition("dispatch_escalation_required", escalation_metadata)

            mock_escalation.assert_called_once_with(escalation_metadata)
            orchestrator.workflow.transition_to.assert_called_with(
                WorkflowState.ERROR,
                "dispatch_escalation_required",
                escalation_metadata
            )


class TestDispatchSystemConfiguration:
    """Test dispatch system configuration validation"""

    def test_minimal_dispatch_config(self):
        """Test minimal dispatch configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            task_file = project_root / "tasks.json"
            task_file.write_text('{"tasks": []}')

            config = {
                "dispatch": {
                    "enabled": True
                }
            }

            orchestrator = SupervisorOrchestrator(
                project_root=project_root,
                task_file=task_file,
                config=config
            )

            assert orchestrator.dispatch_enabled
            assert orchestrator.agent_dispatcher is not None
            assert orchestrator.fix_tracker is not None

    def test_full_dispatch_config(self):
        """Test comprehensive dispatch configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            task_file = project_root / "tasks.json"
            task_file.write_text('{"tasks": []}')

            config = {
                "dispatch": {
                    "enabled": True,
                    "max_concurrent": 3,
                    "default_timeout_ms": 900000,
                    "enable_escalation": True,
                    "code_review": {
                        "primary_model": "custom-model",
                        "fallback_models": ["fallback1", "fallback2"],
                        "max_file_size_mb": 10,
                        "timeout_seconds": 600,
                        "enabled": True,
                        "frequency": "periodic",
                        "period": 2
                    },
                    "fix_tracking": {
                        "max_attempts": 10,
                        "enable_persistence": False
                    },
                    "result_processing": {
                        "confidence_threshold": 0.7,
                        "max_description_length": 1000
                    },
                    "verification": {
                        "minimum_resolution_rate": 0.9,
                        "max_acceptable_regressions": 1,
                        "timeout_ms": 600000
                    }
                }
            }

            orchestrator = SupervisorOrchestrator(
                project_root=project_root,
                task_file=task_file,
                config=config
            )

            assert orchestrator.dispatch_enabled

            # Verify configuration was applied
            assert orchestrator.code_review_agent.config.primary_model == "custom-model"
            assert orchestrator.code_review_agent.config.max_file_size_mb == 10
            assert orchestrator.fix_tracker.max_attempts == 10
            assert orchestrator.result_processor.config.confidence_threshold == 0.7
