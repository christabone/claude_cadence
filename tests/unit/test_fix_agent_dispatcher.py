"""Unit tests for FixAgentDispatcher"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cadence.fix_agent_dispatcher import (
    FixAgentDispatcher,
    FixAttemptStatus,
    FixAttempt,
    IssueContext
)
from cadence.agent_messages import AgentMessage, MessageType, AgentType, MessageContext, SuccessCriteria, CallbackInfo
from cadence.config import FixAgentDispatcherConfig, CircularDependencyConfig


class TestFixAgentDispatcher:
    """Test cases for FixAgentDispatcher"""

    def test_initialization(self):
        """Test basic initialization"""
        config = FixAgentDispatcherConfig(
            max_attempts=5,
            timeout_ms=600000,
            enable_auto_fix=False,
            severity_threshold="critical"
        )
        dispatcher = FixAgentDispatcher(config)

        assert dispatcher.max_attempts == 5
        assert dispatcher.timeout_ms == 600000
        assert dispatcher.enable_auto_fix is False
        assert dispatcher.severity_threshold == "critical"
        assert len(dispatcher.active_fixes) == 0
        assert len(dispatcher.fix_history) == 0

    def test_default_initialization(self):
        """Test initialization with defaults"""
        dispatcher = FixAgentDispatcher()

        assert dispatcher.max_attempts == 3
        assert dispatcher.timeout_ms == 300000  # 5 minutes
        assert dispatcher.enable_auto_fix is True
        assert dispatcher.severity_threshold == "high"

    def test_should_dispatch_fix_auto_fix_disabled(self):
        """Test should_dispatch_fix when auto-fix is disabled"""
        config = FixAgentDispatcherConfig(enable_auto_fix=False)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        assert dispatcher.should_dispatch_fix(issue) is False

    def test_should_dispatch_fix_severity_threshold(self):
        """Test should_dispatch_fix with different severity levels"""
        config = FixAgentDispatcherConfig(severity_threshold="high")
        dispatcher = FixAgentDispatcher(config)

        # Low severity - should not dispatch
        low_issue = IssueContext(
            issue_id="issue-1",
            severity="low",
            issue_type="style",
            description="Minor style issue",
            file_path="/test/file.py"
        )
        assert dispatcher.should_dispatch_fix(low_issue) is False

        # Medium severity - should not dispatch
        medium_issue = IssueContext(
            issue_id="issue-2",
            severity="medium",
            issue_type="warning",
            description="Medium priority warning",
            file_path="/test/file.py"
        )
        assert dispatcher.should_dispatch_fix(medium_issue) is False

        # High severity - should dispatch
        high_issue = IssueContext(
            issue_id="issue-3",
            severity="high",
            issue_type="bug",
            description="High priority bug",
            file_path="/test/file.py"
        )
        assert dispatcher.should_dispatch_fix(high_issue) is True

        # Critical severity - should dispatch
        critical_issue = IssueContext(
            issue_id="issue-4",
            severity="critical",
            issue_type="security",
            description="Critical security issue",
            file_path="/test/file.py"
        )
        assert dispatcher.should_dispatch_fix(critical_issue) is True

    def test_should_dispatch_fix_already_active(self):
        """Test should_dispatch_fix when issue is already being fixed"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Simulate active fix
        dispatcher.active_fixes[issue.issue_id] = issue

        assert dispatcher.should_dispatch_fix(issue) is False

    def test_should_dispatch_fix_max_attempts_exceeded(self):
        """Test should_dispatch_fix when max attempts are exceeded"""
        config = FixAgentDispatcherConfig(max_attempts=2)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Add failed attempts to history
        dispatcher.fix_history[issue.issue_id] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now()
            ),
            FixAttempt(
                attempt_number=2,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now()
            )
        ]

        assert dispatcher.should_dispatch_fix(issue) is False

    def test_get_fix_status_not_found(self):
        """Test get_fix_status for non-existent issue"""
        dispatcher = FixAgentDispatcher()

        status = dispatcher.get_fix_status("non-existent")
        assert status is None

    def test_get_fix_status_active_issue(self):
        """Test get_fix_status for active issue"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        dispatcher.active_fixes[issue.issue_id] = issue

        status = dispatcher.get_fix_status(issue.issue_id)
        assert status is not None
        assert status["issue_id"] == "issue-1"
        assert status["is_active"] is True
        assert len(status["attempts"]) == 0

    def test_get_fix_status_with_history(self):
        """Test get_fix_status with attempt history"""
        dispatcher = FixAgentDispatcher()

        start_time = datetime.now()
        end_time = datetime.now()

        dispatcher.fix_history["issue-1"] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                error_message="Test error",
                files_modified=["file1.py", "file2.py"]
            )
        ]

        status = dispatcher.get_fix_status("issue-1")
        assert status is not None
        assert status["issue_id"] == "issue-1"
        assert status["is_active"] is False
        assert len(status["attempts"]) == 1

        attempt = status["attempts"][0]
        assert attempt["number"] == 1
        assert attempt["status"] == "failed"
        assert attempt["error"] == "Test error"
        assert attempt["files_modified"] == ["file1.py", "file2.py"]

    def test_cleanup(self):
        """Test cleanup method"""
        dispatcher = FixAgentDispatcher()

        # Add some data
        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        dispatcher.active_fixes["issue-1"] = issue
        dispatcher.fix_history["issue-1"] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.SUCCESS,
                start_time=datetime.now()
            )
        ]

        # Cleanup
        dispatcher.cleanup()

        assert len(dispatcher.active_fixes) == 0
        assert len(dispatcher.fix_history) == 0

    def test_thread_safety(self):
        """Test thread safety of should_dispatch_fix"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Test that lock is acquired by patching the lock itself
        mock_lock = MagicMock()
        dispatcher.lock = mock_lock

        dispatcher.should_dispatch_fix(issue)

        # Verify lock was acquired and released
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_classify_issue_type_security(self):
        """Test classifying security issues"""
        dispatcher = FixAgentDispatcher()

        security_issues = [
            IssueContext(
                issue_id="sec-1",
                severity="critical",
                issue_type="",
                description="SQL injection vulnerability in user input",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="sec-2",
                severity="high",
                issue_type="",
                description="Missing authentication check",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="sec-3",
                severity="high",
                issue_type="",
                description="Weak password encryption",
                file_path="/test/file.py"
            ),
        ]

        for issue in security_issues:
            assert dispatcher.classify_issue_type(issue) == "security"

    def test_classify_issue_type_performance(self):
        """Test classifying performance issues"""
        dispatcher = FixAgentDispatcher()

        performance_issues = [
            IssueContext(
                issue_id="perf-1",
                severity="medium",
                issue_type="",
                description="Memory leak in request handler",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="perf-2",
                severity="high",
                issue_type="",
                description="High CPU usage in processing loop",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="perf-3",
                severity="medium",
                issue_type="",
                description="Slow database query causing latency",
                file_path="/test/file.py"
            ),
        ]

        for issue in performance_issues:
            assert dispatcher.classify_issue_type(issue) == "performance"

    def test_classify_issue_type_bug(self):
        """Test classifying bug/error issues"""
        dispatcher = FixAgentDispatcher()

        bug_issues = [
            IssueContext(
                issue_id="bug-1",
                severity="critical",
                issue_type="",
                description="Application crashes on null input",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="bug-2",
                severity="high",
                issue_type="",
                description="Exception thrown when parsing invalid data",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="bug-3",
                severity="medium",
                issue_type="",
                description="Function returns incorrect value",
                file_path="/test/file.py"
            ),
        ]

        for issue in bug_issues:
            assert dispatcher.classify_issue_type(issue) == "bug"

    def test_classify_issue_type_quality(self):
        """Test classifying code quality issues"""
        dispatcher = FixAgentDispatcher()

        quality_issues = [
            IssueContext(
                issue_id="qual-1",
                severity="low",
                issue_type="",
                description="Duplicate code needs refactoring",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="qual-2",
                severity="medium",
                issue_type="",
                description="High complexity in method",
                file_path="/test/file.py"
            ),
            IssueContext(
                issue_id="qual-3",
                severity="low",
                issue_type="",
                description="Code smell: long method",
                file_path="/test/file.py"
            ),
        ]

        for issue in quality_issues:
            assert dispatcher.classify_issue_type(issue) == "quality"

    def test_classify_issue_type_existing_type(self):
        """Test that existing issue type is preserved"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="custom-1",
            severity="high",
            issue_type="custom_type",
            description="Some generic issue",
            file_path="/test/file.py"
        )

        assert dispatcher.classify_issue_type(issue) == "custom_type"

    def test_classify_issue_type_general_fallback(self):
        """Test fallback to general type"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="gen-1",
            severity="medium",
            issue_type="",
            description="Some other issue that doesn't match patterns",
            file_path="/test/file.py"
        )

        assert dispatcher.classify_issue_type(issue) == "general"

    def test_dispatch_fix_agent_not_eligible(self):
        """Test dispatch when issue is not eligible"""
        config = FixAgentDispatcherConfig(enable_auto_fix=False)
        dispatcher = FixAgentDispatcher(config)

        # Mock the agent dispatcher
        mock_agent_dispatcher = Mock()
        dispatcher.agent_dispatcher = mock_agent_dispatcher

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        result = dispatcher.dispatch_fix_agent(issue)
        assert result is None

        # AgentDispatcher should not be called
        mock_agent_dispatcher.dispatch_agent.assert_not_called()

    @patch('cadence.fix_agent_dispatcher.datetime')
    def test_dispatch_fix_agent_success(self, mock_datetime):
        """Test successful fix agent dispatch"""
        mock_now = Mock()
        mock_datetime.now.return_value = mock_now

        dispatcher = FixAgentDispatcher()

        # Mock the agent dispatcher
        mock_agent_dispatcher = Mock()
        mock_agent_dispatcher.dispatch_agent.return_value = "msg-12345678-1234567890"
        mock_agent_dispatcher.generate_message_id.return_value = "msg-12345678-1234567890"
        dispatcher.agent_dispatcher = mock_agent_dispatcher

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="",
            description="Security vulnerability found",
            file_path=tmp_path,
            line_numbers=[10, 20],
            suggested_fix="Use parameterized queries"
        )

        callback = Mock()

        try:
            result = dispatcher.dispatch_fix_agent(issue, callback)

            # Should return message ID
            assert result == "msg-12345678-1234567890"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Issue should be classified
        assert issue.issue_type == "security"

        # Issue should be in active fixes
        assert issue.issue_id in dispatcher.active_fixes

        # History should have new attempt
        assert issue.issue_id in dispatcher.fix_history
        assert len(dispatcher.fix_history[issue.issue_id]) == 1

        attempt = dispatcher.fix_history[issue.issue_id][0]
        assert attempt.attempt_number == 1
        assert attempt.status == FixAttemptStatus.IN_PROGRESS
        assert attempt.fix_agent_id == "msg-12345678-1234567890"

        # Verify agent dispatcher was called correctly
        mock_agent_dispatcher.dispatch_agent.assert_called_once()
        call_args = mock_agent_dispatcher.dispatch_agent.call_args

        # Check the call arguments
        assert call_args[1]['agent_type'] == AgentType.FIX
        assert call_args[1]['timeout_ms'] == 300000  # default timeout

        # Check context
        context = call_args[1]['context']
        assert context.task_id == "fix-issue-1"
        assert context.parent_session == "review-session-issue-1"
        assert context.files_modified == [tmp_path]

        # Check success criteria
        success_criteria = call_args[1]['success_criteria']
        assert len(success_criteria.expected_outcomes) == 3
        assert f"Fix the security issue in {tmp_path}" in success_criteria.expected_outcomes[0]

    def test_dispatch_fix_agent_with_existing_attempts(self):
        """Test dispatching with existing attempts"""
        config = FixAgentDispatcherConfig(max_attempts=3)
        dispatcher = FixAgentDispatcher(config)

        # Mock the agent dispatcher
        mock_agent_dispatcher = Mock()
        mock_agent_dispatcher.dispatch_agent.return_value = "msg-attempt2"
        mock_agent_dispatcher.generate_message_id.return_value = "msg-attempt2"
        dispatcher.agent_dispatcher = mock_agent_dispatcher

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Critical bug",
            file_path=tmp_path
        )

        # Add a previous failed attempt
        dispatcher.fix_history[issue.issue_id] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now(),
                error_message="First attempt failed"
            )
        ]

        try:
            result = dispatcher.dispatch_fix_agent(issue)

            # Should succeed with second attempt
            assert result == "msg-attempt2"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Should have 2 attempts now
        assert len(dispatcher.fix_history[issue.issue_id]) == 2
        assert dispatcher.fix_history[issue.issue_id][1].attempt_number == 2

    def test_handle_fix_response_success(self):
        """Test handling successful fix response"""
        config = FixAgentDispatcherConfig(enable_verification=False)
        dispatcher = FixAgentDispatcher(config)

        # Set up issue and attempt
        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.IN_PROGRESS,
            start_time=datetime.now()
        )

        dispatcher.active_fixes[issue.issue_id] = issue
        dispatcher.fix_history[issue.issue_id] = [attempt]

        # Mock success callback
        success_callback = Mock()
        dispatcher.on_fix_complete = success_callback

        # Create success response
        from cadence.agent_messages import MessageContext, SuccessCriteria, CallbackInfo
        response = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="fix-issue-1",
                parent_session="review-session-issue-1",
                files_modified=["/test/file.py", "/test/file2.py"],
                project_path="/test"
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["Fix completed"],
                validation_steps=["Tests passed"]
            ),
            callback=CallbackInfo(
                handler="fix_complete",
                timeout_ms=300000
            ),
            payload={"result": "Fixed successfully"}
        )

        # Handle response
        dispatcher._handle_fix_response(issue, attempt, response)

        # Check attempt was updated
        assert attempt.status == FixAttemptStatus.SUCCESS
        assert attempt.end_time is not None
        assert attempt.files_modified == ["/test/file.py", "/test/file2.py"]

        # Check issue was removed from active
        assert issue.issue_id not in dispatcher.active_fixes

        # Check callback was called
        success_callback.assert_called_once_with(issue, attempt)

    def test_handle_fix_response_failure_with_retry(self):
        """Test handling failed response with retry available"""
        config = FixAgentDispatcherConfig(max_attempts=3)
        dispatcher = FixAgentDispatcher(config)

        # Set up issue and attempt
        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.IN_PROGRESS,
            start_time=datetime.now()
        )

        dispatcher.active_fixes[issue.issue_id] = issue
        dispatcher.fix_history[issue.issue_id] = [attempt]

        # Create error response
        from cadence.agent_messages import MessageContext, SuccessCriteria, CallbackInfo
        response = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="fix-issue-1",
                parent_session="review-session-issue-1",
                files_modified=[],
                project_path="/test"
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["Fix completed"],
                validation_steps=["Tests passed"]
            ),
            callback=CallbackInfo(
                handler="fix_complete",
                timeout_ms=300000
            ),
            payload={"error": "Failed to apply fix"}
        )

        # Handle response
        dispatcher._handle_fix_response(issue, attempt, response)

        # Check attempt was updated
        assert attempt.status == FixAttemptStatus.FAILED
        assert attempt.error_message == "Failed to apply fix"
        assert attempt.end_time is not None

        # Check issue was removed from active (to allow retry)
        assert issue.issue_id not in dispatcher.active_fixes

        # Give the Timer a chance to execute
        import time
        time.sleep(0.1)

        # Check that a retry was scheduled
        assert len(dispatcher.retry_queue) == 1

        # Cancel the retry timer to avoid test hanging
        if dispatcher.retry_timer:
            dispatcher.retry_timer.cancel()
            dispatcher.retry_timer = None

    def test_handle_fix_response_max_retries_exceeded(self):
        """Test handling failure when max retries exceeded"""
        config = FixAgentDispatcherConfig(max_attempts=2)
        dispatcher = FixAgentDispatcher(config)

        # Set up issue with 2 attempts already
        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt1 = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.FAILED,
            start_time=datetime.now()
        )

        attempt2 = FixAttempt(
            attempt_number=2,
            status=FixAttemptStatus.IN_PROGRESS,
            start_time=datetime.now()
        )

        dispatcher.active_fixes[issue.issue_id] = issue
        dispatcher.fix_history[issue.issue_id] = [attempt1, attempt2]

        # Mock callbacks
        max_retries_callback = Mock()
        dispatcher.on_max_retries = max_retries_callback

        # Create error response
        from cadence.agent_messages import MessageContext, SuccessCriteria, CallbackInfo
        response = AgentMessage(
            message_type=MessageType.ERROR,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id="fix-issue-1",
                parent_session="review-session-issue-1",
                files_modified=[],
                project_path="/test"
            ),
            success_criteria=SuccessCriteria(
                expected_outcomes=["Fix completed"],
                validation_steps=["Tests passed"]
            ),
            callback=CallbackInfo(
                handler="fix_complete",
                timeout_ms=300000
            ),
            payload={"error": "Failed again"}
        )

        # Handle response
        dispatcher._handle_fix_response(issue, attempt2, response)

        # Check attempt was updated
        assert attempt2.status == FixAttemptStatus.EXCEEDED_RETRIES
        assert attempt2.error_message == "Failed again"

        # Check issue was removed from active
        assert issue.issue_id not in dispatcher.active_fixes

        # Check callback was called with all attempts
        max_retries_callback.assert_called_once_with(issue, [attempt1, attempt2])

    def test_get_fix_context_not_found(self):
        """Test getting context for non-existent issue"""
        dispatcher = FixAgentDispatcher()

        context = dispatcher.get_fix_context("non-existent")
        assert context is None

    def test_preserve_and_get_fix_context(self):
        """Test preserving and retrieving fix context"""
        dispatcher = FixAgentDispatcher()

        # Preserve context
        dispatcher.preserve_fix_context("issue-1", {
            "key1": "value1",
            "key2": ["item1", "item2"]
        })

        # Get context
        context = dispatcher.get_fix_context("issue-1")
        assert context is not None
        assert context["key1"] == "value1"
        assert context["key2"] == ["item1", "item2"]

        # Update context
        dispatcher.preserve_fix_context("issue-1", {
            "key2": ["item3"],
            "key3": "value3"
        })

        # Get updated context
        context = dispatcher.get_fix_context("issue-1")
        assert context["key1"] == "value1"  # Original preserved
        assert context["key2"] == ["item3"]  # Updated
        assert context["key3"] == "value3"  # New key added

    def test_get_fix_context_with_attempt_number(self):
        """Test getting context for specific attempt"""
        dispatcher = FixAgentDispatcher()

        # Add issue and attempts
        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test bug",
            file_path="/test/file.py"
        )

        dispatcher.fix_history["issue-1"] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now(),
                error_message="First error",
                files_modified=["file1.py"]
            ),
            FixAttempt(
                attempt_number=2,
                status=FixAttemptStatus.SUCCESS,
                start_time=datetime.now(),
                files_modified=["file1.py", "file2.py"]
            )
        ]

        # Preserve context
        dispatcher.preserve_fix_context("issue-1", {"base": "context"})

        # Get context for specific attempt
        context = dispatcher.get_fix_context("issue-1", attempt_number=2)
        assert context is not None
        assert context["base"] == "context"
        assert "specific_attempt" in context
        assert context["specific_attempt"]["number"] == 2
        assert context["specific_attempt"]["status"] == "success"
        assert context["specific_attempt"]["files_modified"] == ["file1.py", "file2.py"]

    def test_get_fix_scope_security(self):
        """Test fix scope for security issues"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="sec-1",
            severity="critical",
            issue_type="security",
            description="SQL injection vulnerability",
            file_path="/app/db.py",
            line_numbers=[45, 50]
        )

        scope = dispatcher.get_fix_scope(issue)

        assert scope["primary_file"] == "/app/db.py"
        assert scope["affected_lines"] == [45, 50]
        assert scope["fix_type"] == "security"
        assert "sanitize_input" in scope["allowed_operations"]
        assert "add_validation" in scope["allowed_operations"]
        assert "preserve_existing_functionality" in scope["restrictions"]
        assert "no_new_external_dependencies" in scope["restrictions"]

    def test_get_fix_scope_performance(self):
        """Test fix scope for performance issues"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="perf-1",
            severity="high",
            issue_type="performance",
            description="Slow database query",
            file_path="/app/queries.py"
        )

        scope = dispatcher.get_fix_scope(issue)

        assert scope["fix_type"] == "performance"
        assert "optimize_algorithm" in scope["allowed_operations"]
        assert "add_caching" in scope["allowed_operations"]
        assert "maintain_correctness" in scope["restrictions"]
        assert "no_functional_changes" in scope["restrictions"]

    def test_get_fix_scope_with_previous_context(self):
        """Test fix scope includes previous context"""
        dispatcher = FixAgentDispatcher()

        # Add previous context
        dispatcher.preserve_fix_context("bug-1", {
            "related_files": ["test.py", "utils.py"],
            "learned_constraints": ["must_handle_null_values"]
        })

        issue = IssueContext(
            issue_id="bug-1",
            severity="medium",
            issue_type="bug",
            description="Null pointer exception",
            file_path="/app/main.py"
        )

        scope = dispatcher.get_fix_scope(issue)

        assert "test.py" in scope["related_files"]
        assert "utils.py" in scope["related_files"]
        assert "must_handle_null_values" in scope["restrictions"]

    def test_update_fix_scope(self):
        """Test updating fix scope with new information"""
        dispatcher = FixAgentDispatcher()

        # Initial update
        dispatcher.update_fix_scope("issue-1", {
            "discovered_files": ["helper.py", "config.py"],
            "new_constraints": ["must_be_thread_safe"]
        })

        # Get context to verify
        context = dispatcher.get_fix_context("issue-1")
        assert context is not None
        assert "related_files" in context
        assert "helper.py" in context["related_files"]
        assert "config.py" in context["related_files"]
        assert "learned_constraints" in context
        assert "must_be_thread_safe" in context["learned_constraints"]

        # Update again with more files
        dispatcher.update_fix_scope("issue-1", {
            "discovered_files": ["helper.py", "new_file.py"],  # Duplicate should be removed
            "new_constraints": ["handle_concurrent_access"]
        })

        # Verify updates
        context = dispatcher.get_fix_context("issue-1")
        assert len(context["related_files"]) == 3  # No duplicates
        assert "new_file.py" in context["related_files"]
        assert len(context["learned_constraints"]) == 2
        assert "handle_concurrent_access" in context["learned_constraints"]

    def test_cleanup_clears_contexts(self):
        """Test that cleanup clears preserved contexts"""
        dispatcher = FixAgentDispatcher()

        # Add some context
        dispatcher.preserve_fix_context("issue-1", {"data": "test"})

        # Verify it exists
        assert dispatcher.get_fix_context("issue-1") is not None

        # Cleanup
        dispatcher.cleanup()

        # Verify it's gone
        assert dispatcher.get_fix_context("issue-1") is None

    def test_get_retry_delay(self):
        """Test retry delay calculation with exponential backoff"""
        dispatcher = FixAgentDispatcher()

        # Test exponential backoff
        assert dispatcher.get_retry_delay(1) == 1    # 2^0 = 1
        assert dispatcher.get_retry_delay(2) == 2    # 2^1 = 2
        assert dispatcher.get_retry_delay(3) == 4    # 2^2 = 4
        assert dispatcher.get_retry_delay(4) == 8    # 2^3 = 8
        assert dispatcher.get_retry_delay(5) == 16   # 2^4 = 16

        # Test cap at 5 minutes
        assert dispatcher.get_retry_delay(10) == 300  # Capped at 300s
        assert dispatcher.get_retry_delay(20) == 300  # Still capped

    def test_schedule_retry_success(self):
        """Test successful retry scheduling"""
        config = FixAgentDispatcherConfig(max_attempts=3)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Add one failed attempt
        dispatcher.fix_history[issue.issue_id] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now()
            )
        ]

        # Schedule retry
        result = dispatcher.schedule_retry(issue)
        assert result is True

        # Check retry queue
        assert len(dispatcher.retry_queue) == 1
        retry_time, retry_issue = dispatcher.retry_queue[0]
        assert retry_issue.issue_id == issue.issue_id
        assert retry_time > datetime.now()

        # Cleanup timer
        if dispatcher.retry_timer:
            dispatcher.retry_timer.cancel()
            dispatcher.retry_timer = None

    def test_schedule_retry_max_attempts_exceeded(self):
        """Test retry scheduling when max attempts exceeded"""
        config = FixAgentDispatcherConfig(max_attempts=2)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Add two failed attempts
        dispatcher.fix_history[issue.issue_id] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now()
            ),
            FixAttempt(
                attempt_number=2,
                status=FixAttemptStatus.FAILED,
                start_time=datetime.now()
            )
        ]

        # Try to schedule retry
        result = dispatcher.schedule_retry(issue)
        assert result is False

        # Queue should be empty
        assert len(dispatcher.retry_queue) == 0

    def test_get_iteration_stats(self):
        """Test getting iteration statistics"""
        dispatcher = FixAgentDispatcher()

        start_time1 = datetime.now()
        end_time1 = start_time1 + timedelta(seconds=30)
        start_time2 = datetime.now()
        end_time2 = start_time2 + timedelta(seconds=45)

        dispatcher.fix_history["issue-1"] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.FAILED,
                start_time=start_time1,
                end_time=end_time1,
                error_message="Syntax error in fix",
                files_modified=["file1.py"]
            ),
            FixAttempt(
                attempt_number=2,
                status=FixAttemptStatus.SUCCESS,
                start_time=start_time2,
                end_time=end_time2,
                files_modified=["file1.py", "file2.py"]
            )
        ]

        # Get stats
        stats = dispatcher.get_iteration_stats("issue-1")
        assert stats is not None
        assert stats["issue_id"] == "issue-1"
        assert stats["total_attempts"] == 2
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 1
        assert stats["average_duration"] == 37.5  # (30 + 45) / 2
        assert set(stats["total_files_modified"]) == {"file1.py", "file2.py"}
        assert "syntax" in stats["error_patterns"]
        assert stats["error_patterns"]["syntax"] == 1

    def test_get_iteration_stats_not_found(self):
        """Test getting stats for non-existent issue"""
        dispatcher = FixAgentDispatcher()

        stats = dispatcher.get_iteration_stats("non-existent")
        assert stats is None

    def test_classify_error(self):
        """Test error classification"""
        dispatcher = FixAgentDispatcher()

        # Test various error types
        assert dispatcher._classify_error("Operation timed out") == "timeout"
        assert dispatcher._classify_error("Permission denied") == "permission"
        assert dispatcher._classify_error("Access denied to file") == "permission"
        assert dispatcher._classify_error("Syntax error on line 10") == "syntax"
        assert dispatcher._classify_error("Failed to parse JSON") == "syntax"
        assert dispatcher._classify_error("Module not found: numpy") == "import"
        assert dispatcher._classify_error("Import error: cannot import") == "import"
        assert dispatcher._classify_error("Test failed: assertion error") == "test_failure"
        assert dispatcher._classify_error("Unit test failure") == "test_failure"
        assert dispatcher._classify_error("Compilation failed") == "build"
        assert dispatcher._classify_error("Build error occurred") == "build"
        assert dispatcher._classify_error("Some other error") == "other"

    def test_cancel_fix_active(self):
        """Test cancelling an active fix"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Add to active fixes
        dispatcher.active_fixes[issue.issue_id] = issue

        # Add an in-progress attempt
        dispatcher.fix_history[issue.issue_id] = [
            FixAttempt(
                attempt_number=1,
                status=FixAttemptStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
        ]

        # Cancel the fix
        result = dispatcher.cancel_fix(issue.issue_id)
        assert result is True

        # Check it was removed from active
        assert issue.issue_id not in dispatcher.active_fixes

        # Check attempt was marked as failed
        attempt = dispatcher.fix_history[issue.issue_id][0]
        assert attempt.status == FixAttemptStatus.FAILED
        assert attempt.error_message == "Cancelled by user"
        assert attempt.end_time is not None

    def test_cancel_fix_not_found(self):
        """Test cancelling non-existent fix"""
        dispatcher = FixAgentDispatcher()

        result = dispatcher.cancel_fix("non-existent")
        assert result is False

    def test_cancel_fix_removes_from_retry_queue(self):
        """Test that cancelling removes from retry queue"""
        dispatcher = FixAgentDispatcher()

        issue1 = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue 1",
            file_path="/test/file1.py"
        )

        issue2 = IssueContext(
            issue_id="issue-2",
            severity="high",
            issue_type="bug",
            description="Test issue 2",
            file_path="/test/file2.py"
        )

        # Add both to retry queue
        retry_time = datetime.now() + timedelta(seconds=10)
        dispatcher.retry_queue = [(retry_time, issue1), (retry_time, issue2)]

        # Add issue1 to active
        dispatcher.active_fixes[issue1.issue_id] = issue1

        # Cancel issue1
        dispatcher.cancel_fix(issue1.issue_id)

        # Check only issue2 remains in retry queue
        assert len(dispatcher.retry_queue) == 1
        assert dispatcher.retry_queue[0][1].issue_id == issue2.issue_id

    def test_process_retries(self):
        """Test processing retries when timer fires"""
        dispatcher = FixAgentDispatcher()

        # Mock dispatch_fix_agent
        dispatched_issues = []
        def mock_dispatch(issue, callback=None):
            dispatched_issues.append(issue.issue_id)
            return f"msg-{issue.issue_id}"

        dispatcher.dispatch_fix_agent = mock_dispatch

        # Add issues to retry queue - some ready, some not
        now = datetime.now()
        issue1 = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Ready for retry",
            file_path="/test/file1.py"
        )
        issue2 = IssueContext(
            issue_id="issue-2",
            severity="high",
            issue_type="bug",
            description="Also ready",
            file_path="/test/file2.py"
        )
        issue3 = IssueContext(
            issue_id="issue-3",
            severity="high",
            issue_type="bug",
            description="Not ready yet",
            file_path="/test/file3.py"
        )

        dispatcher.retry_queue = [
            (now - timedelta(seconds=1), issue1),  # Ready
            (now - timedelta(seconds=1), issue2),  # Ready
            (now + timedelta(seconds=60), issue3)  # Not ready
        ]

        # Process retries
        dispatcher._process_retries()

        # Check that ready issues were dispatched
        assert len(dispatched_issues) == 2
        assert "issue-1" in dispatched_issues
        assert "issue-2" in dispatched_issues

        # Check that not-ready issue remains in queue
        assert len(dispatcher.retry_queue) == 1
        assert dispatcher.retry_queue[0][1].issue_id == "issue-3"

    def test_verify_fix_success(self):
        """Test successful fix verification"""
        config = FixAgentDispatcherConfig(enable_verification=True)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.SUCCESS,
            start_time=datetime.now(),
            files_modified=[tmp_path]
        )

        try:
            # Verify the fix
            results = dispatcher.verify_fix(issue, attempt)

            assert results["success"] is True
            assert "file_existence" in results["checks_performed"]
            assert len(results["errors"]) == 0
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_verify_fix_file_not_exists(self):
        """Test verification fails when file doesn't exist"""
        config = FixAgentDispatcherConfig(enable_verification=True)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.SUCCESS,
            start_time=datetime.now(),
            files_modified=["/non/existent/file.py"]
        )

        # Verify the fix
        results = dispatcher.verify_fix(issue, attempt)

        assert results["success"] is False
        assert len(results["errors"]) > 0
        assert "no longer exists" in results["errors"][0]

    def test_verify_fix_no_files_modified(self):
        """Test verification fails when no files were modified"""
        config = FixAgentDispatcherConfig(enable_verification=True)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.SUCCESS,
            start_time=datetime.now(),
            files_modified=[]
        )

        # Verify the fix
        results = dispatcher.verify_fix(issue, attempt)

        assert results["success"] is False
        assert len(results["errors"]) > 0
        assert "No files were modified" in results["errors"][0]

    def test_verify_fix_with_custom_verifier(self):
        """Test verification with custom verifier"""
        config = FixAgentDispatcherConfig(enable_verification=True)
        dispatcher = FixAgentDispatcher(config)

        # Set up custom verifier
        def custom_verifier(issue, attempt):
            return {
                "success": True,
                "custom_check": "passed"
            }

        dispatcher.set_fix_verifier(custom_verifier)

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.SUCCESS,
            start_time=datetime.now(),
            files_modified=[tmp_path]
        )

        try:
            # Verify the fix
            results = dispatcher.verify_fix(issue, attempt)

            assert results["success"] is True
            assert "custom_verifier" in results["checks_performed"]
            assert "custom_verification" in results
            assert results["custom_verification"]["custom_check"] == "passed"
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_verify_fix_custom_verifier_fails(self):
        """Test when custom verifier fails"""
        config = FixAgentDispatcherConfig(enable_verification=True)
        dispatcher = FixAgentDispatcher(config)

        # Set up failing custom verifier
        def custom_verifier(issue, attempt):
            return {
                "success": False,
                "error": "Custom check failed"
            }

        dispatcher.set_fix_verifier(custom_verifier)

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.SUCCESS,
            start_time=datetime.now(),
            files_modified=[tmp_path]
        )

        try:
            # Verify the fix
            results = dispatcher.verify_fix(issue, attempt)

            assert results["success"] is False
            assert len(results["errors"]) > 0
            assert "Custom check failed" in results["errors"][0]
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_handle_fix_error_timeout(self):
        """Test handling timeout errors"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        error = TimeoutError("Operation timed out")

        result = dispatcher.handle_fix_error(issue, error)

        assert result["error_type"] == "TimeoutError"
        assert result["recovery_action"] == "retry_with_extended_timeout"
        assert result["issue_id"] == "issue-1"

    def test_handle_fix_error_permission(self):
        """Test handling permission errors"""
        dispatcher = FixAgentDispatcher()

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        error = PermissionError("Permission denied")

        result = dispatcher.handle_fix_error(issue, error)

        assert result["error_type"] == "PermissionError"
        assert result["recovery_action"] == "escalate_permissions"

    def test_validate_fix_request_success(self):
        """Test successful validation of fix request"""
        dispatcher = FixAgentDispatcher()

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path=tmp_path
        )

        try:
            is_valid, error_msg = dispatcher.validate_fix_request(issue)
            assert is_valid is True
            assert error_msg is None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_fix_request_missing_fields(self):
        """Test validation fails for missing required fields"""
        dispatcher = FixAgentDispatcher()

        # Missing issue_id
        issue = IssueContext(
            issue_id="",
            severity="high",
            issue_type="bug",
            description="Test",
            file_path="/test/file.py"
        )

        is_valid, error_msg = dispatcher.validate_fix_request(issue)
        assert is_valid is False
        assert "Issue ID is required" in error_msg

        # Missing file_path
        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test",
            file_path=""
        )

        is_valid, error_msg = dispatcher.validate_fix_request(issue)
        assert is_valid is False
        assert "File path is required" in error_msg

    def test_validate_fix_request_invalid_severity(self):
        """Test validation fails for invalid severity"""
        dispatcher = FixAgentDispatcher()

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        issue = IssueContext(
            issue_id="issue-1",
            severity="extreme",  # Invalid
            issue_type="bug",
            description="Test",
            file_path=tmp_path
        )

        try:
            is_valid, error_msg = dispatcher.validate_fix_request(issue)
            assert is_valid is False
            assert "Invalid severity" in error_msg
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fix_response_with_verification(self):
        """Test fix response handling with verification enabled"""
        config = FixAgentDispatcherConfig(enable_verification=True)
        dispatcher = FixAgentDispatcher(config)

        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.IN_PROGRESS,
            start_time=datetime.now()
        )

        # Track callbacks
        success_callback_called = False
        def on_success(iss, att):
            nonlocal success_callback_called
            success_callback_called = True

        dispatcher.on_fix_complete = on_success
        dispatcher.active_fixes[issue.issue_id] = issue
        dispatcher.fix_history[issue.issue_id] = [attempt]

        # Create success response
        response = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id=f"fix-{issue.issue_id}",
                parent_session="",
                files_modified=[tmp_path],
                project_path=""
            ),
            success_criteria=SuccessCriteria([], []),
            callback=CallbackInfo("test", 0),
            message_id="msg-123"
        )

        try:
            # Handle the response
            dispatcher._handle_fix_response(issue, attempt, response)

            # Check that verification was performed
            assert attempt.verification_results is not None
            assert attempt.status == FixAttemptStatus.SUCCESS
            assert success_callback_called is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fix_response_verification_fails(self):
        """Test fix response when verification fails"""
        config = FixAgentDispatcherConfig(enable_verification=True, max_attempts=1)
        dispatcher = FixAgentDispatcher(config)

        issue = IssueContext(
            issue_id="issue-1",
            severity="high",
            issue_type="bug",
            description="Test issue",
            file_path="/test/file.py"
        )

        attempt = FixAttempt(
            attempt_number=1,
            status=FixAttemptStatus.IN_PROGRESS,
            start_time=datetime.now()
        )

        # Track callbacks
        verification_failed_called = False
        def on_verification_failed(iss, att, results):
            nonlocal verification_failed_called
            verification_failed_called = True

        dispatcher.on_verification_failed = on_verification_failed
        dispatcher.active_fixes[issue.issue_id] = issue
        dispatcher.fix_history[issue.issue_id] = [attempt]

        # Create success response with non-existent file
        response = AgentMessage(
            message_type=MessageType.TASK_COMPLETE,
            agent_type=AgentType.FIX,
            context=MessageContext(
                task_id=f"fix-{issue.issue_id}",
                parent_session="",
                files_modified=["/non/existent/file.py"],
                project_path=""
            ),
            success_criteria=SuccessCriteria([], []),
            callback=CallbackInfo("test", 0),
            message_id="msg-123"
        )

        # Handle the response
        dispatcher._handle_fix_response(issue, attempt, response)

        # Check that verification failed and max retries exceeded
        assert attempt.verification_results is not None
        assert attempt.status == FixAttemptStatus.EXCEEDED_RETRIES  # Max attempts=1, so it's exceeded
        assert verification_failed_called is True
        assert attempt.error_message is not None
        assert "Verification failed" in attempt.error_message

    def test_dispatch_fix_callback_on_invalid(self):
        """Test callback is invoked on invalid fix request"""
        dispatcher = FixAgentDispatcher()

        # Invalid issue (missing file path)
        issue = IssueContext(
            issue_id="issue-1",
            severity="critical",
            issue_type="bug",
            description="Test issue",
            file_path=""  # Invalid - empty path
        )

        # Track callback
        callback_called = False
        callback_response = None
        def callback(response):
            nonlocal callback_called, callback_response
            callback_called = True
            callback_response = response

        # Try to dispatch
        result = dispatcher.dispatch_fix_agent(issue, callback)

        # Should return None but callback should be called
        assert result is None
        assert callback_called is True
        assert callback_response is not None
        assert callback_response.message_type == MessageType.ERROR
        assert "File path is required" in callback_response.payload["error"]
