"""
Unit tests for Fix Verification Workflow
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cadence.fix_verification_workflow import (
    FixVerificationWorkflow, VerificationResult, VerificationConfig, IssueComparison,
    VerificationStatus, VerificationMode, verify_fix_quick
)
from cadence.review_result_parser import ParsedIssue, IssueSeverity, IssueCategory
from cadence.scope_validator import FixProposal, TaskScope
from cadence.code_review_agent import CodeReviewAgent, ReviewResult


class TestVerificationStatus:
    """Test VerificationStatus enum"""

    def test_status_values(self):
        """Test status enum values"""
        assert VerificationStatus.PENDING == "pending"
        assert VerificationStatus.IN_PROGRESS == "in_progress"
        assert VerificationStatus.VERIFIED == "verified"
        assert VerificationStatus.FAILED == "failed"
        assert VerificationStatus.REGRESSION_DETECTED == "regression_detected"
        assert VerificationStatus.PARTIAL_SUCCESS == "partial_success"


class TestVerificationMode:
    """Test VerificationMode enum"""

    def test_mode_values(self):
        """Test mode enum values"""
        assert VerificationMode.FULL_REVIEW == "full_review"
        assert VerificationMode.TARGETED_REVIEW == "targeted_review"
        assert VerificationMode.ISSUE_SPECIFIC == "issue_specific"
        assert VerificationMode.REGRESSION_ONLY == "regression_only"


class TestIssueComparison:
    """Test IssueComparison dataclass"""

    def test_empty_comparison(self):
        """Test empty issue comparison"""
        comparison = IssueComparison()

        assert len(comparison.resolved_issues) == 0
        assert len(comparison.persisting_issues) == 0
        assert len(comparison.new_issues) == 0
        assert len(comparison.modified_issues) == 0
        assert comparison.resolution_rate == 1.0  # No issues = 100% resolved
        assert comparison.regression_count == 0
        assert not comparison.has_regressions

    def test_comparison_with_issues(self):
        """Test comparison with various issues"""
        resolved_issue = ParsedIssue(
            severity=IssueSeverity.HIGH,
            category=IssueCategory.BUG,
            description="Resolved bug"
        )

        persisting_issue = ParsedIssue(
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.PERFORMANCE,
            description="Still exists"
        )

        new_issue = ParsedIssue(
            severity=IssueSeverity.LOW,
            category=IssueCategory.STYLE,
            description="New regression"
        )

        comparison = IssueComparison(
            resolved_issues=[resolved_issue],
            persisting_issues=[persisting_issue],
            new_issues=[new_issue]
        )

        # Test metrics
        assert comparison.resolution_rate == 0.5  # 1 resolved out of 2 original
        assert comparison.regression_count == 1
        assert comparison.has_regressions

    def test_resolution_rate_calculation(self):
        """Test resolution rate calculation edge cases"""
        # Only resolved issues
        comparison = IssueComparison(
            resolved_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 1"),
                ParsedIssue(IssueSeverity.MEDIUM, IssueCategory.BUG, "Fixed 2")
            ]
        )
        assert comparison.resolution_rate == 1.0

        # Only persisting issues
        comparison = IssueComparison(
            persisting_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here")
            ]
        )
        assert comparison.resolution_rate == 0.0


class TestVerificationResult:
    """Test VerificationResult dataclass"""

    def test_default_verification_result(self):
        """Test default verification result creation"""
        comparison = IssueComparison()

        result = VerificationResult(
            verification_id="test-123",
            task_id="task-1",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.TARGETED_REVIEW,
            issue_comparison=comparison
        )

        assert result.verification_id == "test-123"
        assert result.task_id == "task-1"
        assert result.status == VerificationStatus.VERIFIED
        assert result.mode == VerificationMode.TARGETED_REVIEW
        assert result.verification_timestamp is not None  # Auto-set in __post_init__
        assert result.is_successful
        assert not result.requires_attention

    def test_verification_result_properties(self):
        """Test verification result properties"""
        # Successful result
        comparison = IssueComparison()
        result = VerificationResult(
            verification_id="test-1",
            task_id="task-1",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.FULL_REVIEW,
            issue_comparison=comparison
        )

        assert result.is_successful
        assert not result.requires_attention

        # Failed result
        result.status = VerificationStatus.FAILED
        assert not result.is_successful
        assert result.requires_attention

        # Regression detected
        result.status = VerificationStatus.REGRESSION_DETECTED
        assert not result.is_successful
        assert result.requires_attention

        # Partial success
        result.status = VerificationStatus.PARTIAL_SUCCESS
        assert result.is_successful
        assert not result.requires_attention

        # Result with regressions in comparison
        comparison_with_regressions = IssueComparison(
            new_issues=[ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Regression")]
        )
        result = VerificationResult(
            verification_id="test-2",
            task_id="task-2",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.FULL_REVIEW,
            issue_comparison=comparison_with_regressions
        )

        assert result.is_successful  # Status is still VERIFIED
        assert result.requires_attention  # But has regressions


class TestVerificationConfig:
    """Test VerificationConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VerificationConfig()

        assert config.default_mode == VerificationMode.TARGETED_REVIEW
        assert config.enable_regression_detection is True
        assert config.enable_scope_validation is True
        assert config.minimum_resolution_rate == 0.8
        assert config.max_acceptable_regressions == 0
        assert config.verification_timeout_ms == 300000
        assert config.issue_similarity_threshold == 0.8
        assert config.ignore_minor_variations is True
        assert config.include_detailed_metrics is True
        assert config.generate_recommendations is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = VerificationConfig(
            default_mode=VerificationMode.FULL_REVIEW,
            minimum_resolution_rate=0.9,
            max_acceptable_regressions=2,
            verification_timeout_ms=600000,
            enable_scope_validation=False
        )

        assert config.default_mode == VerificationMode.FULL_REVIEW
        assert config.minimum_resolution_rate == 0.9
        assert config.max_acceptable_regressions == 2
        assert config.verification_timeout_ms == 600000
        assert config.enable_scope_validation is False


class TestFixVerificationWorkflow:
    """Test FixVerificationWorkflow class"""

    def test_initialization_default(self):
        """Test workflow initialization with defaults"""
        workflow = FixVerificationWorkflow()

        assert workflow.config is not None
        assert isinstance(workflow.config, VerificationConfig)
        assert workflow.code_review_agent is not None
        assert workflow.result_processor is not None
        assert workflow.scope_validator is not None
        assert len(workflow._verification_history) == 0

    def test_initialization_custom_components(self):
        """Test workflow initialization with custom components"""
        config = VerificationConfig(minimum_resolution_rate=0.9)
        mock_review_agent = Mock(spec=CodeReviewAgent)
        mock_processor = Mock()
        mock_validator = Mock()

        workflow = FixVerificationWorkflow(
            config=config,
            code_review_agent=mock_review_agent,
            result_processor=mock_processor,
            scope_validator=mock_validator
        )

        assert workflow.config.minimum_resolution_rate == 0.9
        assert workflow.code_review_agent is mock_review_agent
        assert workflow.result_processor is mock_processor
        assert workflow.scope_validator is mock_validator

    @patch('time.time')
    def test_verify_fix_success(self, mock_time):
        """Test successful fix verification"""
        mock_time.return_value = 1000.0

        # Setup mocks
        mock_review_agent = Mock(spec=CodeReviewAgent)
        mock_processor = Mock()
        mock_validator = Mock()

        # Configure review agent to return issues
        mock_review_result = Mock()
        mock_review_result.success = True
        mock_review_result.review_output = "Mock review output"
        mock_review_agent.review_files.return_value = mock_review_result

        # Configure processor to return parsed issues
        mock_category_result = Mock()
        mock_category_result.blocking_issues = []
        mock_category_result.required_issues = []
        mock_category_result.recommended_issues = []
        mock_category_result.optional_issues = []
        mock_category_result.informational_issues = []
        mock_processor.process_review_result.return_value = mock_category_result

        # Configure validator
        mock_scope_result = Mock()
        mock_scope_result.is_successful = True
        mock_validator.validate_fix_proposal.return_value = (mock_scope_result, [])

        workflow = FixVerificationWorkflow(
            code_review_agent=mock_review_agent,
            result_processor=mock_processor,
            scope_validator=mock_validator
        )

        # Create test data
        before_issues = [
            ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Original bug")
        ]

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test.py"},
            estimated_line_changes={"test.py": 10}
        )

        # Execute verification
        result = workflow.verify_fix("test-task", before_issues, fix_proposal)

        # Assertions
        assert isinstance(result, VerificationResult)
        assert result.task_id == "test-task"
        assert result.status in VerificationStatus
        assert result.verification_id in workflow._verification_history
        assert result.execution_time_ms is not None

        # Verify mocks were called
        mock_review_agent.review_files.assert_called_once()
        mock_processor.process_review_result.assert_called_once()

    def test_verify_fix_scope_validation_failure(self):
        """Test fix verification with scope validation failure"""
        mock_validator = Mock()

        # Configure validator to fail
        mock_scope_result = Mock()
        mock_scope_result.is_successful = False
        mock_scope_result.error_message = "Scope violation detected"

        # Mock the _validate_fix_scope method to return the failing result
        workflow = FixVerificationWorkflow(scope_validator=mock_validator)
        workflow._validate_fix_scope = Mock(return_value=mock_scope_result)

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test.py"}
        )

        result = workflow.verify_fix("test-task", [], fix_proposal)

        assert result.status == VerificationStatus.FAILED
        assert "Scope validation failed" in result.error_message

    def test_verify_fix_review_failure(self):
        """Test fix verification with review failure"""
        mock_review_agent = Mock(spec=CodeReviewAgent)

        # Configure review agent to fail
        mock_review_result = Mock()
        mock_review_result.success = False
        mock_review_result.error_message = "Review failed"
        mock_review_agent.review_files.return_value = mock_review_result

        workflow = FixVerificationWorkflow(code_review_agent=mock_review_agent)
        workflow._validate_fix_scope = Mock(return_value=Mock(is_successful=True))

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test.py"}
        )

        result = workflow.verify_fix("test-task", [], fix_proposal)

        # Should handle gracefully - review failure doesn't fail verification
        assert isinstance(result, VerificationResult)

    def test_verify_fix_exception_handling(self):
        """Test fix verification exception handling"""
        workflow = FixVerificationWorkflow()

        # Mock a method that would cause an exception in the main flow
        workflow._compare_issues = Mock(side_effect=Exception("Comparison error"))

        # Mock successful earlier steps
        workflow._validate_fix_scope = Mock(return_value=Mock(is_successful=True))
        workflow._perform_verification_review = Mock(return_value=[])

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test.py"}
        )

        result = workflow.verify_fix("test-task", [], fix_proposal)

        assert result.status == VerificationStatus.FAILED
        assert "Comparison error" in result.error_message

    def test_verify_targeted_issues(self):
        """Test targeted issue verification"""
        workflow = FixVerificationWorkflow()

        # Mock the main verify_fix method
        expected_result = VerificationResult(
            verification_id="test-123",
            task_id="test-task",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.ISSUE_SPECIFIC,
            issue_comparison=IssueComparison()
        )

        workflow.verify_fix = Mock(return_value=expected_result)

        target_issues = [
            ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Target bug")
        ]

        fix_proposal = FixProposal(task_id="test-task")

        result = workflow.verify_targeted_issues("test-task", target_issues, fix_proposal)

        assert result == expected_result
        workflow.verify_fix.assert_called_once_with(
            task_id="test-task",
            before_issues=target_issues,
            fix_proposal=fix_proposal,
            verification_mode=VerificationMode.ISSUE_SPECIFIC
        )

    def test_check_for_regressions(self):
        """Test regression checking"""
        workflow = FixVerificationWorkflow()

        # Mock the main verify_fix method
        expected_result = VerificationResult(
            verification_id="test-123",
            task_id="test-task",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.REGRESSION_ONLY,
            issue_comparison=IssueComparison()
        )

        workflow.verify_fix = Mock(return_value=expected_result)

        baseline_issues = [
            ParsedIssue(IssueSeverity.MEDIUM, IssueCategory.PERFORMANCE, "Baseline issue")
        ]

        fix_proposal = FixProposal(task_id="test-task")

        result = workflow.check_for_regressions("test-task", fix_proposal, baseline_issues)

        assert result == expected_result
        workflow.verify_fix.assert_called_once_with(
            task_id="test-task",
            before_issues=baseline_issues,
            fix_proposal=fix_proposal,
            verification_mode=VerificationMode.REGRESSION_ONLY
        )

    def test_check_for_regressions_no_baseline(self):
        """Test regression checking without baseline issues"""
        workflow = FixVerificationWorkflow()
        workflow.verify_fix = Mock()

        fix_proposal = FixProposal(task_id="test-task")

        workflow.check_for_regressions("test-task", fix_proposal)

        # Should pass empty list as before_issues
        workflow.verify_fix.assert_called_once_with(
            task_id="test-task",
            before_issues=[],
            fix_proposal=fix_proposal,
            verification_mode=VerificationMode.REGRESSION_ONLY
        )

    def test_issue_signature_creation(self):
        """Test issue signature creation for comparison"""
        workflow = FixVerificationWorkflow()

        issue1 = ParsedIssue(
            severity=IssueSeverity.HIGH,
            category=IssueCategory.BUG,
            description="Null pointer exception in getUserData",
            file_path="user.py",
            line_number=42
        )

        issue2 = ParsedIssue(
            severity=IssueSeverity.HIGH,
            category=IssueCategory.BUG,
            description="Null pointer exception in getUserData",  # Same description
            file_path="user.py",
            line_number=42
        )

        issue3 = ParsedIssue(
            severity=IssueSeverity.MEDIUM,  # Different severity
            category=IssueCategory.BUG,
            description="Null pointer exception in getUserData",
            file_path="user.py",
            line_number=42
        )

        sig1 = workflow._create_issue_signature(issue1)
        sig2 = workflow._create_issue_signature(issue2)
        sig3 = workflow._create_issue_signature(issue3)

        assert sig1 == sig2  # Same issues should have same signature
        assert sig1 != sig3  # Different severity should give different signature

    def test_compare_issues(self):
        """Test issue comparison logic"""
        workflow = FixVerificationWorkflow()

        # Create test issues
        resolved_issue = ParsedIssue(
            severity=IssueSeverity.HIGH,
            category=IssueCategory.BUG,
            description="Fixed bug",
            file_path="test.py",
            line_number=10
        )

        persisting_issue = ParsedIssue(
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.PERFORMANCE,
            description="Still slow",
            file_path="test.py",
            line_number=20
        )

        new_issue = ParsedIssue(
            severity=IssueSeverity.LOW,
            category=IssueCategory.STYLE,
            description="New formatting issue",
            file_path="test.py",
            line_number=30
        )

        before_issues = [resolved_issue, persisting_issue]
        after_issues = [persisting_issue, new_issue]

        comparison = workflow._compare_issues(before_issues, after_issues)

        assert len(comparison.resolved_issues) == 1
        assert len(comparison.persisting_issues) == 1
        assert len(comparison.new_issues) == 1
        assert comparison.resolution_rate == 0.5  # 1 resolved out of 2 original
        assert comparison.regression_count == 1

    def test_determine_verification_status(self):
        """Test verification status determination logic"""
        config = VerificationConfig(
            minimum_resolution_rate=0.8,
            max_acceptable_regressions=0,
            enable_regression_detection=True
        )
        workflow = FixVerificationWorkflow(config)

        # Test VERIFIED status
        comparison = IssueComparison(
            resolved_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 1"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 2"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 3"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 4")
            ],
            persisting_issues=[
                ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "Still here")
            ]
        )
        status = workflow._determine_verification_status(comparison)
        assert status == VerificationStatus.VERIFIED

        # Test REGRESSION_DETECTED status
        comparison = IssueComparison(
            resolved_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed")
            ],
            new_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Regression 1"),
                ParsedIssue(IssueSeverity.MEDIUM, IssueCategory.BUG, "Regression 2")
            ]
        )
        status = workflow._determine_verification_status(comparison)
        assert status == VerificationStatus.REGRESSION_DETECTED

        # Test FAILED status (low resolution rate)
        comparison = IssueComparison(
            persisting_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 1"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 2"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 3"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 4")
            ],
            resolved_issues=[
                ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "Fixed one")
            ]
        )
        status = workflow._determine_verification_status(comparison)
        assert status == VerificationStatus.FAILED

        # Test PARTIAL_SUCCESS status (good resolution + acceptable regressions)
        config.max_acceptable_regressions = 1
        workflow.config = config

        comparison = IssueComparison(
            resolved_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 1"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 2"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 3"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 4")
            ],
            persisting_issues=[
                ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "Still here")
            ],
            new_issues=[
                ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "Minor regression")
            ]
        )
        status = workflow._determine_verification_status(comparison)
        assert status == VerificationStatus.PARTIAL_SUCCESS

    def test_calculate_success_metrics(self):
        """Test success metrics calculation"""
        config = VerificationConfig(include_detailed_metrics=True)
        workflow = FixVerificationWorkflow(config)

        comparison = IssueComparison(
            resolved_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 1"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed 2")
            ],
            persisting_issues=[
                ParsedIssue(IssueSeverity.MEDIUM, IssueCategory.PERFORMANCE, "Still here")
            ],
            new_issues=[
                ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "New issue")
            ]
        )

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"file1.py", "file2.py"},
            files_to_create={"file3.py"},
            estimated_line_changes={"file1.py": 50, "file2.py": 30, "file3.py": 20}
        )

        metrics = workflow._calculate_success_metrics(comparison, fix_proposal)

        assert metrics["resolution_rate"] == 2/3  # 2 resolved out of 3 original
        assert metrics["issues_resolved"] == 2
        assert metrics["issues_persisting"] == 1
        assert metrics["new_issues_introduced"] == 1
        assert metrics["regression_count"] == 1
        assert metrics["files_modified"] == 3
        assert metrics["lines_changed"] == 100
        assert "verification_efficiency" in metrics
        assert "regression_rate" in metrics

    def test_calculate_success_metrics_disabled(self):
        """Test success metrics when disabled"""
        config = VerificationConfig(include_detailed_metrics=False)
        workflow = FixVerificationWorkflow(config)

        comparison = IssueComparison()
        fix_proposal = FixProposal(task_id="test-task")

        metrics = workflow._calculate_success_metrics(comparison, fix_proposal)

        assert metrics == {}

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        config = VerificationConfig(generate_recommendations=True)
        workflow = FixVerificationWorkflow(config)

        # Test REGRESSION_DETECTED recommendations
        comparison = IssueComparison(
            new_issues=[
                ParsedIssue(IssueSeverity.CRITICAL, IssueCategory.SECURITY, "Critical regression"),
                ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "Minor regression")
            ]
        )

        fix_proposal = FixProposal(
            task_id="test-task",
            estimated_line_changes={"file.py": 50}
        )

        recommendations = workflow._generate_recommendations(
            comparison, VerificationStatus.REGRESSION_DETECTED, fix_proposal
        )

        assert any("Regressions detected" in rec for rec in recommendations)
        assert any("high-severity regressions" in rec for rec in recommendations)

        # Test FAILED recommendations
        comparison = IssueComparison(
            resolved_issues=[ParsedIssue(IssueSeverity.LOW, IssueCategory.STYLE, "Fixed one")],
            persisting_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 1"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 2"),
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Still here 3")
            ]
        )

        recommendations = workflow._generate_recommendations(
            comparison, VerificationStatus.FAILED, fix_proposal
        )

        assert any("resolution rate" in rec for rec in recommendations)

        # Test VERIFIED recommendations
        comparison = IssueComparison(
            resolved_issues=[
                ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed")
            ]
        )

        recommendations = workflow._generate_recommendations(
            comparison, VerificationStatus.VERIFIED, fix_proposal
        )

        assert any("successfully verified" in rec for rec in recommendations)

        # Test large change recommendations
        large_fix_proposal = FixProposal(
            task_id="test-task",
            estimated_line_changes={"file.py": 300}
        )

        recommendations = workflow._generate_recommendations(
            comparison, VerificationStatus.VERIFIED, large_fix_proposal
        )

        assert any("breaking large fix" in rec for rec in recommendations)

    def test_generate_recommendations_disabled(self):
        """Test recommendation generation when disabled in full workflow"""
        config = VerificationConfig(
            generate_recommendations=False,
            enable_scope_validation=False  # Disable to simplify test
        )
        workflow = FixVerificationWorkflow(config)

        # Mock the review process to avoid complexity
        workflow._perform_verification_review = Mock(return_value=[])

        fix_proposal = FixProposal(task_id="test-task")

        result = workflow.verify_fix("test-task", [], fix_proposal)

        # Recommendations should be empty when disabled
        assert result.recommendations == []

    @patch('time.time')
    def test_verification_history(self, mock_time):
        """Test verification history tracking"""
        # Mock time to return different values for unique IDs - need enough for all calls
        mock_time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0]

        workflow = FixVerificationWorkflow()

        # Mock successful verification
        workflow._validate_fix_scope = Mock(return_value=Mock(is_successful=True))
        workflow._perform_verification_review = Mock(return_value=[])

        fix_proposal = FixProposal(task_id="test-task")

        # Perform multiple verifications
        result1 = workflow.verify_fix("task-1", [], fix_proposal)
        result2 = workflow.verify_fix("task-2", [], fix_proposal)
        result3 = workflow.verify_fix("task-1", [], fix_proposal)  # Same task again

        # Check history
        all_history = workflow.get_verification_history()
        assert len(all_history) == 3

        # Check task-specific history
        task1_history = workflow.get_verification_history("task-1")
        assert len(task1_history) == 2

        task2_history = workflow.get_verification_history("task-2")
        assert len(task2_history) == 1

        nonexistent_history = workflow.get_verification_history("nonexistent")
        assert len(nonexistent_history) == 0

    def test_verification_summary(self):
        """Test verification summary generation"""
        workflow = FixVerificationWorkflow()

        # Add some mock results to history
        comparison1 = IssueComparison(
            resolved_issues=[ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Fixed")]
        )
        result1 = VerificationResult(
            verification_id="test-1",
            task_id="task-1",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.TARGETED_REVIEW,
            issue_comparison=comparison1,
            verification_timestamp="2023-01-01T10:00:00Z"
        )

        comparison2 = IssueComparison(
            persisting_issues=[ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Unresolved")],
            new_issues=[ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Regression")]
        )
        result2 = VerificationResult(
            verification_id="test-2",
            task_id="task-1",
            status=VerificationStatus.REGRESSION_DETECTED,
            mode=VerificationMode.FULL_REVIEW,
            issue_comparison=comparison2,
            verification_timestamp="2023-01-01T11:00:00Z"
        )

        workflow._verification_history = {
            "test-1": result1,
            "test-2": result2
        }

        summary = workflow.get_verification_summary("task-1")

        assert summary["task_id"] == "task-1"
        assert summary["verification_count"] == 2
        assert summary["success_rate"] == 0.5  # 1 successful out of 2
        assert summary["latest_status"] == "regression_detected"
        assert summary["latest_resolution_rate"] == 0.0
        assert summary["total_regressions_detected"] == 1
        assert summary["requires_attention"] is True

    def test_verification_summary_no_history(self):
        """Test verification summary with no history"""
        workflow = FixVerificationWorkflow()

        summary = workflow.get_verification_summary("nonexistent-task")

        assert summary["task_id"] == "nonexistent-task"
        assert summary["verification_count"] == 0


class TestPerformVerificationReview:
    """Test the _perform_verification_review method"""

    def test_perform_verification_review_success(self):
        """Test successful verification review"""
        mock_review_agent = Mock(spec=CodeReviewAgent)
        mock_processor = Mock()

        # Configure mocks
        mock_review_result = Mock()
        mock_review_result.success = True
        mock_review_result.review_output = "Mock review output"
        mock_review_agent.review_files.return_value = mock_review_result

        mock_category_result = Mock()
        expected_issues = [
            ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Issue 1"),
            ParsedIssue(IssueSeverity.MEDIUM, IssueCategory.PERFORMANCE, "Issue 2")
        ]
        mock_category_result.blocking_issues = [expected_issues[0]]
        mock_category_result.required_issues = []
        mock_category_result.recommended_issues = [expected_issues[1]]
        mock_category_result.optional_issues = []
        mock_category_result.informational_issues = []
        mock_processor.process_review_result.return_value = mock_category_result

        workflow = FixVerificationWorkflow(
            code_review_agent=mock_review_agent,
            result_processor=mock_processor
        )

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test1.py", "test2.py"}
        )

        result = workflow._perform_verification_review(fix_proposal, VerificationMode.TARGETED_REVIEW)

        assert len(result) == 2
        assert result == expected_issues

        # Verify calls (sorted file paths for consistency)
        mock_review_agent.review_files.assert_called_once_with(
            file_paths=["test1.py", "test2.py"],  # sorted order
            severity_filter="all"
        )
        mock_processor.process_review_result.assert_called_once_with("Mock review output")

    def test_perform_verification_review_no_files(self):
        """Test verification review with no files"""
        workflow = FixVerificationWorkflow()

        fix_proposal = FixProposal(task_id="test-task")  # No files

        result = workflow._perform_verification_review(fix_proposal, VerificationMode.TARGETED_REVIEW)

        assert result == []

    def test_perform_verification_review_failure(self):
        """Test verification review with review failure"""
        mock_review_agent = Mock(spec=CodeReviewAgent)

        # Configure review to fail
        mock_review_result = Mock()
        mock_review_result.success = False
        mock_review_result.error_message = "Review failed"
        mock_review_agent.review_files.return_value = mock_review_result

        workflow = FixVerificationWorkflow(code_review_agent=mock_review_agent)

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test.py"}
        )

        result = workflow._perform_verification_review(fix_proposal, VerificationMode.FULL_REVIEW)

        assert result == []

    def test_perform_verification_review_exception(self):
        """Test verification review with exception"""
        mock_review_agent = Mock(spec=CodeReviewAgent)
        mock_review_agent.review_files.side_effect = Exception("Review error")

        workflow = FixVerificationWorkflow(code_review_agent=mock_review_agent)

        fix_proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"test.py"}
        )

        result = workflow._perform_verification_review(fix_proposal, VerificationMode.REGRESSION_ONLY)

        assert result == []


class TestConvenienceFunction:
    """Test the convenience function"""

    @patch('cadence.fix_verification_workflow.FixVerificationWorkflow')
    def test_verify_fix_quick(self, mock_workflow_class):
        """Test verify_fix_quick convenience function"""
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow

        expected_result = VerificationResult(
            verification_id="test-123",
            task_id="test-task",
            status=VerificationStatus.VERIFIED,
            mode=VerificationMode.TARGETED_REVIEW,
            issue_comparison=IssueComparison()
        )
        mock_workflow.verify_fix.return_value = expected_result

        before_issues = [
            ParsedIssue(IssueSeverity.HIGH, IssueCategory.BUG, "Test issue")
        ]
        files_modified = ["test1.py", "test2.py"]
        estimated_changes = {"test1.py": 10, "test2.py": 5}
        config = VerificationConfig(minimum_resolution_rate=0.9)

        result = verify_fix_quick(
            task_id="test-task",
            before_issues=before_issues,
            files_modified=files_modified,
            estimated_changes=estimated_changes,
            config=config
        )

        # Verify workflow was created with config
        mock_workflow_class.assert_called_once_with(config)

        # Verify verify_fix was called with correct parameters
        mock_workflow.verify_fix.assert_called_once()
        call_args = mock_workflow.verify_fix.call_args

        assert call_args[0][0] == "test-task"  # task_id
        assert call_args[0][1] == before_issues  # before_issues

        # Check the fix_proposal
        fix_proposal = call_args[0][2]
        assert isinstance(fix_proposal, FixProposal)
        assert fix_proposal.task_id == "test-task"
        assert fix_proposal.files_to_modify == {"test1.py", "test2.py"}
        assert fix_proposal.estimated_line_changes == estimated_changes

        assert result == expected_result


if __name__ == "__main__":
    pytest.main([__file__])
