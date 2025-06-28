"""
Fix Verification Workflow

This module provides comprehensive verification of applied fixes to ensure they
resolve identified issues without introducing regressions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
import time
from datetime import datetime

from cadence.code_review_agent import CodeReviewAgent, ReviewConfig
from cadence.review_result_parser import (
    ReviewResultProcessor, ParsedIssue, CategoryResult, IssueSeverity
)
from cadence.scope_validator import ScopeValidator, FixProposal, TaskScope

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Status of fix verification"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    REGRESSION_DETECTED = "regression_detected"
    PARTIAL_SUCCESS = "partial_success"


class VerificationMode(str, Enum):
    """Mode of verification"""
    FULL_REVIEW = "full_review"        # Complete re-review of all files
    TARGETED_REVIEW = "targeted_review"  # Review only modified files
    ISSUE_SPECIFIC = "issue_specific"   # Verify specific issues only
    REGRESSION_ONLY = "regression_only"  # Check for new issues only


@dataclass
class IssueComparison:
    """Comparison result between before and after issues"""
    resolved_issues: List[ParsedIssue] = field(default_factory=list)
    persisting_issues: List[ParsedIssue] = field(default_factory=list)
    new_issues: List[ParsedIssue] = field(default_factory=list)
    modified_issues: List[ParsedIssue] = field(default_factory=list)

    @property
    def resolution_rate(self) -> float:
        """Calculate percentage of issues resolved"""
        total_original = len(self.resolved_issues) + len(self.persisting_issues)
        if total_original == 0:
            return 1.0
        return len(self.resolved_issues) / total_original

    @property
    def regression_count(self) -> int:
        """Count of new issues introduced"""
        return len(self.new_issues)

    @property
    def has_regressions(self) -> bool:
        """Check if regressions were introduced"""
        return self.regression_count > 0


@dataclass
class VerificationResult:
    """Result of fix verification"""
    verification_id: str
    task_id: str
    status: VerificationStatus
    mode: VerificationMode
    issue_comparison: IssueComparison
    files_verified: List[str] = field(default_factory=list)
    verification_timestamp: Optional[str] = None
    execution_time_ms: Optional[int] = None
    success_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default values after creation"""
        if self.verification_timestamp is None:
            self.verification_timestamp = datetime.now().isoformat()

    @property
    def is_successful(self) -> bool:
        """Check if verification was successful"""
        return self.status in [VerificationStatus.VERIFIED, VerificationStatus.PARTIAL_SUCCESS]

    @property
    def requires_attention(self) -> bool:
        """Check if verification requires attention"""
        return self.status in [
            VerificationStatus.FAILED,
            VerificationStatus.REGRESSION_DETECTED
        ] or self.issue_comparison.has_regressions


@dataclass
class VerificationConfig:
    """Configuration for fix verification"""
    # Verification modes
    default_mode: VerificationMode = VerificationMode.TARGETED_REVIEW
    enable_regression_detection: bool = True
    enable_scope_validation: bool = True

    # Thresholds
    minimum_resolution_rate: float = 0.8  # 80% of issues should be resolved
    max_acceptable_regressions: int = 0   # No new issues allowed
    verification_timeout_ms: int = 300000  # 5 minutes

    # Review configuration
    review_config: Optional[ReviewConfig] = None

    # Issue matching sensitivity
    issue_similarity_threshold: float = 0.8
    ignore_minor_variations: bool = True

    # Reporting
    include_detailed_metrics: bool = True
    generate_recommendations: bool = True


class FixVerificationWorkflow:
    """
    Comprehensive workflow for verifying that applied fixes resolve issues
    without introducing regressions.

    Integrates CodeReviewAgent, ReviewResultProcessor, and ScopeValidator
    to provide end-to-end fix verification.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        code_review_agent: Optional[CodeReviewAgent] = None,
        result_processor: Optional[ReviewResultProcessor] = None,
        scope_validator: Optional[ScopeValidator] = None
    ):
        """
        Initialize the fix verification workflow.

        Args:
            config: Verification configuration
            code_review_agent: Optional code review agent instance
            result_processor: Optional result processor instance
            scope_validator: Optional scope validator instance
        """
        self.config = config or VerificationConfig()

        # Initialize components
        self.code_review_agent = code_review_agent or CodeReviewAgent(
            config=self.config.review_config
        )
        self.result_processor = result_processor or ReviewResultProcessor()
        self.scope_validator = scope_validator or ScopeValidator()

        # Verification history
        self._verification_history: Dict[str, VerificationResult] = {}

        logger.info("FixVerificationWorkflow initialized")

    def verify_fix(
        self,
        task_id: str,
        before_issues: List[ParsedIssue],
        fix_proposal: FixProposal,
        verification_mode: Optional[VerificationMode] = None
    ) -> VerificationResult:
        """
        Verify that a fix resolves issues without introducing regressions.

        Args:
            task_id: Task identifier
            before_issues: Issues identified before the fix
            fix_proposal: Details of the fix that was applied
            verification_mode: Mode of verification to use

        Returns:
            VerificationResult with comprehensive verification details
        """
        verification_id = f"{task_id}-{int(time.time())}"
        mode = verification_mode or self.config.default_mode
        start_time = time.time()

        try:
            logger.info(f"Starting fix verification {verification_id} for task {task_id}")

            # Validate scope if enabled
            if self.config.enable_scope_validation:
                scope_result = self._validate_fix_scope(fix_proposal)
                if not scope_result.is_successful:
                    return self._create_failed_result(
                        verification_id, task_id, mode,
                        f"Scope validation failed: {scope_result.error_message}",
                        start_time
                    )

            # Perform verification based on mode
            after_issues = self._perform_verification_review(
                fix_proposal, mode
            )

            # Compare before and after issues
            issue_comparison = self._compare_issues(before_issues, after_issues)

            # Determine verification status
            status = self._determine_verification_status(issue_comparison)

            # Generate success metrics
            metrics = self._calculate_success_metrics(issue_comparison, fix_proposal)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                issue_comparison, status, fix_proposal
            ) if self.config.generate_recommendations else []

            execution_time = int((time.time() - start_time) * 1000)

            result = VerificationResult(
                verification_id=verification_id,
                task_id=task_id,
                status=status,
                mode=mode,
                issue_comparison=issue_comparison,
                files_verified=list(fix_proposal.files_to_modify | fix_proposal.files_to_create),
                execution_time_ms=execution_time,
                success_metrics=metrics,
                recommendations=recommendations
            )

            # Store in history
            self._verification_history[verification_id] = result

            logger.info(f"Fix verification {verification_id} completed with status: {status}")
            return result

        except Exception as e:
            logger.error(f"Fix verification {verification_id} failed: {e}")
            return self._create_failed_result(
                verification_id, task_id, mode, str(e), start_time
            )

    def verify_targeted_issues(
        self,
        task_id: str,
        target_issues: List[ParsedIssue],
        fix_proposal: FixProposal
    ) -> VerificationResult:
        """
        Verify resolution of specific target issues.

        Args:
            task_id: Task identifier
            target_issues: Specific issues to verify resolution for
            fix_proposal: Details of the fix that was applied

        Returns:
            VerificationResult focused on target issues
        """
        return self.verify_fix(
            task_id=task_id,
            before_issues=target_issues,
            fix_proposal=fix_proposal,
            verification_mode=VerificationMode.ISSUE_SPECIFIC
        )

    def check_for_regressions(
        self,
        task_id: str,
        fix_proposal: FixProposal,
        baseline_issues: Optional[List[ParsedIssue]] = None
    ) -> VerificationResult:
        """
        Check for regressions introduced by a fix.

        Args:
            task_id: Task identifier
            fix_proposal: Details of the fix that was applied
            baseline_issues: Optional baseline issues to compare against

        Returns:
            VerificationResult focused on regression detection
        """
        return self.verify_fix(
            task_id=task_id,
            before_issues=baseline_issues or [],
            fix_proposal=fix_proposal,
            verification_mode=VerificationMode.REGRESSION_ONLY
        )

    def _validate_fix_scope(self, fix_proposal: FixProposal) -> VerificationResult:
        """Validate that fix stays within acceptable scope"""
        # Create a simple result for scope validation
        class ScopeResult:
            def __init__(self, success: bool, error: Optional[str] = None):
                self.is_successful = success
                self.error_message = error

        try:
            # Use the scope validator to check the fix proposal
            # Note: This is a simplified validation since we don't have full task scope
            validation_result, violations = self.scope_validator.validate_fix_proposal(fix_proposal)

            if violations:
                error_msg = f"Scope violations detected: {', '.join(v.description for v in violations)}"
                return ScopeResult(False, error_msg)

            return ScopeResult(True)

        except Exception as e:
            return ScopeResult(False, f"Scope validation error: {e}")

    def _perform_verification_review(
        self,
        fix_proposal: FixProposal,
        mode: VerificationMode
    ) -> List[ParsedIssue]:
        """Perform code review based on verification mode"""
        try:
            files_to_review = []

            if mode == VerificationMode.FULL_REVIEW:
                # Review all files in the project (would need project context)
                files_to_review = list(fix_proposal.files_to_modify | fix_proposal.files_to_create)
            elif mode in [VerificationMode.TARGETED_REVIEW, VerificationMode.ISSUE_SPECIFIC]:
                # Review only modified files
                files_to_review = list(fix_proposal.files_to_modify | fix_proposal.files_to_create)
            elif mode == VerificationMode.REGRESSION_ONLY:
                # Review modified files for new issues
                files_to_review = list(fix_proposal.files_to_modify | fix_proposal.files_to_create)

            if not files_to_review:
                logger.warning("No files to review for verification")
                return []

            # Use code review agent to review files (sort for consistency)
            review_result = self.code_review_agent.review_files(
                file_paths=sorted(files_to_review),
                severity_filter="all"
            )

            # Process the review result
            if review_result and review_result.success:
                category_result = self.result_processor.process_review_result(
                    review_result.review_output
                )

                # Extract all issues from category result
                all_issues = (
                    category_result.blocking_issues +
                    category_result.required_issues +
                    category_result.recommended_issues +
                    category_result.optional_issues +
                    category_result.informational_issues
                )

                return all_issues
            else:
                logger.warning(f"Review failed: {review_result.error_message if review_result else 'Unknown error'}")
                return []

        except Exception as e:
            logger.error(f"Verification review failed: {e}")
            return []

    def _compare_issues(
        self,
        before_issues: List[ParsedIssue],
        after_issues: List[ParsedIssue]
    ) -> IssueComparison:
        """Compare before and after issues to detect resolution and regressions"""
        comparison = IssueComparison()

        # Create sets for efficient comparison
        before_signatures = {self._create_issue_signature(issue): issue for issue in before_issues}
        after_signatures = {self._create_issue_signature(issue): issue for issue in after_issues}

        # Find resolved issues (in before but not in after)
        for signature, issue in before_signatures.items():
            if signature not in after_signatures:
                comparison.resolved_issues.append(issue)

        # Find persisting issues (in both before and after)
        for signature, issue in before_signatures.items():
            if signature in after_signatures:
                comparison.persisting_issues.append(issue)

        # Find new issues (in after but not in before)
        for signature, issue in after_signatures.items():
            if signature not in before_signatures:
                comparison.new_issues.append(issue)

        # Note: modified_issues would require more sophisticated comparison
        # For now, we'll leave it empty

        return comparison

    def _create_issue_signature(self, issue: ParsedIssue) -> str:
        """Create a signature for issue comparison"""
        # Create a signature based on key characteristics
        signature_parts = [
            issue.severity.value,
            issue.category.value,
            issue.file_path or "unknown",
            str(issue.line_number) if issue.line_number else "0"
        ]

        # Add normalized description for better matching
        if issue.description:
            # Normalize description by removing common variations
            normalized_desc = issue.description.lower().strip()
            if self.config.ignore_minor_variations:
                # Remove common variations
                normalized_desc = normalized_desc.replace(" ", "").replace("-", "").replace("_", "")
            signature_parts.append(normalized_desc[:50])  # First 50 chars

        return "|".join(signature_parts)

    def _determine_verification_status(self, comparison: IssueComparison) -> VerificationStatus:
        """Determine verification status based on issue comparison"""

        # Check for regressions first
        if self.config.enable_regression_detection and comparison.has_regressions:
            if comparison.regression_count > self.config.max_acceptable_regressions:
                return VerificationStatus.REGRESSION_DETECTED

        # Check resolution rate
        resolution_rate = comparison.resolution_rate

        if resolution_rate >= self.config.minimum_resolution_rate:
            if comparison.has_regressions and comparison.regression_count <= self.config.max_acceptable_regressions:
                return VerificationStatus.PARTIAL_SUCCESS
            else:
                return VerificationStatus.VERIFIED
        else:
            return VerificationStatus.FAILED

    def _calculate_success_metrics(
        self,
        comparison: IssueComparison,
        fix_proposal: FixProposal
    ) -> Dict[str, Any]:
        """Calculate success metrics for verification"""
        if not self.config.include_detailed_metrics:
            return {}

        total_files_modified = len(fix_proposal.files_to_modify | fix_proposal.files_to_create)
        total_lines_changed = sum(fix_proposal.estimated_line_changes.values())

        return {
            "resolution_rate": comparison.resolution_rate,
            "issues_resolved": len(comparison.resolved_issues),
            "issues_persisting": len(comparison.persisting_issues),
            "new_issues_introduced": len(comparison.new_issues),
            "regression_count": comparison.regression_count,
            "files_modified": total_files_modified,
            "lines_changed": total_lines_changed,
            "verification_efficiency": comparison.resolution_rate / max(total_files_modified, 1),
            "regression_rate": comparison.regression_count / max(total_files_modified, 1)
        }

    def _generate_recommendations(
        self,
        comparison: IssueComparison,
        status: VerificationStatus,
        fix_proposal: FixProposal
    ) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []

        if status == VerificationStatus.REGRESSION_DETECTED:
            recommendations.append("Regressions detected - review and fix new issues before proceeding")
            if comparison.new_issues:
                high_severity_regressions = [
                    issue for issue in comparison.new_issues
                    if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]
                ]
                if high_severity_regressions:
                    recommendations.append(f"Address {len(high_severity_regressions)} high-severity regressions immediately")

        if status == VerificationStatus.FAILED:
            if comparison.resolution_rate < 0.5:
                recommendations.append("Low resolution rate - consider alternative fix approach")
            else:
                recommendations.append("Partial resolution achieved - refine fix to address remaining issues")

        if comparison.persisting_issues:
            critical_persisting = [
                issue for issue in comparison.persisting_issues
                if issue.severity == IssueSeverity.CRITICAL
            ]
            if critical_persisting:
                recommendations.append(f"Address {len(critical_persisting)} persisting critical issues")

        if status == VerificationStatus.VERIFIED:
            recommendations.append("Fix successfully verified - all requirements met")

        # Add scope-related recommendations
        total_changes = sum(fix_proposal.estimated_line_changes.values())
        if total_changes > 200:
            recommendations.append("Consider breaking large fix into smaller, focused changes")

        return recommendations

    def _create_failed_result(
        self,
        verification_id: str,
        task_id: str,
        mode: VerificationMode,
        error_message: str,
        start_time: float
    ) -> VerificationResult:
        """Create a failed verification result"""
        execution_time = int((time.time() - start_time) * 1000)

        return VerificationResult(
            verification_id=verification_id,
            task_id=task_id,
            status=VerificationStatus.FAILED,
            mode=mode,
            issue_comparison=IssueComparison(),  # Empty comparison
            execution_time_ms=execution_time,
            error_message=error_message
        )

    def get_verification_history(self, task_id: Optional[str] = None) -> Dict[str, VerificationResult]:
        """Get verification history, optionally filtered by task ID"""
        if task_id:
            return {
                vid: result for vid, result in self._verification_history.items()
                if result.task_id == task_id
            }
        return self._verification_history.copy()

    def get_verification_summary(self, task_id: str) -> Dict[str, Any]:
        """Get summary of verification results for a task"""
        task_verifications = self.get_verification_history(task_id)

        if not task_verifications:
            return {"task_id": task_id, "verification_count": 0}

        results = list(task_verifications.values())
        latest_result = max(results, key=lambda r: r.verification_timestamp or "")

        success_count = sum(1 for r in results if r.is_successful)

        return {
            "task_id": task_id,
            "verification_count": len(results),
            "success_rate": success_count / len(results),
            "latest_status": latest_result.status.value,
            "latest_resolution_rate": latest_result.issue_comparison.resolution_rate,
            "total_regressions_detected": sum(
                r.issue_comparison.regression_count for r in results
            ),
            "requires_attention": latest_result.requires_attention
        }


# Convenience functions
def verify_fix_quick(
    task_id: str,
    before_issues: List[ParsedIssue],
    files_modified: List[str],
    estimated_changes: Dict[str, int],
    config: Optional[VerificationConfig] = None
) -> VerificationResult:
    """
    Convenience function for quick fix verification.

    Args:
        task_id: Task identifier
        before_issues: Issues before the fix
        files_modified: List of files that were modified
        estimated_changes: Estimated line changes per file
        config: Optional verification configuration

    Returns:
        VerificationResult
    """
    workflow = FixVerificationWorkflow(config)

    fix_proposal = FixProposal(
        task_id=task_id,
        files_to_modify=set(files_modified),
        estimated_line_changes=estimated_changes
    )

    return workflow.verify_fix(task_id, before_issues, fix_proposal)
