"""
Scope Creep Detection and Validation

This module provides comprehensive validation to ensure fix actions stay within
original task boundaries and prevent uncontrolled scope expansion.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ScopeViolationType(str, Enum):
    """Types of scope violations"""
    FILE_LIMIT_EXCEEDED = "file_limit_exceeded"
    LINE_LIMIT_EXCEEDED = "line_limit_exceeded"
    PATTERN_VIOLATION = "pattern_violation"
    PATH_TRAVERSAL = "path_traversal"
    DIRECTORY_VIOLATION = "directory_violation"
    BINARY_FILE_MODIFICATION = "binary_file_modification"
    SYSTEM_FILE_ACCESS = "system_file_access"


class ScopeValidationResult(str, Enum):
    """Results of scope validation"""
    VALID = "valid"
    WARNING = "warning"
    VIOLATION = "violation"
    BLOCKED = "blocked"


@dataclass
class ScopeViolation:
    """Details of a scope validation violation"""
    violation_type: ScopeViolationType
    severity: ScopeValidationResult
    description: str
    file_path: Optional[str] = None
    line_count: Optional[int] = None
    pattern_violated: Optional[str] = None
    suggested_action: Optional[str] = None


@dataclass
class TaskScope:
    """Definition of original task scope and boundaries"""
    task_id: str
    original_files: Set[str] = field(default_factory=set)
    allowed_directories: Set[str] = field(default_factory=set)
    allowed_file_patterns: List[str] = field(default_factory=list)
    prohibited_patterns: List[str] = field(default_factory=list)
    max_file_changes: int = 10
    max_line_changes: int = 500
    max_new_files: int = 3
    context_description: Optional[str] = None


@dataclass
class FixProposal:
    """Proposed fix changes for validation"""
    task_id: str
    files_to_modify: Set[str] = field(default_factory=set)
    files_to_create: Set[str] = field(default_factory=set)
    files_to_delete: Set[str] = field(default_factory=set)
    estimated_line_changes: Dict[str, int] = field(default_factory=dict)
    change_description: Optional[str] = None
    diff_content: Optional[str] = None


@dataclass
class ScopeValidationConfig:
    """Configuration for scope validation behavior"""
    # File limits
    default_max_file_changes: int = 10
    default_max_line_changes: int = 500
    default_max_new_files: int = 3

    # Pattern enforcement
    enforce_file_patterns: bool = True
    allow_test_files: bool = True
    allow_documentation: bool = True

    # Security restrictions
    prevent_system_files: bool = True
    prevent_binary_files: bool = True
    prevent_path_traversal: bool = True

    # Violation handling
    warning_threshold: float = 0.8  # Warn at 80% of limits
    auto_block_violations: bool = True
    strict_mode: bool = False

    # File pattern configurations
    test_file_patterns: List[str] = field(default_factory=lambda: [
        r'.*test.*\.py$', r'.*_test\.py$', r'test_.*\.py$',
        r'.*\.test\.js$', r'.*\.spec\.js$'
    ])

    documentation_patterns: List[str] = field(default_factory=lambda: [
        r'.*\.md$', r'.*\.rst$', r'.*\.txt$', r'README.*', r'CHANGELOG.*'
    ])

    system_file_patterns: List[str] = field(default_factory=lambda: [
        r'/etc/.*', r'/sys/.*', r'/proc/.*', r'/dev/.*',
        r'.*\.exe$', r'.*\.dll$', r'.*\.so$'
    ])

    binary_file_patterns: List[str] = field(default_factory=lambda: [
        r'.*\.bin$', r'.*\.exe$', r'.*\.dll$', r'.*\.so$',
        r'.*\.jpg$', r'.*\.png$', r'.*\.gif$', r'.*\.pdf$'
    ])


class ScopeValidator:
    """
    Comprehensive scope validation to prevent fix operations from exceeding boundaries.

    Validates proposed fixes against original task scope to prevent:
    - Excessive file modifications
    - Unauthorized directory access
    - System file modifications
    - Binary file changes
    - Path traversal attacks
    """

    def __init__(self, config: Optional[ScopeValidationConfig] = None):
        """
        Initialize the scope validator.

        Args:
            config: Optional validation configuration
        """
        self.config = config or ScopeValidationConfig()
        self._task_scopes: Dict[str, TaskScope] = {}

        # Pre-compile regex patterns for performance
        self._compile_patterns()

        logger.info("ScopeValidator initialized")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching"""
        self.test_patterns = [re.compile(pattern, re.IGNORECASE)
                             for pattern in self.config.test_file_patterns]

        self.doc_patterns = [re.compile(pattern, re.IGNORECASE)
                            for pattern in self.config.documentation_patterns]

        self.system_patterns = [re.compile(pattern, re.IGNORECASE)
                               for pattern in self.config.system_file_patterns]

        self.binary_patterns = [re.compile(pattern, re.IGNORECASE)
                               for pattern in self.config.binary_file_patterns]

    def register_task_scope(self, scope: TaskScope) -> None:
        """
        Register the original scope for a task.

        Args:
            scope: Task scope definition
        """
        self._task_scopes[scope.task_id] = scope
        logger.info(f"Registered scope for task {scope.task_id}")

    def validate_fix_proposal(
        self,
        proposal: FixProposal,
        task_scope: Optional[TaskScope] = None
    ) -> Tuple[ScopeValidationResult, List[ScopeViolation]]:
        """
        Validate a proposed fix against task scope boundaries.

        Args:
            proposal: The fix proposal to validate
            task_scope: Optional task scope (uses registered if not provided)

        Returns:
            Tuple of validation result and list of violations
        """
        # Get task scope
        if task_scope is None:
            task_scope = self._task_scopes.get(proposal.task_id)
            if task_scope is None:
                logger.warning(f"No scope registered for task {proposal.task_id}")
                return ScopeValidationResult.WARNING, [
                    ScopeViolation(
                        violation_type=ScopeViolationType.PATTERN_VIOLATION,
                        severity=ScopeValidationResult.WARNING,
                        description="No task scope registered - cannot validate boundaries"
                    )
                ]

        violations = []

        # Check file count limits
        violations.extend(self._check_file_limits(proposal, task_scope))

        # Check line change limits
        violations.extend(self._check_line_limits(proposal, task_scope))

        # Check file patterns and security
        violations.extend(self._check_file_patterns(proposal, task_scope))

        # Check directory boundaries
        violations.extend(self._check_directory_boundaries(proposal, task_scope))

        # Check for system/binary file access
        violations.extend(self._check_security_restrictions(proposal))

        # Determine overall result
        if any(v.severity == ScopeValidationResult.BLOCKED for v in violations):
            return ScopeValidationResult.BLOCKED, violations
        elif any(v.severity == ScopeValidationResult.VIOLATION for v in violations):
            return ScopeValidationResult.VIOLATION, violations
        elif any(v.severity == ScopeValidationResult.WARNING for v in violations):
            return ScopeValidationResult.WARNING, violations
        else:
            return ScopeValidationResult.VALID, violations

    def _check_file_limits(
        self,
        proposal: FixProposal,
        task_scope: TaskScope
    ) -> List[ScopeViolation]:
        """Check if file modification limits are exceeded"""
        violations = []

        total_files = len(proposal.files_to_modify) + len(proposal.files_to_create)
        max_files = task_scope.max_file_changes

        if total_files > max_files:
            violations.append(ScopeViolation(
                violation_type=ScopeViolationType.FILE_LIMIT_EXCEEDED,
                severity=ScopeValidationResult.VIOLATION,
                description=f"Proposed changes affect {total_files} files, exceeding limit of {max_files}",
                suggested_action=f"Reduce scope to modify at most {max_files} files"
            ))
        elif total_files >= max_files * self.config.warning_threshold:
            violations.append(ScopeViolation(
                violation_type=ScopeViolationType.FILE_LIMIT_EXCEEDED,
                severity=ScopeValidationResult.WARNING,
                description=f"Proposed changes affect {total_files} files, approaching limit of {max_files}",
                suggested_action="Consider if all file changes are necessary"
            ))

        # Check new file creation
        new_files = len(proposal.files_to_create)
        max_new = task_scope.max_new_files

        if new_files > max_new:
            violations.append(ScopeViolation(
                violation_type=ScopeViolationType.FILE_LIMIT_EXCEEDED,
                severity=ScopeValidationResult.VIOLATION,
                description=f"Proposed to create {new_files} new files, exceeding limit of {max_new}",
                suggested_action=f"Reduce new file creation to at most {max_new} files"
            ))

        return violations

    def _check_line_limits(
        self,
        proposal: FixProposal,
        task_scope: TaskScope
    ) -> List[ScopeViolation]:
        """Check if line change limits are exceeded"""
        violations = []

        total_lines = sum(proposal.estimated_line_changes.values())
        max_lines = task_scope.max_line_changes

        if total_lines > max_lines:
            violations.append(ScopeViolation(
                violation_type=ScopeViolationType.LINE_LIMIT_EXCEEDED,
                severity=ScopeValidationResult.VIOLATION,
                description=f"Proposed changes affect {total_lines} lines, exceeding limit of {max_lines}",
                suggested_action=f"Reduce scope to modify at most {max_lines} lines"
            ))
        elif total_lines >= max_lines * self.config.warning_threshold:
            violations.append(ScopeViolation(
                violation_type=ScopeViolationType.LINE_LIMIT_EXCEEDED,
                severity=ScopeValidationResult.WARNING,
                description=f"Proposed changes affect {total_lines} lines, approaching limit of {max_lines}",
                suggested_action="Consider if all line changes are necessary"
            ))

        # Check individual file line changes
        for file_path, line_count in proposal.estimated_line_changes.items():
            if line_count > max_lines * 0.5:  # Single file shouldn't exceed 50% of total
                violations.append(ScopeViolation(
                    violation_type=ScopeViolationType.LINE_LIMIT_EXCEEDED,
                    severity=ScopeValidationResult.WARNING,
                    description=f"File {file_path} has {line_count} line changes (>{max_lines * 0.5})",
                    file_path=file_path,
                    line_count=line_count,
                    suggested_action="Consider breaking up large changes into smaller files"
                ))

        return violations

    def _check_file_patterns(
        self,
        proposal: FixProposal,
        task_scope: TaskScope
    ) -> List[ScopeViolation]:
        """Check if files match allowed patterns and don't violate restrictions"""
        violations = []

        if not self.config.enforce_file_patterns:
            return violations

        all_files = proposal.files_to_modify | proposal.files_to_create

        for file_path in all_files:
            # Check against allowed patterns
            if task_scope.allowed_file_patterns:
                allowed = any(re.match(pattern, file_path, re.IGNORECASE)
                             for pattern in task_scope.allowed_file_patterns)

                # Allow test and documentation files if configured
                if not allowed and self.config.allow_test_files:
                    allowed = any(pattern.match(file_path) for pattern in self.test_patterns)

                if not allowed and self.config.allow_documentation:
                    allowed = any(pattern.match(file_path) for pattern in self.doc_patterns)

                if not allowed:
                    violations.append(ScopeViolation(
                        violation_type=ScopeViolationType.PATTERN_VIOLATION,
                        severity=ScopeValidationResult.VIOLATION,
                        description=f"File {file_path} doesn't match allowed patterns",
                        file_path=file_path,
                        suggested_action="Ensure file is within task scope"
                    ))

            # Check against prohibited patterns
            for pattern in task_scope.prohibited_patterns:
                if re.match(pattern, file_path, re.IGNORECASE):
                    violations.append(ScopeViolation(
                        violation_type=ScopeViolationType.PATTERN_VIOLATION,
                        severity=ScopeValidationResult.BLOCKED,
                        description=f"File {file_path} matches prohibited pattern: {pattern}",
                        file_path=file_path,
                        pattern_violated=pattern,
                        suggested_action="Remove this file from the fix proposal"
                    ))

        return violations

    def _check_directory_boundaries(
        self,
        proposal: FixProposal,
        task_scope: TaskScope
    ) -> List[ScopeViolation]:
        """Check if files are within allowed directory boundaries"""
        violations = []

        if not task_scope.allowed_directories:
            return violations

        all_files = proposal.files_to_modify | proposal.files_to_create

        for file_path in all_files:
            file_dir = str(Path(file_path).parent)

            # Normalize paths for comparison - ensure trailing slash consistency
            normalized_file_dir = file_dir + "/" if not file_dir.endswith("/") and file_dir != "." else file_dir

            # Check if file is within allowed directories
            allowed = any(
                normalized_file_dir.startswith(allowed_dir) or
                (allowed_dir.rstrip("/") == file_dir) or
                file_dir.startswith(allowed_dir.rstrip("/") + "/")
                for allowed_dir in task_scope.allowed_directories
            )

            if not allowed:
                violations.append(ScopeViolation(
                    violation_type=ScopeViolationType.DIRECTORY_VIOLATION,
                    severity=ScopeValidationResult.VIOLATION,
                    description=f"File {file_path} is outside allowed directories",
                    file_path=file_path,
                    suggested_action=f"Keep changes within: {', '.join(task_scope.allowed_directories)}"
                ))

        return violations

    def _check_security_restrictions(self, proposal: FixProposal) -> List[ScopeViolation]:
        """Check for security-related restrictions"""
        violations = []

        all_files = proposal.files_to_modify | proposal.files_to_create

        for file_path in all_files:
            # Check for path traversal
            if self.config.prevent_path_traversal and '..' in file_path:
                violations.append(ScopeViolation(
                    violation_type=ScopeViolationType.PATH_TRAVERSAL,
                    severity=ScopeValidationResult.BLOCKED,
                    description=f"Path traversal detected in {file_path}",
                    file_path=file_path,
                    suggested_action="Use absolute paths without '..' components"
                ))

            # Check for system files
            if self.config.prevent_system_files:
                if any(pattern.match(file_path) for pattern in self.system_patterns):
                    violations.append(ScopeViolation(
                        violation_type=ScopeViolationType.SYSTEM_FILE_ACCESS,
                        severity=ScopeValidationResult.BLOCKED,
                        description=f"System file access attempted: {file_path}",
                        file_path=file_path,
                        suggested_action="Do not modify system files"
                    ))

            # Check for binary files
            if self.config.prevent_binary_files:
                if any(pattern.match(file_path) for pattern in self.binary_patterns):
                    violations.append(ScopeViolation(
                        violation_type=ScopeViolationType.BINARY_FILE_MODIFICATION,
                        severity=ScopeValidationResult.BLOCKED,
                        description=f"Binary file modification attempted: {file_path}",
                        file_path=file_path,
                        suggested_action="Binary files should not be modified by automated fixes"
                    ))

        return violations

    def analyze_diff(self, diff_content: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze diff content to measure change magnitude.

        Args:
            diff_content: Unified diff content
            file_path: Path to the file being changed

        Returns:
            Analysis results including line counts and change types
        """
        lines = diff_content.split('\n')

        analysis = {
            'file_path': file_path,
            'lines_added': 0,
            'lines_deleted': 0,
            'lines_modified': 0,
            'total_changes': 0,
            'change_ratio': 0.0,
            'is_major_rewrite': False,
            'function_changes': [],
            'import_changes': []
        }

        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                analysis['lines_added'] += 1
            elif line.startswith('-') and not line.startswith('---'):
                analysis['lines_deleted'] += 1
            elif line.startswith('@@'):
                # Parse hunk header for context
                continue

        # Calculate totals
        analysis['total_changes'] = analysis['lines_added'] + analysis['lines_deleted']

        # Estimate original file size from diff context
        original_lines = analysis['lines_deleted'] + len([
            line for line in lines
            if not line.startswith(('+', '-', '@', '\\'))
        ])

        if original_lines > 0:
            analysis['change_ratio'] = analysis['total_changes'] / original_lines
            analysis['is_major_rewrite'] = analysis['change_ratio'] > 0.7

        # Detect function and import changes - check all lines for context
        for line in lines:
            # Check changed lines and context lines for function/class definitions
            if ('def ' in line or 'function ' in line or 'class ' in line) and not line.startswith('@@'):
                analysis['function_changes'].append(line.strip())
            elif ('import ' in line or 'from ' in line) and line.startswith(('+', '-')):
                analysis['import_changes'].append(line.strip())

        return analysis

    def estimate_line_changes_from_diff(self, diff_content: str) -> int:
        """
        Estimate total line changes from diff content.

        Args:
            diff_content: Unified diff content

        Returns:
            Estimated number of line changes
        """
        if not diff_content:
            return 0

        lines = diff_content.split('\n')
        changes = 0

        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                changes += 1
            elif line.startswith('-') and not line.startswith('---'):
                changes += 1

        return changes

    def get_scope_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of task scope and current status.

        Args:
            task_id: Task identifier

        Returns:
            Scope summary dictionary
        """
        scope = self._task_scopes.get(task_id)
        if not scope:
            return None

        return {
            'task_id': task_id,
            'original_files': list(scope.original_files),
            'allowed_directories': list(scope.allowed_directories),
            'allowed_patterns': scope.allowed_file_patterns,
            'prohibited_patterns': scope.prohibited_patterns,
            'limits': {
                'max_file_changes': scope.max_file_changes,
                'max_line_changes': scope.max_line_changes,
                'max_new_files': scope.max_new_files
            },
            'context': scope.context_description
        }

    def create_task_scope_from_context(
        self,
        task_id: str,
        original_files: List[str],
        context_description: Optional[str] = None,
        custom_limits: Optional[Dict[str, int]] = None
    ) -> TaskScope:
        """
        Create a task scope from basic context information.

        Args:
            task_id: Task identifier
            original_files: List of files mentioned in original task
            context_description: Description of task context
            custom_limits: Optional custom limits

        Returns:
            Configured TaskScope
        """
        # Extract directory boundaries from original files
        directories = set()
        for file_path in original_files:
            directories.add(str(Path(file_path).parent))

        # Create file patterns based on original files
        file_extensions = set()
        for file_path in original_files:
            ext = Path(file_path).suffix
            if ext:
                file_extensions.add(ext)

        patterns = [f".*\\{ext}$" for ext in file_extensions] if file_extensions else []

        # Apply custom limits
        limits = {
            'max_file_changes': self.config.default_max_file_changes,
            'max_line_changes': self.config.default_max_line_changes,
            'max_new_files': self.config.default_max_new_files
        }

        if custom_limits:
            limits.update(custom_limits)

        scope = TaskScope(
            task_id=task_id,
            original_files=set(original_files),
            allowed_directories=directories,
            allowed_file_patterns=patterns,
            context_description=context_description,
            **limits
        )

        self.register_task_scope(scope)
        return scope


# Convenience functions
def validate_fix_scope(
    task_id: str,
    files_to_modify: List[str],
    estimated_changes: Dict[str, int],
    original_files: List[str],
    config: Optional[ScopeValidationConfig] = None
) -> Tuple[ScopeValidationResult, List[ScopeViolation]]:
    """
    Convenience function for quick scope validation.

    Args:
        task_id: Task identifier
        files_to_modify: List of files to be modified
        estimated_changes: Estimated line changes per file
        original_files: Original files in task scope
        config: Optional validation configuration

    Returns:
        Validation result and violations
    """
    validator = ScopeValidator(config)

    # Create scope from context
    scope = validator.create_task_scope_from_context(task_id, original_files)

    # Create proposal
    proposal = FixProposal(
        task_id=task_id,
        files_to_modify=set(files_to_modify),
        estimated_line_changes=estimated_changes
    )

    return validator.validate_fix_proposal(proposal, scope)
