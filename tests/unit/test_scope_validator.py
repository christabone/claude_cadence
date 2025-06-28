"""
Unit tests for Scope Validator
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Set

from cadence.scope_validator import (
    ScopeValidator, ScopeValidationConfig, TaskScope, FixProposal,
    ScopeViolation, ScopeViolationType, ScopeValidationResult,
    validate_fix_scope
)


class TestScopeValidationConfig:
    """Test ScopeValidationConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ScopeValidationConfig()

        assert config.default_max_file_changes == 10
        assert config.default_max_line_changes == 500
        assert config.default_max_new_files == 3
        assert config.enforce_file_patterns is True
        assert config.prevent_system_files is True
        assert config.prevent_binary_files is True
        assert config.warning_threshold == 0.8

        # Check pattern lists are populated
        assert len(config.test_file_patterns) > 0
        assert len(config.documentation_patterns) > 0
        assert len(config.system_file_patterns) > 0
        assert len(config.binary_file_patterns) > 0

    def test_custom_config(self):
        """Test custom configuration"""
        config = ScopeValidationConfig(
            default_max_file_changes=5,
            warning_threshold=0.9,
            strict_mode=True
        )

        assert config.default_max_file_changes == 5
        assert config.warning_threshold == 0.9
        assert config.strict_mode is True


class TestTaskScope:
    """Test TaskScope dataclass"""

    def test_default_scope(self):
        """Test default scope creation"""
        scope = TaskScope(task_id="test-1")

        assert scope.task_id == "test-1"
        assert len(scope.original_files) == 0
        assert len(scope.allowed_directories) == 0
        assert scope.max_file_changes == 10
        assert scope.max_line_changes == 500
        assert scope.max_new_files == 3

    def test_complete_scope(self):
        """Test scope with all fields"""
        scope = TaskScope(
            task_id="test-2",
            original_files={"file1.py", "file2.py"},
            allowed_directories={"src/", "tests/"},
            allowed_file_patterns=["*.py", "*.md"],
            prohibited_patterns=["*.exe"],
            max_file_changes=5,
            max_line_changes=200,
            max_new_files=2,
            context_description="Test task scope"
        )

        assert scope.task_id == "test-2"
        assert len(scope.original_files) == 2
        assert "file1.py" in scope.original_files
        assert len(scope.allowed_directories) == 2
        assert scope.max_file_changes == 5
        assert scope.context_description == "Test task scope"


class TestFixProposal:
    """Test FixProposal dataclass"""

    def test_default_proposal(self):
        """Test default proposal creation"""
        proposal = FixProposal(task_id="test-1")

        assert proposal.task_id == "test-1"
        assert len(proposal.files_to_modify) == 0
        assert len(proposal.files_to_create) == 0
        assert len(proposal.files_to_delete) == 0
        assert len(proposal.estimated_line_changes) == 0

    def test_complete_proposal(self):
        """Test proposal with all fields"""
        proposal = FixProposal(
            task_id="test-2",
            files_to_modify={"file1.py", "file2.py"},
            files_to_create={"file3.py"},
            files_to_delete={"old_file.py"},
            estimated_line_changes={"file1.py": 50, "file2.py": 30},
            change_description="Test changes",
            diff_content="sample diff"
        )

        assert proposal.task_id == "test-2"
        assert len(proposal.files_to_modify) == 2
        assert "file1.py" in proposal.files_to_modify
        assert len(proposal.files_to_create) == 1
        assert proposal.estimated_line_changes["file1.py"] == 50


class TestScopeValidator:
    """Test ScopeValidator class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = ScopeValidationConfig(
            default_max_file_changes=5,
            default_max_line_changes=100,
            default_max_new_files=2
        )
        self.validator = ScopeValidator(self.config)

        # Create test scope
        self.test_scope = TaskScope(
            task_id="test-task",
            original_files={"src/main.py", "src/utils.py"},
            allowed_directories={"src/", "tests/"},
            allowed_file_patterns=[r".*\.py$", r".*\.md$"],
            prohibited_patterns=[r".*\.exe$"],
            max_file_changes=3,
            max_line_changes=150,
            max_new_files=1
        )

        self.validator.register_task_scope(self.test_scope)

    def test_initialization_default(self):
        """Test validator initialization with defaults"""
        validator = ScopeValidator()

        assert validator.config is not None
        assert hasattr(validator, 'test_patterns')
        assert hasattr(validator, 'doc_patterns')
        assert hasattr(validator, 'system_patterns')
        assert hasattr(validator, 'binary_patterns')

    def test_initialization_custom_config(self):
        """Test validator initialization with custom config"""
        custom_config = ScopeValidationConfig(strict_mode=True)
        validator = ScopeValidator(custom_config)

        assert validator.config.strict_mode is True

    def test_register_task_scope(self):
        """Test task scope registration"""
        scope = TaskScope(task_id="new-task")
        self.validator.register_task_scope(scope)

        assert "new-task" in self.validator._task_scopes
        assert self.validator._task_scopes["new-task"] == scope

    def test_validate_fix_proposal_valid(self):
        """Test validation of valid fix proposal"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/main.py"},
            estimated_line_changes={"src/main.py": 20}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.VALID
        assert len(violations) == 0

    def test_validate_fix_proposal_file_limit_exceeded(self):
        """Test validation when file limits are exceeded"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/main.py", "src/utils.py", "src/config.py", "src/helpers.py"},
            estimated_line_changes={
                "src/main.py": 20,
                "src/utils.py": 15,
                "src/config.py": 10,
                "src/helpers.py": 5
            }
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.VIOLATION
        assert any(v.violation_type == ScopeViolationType.FILE_LIMIT_EXCEEDED for v in violations)

    def test_validate_fix_proposal_line_limit_exceeded(self):
        """Test validation when line limits are exceeded"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/main.py"},
            estimated_line_changes={"src/main.py": 200}  # Exceeds limit of 150
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.VIOLATION
        assert any(v.violation_type == ScopeViolationType.LINE_LIMIT_EXCEEDED for v in violations)

    def test_validate_fix_proposal_pattern_violation(self):
        """Test validation when file patterns are violated"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/main.exe"},  # Prohibited pattern
            estimated_line_changes={"src/main.exe": 10}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.BLOCKED
        assert any(v.violation_type == ScopeViolationType.PATTERN_VIOLATION for v in violations)

    def test_validate_fix_proposal_directory_violation(self):
        """Test validation when directory boundaries are violated"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"lib/external.py"},  # Outside allowed directories
            estimated_line_changes={"lib/external.py": 10}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.VIOLATION
        assert any(v.violation_type == ScopeViolationType.DIRECTORY_VIOLATION for v in violations)

    def test_validate_fix_proposal_warning_threshold(self):
        """Test warning when approaching limits"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/main.py", "src/utils.py"},  # 2 files, warning at 80% of 3
            estimated_line_changes={"src/main.py": 60, "src/utils.py": 60}  # 120 lines, warning at 80% of 150
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.WARNING
        assert any(v.severity == ScopeValidationResult.WARNING for v in violations)

    def test_validate_fix_proposal_no_scope_registered(self):
        """Test validation when no scope is registered"""
        proposal = FixProposal(
            task_id="unknown-task",
            files_to_modify={"src/main.py"}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.WARNING
        assert len(violations) == 1
        assert "No task scope registered" in violations[0].description

    def test_check_security_restrictions_path_traversal(self):
        """Test security check for path traversal"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"../../../etc/passwd"},
            estimated_line_changes={"../../../etc/passwd": 1}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.BLOCKED
        assert any(v.violation_type == ScopeViolationType.PATH_TRAVERSAL for v in violations)

    def test_check_security_restrictions_system_files(self):
        """Test security check for system files"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"/etc/hosts"},
            estimated_line_changes={"/etc/hosts": 1}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.BLOCKED
        assert any(v.violation_type == ScopeViolationType.SYSTEM_FILE_ACCESS for v in violations)

    def test_check_security_restrictions_binary_files(self):
        """Test security check for binary files"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/app.exe"},
            estimated_line_changes={"src/app.exe": 1}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.BLOCKED
        assert any(v.violation_type == ScopeViolationType.BINARY_FILE_MODIFICATION for v in violations)

    def test_allow_test_files(self):
        """Test that test files are allowed when configured"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"tests/test_main.py"},
            estimated_line_changes={"tests/test_main.py": 20}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        # Should be valid even though test files might not match specific patterns
        assert result == ScopeValidationResult.VALID

    def test_allow_documentation_files(self):
        """Test that documentation files are allowed when configured"""
        proposal = FixProposal(
            task_id="test-task",
            files_to_modify={"src/README.md"},
            estimated_line_changes={"src/README.md": 10}
        )

        result, violations = self.validator.validate_fix_proposal(proposal)

        assert result == ScopeValidationResult.VALID

    def test_analyze_diff_basic(self):
        """Test basic diff analysis"""
        diff_content = """--- a/file.py
+++ b/file.py
@@ -1,5 +1,7 @@
 def hello():
-    print("hello")
+    print("hello world")
+    print("new line")

 def goodbye():
     print("goodbye")
+    return True"""

        analysis = self.validator.analyze_diff(diff_content, "file.py")

        assert analysis['file_path'] == "file.py"
        assert analysis['lines_added'] == 3
        assert analysis['lines_deleted'] == 1
        assert analysis['total_changes'] == 4
        assert len(analysis['function_changes']) > 0

    def test_analyze_diff_empty(self):
        """Test diff analysis with empty content"""
        analysis = self.validator.analyze_diff("", "file.py")

        assert analysis['file_path'] == "file.py"
        assert analysis['lines_added'] == 0
        assert analysis['lines_deleted'] == 0
        assert analysis['total_changes'] == 0

    def test_analyze_diff_major_rewrite(self):
        """Test diff analysis detecting major rewrites"""
        # Create a diff that represents 80% of file being changed
        diff_lines = ["--- a/file.py", "+++ b/file.py"]
        for i in range(40):  # 40 deletions
            diff_lines.append(f"-    old_line_{i}")
        for i in range(40):  # 40 additions
            diff_lines.append(f"+    new_line_{i}")
        for i in range(10):  # 10 unchanged context lines
            diff_lines.append(f"     context_{i}")

        diff_content = "\n".join(diff_lines)
        analysis = self.validator.analyze_diff(diff_content, "file.py")

        assert analysis['lines_added'] == 40
        assert analysis['lines_deleted'] == 40
        assert analysis['total_changes'] == 80
        assert analysis['is_major_rewrite'] is True

    def test_estimate_line_changes_from_diff(self):
        """Test line change estimation from diff"""
        diff_content = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,5 @@
 def test():
-    pass
+    print("hello")
+    return True"""

        changes = self.validator.estimate_line_changes_from_diff(diff_content)
        assert changes == 3  # 1 deletion + 2 additions

    def test_estimate_line_changes_empty_diff(self):
        """Test line change estimation with empty diff"""
        changes = self.validator.estimate_line_changes_from_diff("")
        assert changes == 0

    def test_get_scope_summary(self):
        """Test scope summary retrieval"""
        summary = self.validator.get_scope_summary("test-task")

        assert summary is not None
        assert summary['task_id'] == "test-task"
        assert 'original_files' in summary
        assert 'allowed_directories' in summary
        assert 'limits' in summary
        assert summary['limits']['max_file_changes'] == 3

    def test_get_scope_summary_not_found(self):
        """Test scope summary for non-existent task"""
        summary = self.validator.get_scope_summary("unknown-task")
        assert summary is None

    def test_create_task_scope_from_context(self):
        """Test creating task scope from context"""
        original_files = ["src/app.py", "src/config.py", "tests/test_app.py"]

        scope = self.validator.create_task_scope_from_context(
            task_id="context-task",
            original_files=original_files,
            context_description="Test context",
            custom_limits={"max_file_changes": 8}
        )

        assert scope.task_id == "context-task"
        assert len(scope.original_files) == 3
        assert "src/app.py" in scope.original_files
        assert scope.max_file_changes == 8  # Custom limit
        assert scope.context_description == "Test context"

        # Check that directories were extracted
        assert any("src" in d for d in scope.allowed_directories)
        assert any("tests" in d for d in scope.allowed_directories)

        # Check that file patterns were created from extensions
        assert any(".py" in pattern for pattern in scope.allowed_file_patterns)

    def test_create_task_scope_no_extensions(self):
        """Test creating scope when files have no extensions"""
        original_files = ["Makefile", "README", "CHANGELOG"]

        scope = self.validator.create_task_scope_from_context(
            task_id="no-ext-task",
            original_files=original_files
        )

        assert scope.task_id == "no-ext-task"
        assert len(scope.original_files) == 3
        # Should handle files without extensions gracefully
        assert len(scope.allowed_file_patterns) == 0


class TestScopeViolation:
    """Test ScopeViolation dataclass"""

    def test_basic_violation(self):
        """Test basic violation creation"""
        violation = ScopeViolation(
            violation_type=ScopeViolationType.FILE_LIMIT_EXCEEDED,
            severity=ScopeValidationResult.VIOLATION,
            description="Too many files"
        )

        assert violation.violation_type == ScopeViolationType.FILE_LIMIT_EXCEEDED
        assert violation.severity == ScopeValidationResult.VIOLATION
        assert violation.description == "Too many files"
        assert violation.file_path is None

    def test_complete_violation(self):
        """Test violation with all fields"""
        violation = ScopeViolation(
            violation_type=ScopeViolationType.PATTERN_VIOLATION,
            severity=ScopeValidationResult.BLOCKED,
            description="Pattern not allowed",
            file_path="src/bad.exe",
            line_count=100,
            pattern_violated="*.exe",
            suggested_action="Remove the file"
        )

        assert violation.file_path == "src/bad.exe"
        assert violation.line_count == 100
        assert violation.pattern_violated == "*.exe"
        assert violation.suggested_action == "Remove the file"


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_validate_fix_scope_basic(self):
        """Test basic fix scope validation"""
        result, violations = validate_fix_scope(
            task_id="quick-test",
            files_to_modify=["src/main.py", "src/utils.py"],
            estimated_changes={"src/main.py": 50, "src/utils.py": 30},
            original_files=["src/main.py", "src/utils.py", "tests/test_main.py"]
        )

        # Should be valid since within default limits
        assert result == ScopeValidationResult.VALID

    def test_validate_fix_scope_exceeded_limits(self):
        """Test fix scope validation with exceeded limits"""
        # Create many files to exceed default limits
        files_to_modify = [f"src/file_{i}.py" for i in range(15)]  # Exceeds default limit of 10
        estimated_changes = {f: 20 for f in files_to_modify}

        result, violations = validate_fix_scope(
            task_id="exceed-test",
            files_to_modify=files_to_modify,
            estimated_changes=estimated_changes,
            original_files=files_to_modify[:3]  # Only 3 original files
        )

        assert result == ScopeValidationResult.VIOLATION
        assert any(v.violation_type == ScopeViolationType.FILE_LIMIT_EXCEEDED for v in violations)

    def test_validate_fix_scope_custom_config(self):
        """Test fix scope validation with custom config"""
        config = ScopeValidationConfig(
            default_max_file_changes=2,
            default_max_line_changes=50
        )

        result, violations = validate_fix_scope(
            task_id="custom-test",
            files_to_modify=["src/file1.py", "src/file2.py", "src/file3.py"],  # 3 files > limit of 2
            estimated_changes={"src/file1.py": 20, "src/file2.py": 20, "src/file3.py": 20},
            original_files=["src/file1.py"],
            config=config
        )

        assert result == ScopeValidationResult.VIOLATION
        assert any(v.violation_type == ScopeViolationType.FILE_LIMIT_EXCEEDED for v in violations)


class TestSecurityFeatures:
    """Test security-specific features"""

    def setup_method(self):
        """Set up security test fixtures"""
        self.config = ScopeValidationConfig(
            prevent_system_files=True,
            prevent_binary_files=True,
            prevent_path_traversal=True
        )
        self.validator = ScopeValidator(self.config)

    def test_system_file_patterns(self):
        """Test system file pattern recognition"""
        system_files = [
            "/etc/passwd",
            "/sys/kernel/config",
            "/proc/cpuinfo",
            "/dev/null",
            "app.exe",
            "library.dll",
            "module.so"
        ]

        scope = TaskScope(task_id="security-test")
        self.validator.register_task_scope(scope)

        for file_path in system_files:
            proposal = FixProposal(
                task_id="security-test",
                files_to_modify={file_path}
            )

            result, violations = self.validator.validate_fix_proposal(proposal)

            # Should be blocked for system files
            assert result == ScopeValidationResult.BLOCKED
            assert any(
                v.violation_type in [
                    ScopeViolationType.SYSTEM_FILE_ACCESS,
                    ScopeViolationType.BINARY_FILE_MODIFICATION
                ] for v in violations
            ), f"Failed for file: {file_path}"

    def test_path_traversal_detection(self):
        """Test path traversal attack detection"""
        malicious_paths = [
            "../../../etc/passwd",
            "dir/../../../home/user/.ssh/id_rsa",
            "normal/path/../../secret/file.txt",
            "../config.ini"
        ]

        scope = TaskScope(task_id="traversal-test")
        self.validator.register_task_scope(scope)

        for path in malicious_paths:
            proposal = FixProposal(
                task_id="traversal-test",
                files_to_modify={path}
            )

            result, violations = self.validator.validate_fix_proposal(proposal)

            assert result == ScopeValidationResult.BLOCKED
            assert any(v.violation_type == ScopeViolationType.PATH_TRAVERSAL for v in violations)

    def test_binary_file_detection(self):
        """Test binary file detection"""
        binary_files = [
            "app.exe",
            "library.dll",
            "module.so",
            "image.jpg",
            "document.pdf",
            "archive.bin"
        ]

        scope = TaskScope(task_id="binary-test")
        self.validator.register_task_scope(scope)

        for file_path in binary_files:
            proposal = FixProposal(
                task_id="binary-test",
                files_to_modify={file_path}
            )

            result, violations = self.validator.validate_fix_proposal(proposal)

            assert result == ScopeValidationResult.BLOCKED
            assert any(v.violation_type == ScopeViolationType.BINARY_FILE_MODIFICATION for v in violations)

    def test_security_disabled(self):
        """Test with security checks disabled"""
        config = ScopeValidationConfig(
            prevent_system_files=False,
            prevent_binary_files=False,
            prevent_path_traversal=False
        )
        validator = ScopeValidator(config)

        scope = TaskScope(task_id="no-security-test")
        validator.register_task_scope(scope)

        proposal = FixProposal(
            task_id="no-security-test",
            files_to_modify={"../config.ini", "app.exe"},
            estimated_line_changes={"../config.ini": 5, "app.exe": 1}
        )

        result, violations = validator.validate_fix_proposal(proposal)

        # Should not have security violations when disabled
        security_violations = [v for v in violations if v.violation_type in [
            ScopeViolationType.PATH_TRAVERSAL,
            ScopeViolationType.SYSTEM_FILE_ACCESS,
            ScopeViolationType.BINARY_FILE_MODIFICATION
        ]]

        assert len(security_violations) == 0


if __name__ == "__main__":
    pytest.main([__file__])
