"""
Unit tests for Review Result Parser
"""

import json
import pytest
from typing import Dict, Any, List

from cadence.review_result_parser import (
    ReviewResultProcessor, ProcessingConfig, ParsedIssue, CategoryResult,
    IssueSeverity, IssueCategory, ActionType, quick_parse
)


class TestProcessingConfig:
    """Test ProcessingConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ProcessingConfig()

        assert len(config.critical_keywords) > 0
        assert "critical" in config.critical_keywords
        assert "security vulnerability" in config.critical_keywords

        assert len(config.high_keywords) > 0
        assert "error" in config.high_keywords

        assert len(config.security_keywords) > 0
        assert "security" in config.security_keywords

        assert config.extract_line_numbers is True
        assert config.extract_file_paths is True
        assert config.confidence_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration"""
        config = ProcessingConfig(
            critical_keywords=["custom_critical"],
            confidence_threshold=0.8,
            max_description_length=100
        )

        assert config.critical_keywords == ["custom_critical"]
        assert config.confidence_threshold == 0.8
        assert config.max_description_length == 100


class TestParsedIssue:
    """Test ParsedIssue dataclass"""

    def test_default_issue(self):
        """Test default issue creation"""
        issue = ParsedIssue(
            severity=IssueSeverity.HIGH,
            category=IssueCategory.BUG,
            description="Test issue"
        )

        assert issue.severity == IssueSeverity.HIGH
        assert issue.category == IssueCategory.BUG
        assert issue.description == "Test issue"
        assert issue.file_path is None
        assert issue.line_number is None
        assert issue.confidence == 1.0

    def test_complete_issue(self):
        """Test issue with all fields"""
        issue = ParsedIssue(
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            description="Security vulnerability found",
            file_path="/test/file.py",
            line_number=42,
            column_number=10,
            suggested_fix="Use parameterized queries",
            code_snippet="cursor.execute(query)",
            rule_id="SQL001",
            confidence=0.95,
            raw_text="Original text"
        )

        assert issue.severity == IssueSeverity.CRITICAL
        assert issue.file_path == "/test/file.py"
        assert issue.line_number == 42
        assert issue.column_number == 10
        assert issue.suggested_fix == "Use parameterized queries"
        assert issue.confidence == 0.95

    def test_to_dict(self):
        """Test dictionary conversion"""
        issue = ParsedIssue(
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.PERFORMANCE,
            description="Performance issue",
            file_path="test.py",
            line_number=100
        )

        result = issue.to_dict()

        assert result["severity"] == "medium"
        assert result["category"] == "performance"
        assert result["description"] == "Performance issue"
        assert result["file_path"] == "test.py"
        assert result["line_number"] == 100


class TestCategoryResult:
    """Test CategoryResult dataclass"""

    def test_empty_result(self):
        """Test empty category result"""
        result = CategoryResult(action_type=ActionType.INFORMATIONAL)

        assert result.action_type == ActionType.INFORMATIONAL
        assert result.total_issues == 0
        assert len(result.actionable_issues) == 0

    def test_mixed_result(self):
        """Test result with mixed issues"""
        blocking_issue = ParsedIssue(
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            description="Critical security issue"
        )

        optional_issue = ParsedIssue(
            severity=IssueSeverity.LOW,
            category=IssueCategory.STYLE,
            description="Style issue"
        )

        result = CategoryResult(
            action_type=ActionType.BLOCKING,
            blocking_issues=[blocking_issue],
            optional_issues=[optional_issue]
        )

        assert result.total_issues == 2
        assert len(result.actionable_issues) == 1  # Only blocking is actionable
        assert result.actionable_issues[0] == blocking_issue


class TestReviewResultProcessor:
    """Test ReviewResultProcessor class"""

    def test_initialization_default(self):
        """Test processor initialization with defaults"""
        processor = ReviewResultProcessor()

        assert processor.config is not None
        assert hasattr(processor, 'file_path_pattern')
        assert hasattr(processor, 'line_number_pattern')
        assert hasattr(processor, 'severity_patterns')

    def test_initialization_custom_config(self):
        """Test processor initialization with custom config"""
        config = ProcessingConfig(confidence_threshold=0.8)
        processor = ReviewResultProcessor(config)

        assert processor.config.confidence_threshold == 0.8

    def test_parse_string_output_simple(self):
        """Test parsing simple string output"""
        processor = ReviewResultProcessor()

        output = """
        1. Critical security vulnerability in user authentication
        File: auth.py, Line: 42
        Fix: Use bcrypt for password hashing

        2. Performance issue in database query
        File: db.py, Line: 100
        Consider adding indexes
        """

        result = processor.process_review_result(output)

        assert result.total_issues == 2
        assert len(result.blocking_issues) == 1
        assert len(result.recommended_issues) == 1
        assert result.action_type == ActionType.BLOCKING

    def test_parse_dict_output_zen_format(self):
        """Test parsing zen MCP codereview format"""
        processor = ReviewResultProcessor()

        output = {
            "issues_found": [
                {
                    "severity": "high",
                    "description": "Potential SQL injection vulnerability",
                    "file_path": "database.py",
                    "line_number": 25,
                    "suggested_fix": "Use parameterized queries"
                },
                {
                    "severity": "low",
                    "description": "Missing docstring for function",
                    "file_path": "utils.py",
                    "line_number": 10
                }
            ]
        }

        result = processor.process_review_result(output)

        assert result.total_issues == 2
        assert len(result.required_issues) == 1
        assert len(result.optional_issues) == 1
        assert result.action_type == ActionType.REQUIRED

        # Check specific issue details
        high_issue = result.required_issues[0]
        assert high_issue.severity == IssueSeverity.HIGH
        assert high_issue.file_path == "database.py"
        assert high_issue.line_number == 25
        assert high_issue.suggested_fix == "Use parameterized queries"

    def test_parse_list_output(self):
        """Test parsing list of issues"""
        processor = ReviewResultProcessor()

        output = [
            {
                "severity": "critical",
                "category": "security",
                "description": "Buffer overflow detected",
                "file": "buffer.c",
                "line": 150
            },
            {
                "severity": "medium",
                "category": "performance",
                "description": "Inefficient loop detected",
                "file": "loop.py",
                "line": 75
            }
        ]

        result = processor.process_review_result(output)

        assert result.total_issues == 2
        assert len(result.blocking_issues) == 1
        assert len(result.recommended_issues) == 1
        assert result.action_type == ActionType.BLOCKING

    def test_severity_classification(self):
        """Test severity classification from text"""
        processor = ReviewResultProcessor()

        test_cases = [
            ("Critical security vulnerability found", IssueSeverity.CRITICAL),
            ("Error in function call", IssueSeverity.HIGH),
            ("Warning: deprecated function used", IssueSeverity.MEDIUM),
            ("Minor style issue with formatting", IssueSeverity.LOW),
            ("This is a normal description", IssueSeverity.MEDIUM)  # Default
        ]

        for text, expected_severity in test_cases:
            classified = processor._classify_severity(text)
            assert classified == expected_severity, f"Failed for: {text}"

    def test_category_classification(self):
        """Test category classification from text"""
        processor = ReviewResultProcessor()

        test_cases = [
            ("SQL injection vulnerability", IssueCategory.SECURITY),
            ("Performance bottleneck in loop", IssueCategory.PERFORMANCE),
            ("Bug causing null pointer exception", IssueCategory.BUG),
            ("Missing unit test for function", IssueCategory.TESTING),
            ("Missing documentation for class", IssueCategory.DOCUMENTATION),
            ("Incorrect variable naming convention", IssueCategory.STYLE),
            ("Code is too complex", IssueCategory.MAINTAINABILITY),
            ("Unknown issue type", IssueCategory.CODE_QUALITY)  # Default
        ]

        for text, expected_category in test_cases:
            classified = processor._classify_category(text)
            assert classified == expected_category, f"Failed for: {text}"

    def test_file_path_extraction(self):
        """Test file path extraction from text"""
        processor = ReviewResultProcessor()

        test_cases = [
            ("Error in file auth.py", "auth.py"),
            ("File: /path/to/database.py", "/path/to/database.py"),
            ("In 'utils/helper.js'", "utils/helper.js"),
            ("At src/main.cpp line 42", "src/main.cpp"),
            ("No file mentioned here", None)
        ]

        for text, expected_path in test_cases:
            extracted = processor._extract_file_path(text)
            assert extracted == expected_path, f"Failed for: {text}"

    def test_line_number_extraction(self):
        """Test line number extraction from text"""
        processor = ReviewResultProcessor()

        test_cases = [
            ("Error at line 42", 42),
            ("Line: 100", 100),
            ("LINE #250", 250),
            ("line:75", 75),
            ("No line number here", None)
        ]

        for text, expected_line in test_cases:
            extracted = processor._extract_line_number(text)
            assert extracted == expected_line, f"Failed for: {text}"

    def test_code_snippet_extraction(self):
        """Test code snippet extraction from text"""
        processor = ReviewResultProcessor()

        test_cases = [
            ("```python\nprint('hello')\n```", "print('hello')"),
            ("Use `var x = 5` instead", "var x = 5"),
            ("Problem:\n    def bad_function():\n        pass", "def bad_function():\n    pass"),
            ("No code here", None)
        ]

        for text, expected_snippet in test_cases:
            extracted = processor._extract_code_snippet(text)
            if expected_snippet:
                assert expected_snippet in (extracted or ""), f"Failed for: {text}"
            else:
                assert extracted is None, f"Failed for: {text}"

    def test_suggested_fix_extraction(self):
        """Test suggested fix extraction from text"""
        processor = ReviewResultProcessor()

        test_cases = [
            ("Fix: Use bcrypt for hashing", "Use bcrypt for hashing"),
            ("Suggestion: Add input validation", "Add input validation"),
            ("Should use parameterized queries", "use parameterized queries"),
            ("Consider refactoring this method", "refactoring this method"),
            ("No fix suggested", None)
        ]

        for text, expected_fix in test_cases:
            extracted = processor._extract_suggested_fix(text)
            if expected_fix:
                assert expected_fix.lower() in (extracted or "").lower(), f"Failed for: {text}"
            else:
                assert extracted is None, f"Failed for: {text}"

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        processor = ReviewResultProcessor()

        # High confidence: has file path, line number, and sufficient detail
        high_conf_text = "Critical error in auth.py line 42: SQL injection vulnerability detected"
        high_conf = processor._calculate_confidence(high_conf_text)
        assert high_conf >= 0.8

        # Low confidence: minimal information
        low_conf_text = "Error"
        low_conf = processor._calculate_confidence(low_conf_text)
        assert low_conf <= 0.6

    def test_confidence_filtering(self):
        """Test that low-confidence issues are filtered out"""
        config = ProcessingConfig(confidence_threshold=0.8)
        processor = ReviewResultProcessor(config)

        # Create a mock issue that would have low confidence
        output = "Err"  # Very short, low confidence

        result = processor.process_review_result(output)

        # Should have no issues due to confidence filtering
        assert result.total_issues == 0

    def test_action_type_determination(self):
        """Test action type determination based on issue mix"""
        processor = ReviewResultProcessor()

        # Test blocking (critical issues present)
        critical_output = {"issues_found": [{"severity": "critical", "description": "Critical issue"}]}
        result = processor.process_review_result(critical_output)
        assert result.action_type == ActionType.BLOCKING

        # Test required (high issues present, no critical)
        high_output = {"issues_found": [{"severity": "high", "description": "High issue"}]}
        result = processor.process_review_result(high_output)
        assert result.action_type == ActionType.REQUIRED

        # Test recommended (medium issues present, no high/critical)
        medium_output = {"issues_found": [{"severity": "medium", "description": "Medium issue"}]}
        result = processor.process_review_result(medium_output)
        assert result.action_type == ActionType.RECOMMENDED

        # Test optional (only low issues)
        low_output = {"issues_found": [{"severity": "low", "description": "Low issue"}]}
        result = processor.process_review_result(low_output)
        assert result.action_type == ActionType.OPTIONAL

        # Test informational (no issues)
        empty_output = {"issues_found": []}
        result = processor.process_review_result(empty_output)
        assert result.action_type == ActionType.INFORMATIONAL

    def test_export_json_format(self):
        """Test JSON export format"""
        processor = ReviewResultProcessor()

        output = {
            "issues_found": [
                {"severity": "critical", "description": "Critical issue"},
                {"severity": "low", "description": "Low issue"}
            ]
        }

        result = processor.process_review_result(output)
        json_export = processor.export_issues(result, "json")

        assert isinstance(json_export, dict)
        assert json_export["action_type"] == "blocking"
        assert json_export["summary"]["total_issues"] == 2
        assert json_export["summary"]["blocking"] == 1
        assert json_export["summary"]["optional"] == 1
        assert "issues" in json_export
        assert len(json_export["issues"]["blocking"]) == 1
        assert len(json_export["issues"]["optional"]) == 1

    def test_export_text_format(self):
        """Test text export format"""
        processor = ReviewResultProcessor()

        output = {
            "issues_found": [
                {
                    "severity": "high",
                    "description": "High priority issue",
                    "file_path": "test.py",
                    "line_number": 42,
                    "suggested_fix": "Fix this issue"
                }
            ]
        }

        result = processor.process_review_result(output)
        text_export = processor.export_issues(result, "text")

        assert isinstance(text_export, str)
        assert "Code Review Results" in text_export
        assert "REQUIRED" in text_export
        assert "High priority issue" in text_export
        assert "File: test.py" in text_export
        assert "Line: 42" in text_export
        assert "Fix: Fix this issue" in text_export

    def test_export_invalid_format(self):
        """Test export with invalid format"""
        processor = ReviewResultProcessor()
        result = CategoryResult(action_type=ActionType.INFORMATIONAL)

        with pytest.raises(ValueError, match="Unsupported format type"):
            processor.export_issues(result, "invalid_format")

    def test_malformed_input_handling(self):
        """Test handling of malformed input"""
        processor = ReviewResultProcessor()

        test_cases = [
            None,
            [],
            {},
            "",
            {"invalid": "structure"},
            123,  # Invalid type
            [{"description": ""}]  # Empty description
        ]

        for malformed_input in test_cases:
            result = processor.process_review_result(malformed_input)
            # Should handle gracefully without crashing
            assert isinstance(result, CategoryResult)
            assert result.action_type in ActionType

    def test_large_input_handling(self):
        """Test handling of large input volumes"""
        processor = ReviewResultProcessor()

        # Create large input with many issues
        large_output = {
            "issues_found": [
                {
                    "severity": "medium",
                    "description": f"Issue {i}",
                    "file_path": f"file_{i}.py",
                    "line_number": i
                }
                for i in range(100)
            ]
        }

        result = processor.process_review_result(large_output)

        assert result.total_issues == 100
        assert len(result.recommended_issues) == 100
        assert result.action_type == ActionType.RECOMMENDED

    def test_edge_cases(self):
        """Test various edge cases"""
        processor = ReviewResultProcessor()

        # Very long description
        long_desc = "A" * 1000
        output = {"issues_found": [{"severity": "medium", "description": long_desc}]}
        result = processor.process_review_result(output)
        assert len(result.recommended_issues[0].description) <= processor.config.max_description_length

        # Special characters in description
        special_output = {"issues_found": [{"severity": "medium", "description": "Issue with special chars: <>&'\""}]}
        result = processor.process_review_result(special_output)
        assert result.total_issues == 1

        # Unicode characters
        unicode_output = {"issues_found": [{"severity": "medium", "description": "Unicode issue: 测试问题"}]}
        result = processor.process_review_result(unicode_output)
        assert result.total_issues == 1


class TestQuickParse:
    """Test quick_parse convenience function"""

    def test_quick_parse_function(self):
        """Test quick parse convenience function"""
        output = "Critical security vulnerability in auth.py"

        result = quick_parse(output)

        assert isinstance(result, CategoryResult)
        assert result.total_issues >= 1
        assert result.action_type == ActionType.BLOCKING

    def test_quick_parse_with_config(self):
        """Test quick parse with custom config"""
        config = ProcessingConfig(confidence_threshold=0.9)
        output = "Error"  # Low confidence

        result = quick_parse(output, config)

        # Should filter out low confidence issues
        assert result.total_issues == 0


class TestIntegration:
    """Integration tests with realistic scenarios"""

    def test_realistic_string_review(self):
        """Test with realistic string review output"""
        output = """
        Code Review Results:

        1. CRITICAL: SQL Injection vulnerability detected in user_login function
           File: auth/login.py, Line: 45
           The code directly concatenates user input into SQL query without sanitization
           Fix: Use parameterized queries or ORM methods

        2. HIGH: Memory leak in file processing
           File: utils/file_processor.py, Line: 123
           File handles are not properly closed in exception scenarios
           Fix: Use context managers (with statements) for file operations

        3. MEDIUM: Inefficient database query in user search
           File: models/user.py, Line: 78
           N+1 query problem detected
           Fix: Use select_related() or prefetch_related()

        4. LOW: Missing docstring for helper function
           File: utils/helpers.py, Line: 12
           Function lacks documentation
           Fix: Add comprehensive docstring
        """

        processor = ReviewResultProcessor()
        result = processor.process_review_result(output)

        assert result.total_issues == 4
        assert len(result.blocking_issues) == 1
        assert len(result.required_issues) == 1
        assert len(result.recommended_issues) == 1
        assert len(result.optional_issues) == 1
        assert result.action_type == ActionType.BLOCKING

        # Check specific issue details
        critical_issue = result.blocking_issues[0]
        assert critical_issue.severity == IssueSeverity.CRITICAL
        assert critical_issue.category == IssueCategory.SECURITY
        assert "SQL Injection" in critical_issue.description
        assert critical_issue.file_path == "auth/login.py"
        assert critical_issue.line_number == 45
        assert "parameterized queries" in critical_issue.suggested_fix

    def test_zen_mcp_integration(self):
        """Test integration with zen MCP codereview format"""
        # Simulate zen MCP codereview tool output
        zen_output = {
            "status": "analyze_complete",
            "issues_found": [
                {
                    "severity": "high",
                    "description": "Potential XSS vulnerability in template rendering",
                    "file_path": "templates/user_profile.html",
                    "line_number": 23,
                    "category": "security",
                    "suggested_fix": "Escape user input before rendering"
                },
                {
                    "severity": "medium",
                    "description": "Performance bottleneck in data processing loop",
                    "file_path": "processors/data_processor.py",
                    "line_number": 156,
                    "category": "performance",
                    "suggested_fix": "Consider using vectorized operations"
                },
                {
                    "severity": "low",
                    "description": "Inconsistent naming convention for variables",
                    "file_path": "models/base.py",
                    "line_number": 89,
                    "category": "style"
                }
            ],
            "confidence": "high",
            "findings": "Found 3 issues requiring attention"
        }

        processor = ReviewResultProcessor()
        result = processor.process_review_result(zen_output)

        assert result.total_issues == 3
        assert len(result.required_issues) == 1
        assert len(result.recommended_issues) == 1
        assert len(result.optional_issues) == 1
        assert result.action_type == ActionType.REQUIRED

        # Verify categories were preserved
        high_issue = result.required_issues[0]
        assert high_issue.category == IssueCategory.SECURITY

        medium_issue = result.recommended_issues[0]
        assert medium_issue.category == IssueCategory.PERFORMANCE

        low_issue = result.optional_issues[0]
        assert low_issue.category == IssueCategory.STYLE

    def test_multi_format_processing(self):
        """Test processing multiple formats in sequence"""
        processor = ReviewResultProcessor()

        # Process string format
        string_output = "Critical error in main.py line 50"
        string_result = processor.process_review_result(string_output)

        # Process dict format
        dict_output = {"issues_found": [{"severity": "high", "description": "High severity issue"}]}
        dict_result = processor.process_review_result(dict_output)

        # Process list format
        list_output = [{"severity": "medium", "description": "Medium issue"}]
        list_result = processor.process_review_result(list_output)

        # All should produce valid results
        assert string_result.total_issues >= 1
        assert dict_result.total_issues == 1
        assert list_result.total_issues == 1

        # Verify action types are appropriate
        assert string_result.action_type == ActionType.BLOCKING
        assert dict_result.action_type == ActionType.REQUIRED
        assert list_result.action_type == ActionType.RECOMMENDED


if __name__ == "__main__":
    pytest.main([__file__])
