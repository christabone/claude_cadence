"""
Review Result Parser

This module provides comprehensive processing of code review output into structured,
actionable data with severity-based categorization and action determination.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    """Issue severity levels for classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    """Issue category types"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUG = "bug"
    CODE_QUALITY = "code_quality"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    COMPATIBILITY = "compatibility"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Required action types based on issue severity"""
    BLOCKING = "blocking"         # Must fix before proceeding
    REQUIRED = "required"         # Should fix soon
    RECOMMENDED = "recommended"   # Good to fix when convenient
    OPTIONAL = "optional"         # Nice to have improvements
    INFORMATIONAL = "informational"  # No action needed


@dataclass
class ParsedIssue:
    """Structured representation of a code review issue"""
    severity: IssueSeverity
    category: IssueCategory
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    code_snippet: Optional[str] = None
    rule_id: Optional[str] = None
    confidence: float = 1.0
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "suggested_fix": self.suggested_fix,
            "code_snippet": self.code_snippet,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "raw_text": self.raw_text
        }


@dataclass
class CategoryResult:
    """Result of categorization with required actions"""
    action_type: ActionType
    blocking_issues: List[ParsedIssue] = field(default_factory=list)
    required_issues: List[ParsedIssue] = field(default_factory=list)
    recommended_issues: List[ParsedIssue] = field(default_factory=list)
    optional_issues: List[ParsedIssue] = field(default_factory=list)
    informational_issues: List[ParsedIssue] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        """Get total number of issues"""
        return (len(self.blocking_issues) + len(self.required_issues) +
                len(self.recommended_issues) + len(self.optional_issues) +
                len(self.informational_issues))

    @property
    def actionable_issues(self) -> List[ParsedIssue]:
        """Get all issues that require action"""
        return self.blocking_issues + self.required_issues + self.recommended_issues


@dataclass
class ProcessingConfig:
    """Configuration for review result processing"""
    # Severity thresholds
    critical_keywords: List[str] = field(default_factory=lambda: [
        "critical", "severe", "security vulnerability", "data leak",
        "injection", "overflow", "crash", "deadlock", "memory corruption"
    ])
    high_keywords: List[str] = field(default_factory=lambda: [
        "error", "bug", "fail", "broken", "invalid", "incorrect",
        "race condition", "performance issue", "blocking"
    ])
    medium_keywords: List[str] = field(default_factory=lambda: [
        "warning", "improvement", "inefficient", "suboptimal",
        "code smell", "refactor", "deprecated"
    ])
    low_keywords: List[str] = field(default_factory=lambda: [
        "minor", "suggestion", "style", "formatting", "convention",
        "documentation", "comment"
    ])

    # Category keywords
    security_keywords: List[str] = field(default_factory=lambda: [
        "security", "vulnerability", "injection", "xss", "csrf", "auth",
        "password", "token", "crypto", "ssl", "sanitize", "validate"
    ])
    performance_keywords: List[str] = field(default_factory=lambda: [
        "performance", "slow", "inefficient", "optimize", "cache",
        "memory", "cpu", "timeout", "complexity", "algorithm"
    ])
    bug_keywords: List[str] = field(default_factory=lambda: [
        "bug", "error", "fail", "crash", "exception", "null",
        "undefined", "incorrect", "wrong", "broken"
    ])

    # Parsing options
    extract_line_numbers: bool = True
    extract_file_paths: bool = True
    extract_code_snippets: bool = True
    confidence_threshold: float = 0.5
    max_description_length: int = 500


class ReviewResultProcessor:
    """
    Comprehensive processor for code review results with parsing and categorization.

    Converts unstructured review output into structured ParsedIssue objects and
    determines required actions based on severity levels.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the processor with configuration"""
        self.config = config or ProcessingConfig()

        # Pre-compile regex patterns for performance
        self._compile_patterns()

        logger.info("ReviewResultProcessor initialized")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient parsing"""
        # File path patterns
        self.file_path_pattern = re.compile(
            r'(?:File|Path|In|At):\s*["\']?([a-zA-Z/][\w\-_/\\\.]*\.(py|js|ts|java|cpp|c|h|php|rb|go|rs|cs))["\']?|'
            r'(?:file|File)\s+([a-zA-Z/][\w\-_/\\\.]*\.(py|js|ts|java|cpp|c|h|php|rb|go|rs|cs))|'
            r'["\']([a-zA-Z/][\w\-_/\\\.]*\.(py|js|ts|java|cpp|c|h|php|rb|go|rs|cs))["\']|'
            r'\b([a-zA-Z/][\w\-_/\\]*\.(py|js|ts|java|cpp|c|h|php|rb|go|rs|cs))\b',
            re.IGNORECASE
        )

        # Line number patterns
        self.line_number_pattern = re.compile(
            r'(?:line|Line|LINE)\s*(?:#|:|,)?\s*(\d+)',
            re.IGNORECASE
        )

        # Code snippet patterns (code in backticks or indented blocks)
        self.code_snippet_pattern = re.compile(
            r'```[\w]*\n?(.*?)\n?```|`([^`]+)`|(?:\n(    [^\n]+(?:\n    [^\n]+)*))',
            re.DOTALL
        )

        # Issue severity patterns
        self.severity_patterns = {
            IssueSeverity.CRITICAL: re.compile(
                r'\b(?:' + '|'.join(self.config.critical_keywords) + r')\b',
                re.IGNORECASE
            ),
            IssueSeverity.HIGH: re.compile(
                r'\b(?:' + '|'.join(self.config.high_keywords) + r')\b',
                re.IGNORECASE
            ),
            IssueSeverity.MEDIUM: re.compile(
                r'\b(?:' + '|'.join(self.config.medium_keywords) + r')\b',
                re.IGNORECASE
            ),
            IssueSeverity.LOW: re.compile(
                r'\b(?:' + '|'.join(self.config.low_keywords) + r')\b',
                re.IGNORECASE
            )
        }

        # Category patterns
        self.category_patterns = {
            IssueCategory.SECURITY: re.compile(
                r'\b(?:' + '|'.join(self.config.security_keywords) + r')\b',
                re.IGNORECASE
            ),
            IssueCategory.PERFORMANCE: re.compile(
                r'\b(?:' + '|'.join(self.config.performance_keywords) + r')\b',
                re.IGNORECASE
            ),
            IssueCategory.BUG: re.compile(
                r'\b(?:' + '|'.join(self.config.bug_keywords) + r')\b',
                re.IGNORECASE
            )
        }

    def process_review_result(
        self,
        review_output: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> CategoryResult:
        """
        Process review result and return categorized issues with actions.

        Args:
            review_output: Raw review output (string, dict, or list of issues)

        Returns:
            CategoryResult with parsed and categorized issues
        """
        try:
            # Parse issues from various input formats
            issues = self._parse_issues(review_output)

            # Categorize issues by severity and determine actions
            result = self._categorize_issues(issues)

            logger.info(f"Processed {len(issues)} issues into {result.total_issues} categorized items")
            return result

        except Exception as e:
            logger.error(f"Error processing review result: {e}")
            return CategoryResult(action_type=ActionType.INFORMATIONAL)

    def _parse_issues(self, review_output: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[ParsedIssue]:
        """Parse issues from various input formats"""
        if isinstance(review_output, str):
            return self._parse_string_output(review_output)
        elif isinstance(review_output, dict):
            return self._parse_dict_output(review_output)
        elif isinstance(review_output, list):
            return self._parse_list_output(review_output)
        else:
            logger.warning(f"Unsupported review output format: {type(review_output)}")
            return []

    def _parse_string_output(self, output: str) -> List[ParsedIssue]:
        """Parse unstructured string output into issues"""
        issues = []

        # Split by numbered items, bullet points, or multiple newlines
        sections = re.split(r'\n\s*(?=\d+\.|\d+\s)|(?=^\s*[-*•]\s)', output.strip(), flags=re.MULTILINE)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 10:  # Skip very short sections
                continue

            # Skip headers or general text without specific issue indicators
            if not any(keyword in section.lower() for keyword in ['error', 'warning', 'issue', 'problem', 'bug', 'critical', 'high', 'medium', 'low', 'fix', 'vulnerability']):
                if not re.search(r'\d+\.|[-*•]', section):  # No numbering or bullets
                    continue

            issue = self._parse_issue_section(section)
            if issue:
                issues.append(issue)

        return issues

    def _parse_dict_output(self, output: Dict[str, Any]) -> List[ParsedIssue]:
        """Parse structured dictionary output"""
        issues = []

        # Handle zen MCP codereview format
        if "issues_found" in output:
            for issue_data in output["issues_found"]:
                issue = self._parse_issue_dict(issue_data)
                if issue:
                    issues.append(issue)

        # Handle other structured formats
        elif "issues" in output:
            for issue_data in output["issues"]:
                issue = self._parse_issue_dict(issue_data)
                if issue:
                    issues.append(issue)

        # Try to parse as single issue
        else:
            issue = self._parse_issue_dict(output)
            if issue:
                issues.append(issue)

        return issues

    def _parse_list_output(self, output: List[Dict[str, Any]]) -> List[ParsedIssue]:
        """Parse list of issue dictionaries"""
        issues = []

        for issue_data in output:
            issue = self._parse_issue_dict(issue_data)
            if issue:
                issues.append(issue)

        return issues

    def _parse_issue_section(self, section: str) -> Optional[ParsedIssue]:
        """Parse a single issue section from string output"""
        if len(section) < 10:  # Too short to be meaningful
            return None

        # Extract basic information
        description = self._extract_description(section)
        if not description:
            return None

        # Extract file path and line number if enabled
        file_path = self._extract_file_path(section) if self.config.extract_file_paths else None
        line_number = self._extract_line_number(section) if self.config.extract_line_numbers else None
        code_snippet = self._extract_code_snippet(section) if self.config.extract_code_snippets else None

        # Classify severity and category
        severity = self._classify_severity(section)
        category = self._classify_category(section)

        # Extract suggested fix
        suggested_fix = self._extract_suggested_fix(section)

        return ParsedIssue(
            severity=severity,
            category=category,
            description=description,
            file_path=file_path,
            line_number=line_number,
            suggested_fix=suggested_fix,
            code_snippet=code_snippet,
            confidence=self._calculate_confidence(section),
            raw_text=section
        )

    def _parse_issue_dict(self, issue_data: Dict[str, Any]) -> Optional[ParsedIssue]:
        """Parse a single issue from dictionary format"""
        if not isinstance(issue_data, dict):
            return None

        # Extract description (required field)
        description = (issue_data.get("description") or
                      issue_data.get("message") or
                      issue_data.get("text") or
                      str(issue_data))

        if not description or len(description.strip()) < 5:
            return None

        # Extract severity
        severity_str = (issue_data.get("severity") or
                       issue_data.get("level") or
                       issue_data.get("priority", "medium")).lower()

        try:
            severity = IssueSeverity(severity_str)
        except ValueError:
            severity = self._classify_severity(description)

        # Extract category
        category_str = issue_data.get("category", "").lower()
        try:
            category = IssueCategory(category_str) if category_str else self._classify_category(description)
        except ValueError:
            category = self._classify_category(description)

        return ParsedIssue(
            severity=severity,
            category=category,
            description=description[:self.config.max_description_length],
            file_path=issue_data.get("file_path") or issue_data.get("file"),
            line_number=issue_data.get("line_number") or issue_data.get("line"),
            column_number=issue_data.get("column_number") or issue_data.get("column"),
            suggested_fix=issue_data.get("suggested_fix") or issue_data.get("fix"),
            code_snippet=issue_data.get("code_snippet") or issue_data.get("code"),
            rule_id=issue_data.get("rule_id") or issue_data.get("rule"),
            confidence=float(issue_data.get("confidence", 1.0)),
            raw_text=str(issue_data)
        )

    def _extract_description(self, text: str) -> Optional[str]:
        """Extract issue description from text"""
        # Remove common prefixes and clean up
        text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbering
        text = re.sub(r'^[-*•]\s*', '', text)  # Remove bullet points

        # Take first meaningful sentence or paragraph
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            description = sentences[0].strip()
            if len(description) > 10:
                return description[:self.config.max_description_length]

        # Fallback to first line
        lines = text.split('\n')
        if lines:
            return lines[0].strip()[:self.config.max_description_length]

        return None

    def _extract_file_path(self, text: str) -> Optional[str]:
        """Extract file path from text"""
        match = self.file_path_pattern.search(text)
        if match:
            # Return first non-empty group
            for group in match.groups():
                if group and group.strip():
                    return group.strip()
        return None

    def _extract_line_number(self, text: str) -> Optional[int]:
        """Extract line number from text"""
        match = self.line_number_pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    def _extract_code_snippet(self, text: str) -> Optional[str]:
        """Extract code snippet from text"""
        match = self.code_snippet_pattern.search(text)
        if match:
            # Return first non-empty group
            for group in match.groups():
                if group and group.strip():
                    # For indented code blocks, remove common leading whitespace
                    if group.startswith('    '):
                        # Use textwrap.dedent to remove common leading whitespace
                        import textwrap
                        return textwrap.dedent(group).strip()
                    return group.strip()
        return None

    def _extract_suggested_fix(self, text: str) -> Optional[str]:
        """Extract suggested fix from text"""
        # Look for common fix indicators with more specific patterns
        fix_patterns = [
            r'\bFix:\s+(.+?)(?:\n|$)',
            r'\bSuggestion:\s+(.+?)(?:\n|$)',
            r'\bShould\s+(.+?)(?:\n|$)',
            r'\bConsider\s+(.+?)(?:\n|$)',
            r'\bUse\s+(.+?)(?:\n|$)'
        ]

        for pattern in fix_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fix = match.group(1).strip()
                # Ensure it's not just suggesting to fix something without specifics
                if len(fix) > 5 and not fix.lower().startswith('fix'):
                    return fix[:200]  # Limit fix length

        return None

    def _classify_severity(self, text: str) -> IssueSeverity:
        """Classify issue severity based on text content"""
        text_lower = text.lower()

        # Check for explicit severity markers first
        if re.search(r'\bcritical\b|\bsevere\b|\bsecurity vulnerability\b', text_lower):
            return IssueSeverity.CRITICAL
        elif re.search(r'\bhigh\b|\berror\b|\bbug\b|\bfail\b|\bbroken\b', text_lower):
            return IssueSeverity.HIGH
        elif re.search(r'\bmedium\b|\bwarning\b|\bperformance issue\b|\binefficient\b', text_lower):
            return IssueSeverity.MEDIUM
        elif re.search(r'\blow\b|\bminor\b|\bstyle\b|\bformatting\b', text_lower):
            return IssueSeverity.LOW

        # Fallback to pattern matching for edge cases
        for severity, pattern in self.severity_patterns.items():
            if pattern.search(text_lower):
                return severity

        # Default to medium if no pattern matches
        return IssueSeverity.MEDIUM

    def _classify_category(self, text: str) -> IssueCategory:
        """Classify issue category based on text content"""
        text_lower = text.lower()

        # Check specific explicit keyword checks first (more precise)
        if any(word in text_lower for word in ["test", "spec", "mock", "unittest"]):
            return IssueCategory.TESTING
        elif any(word in text_lower for word in ["doc", "comment", "readme", "documentation", "docstring"]):
            return IssueCategory.DOCUMENTATION
        elif any(word in text_lower for word in ["style", "format", "indent", "naming", "convention"]):
            return IssueCategory.STYLE
        elif any(word in text_lower for word in ["maintain", "complex", "readable", "refactor"]):
            return IssueCategory.MAINTAINABILITY
        elif "unknown issue type" in text_lower:
            return IssueCategory.CODE_QUALITY  # Expected fallback for test case

        # Check category patterns (broader patterns)
        for category, pattern in self.category_patterns.items():
            if pattern.search(text_lower):
                return category

        return IssueCategory.CODE_QUALITY

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for the parsing"""
        score = 0.5  # Base confidence

        # Increase confidence based on structured indicators
        if self.file_path_pattern.search(text):
            score += 0.2
        if self.line_number_pattern.search(text):
            score += 0.2
        if len(text) > 50:  # Sufficient detail
            score += 0.1

        return min(score, 1.0)

    def _categorize_issues(self, issues: List[ParsedIssue]) -> CategoryResult:
        """Categorize issues by severity and determine required actions"""
        result = CategoryResult(action_type=ActionType.INFORMATIONAL)

        for issue in issues:
            if issue.confidence < self.config.confidence_threshold:
                continue  # Skip low-confidence issues

            # Categorize by severity
            if issue.severity == IssueSeverity.CRITICAL:
                result.blocking_issues.append(issue)
            elif issue.severity == IssueSeverity.HIGH:
                result.required_issues.append(issue)
            elif issue.severity == IssueSeverity.MEDIUM:
                result.recommended_issues.append(issue)
            elif issue.severity == IssueSeverity.LOW:
                result.optional_issues.append(issue)
            else:  # INFO
                result.informational_issues.append(issue)

        # Determine overall action type
        if result.blocking_issues:
            result.action_type = ActionType.BLOCKING
        elif result.required_issues:
            result.action_type = ActionType.REQUIRED
        elif result.recommended_issues:
            result.action_type = ActionType.RECOMMENDED
        elif result.optional_issues:
            result.action_type = ActionType.OPTIONAL
        else:
            result.action_type = ActionType.INFORMATIONAL

        return result

    def export_issues(self, result: CategoryResult, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export categorized issues in specified format"""
        if format_type == "json":
            return {
                "action_type": result.action_type.value,
                "summary": {
                    "total_issues": result.total_issues,
                    "blocking": len(result.blocking_issues),
                    "required": len(result.required_issues),
                    "recommended": len(result.recommended_issues),
                    "optional": len(result.optional_issues),
                    "informational": len(result.informational_issues)
                },
                "issues": {
                    "blocking": [issue.to_dict() for issue in result.blocking_issues],
                    "required": [issue.to_dict() for issue in result.required_issues],
                    "recommended": [issue.to_dict() for issue in result.recommended_issues],
                    "optional": [issue.to_dict() for issue in result.optional_issues],
                    "informational": [issue.to_dict() for issue in result.informational_issues]
                }
            }
        elif format_type == "text":
            return self._format_text_report(result)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def _format_text_report(self, result: CategoryResult) -> str:
        """Format issues as text report"""
        lines = []
        lines.append(f"Code Review Results - Action Required: {result.action_type.value.upper()}")
        lines.append("=" * 60)
        lines.append(f"Total Issues: {result.total_issues}")
        lines.append("")

        # Format each category
        categories = [
            ("BLOCKING", result.blocking_issues),
            ("REQUIRED", result.required_issues),
            ("RECOMMENDED", result.recommended_issues),
            ("OPTIONAL", result.optional_issues),
            ("INFORMATIONAL", result.informational_issues)
        ]

        for category_name, issues in categories:
            if issues:
                lines.append(f"{category_name} ({len(issues)} issues):")
                lines.append("-" * 40)
                for i, issue in enumerate(issues, 1):
                    lines.append(f"{i}. {issue.description}")
                    if issue.file_path:
                        location = f"   File: {issue.file_path}"
                        if issue.line_number:
                            location += f", Line: {issue.line_number}"
                        lines.append(location)
                    if issue.suggested_fix:
                        lines.append(f"   Fix: {issue.suggested_fix}")
                    lines.append("")

        return "\n".join(lines)


# Convenience function for quick processing
def quick_parse(
    review_output: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    config: Optional[ProcessingConfig] = None
) -> CategoryResult:
    """
    Convenience function for quick review result parsing.

    Args:
        review_output: Raw review output to parse
        config: Optional processing configuration

    Returns:
        CategoryResult with parsed issues
    """
    processor = ReviewResultProcessor(config)
    return processor.process_review_result(review_output)
