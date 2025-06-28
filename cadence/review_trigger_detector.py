"""
Review Trigger Detector for Claude Cadence

This module provides the ReviewTriggerDetector class which serves as the primary
entry point for the automated review system. It detects patterns in streaming
output that should trigger code reviews, task completions, or help requests.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Pattern, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Types of triggers that can be detected"""
    TASK_COMPLETE = "task_complete"
    ALL_TASKS_COMPLETE = "all_tasks_complete"
    HELP_NEEDED = "help_needed"
    CODE_REVIEW_REQUEST = "code_review_request"
    ERROR_PATTERN = "error_pattern"
    ZEN_MCP_CALL = "zen_mcp_call"
    JSON_DECISION = "json_decision"
    STATUS_CHANGE = "status_change"


@dataclass
class TriggerContext:
    """Context information extracted when a trigger fires"""
    trigger_type: TriggerType
    confidence: float  # 0.0 to 1.0
    matched_text: str
    line_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Extracted context data
    task_id: Optional[str] = None
    task_title: Optional[str] = None
    modified_files: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    project_root: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    help_category: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerPattern:
    """Configuration for a trigger pattern"""
    name: str
    trigger_type: TriggerType
    pattern: Union[str, Pattern]
    context_extractors: List[str] = field(default_factory=list)
    confidence: float = 1.0
    enabled: bool = True
    priority: int = 50  # Lower number = higher priority
    description: Optional[str] = None


class ReviewTriggerDetector:
    """
    Primary entry point for the Claude Cadence automated review system.

    Detects patterns in streaming output that should trigger code reviews,
    task completions, help requests, or other workflow events. Supports
    configurable trigger patterns and context extraction.

    Features:
    - Configurable regex and JSON path trigger patterns
    - Context extraction (files, task IDs, error details)
    - Priority-based pattern matching
    - Integration with existing config.yaml structure
    - Line-by-line and buffer-based detection modes
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trigger detector.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config or {}
        self.patterns: List[TriggerPattern] = []
        self.callbacks: Dict[TriggerType, List[Callable]] = {}
        self.buffer = ""
        self.line_count = 0

        # Initialize default patterns
        self._load_default_patterns()

        # Load custom patterns from config
        self._load_config_patterns()

        # Cache sorted patterns for performance (fix for line 261 O(n*m) issue)
        self._sorted_patterns = sorted(self.patterns, key=lambda p: p.priority)

        # Create extractor dispatch map for security (fix for line 307 getattr risk)
        self._extractor_map = {
            "extract_json_task_data": self.extract_json_task_data,
            "extract_json_decision": self.extract_json_decision,
            "extract_error_details": self.extract_error_details,
            "extract_mcp_context": self.extract_mcp_context
        }

        logger.info(f"ReviewTriggerDetector initialized with {len(self.patterns)} patterns")

    def _load_default_patterns(self) -> None:
        """Load built-in trigger patterns based on research findings"""

        # Task completion patterns
        self.patterns.extend([
            TriggerPattern(
                name="all_tasks_complete",
                trigger_type=TriggerType.ALL_TASKS_COMPLETE,
                pattern=re.compile(r"ALL TASKS COMPLETE", re.IGNORECASE),
                confidence=0.95,
                priority=10,
                description="Primary project completion signal"
            ),
            TriggerPattern(
                name="task_complete_structured",
                trigger_type=TriggerType.TASK_COMPLETE,
                pattern=re.compile(r'\{"task_complete":\s*\{[^}]+\}\}'),
                context_extractors=["extract_json_task_data"],
                confidence=0.9,
                priority=15,
                description="Structured JSON task completion"
            ),
            TriggerPattern(
                name="status_complete",
                trigger_type=TriggerType.TASK_COMPLETE,
                pattern=re.compile(r"Status:\s*COMPLETE", re.IGNORECASE),
                confidence=0.8,
                priority=20,
                description="Status completion marker"
            ),
        ])

        # Help and stuck patterns
        self.patterns.extend([
            TriggerPattern(
                name="help_needed",
                trigger_type=TriggerType.HELP_NEEDED,
                pattern=re.compile(r"HELP NEEDED", re.IGNORECASE),
                confidence=0.9,
                priority=5,
                description="Primary help request signal"
            ),
            TriggerPattern(
                name="status_stuck",
                trigger_type=TriggerType.HELP_NEEDED,
                pattern=re.compile(r"Status:\s*STUCK", re.IGNORECASE),
                confidence=0.85,
                priority=10,
                description="Stuck status indicator"
            ),
        ])

        # JSON decision patterns
        self.patterns.extend([
            TriggerPattern(
                name="json_decision",
                trigger_type=TriggerType.JSON_DECISION,
                pattern=re.compile(r'\{\s*"action":\s*"(execute|skip|complete)"[^}]*\}'),
                context_extractors=["extract_json_decision"],
                confidence=0.9,
                priority=15,
                description="JSON decision object from orchestrator"
            ),
        ])

        # Zen MCP patterns
        self.patterns.extend([
            TriggerPattern(
                name="zen_codereview_call",
                trigger_type=TriggerType.ZEN_MCP_CALL,
                pattern=re.compile(r"mcp__zen__codereview"),
                context_extractors=["extract_mcp_context"],
                confidence=0.95,
                priority=5,
                description="Zen MCP code review call"
            ),
        ])

        # Error patterns
        self.patterns.extend([
            TriggerPattern(
                name="import_error",
                trigger_type=TriggerType.ERROR_PATTERN,
                pattern=re.compile(r"(ModuleNotFoundError|ImportError|cannot import name)", re.IGNORECASE),
                context_extractors=["extract_error_details"],
                confidence=0.8,
                priority=25,
                description="Python import errors"
            ),
            TriggerPattern(
                name="file_error",
                trigger_type=TriggerType.ERROR_PATTERN,
                pattern=re.compile(r"(FileNotFoundError|No such file|does not exist)", re.IGNORECASE),
                context_extractors=["extract_error_details"],
                confidence=0.8,
                priority=25,
                description="File system errors"
            ),
        ])

    def _load_config_patterns(self) -> None:
        """Load trigger patterns from configuration"""
        triggers_config = self.config.get("triggers", {})
        custom_patterns = triggers_config.get("patterns", [])

        for pattern_config in custom_patterns:
            try:
                pattern = TriggerPattern(
                    name=pattern_config["name"],
                    trigger_type=TriggerType(pattern_config["type"]),
                    pattern=re.compile(pattern_config["pattern"],
                                     pattern_config.get("flags", 0)),
                    context_extractors=pattern_config.get("extractors", []),
                    confidence=pattern_config.get("confidence", 1.0),
                    enabled=pattern_config.get("enabled", True),
                    priority=pattern_config.get("priority", 50),
                    description=pattern_config.get("description")
                )
                self.patterns.append(pattern)
                logger.debug(f"Loaded custom pattern: {pattern.name}")
            except Exception as e:
                logger.error(f"Failed to load custom pattern {pattern_config.get('name', 'unknown')}: {e}")

    def register_callback(self, trigger_type: TriggerType, callback: Callable[[TriggerContext], None]) -> None:
        """
        Register a callback for a specific trigger type.

        Args:
            trigger_type: Type of trigger to listen for
            callback: Function to call when trigger fires
        """
        if trigger_type not in self.callbacks:
            self.callbacks[trigger_type] = []
        self.callbacks[trigger_type].append(callback)
        logger.debug(f"Registered callback for {trigger_type.value}")

    def process_line(self, line: str) -> List[TriggerContext]:
        """
        Process a single line of output for trigger detection.

        Args:
            line: Line of output to analyze

        Returns:
            List of triggered contexts
        """
        self.line_count += 1
        self.buffer += line + "\n"

        # Keep buffer size reasonable
        if len(self.buffer) > 50000:  # 50KB limit
            # Keep last 25KB
            self.buffer = self.buffer[-25000:]

        triggers = []

        # Check patterns against the current line (using cached sorted patterns)
        for pattern in self._sorted_patterns:
            if not pattern.enabled:
                continue

            match = pattern.pattern.search(line)
            if match:
                context = self._extract_context(pattern, match, line)
                context.line_number = self.line_count
                triggers.append(context)

                # Trigger callbacks
                self._trigger_callbacks(context)

                logger.debug(f"Triggered {pattern.name} on line {self.line_count}")

        return triggers

    def process_buffer(self, text: str) -> List[TriggerContext]:
        """
        Process a block of text for trigger detection.

        Args:
            text: Text block to analyze

        Returns:
            List of triggered contexts
        """
        triggers = []
        lines = text.split("\n")

        for line in lines:
            line_triggers = self.process_line(line)
            triggers.extend(line_triggers)

        return triggers

    def _extract_context(self, pattern: TriggerPattern, match: re.Match, line: str) -> TriggerContext:
        """Extract context information from a pattern match"""
        context = TriggerContext(
            trigger_type=pattern.trigger_type,
            confidence=pattern.confidence,
            matched_text=match.group(0)
        )

        # Run context extractors (using secure dispatch map)
        for extractor_name in pattern.context_extractors:
            extractor = self._extractor_map.get(extractor_name)
            if extractor:
                try:
                    extractor(context, match, line, self.buffer)
                except Exception as e:
                    logger.error(f"Error in context extractor {extractor_name}: {e}")
            else:
                logger.warning(f"Unknown context extractor: {extractor_name}")

        return context

    def extract_json_task_data(self, context: TriggerContext, match: re.Match, line: str, buffer: str) -> None:
        """Extract task data from JSON task completion"""
        try:
            json_text = match.group(0)
            data = json.loads(json_text)

            if "task_complete" in data:
                task_data = data["task_complete"]
                context.task_id = task_data.get("task_id")
                context.task_title = task_data.get("title")
                context.json_data = data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON task data: {e}")

    def extract_json_decision(self, context: TriggerContext, match: re.Match, line: str, buffer: str) -> None:
        """Extract data from JSON decision object"""
        try:
            json_text = match.group(0)
            data = json.loads(json_text)

            context.task_id = data.get("task_id")
            context.session_id = data.get("session_id")
            context.project_root = data.get("project_root")
            context.json_data = data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON decision: {e}")

    def extract_mcp_context(self, context: TriggerContext, match: re.Match, line: str, buffer: str) -> None:
        """Extract context from MCP tool calls"""
        # Look for common MCP patterns in surrounding lines
        lines = buffer.split("\n")
        for i, buf_line in enumerate(lines):
            if match.group(0) in buf_line:
                # Check surrounding lines for context
                start_idx = max(0, i - 3)
                end_idx = min(len(lines), i + 4)

                for j in range(start_idx, end_idx):
                    # Look for file paths
                    file_match = re.search(r'"file_path":\s*"([^"]+)"', lines[j])
                    if file_match:
                        context.modified_files.append(file_match.group(1))

                    # Look for step numbers or task references
                    task_match = re.search(r'task[_\s]*(\d+)', lines[j], re.IGNORECASE)
                    if task_match:
                        context.task_id = task_match.group(1)
                break

    def extract_error_details(self, context: TriggerContext, match: re.Match, line: str, buffer: str) -> None:
        """Extract details from error patterns"""
        context.error_details = {
            "error_type": match.group(1),
            "full_line": line.strip(),
            "context": line
        }

        # Look for file paths in the error
        file_match = re.search(r'["\']([^"\']*\.(py|js|ts|json|yaml|yml))["\']', line)
        if file_match:
            context.modified_files.append(file_match.group(1))

    def _trigger_callbacks(self, context: TriggerContext) -> None:
        """Trigger registered callbacks for a context"""
        callbacks = self.callbacks.get(context.trigger_type, [])
        for callback in callbacks:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Error in trigger callback for {context.trigger_type.value}: {e}")

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns"""
        return {
            "total_patterns": len(self.patterns),
            "enabled_patterns": len([p for p in self.patterns if p.enabled]),
            "patterns_by_type": {
                trigger_type.value: len([p for p in self.patterns if p.trigger_type == trigger_type])
                for trigger_type in TriggerType
            },
            "registered_callbacks": {
                trigger_type.value: len(callbacks)
                for trigger_type, callbacks in self.callbacks.items()
            }
        }

    def clear_buffer(self) -> None:
        """Clear the internal buffer"""
        self.buffer = ""
        self.line_count = 0
        logger.debug("Cleared trigger detector buffer")
