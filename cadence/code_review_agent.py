"""
Code Review Agent Wrapper

This module provides a controlled wrapper around zen MCP code review functionality
with enhanced error handling, model fallback, and token management.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
# Note: retry_utils not needed for MCP responses - they don't need retry

logger = logging.getLogger(__name__)


class ReviewSeverity(str, Enum):
    """Review severity levels for filtering"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ALL = "all"


class ReviewType(str, Enum):
    """Types of code review to perform"""
    FULL = "full"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUICK = "quick"


class ModelProvider(str, Enum):
    """Supported model providers"""
    GEMINI = "gemini-2.5-pro"
    O3 = "o3"
    O3_MINI = "o3-mini"
    FLASH = "gemini-2.5-flash"


@dataclass
class ReviewConfig:
    """Configuration for code review parameters"""
    review_type: ReviewType = ReviewType.FULL
    severity_filter: ReviewSeverity = ReviewSeverity.MEDIUM
    focus_areas: List[str] = field(default_factory=list)
    max_steps: int = 3
    primary_model: ModelProvider = ModelProvider.GEMINI
    fallback_models: List[ModelProvider] = field(default_factory=lambda: [ModelProvider.O3_MINI, ModelProvider.FLASH])
    max_retries: int = 2
    retry_delay: float = 1.0
    chunk_files: bool = True
    max_files_per_chunk: int = 3
    temperature: float = 0.1
    thinking_mode: str = "medium"


@dataclass
class ReviewResult:
    """Result of a code review operation"""
    success: bool
    model_used: Optional[ModelProvider] = None
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "unknown"
    step_number: int = 0
    total_steps: int = 0
    findings: str = ""
    error_message: Optional[str] = None
    token_limit_exceeded: bool = False
    files_reviewed: List[str] = field(default_factory=list)
    review_metadata: Dict[str, Any] = field(default_factory=dict)


class CodeReviewAgent:
    """
    Wrapper around zen MCP code review functionality with enhanced control.

    Provides structured configuration, model fallback, error handling,
    and token management for reliable code review operations.
    """

    def __init__(
        self,
        config: Optional[ReviewConfig] = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the code review agent.

        Args:
            config: Review configuration
            mcp_client: Optional MCP client for dependency injection
        """
        self.config = config or ReviewConfig()
        self.mcp_client = mcp_client
        self._review_history: List[ReviewResult] = []

        logger.info(f"CodeReviewAgent initialized with {self.config.primary_model} primary model")

    def review_files(
        self,
        file_paths: List[Union[str, Path]],
        context_description: Optional[str] = None,
        custom_config: Optional[ReviewConfig] = None
    ) -> ReviewResult:
        """
        Review specified files with comprehensive error handling.

        Args:
            file_paths: List of file paths to review
            context_description: Optional context about the changes
            custom_config: Optional configuration override

        Returns:
            ReviewResult with findings and metadata
        """
        config = custom_config or self.config

        # Convert paths to strings and validate
        validated_paths = self._validate_file_paths(file_paths)
        if not validated_paths:
            return ReviewResult(
                success=False,
                error_message="No valid files provided for review"
            )

        # Chunk files if needed to avoid token limits
        if config.chunk_files and len(validated_paths) > config.max_files_per_chunk:
            return self._review_chunked_files(validated_paths, context_description, config)

        # Attempt review with primary model and fallbacks
        for attempt, model in enumerate([config.primary_model] + config.fallback_models):
            try:
                logger.info(f"Attempting code review with {model} (attempt {attempt + 1})")

                result = self._perform_review(
                    file_paths=validated_paths,
                    model=model,
                    config=config,
                    context_description=context_description
                )

                if result.success:
                    result.model_used = model
                    self._review_history.append(result)
                    logger.info(f"Code review completed successfully with {model}")
                    return result

                # Check for token limit issues
                if result.token_limit_exceeded and attempt < len(config.fallback_models):
                    logger.warning(f"Token limit exceeded with {model}, trying fallback")
                    continue

                # For other errors, try next model after delay
                if attempt < len(config.fallback_models):
                    logger.warning(f"Review failed with {model}: {result.error_message}, retrying with fallback")
                    time.sleep(config.retry_delay)
                    continue

                # Last attempt failed
                return result

            except Exception as e:
                logger.error(f"Unexpected error with {model}: {e}")
                if attempt == len(config.fallback_models):
                    return ReviewResult(
                        success=False,
                        error_message=f"All models failed. Last error: {str(e)}"
                    )
                time.sleep(config.retry_delay)

        return ReviewResult(
            success=False,
            error_message="Exhausted all model options"
        )

    def _validate_file_paths(self, file_paths: List[Union[str, Path]]) -> List[str]:
        """Validate and convert file paths to absolute strings"""
        validated = []

        for path in file_paths:
            try:
                path_obj = Path(path)
                if path_obj.exists() and path_obj.is_file():
                    validated.append(str(path_obj.absolute()))
                else:
                    logger.warning(f"File not found or not a file: {path}")
            except Exception as e:
                logger.warning(f"Invalid path {path}: {e}")

        return validated

    def _review_chunked_files(
        self,
        file_paths: List[str],
        context_description: Optional[str],
        config: ReviewConfig
    ) -> ReviewResult:
        """Review files in chunks to manage token limits"""
        chunks = [
            file_paths[i:i + config.max_files_per_chunk]
            for i in range(0, len(file_paths), config.max_files_per_chunk)
        ]

        all_issues = []
        all_findings = []
        all_files_reviewed = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Reviewing chunk {i + 1}/{len(chunks)} with {len(chunk)} files")

            chunk_context = f"{context_description} (Chunk {i + 1}/{len(chunks)})" if context_description else f"Chunk {i + 1}/{len(chunks)}"

            result = self._perform_review(
                file_paths=chunk,
                model=config.primary_model,
                config=config,
                context_description=chunk_context
            )

            if result.success:
                all_issues.extend(result.issues_found)
                all_findings.append(f"Chunk {i + 1}: {result.findings}")
                all_files_reviewed.extend(result.files_reviewed)
            else:
                logger.warning(f"Chunk {i + 1} failed: {result.error_message}")
                # Continue with other chunks

        return ReviewResult(
            success=len(all_files_reviewed) > 0,
            model_used=config.primary_model,
            issues_found=all_issues,
            findings="\n\n".join(all_findings),
            files_reviewed=all_files_reviewed,
            confidence="medium" if all_files_reviewed else "low"
        )

    def _perform_review(
        self,
        file_paths: List[str],
        model: ModelProvider,
        config: ReviewConfig,
        context_description: Optional[str] = None
    ) -> ReviewResult:
        """
        Perform the actual code review using zen MCP tools.

        This method interfaces with the MCP codereview tool and handles
        the specific tool interactions and response parsing.

        Note: This is a placeholder for MCP tool integration. In actual usage,
        this would need to be called through the MCP client interface.
        """
        try:
            # Prepare review parameters for MCP tool call
            review_params = {
                "step": self._build_review_step_description(file_paths, context_description),
                "step_number": 1,
                "total_steps": config.max_steps,
                "next_step_required": config.max_steps > 1,
                "findings": f"Starting {config.review_type.value} code review of {len(file_paths)} files",
                "model": model.value,
                "relevant_files": file_paths,
                "review_type": config.review_type.value,
                "severity_filter": config.severity_filter.value,
                "confidence": "exploring",
                "temperature": config.temperature,
                "thinking_mode": config.thinking_mode
            }

            # Add focus areas if specified
            if config.focus_areas:
                review_params["focus_areas"] = config.focus_areas

            # If MCP client is provided, use it; otherwise this is a simulation
            if self.mcp_client:
                try:
                    response = self.mcp_client.call_tool("mcp__zen__codereview", review_params)
                except Exception as e:
                    error_msg = str(e)
                    token_exceeded = "exceeds maximum allowed tokens" in error_msg

                    return ReviewResult(
                        success=False,
                        token_limit_exceeded=token_exceeded,
                        error_message=error_msg,
                        files_reviewed=file_paths if token_exceeded else []
                    )
            else:
                # Simulation mode for testing - return a mock successful result
                logger.warning("No MCP client provided, running in simulation mode")
                return ReviewResult(
                    success=True,
                    model_used=model,
                    confidence="simulated",
                    findings=f"Simulated review of {len(file_paths)} files completed",
                    files_reviewed=file_paths,
                    issues_found=[
                        {
                            "severity": "medium",
                            "description": f"Simulated issue in {Path(file_paths[0]).name}" if file_paths else "No files"
                        }
                    ]
                )

            # Parse response - no retry needed for MCP responses
            try:
                # If response is a string, parse it as JSON
                if isinstance(response, str):
                    parsed_response = json.loads(response)
                else:
                    parsed_response = response

                # Check for token limit issues
                if isinstance(parsed_response, dict):
                    if "error" in parsed_response and "exceeds maximum allowed tokens" in str(parsed_response["error"]):
                        return ReviewResult(
                            success=False,
                            token_limit_exceeded=True,
                            error_message="Token limit exceeded",
                            files_reviewed=file_paths
                        )

                    return self._parse_review_response(parsed_response, file_paths)

                return ReviewResult(
                    success=False,
                    error_message="Unexpected response format from review tool"
                )

            except json.JSONDecodeError as e:
                # Truncate response for logging if too long
                truncated_response = response if len(str(response)) <= 500 else str(response)[:500] + "..."
                logger.error(f"Failed to parse MCP response - JSONDecodeError: {e}\n"
                           f"Error position: line {e.lineno}, column {e.colno}\n"
                           f"Response preview: {truncated_response}")
                return ReviewResult(
                    success=False,
                    error_message=f"Failed to parse review response: {e}",
                    files_reviewed=file_paths
                )

        except Exception as e:
            error_msg = str(e)
            token_exceeded = "exceeds maximum allowed tokens" in error_msg

            return ReviewResult(
                success=False,
                token_limit_exceeded=token_exceeded,
                error_message=error_msg,
                files_reviewed=file_paths if token_exceeded else []
            )

    def _build_review_step_description(
        self,
        file_paths: List[str],
        context_description: Optional[str]
    ) -> str:
        """Build descriptive step text for the review"""
        base_desc = f"I need to perform a {self.config.review_type.value} code review of {len(file_paths)} files"

        if context_description:
            base_desc += f". Context: {context_description}"

        if self.config.focus_areas:
            base_desc += f". Focus areas: {', '.join(self.config.focus_areas)}"

        base_desc += f". Looking for {self.config.severity_filter.value}+ severity issues."

        return base_desc

    def _parse_review_response(
        self,
        response: Dict[str, Any],
        file_paths: List[str]
    ) -> ReviewResult:
        """Parse the response from zen codereview tool"""

        # Handle successful response
        if response.get("status") == "analyze_complete":
            return ReviewResult(
                success=True,
                issues_found=response.get("issues_found", []),
                confidence=response.get("confidence", "unknown"),
                step_number=response.get("step_number", 1),
                total_steps=response.get("total_steps", 1),
                findings=response.get("findings", ""),
                files_reviewed=file_paths,
                review_metadata=response.get("code_review_status", {})
            )

        # Handle pause/continuation responses
        if response.get("status") == "pause_for_code_review":
            # This indicates the tool wants us to continue investigation
            return ReviewResult(
                success=False,
                error_message="Review tool requires additional investigation steps",
                files_reviewed=file_paths
            )

        # Handle error responses
        return ReviewResult(
            success=False,
            error_message=response.get("error", "Unknown error from review tool"),
            files_reviewed=file_paths
        )

    def get_review_history(self) -> List[ReviewResult]:
        """Get history of all reviews performed"""
        return self._review_history.copy()

    def clear_history(self) -> None:
        """Clear review history"""
        self._review_history.clear()
        logger.info("Review history cleared")

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status and statistics"""
        total_reviews = len(self._review_history)
        successful_reviews = sum(1 for r in self._review_history if r.success)

        model_usage = {}
        for result in self._review_history:
            if result.model_used:
                model_usage[result.model_used.value] = model_usage.get(result.model_used.value, 0) + 1

        return {
            "total_reviews": total_reviews,
            "successful_reviews": successful_reviews,
            "success_rate": successful_reviews / total_reviews if total_reviews > 0 else 0,
            "model_usage": model_usage,
            "config": {
                "primary_model": self.config.primary_model.value,
                "fallback_models": [m.value for m in self.config.fallback_models],
                "max_retries": self.config.max_retries
            }
        }


# Convenience function for quick reviews
def quick_review(
    file_paths: List[Union[str, Path]],
    review_type: ReviewType = ReviewType.QUICK,
    severity_filter: ReviewSeverity = ReviewSeverity.MEDIUM
) -> ReviewResult:
    """
    Convenience function for quick code reviews.

    Args:
        file_paths: Files to review
        review_type: Type of review to perform
        severity_filter: Minimum severity to report

    Returns:
        ReviewResult
    """
    config = ReviewConfig(
        review_type=review_type,
        severity_filter=severity_filter,
        max_steps=2,
        chunk_files=True
    )

    agent = CodeReviewAgent(config=config)
    return agent.review_files(file_paths)
