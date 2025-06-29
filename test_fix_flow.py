#!/usr/bin/env python3
"""
Test script to verify fix agent flow works end-to-end.
"""

import os
import sys
from pathlib import Path

# Add cadence to path
sys.path.insert(0, str(Path(__file__).parent))

from cadence.orchestrator import SupervisorOrchestrator
from cadence.code_review_agent import CodeReviewAgent, ReviewConfig, ReviewType, ReviewSeverity, ModelProvider
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_fix_agent_flow():
    """Test the fix agent flow with a simple code review."""

    logger.info("Starting fix agent flow test...")

    # Initialize orchestrator
    project_root = Path(__file__).parent
    orchestrator = SupervisorOrchestrator(project_root)

    # Create a code review agent
    review_config = ReviewConfig(
        review_type=ReviewType.SECURITY,  # Focus on security to catch eval() issue
        severity_filter=ReviewSeverity.MEDIUM,
        max_steps=1,
        primary_model=ModelProvider.FLASH  # Use fast model for testing
    )

    # Note: In real usage, MCP client would be provided
    review_agent = CodeReviewAgent(config=review_config)

    # Review the test file
    test_file = project_root / "test_fix_agent.py"
    logger.info(f"Running code review on {test_file}...")

    review_result = review_agent.review_files([test_file])

    if review_result.success:
        logger.info(f"Code review completed. Found {len(review_result.issues_found)} issues")
        for issue in review_result.issues_found:
            logger.info(f"  - [{issue.get('severity', 'unknown')}] {issue.get('description', 'No description')}")

        # Now test the fix agent
        if review_result.issues_found:
            logger.info("Testing fix agent execution...")

            # Format issues for fix agent
            issues = []
            for issue in review_result.issues_found:
                issues.append({
                    "severity": issue.get("severity", "medium"),
                    "description": issue.get("description", "Unknown issue"),
                    "file": str(test_file),
                    "line": issue.get("line", 0)
                })

            # Run fix agent
            fix_result = orchestrator.run_fix_agent(
                task_id="test-fix-001",
                issues=issues,
                files=[str(test_file)],
                use_continue=False
            )

            if fix_result.success:
                logger.info("Fix agent completed successfully!")
                logger.info(f"  Execution time: {fix_result.execution_time:.2f}s")
            else:
                logger.error("Fix agent failed!")
                for error in fix_result.errors:
                    logger.error(f"  - {error}")
    else:
        logger.error(f"Code review failed: {review_result.error_message}")

    logger.info("Fix agent flow test completed.")


if __name__ == "__main__":
    test_fix_agent_flow()
