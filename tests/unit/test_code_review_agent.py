"""
Unit tests for Code Review Agent
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, call, MagicMock

import pytest

from cadence.code_review_agent import (
    CodeReviewAgent, ReviewConfig, ReviewResult, ReviewSeverity,
    ReviewType, ModelProvider, quick_review
)


class TestReviewConfig:
    """Test ReviewConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ReviewConfig()

        assert config.review_type == ReviewType.FULL
        assert config.severity_filter == ReviewSeverity.MEDIUM
        assert config.focus_areas == []
        assert config.max_steps == 3
        assert config.primary_model == ModelProvider.GEMINI
        assert config.fallback_models == [ModelProvider.O3_MINI, ModelProvider.FLASH]
        assert config.max_retries == 2
        assert config.retry_delay == 1.0
        assert config.chunk_files is True
        assert config.max_files_per_chunk == 3

    def test_custom_config(self):
        """Test custom configuration"""
        config = ReviewConfig(
            review_type=ReviewType.SECURITY,
            severity_filter=ReviewSeverity.HIGH,
            focus_areas=["authentication", "sql_injection"],
            primary_model=ModelProvider.O3,
            max_steps=5
        )

        assert config.review_type == ReviewType.SECURITY
        assert config.severity_filter == ReviewSeverity.HIGH
        assert config.focus_areas == ["authentication", "sql_injection"]
        assert config.primary_model == ModelProvider.O3
        assert config.max_steps == 5


class TestReviewResult:
    """Test ReviewResult dataclass"""

    def test_default_result(self):
        """Test default result values"""
        result = ReviewResult(success=True)

        assert result.success is True
        assert result.model_used is None
        assert result.issues_found == []
        assert result.confidence == "unknown"
        assert result.step_number == 0
        assert result.total_steps == 0
        assert result.findings == ""
        assert result.error_message is None
        assert result.token_limit_exceeded is False
        assert result.files_reviewed == []
        assert result.review_metadata == {}

    def test_complete_result(self):
        """Test complete result with all fields"""
        issues = [{"severity": "high", "description": "Test issue"}]
        files = ["/test/file1.py", "/test/file2.py"]
        metadata = {"files_checked": 2, "issues_count": 1}

        result = ReviewResult(
            success=True,
            model_used=ModelProvider.GEMINI,
            issues_found=issues,
            confidence="high",
            step_number=2,
            total_steps=3,
            findings="Found 1 issue",
            files_reviewed=files,
            review_metadata=metadata
        )

        assert result.success is True
        assert result.model_used == ModelProvider.GEMINI
        assert result.issues_found == issues
        assert result.confidence == "high"
        assert result.files_reviewed == files
        assert result.review_metadata == metadata


class TestCodeReviewAgent:
    """Test CodeReviewAgent class"""

    def test_initialization_default(self):
        """Test agent initialization with defaults"""
        agent = CodeReviewAgent()

        assert agent.config.review_type == ReviewType.FULL
        assert agent.config.primary_model == ModelProvider.GEMINI
        assert agent.mcp_client is None
        assert len(agent._review_history) == 0

    def test_initialization_custom_config(self):
        """Test agent initialization with custom config"""
        config = ReviewConfig(
            review_type=ReviewType.SECURITY,
            primary_model=ModelProvider.O3
        )
        mock_client = Mock()

        agent = CodeReviewAgent(config=config, mcp_client=mock_client)

        assert agent.config.review_type == ReviewType.SECURITY
        assert agent.config.primary_model == ModelProvider.O3
        assert agent.mcp_client == mock_client

    def test_validate_file_paths_valid_files(self):
        """Test file path validation with valid files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1 = Path(temp_dir) / "test1.py"
            file2 = Path(temp_dir) / "test2.py"
            file1.write_text("# Test file 1")
            file2.write_text("# Test file 2")

            agent = CodeReviewAgent()
            validated = agent._validate_file_paths([str(file1), str(file2)])

            assert len(validated) == 2
            assert str(file1.absolute()) in validated
            assert str(file2.absolute()) in validated

    def test_validate_file_paths_invalid_files(self):
        """Test file path validation with invalid files"""
        agent = CodeReviewAgent()
        validated = agent._validate_file_paths([
            "/nonexistent/file.py",
            "/another/missing/file.py"
        ])

        assert len(validated) == 0

    def test_validate_file_paths_mixed(self):
        """Test file path validation with mix of valid and invalid files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create one valid file
            valid_file = Path(temp_dir) / "valid.py"
            valid_file.write_text("# Valid file")

            agent = CodeReviewAgent()
            validated = agent._validate_file_paths([
                str(valid_file),
                "/nonexistent/file.py",
                str(Path(temp_dir))  # Directory, not file
            ])

            assert len(validated) == 1
            assert str(valid_file.absolute()) in validated

    def test_review_files_no_valid_files(self):
        """Test review with no valid files"""
        agent = CodeReviewAgent()

        result = agent.review_files(["/nonexistent/file.py"])

        assert result.success is False
        assert "No valid files provided" in result.error_message

    def test_review_files_simulation_mode(self):
        """Test review in simulation mode (no MCP client)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test_function():\n    pass")

            agent = CodeReviewAgent()
            result = agent.review_files([str(test_file)])

            assert result.success is True
            assert result.model_used == ModelProvider.GEMINI
            assert result.confidence == "simulated"
            assert len(result.files_reviewed) == 1
            assert len(result.issues_found) == 1

    def test_review_files_with_mcp_client_success(self):
        """Test review with successful MCP client response"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test_function():\n    pass")

            # Mock successful MCP response
            mock_client = Mock()
            mock_response = {
                "status": "analyze_complete",
                "issues_found": [{"severity": "low", "description": "Test issue"}],
                "confidence": "high",
                "step_number": 1,
                "total_steps": 1,
                "findings": "Review completed successfully"
            }
            mock_client.call_tool.return_value = mock_response

            agent = CodeReviewAgent(mcp_client=mock_client)
            result = agent.review_files([str(test_file)])

            assert result.success is True
            assert result.confidence == "high"
            assert len(result.issues_found) == 1
            assert result.findings == "Review completed successfully"

            # Verify MCP client was called correctly
            mock_client.call_tool.assert_called_once()
            call_args = mock_client.call_tool.call_args
            assert call_args[0][0] == "mcp__zen__codereview"
            assert "relevant_files" in call_args[0][1]

    def test_review_files_with_mcp_client_token_limit(self):
        """Test review with token limit exceeded"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test_function():\n    pass")

            # Mock token limit error
            mock_client = Mock()
            mock_client.call_tool.side_effect = Exception("exceeds maximum allowed tokens")

            config = ReviewConfig(fallback_models=[ModelProvider.O3_MINI])
            agent = CodeReviewAgent(config=config, mcp_client=mock_client)

            result = agent.review_files([str(test_file)])

            # Should attempt fallback but ultimately fail
            assert result.success is False
            assert result.token_limit_exceeded is True
            assert "exceeds maximum allowed tokens" in result.error_message

    def test_review_files_with_fallback_success(self):
        """Test successful fallback to secondary model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test_function():\n    pass")

            mock_client = Mock()

            # First call fails, second succeeds
            mock_client.call_tool.side_effect = [
                Exception("Primary model error"),
                {
                    "status": "analyze_complete",
                    "issues_found": [],
                    "confidence": "medium",
                    "findings": "Fallback review completed"
                }
            ]

            config = ReviewConfig(
                primary_model=ModelProvider.GEMINI,
                fallback_models=[ModelProvider.O3_MINI],
                retry_delay=0.1  # Speed up test
            )
            agent = CodeReviewAgent(config=config, mcp_client=mock_client)

            result = agent.review_files([str(test_file)])

            assert result.success is True
            assert result.model_used == ModelProvider.O3_MINI
            assert result.findings == "Fallback review completed"
            assert mock_client.call_tool.call_count == 2

    def test_review_files_chunked(self):
        """Test chunked file review for large file sets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test files
            files = []
            for i in range(5):
                test_file = Path(temp_dir) / f"test{i}.py"
                test_file.write_text(f"def test_function_{i}():\n    pass")
                files.append(str(test_file))

            config = ReviewConfig(
                max_files_per_chunk=2,
                chunk_files=True
            )
            agent = CodeReviewAgent(config=config)

            result = agent.review_files(files)

            assert result.success is True
            assert len(result.files_reviewed) == 5
            # Should have chunked findings
            assert "Chunk" in result.findings

    def test_build_review_step_description(self):
        """Test building review step description"""
        config = ReviewConfig(
            review_type=ReviewType.SECURITY,
            severity_filter=ReviewSeverity.HIGH,
            focus_areas=["authentication", "injection"]
        )
        agent = CodeReviewAgent(config=config)

        files = ["/test/file1.py", "/test/file2.py"]
        context = "Testing authentication module"

        description = agent._build_review_step_description(files, context)

        assert "security code review" in description
        assert "2 files" in description
        assert "Testing authentication module" in description
        assert "authentication, injection" in description
        assert "high+ severity" in description

    def test_parse_review_response_success(self):
        """Test parsing successful review response"""
        agent = CodeReviewAgent()
        files = ["/test/file.py"]

        response = {
            "status": "analyze_complete",
            "issues_found": [{"severity": "medium", "description": "Test issue"}],
            "confidence": "high",
            "step_number": 2,
            "total_steps": 3,
            "findings": "Review completed",
            "code_review_status": {"files_checked": 1}
        }

        result = agent._parse_review_response(response, files)

        assert result.success is True
        assert result.confidence == "high"
        assert len(result.issues_found) == 1
        assert result.step_number == 2
        assert result.total_steps == 3
        assert result.findings == "Review completed"
        assert result.files_reviewed == files
        assert result.review_metadata == {"files_checked": 1}

    def test_parse_review_response_pause(self):
        """Test parsing pause response"""
        agent = CodeReviewAgent()
        files = ["/test/file.py"]

        response = {
            "status": "pause_for_code_review"
        }

        result = agent._parse_review_response(response, files)

        assert result.success is False
        assert "additional investigation" in result.error_message
        assert result.files_reviewed == files

    def test_parse_review_response_error(self):
        """Test parsing error response"""
        agent = CodeReviewAgent()
        files = ["/test/file.py"]

        response = {
            "error": "Something went wrong"
        }

        result = agent._parse_review_response(response, files)

        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.files_reviewed == files

    def test_get_review_history(self):
        """Test getting review history"""
        agent = CodeReviewAgent()

        # Add some mock history
        result1 = ReviewResult(success=True, model_used=ModelProvider.GEMINI)
        result2 = ReviewResult(success=False, error_message="Test error")
        agent._review_history.extend([result1, result2])

        history = agent.get_review_history()

        assert len(history) == 2
        assert history[0].success is True
        assert history[1].success is False

        # Should be a copy, not the original
        assert history is not agent._review_history

    def test_clear_history(self):
        """Test clearing review history"""
        agent = CodeReviewAgent()

        # Add some mock history
        agent._review_history.append(ReviewResult(success=True))
        assert len(agent._review_history) == 1

        agent.clear_history()
        assert len(agent._review_history) == 0

    def test_get_health_status(self):
        """Test getting agent health status"""
        agent = CodeReviewAgent()

        # Add some mock history
        agent._review_history.extend([
            ReviewResult(success=True, model_used=ModelProvider.GEMINI),
            ReviewResult(success=True, model_used=ModelProvider.GEMINI),
            ReviewResult(success=False, model_used=ModelProvider.O3_MINI),
            ReviewResult(success=True, model_used=ModelProvider.O3_MINI)
        ])

        status = agent.get_health_status()

        assert status["total_reviews"] == 4
        assert status["successful_reviews"] == 3
        assert status["success_rate"] == 0.75
        assert status["model_usage"]["gemini-2.5-pro"] == 2
        assert status["model_usage"]["o3-mini"] == 2
        assert status["config"]["primary_model"] == "gemini-2.5-pro"

    def test_get_health_status_empty(self):
        """Test health status with no history"""
        agent = CodeReviewAgent()

        status = agent.get_health_status()

        assert status["total_reviews"] == 0
        assert status["successful_reviews"] == 0
        assert status["success_rate"] == 0
        assert status["model_usage"] == {}


class TestQuickReview:
    """Test quick_review convenience function"""

    def test_quick_review_function(self):
        """Test quick review convenience function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test():\n    pass")

            result = quick_review(
                [str(test_file)],
                review_type=ReviewType.QUICK,
                severity_filter=ReviewSeverity.HIGH
            )

            assert result.success is True
            assert len(result.files_reviewed) == 1


class TestIntegration:
    """Integration tests with real file scenarios"""

    def test_review_actual_project_file(self):
        """Test reviewing an actual project file"""
        # Use the agent file itself as test subject
        agent_file = Path(__file__).parent.parent.parent / "cadence" / "code_review_agent.py"

        if agent_file.exists():
            agent = CodeReviewAgent()
            result = agent.review_files([str(agent_file)])

            # In simulation mode, should always succeed
            assert result.success is True
            assert len(result.files_reviewed) == 1
            assert result.confidence == "simulated"

    def test_review_multiple_real_files(self):
        """Test reviewing multiple real files"""
        cadence_dir = Path(__file__).parent.parent.parent / "cadence"
        python_files = list(cadence_dir.glob("*.py"))[:3]  # Take first 3 files

        if python_files:
            config = ReviewConfig(
                review_type=ReviewType.QUICK,
                max_files_per_chunk=2,
                chunk_files=True
            )
            agent = CodeReviewAgent(config=config)

            result = agent.review_files([str(f) for f in python_files])

            assert result.success is True
            assert len(result.files_reviewed) >= 1

    @pytest.mark.parametrize("review_type,severity", [
        (ReviewType.SECURITY, ReviewSeverity.HIGH),
        (ReviewType.PERFORMANCE, ReviewSeverity.MEDIUM),
        (ReviewType.QUICK, ReviewSeverity.LOW)
    ])
    def test_different_review_configurations(self, review_type, severity):
        """Test different review type and severity combinations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def example():\n    return 'test'")

            config = ReviewConfig(
                review_type=review_type,
                severity_filter=severity
            )
            agent = CodeReviewAgent(config=config)

            result = agent.review_files([str(test_file)])

            assert result.success is True

    def test_concurrent_reviews(self):
        """Test multiple concurrent reviews (simulation)"""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            test_files = []
            for i in range(3):
                test_file = Path(temp_dir) / f"test{i}.py"
                test_file.write_text(f"def test_{i}():\n    pass")
                test_files.append(str(test_file))

            agent = CodeReviewAgent()
            results = []
            errors = []

            def worker(file_path):
                try:
                    result = agent.review_files([file_path])
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(f,)) for f in test_files]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            assert len(errors) == 0
            assert len(results) == 3
            assert all(r.success for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
