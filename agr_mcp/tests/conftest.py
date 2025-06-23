"""Shared fixtures and configuration for AGR MCP Server tests.

This module provides common test fixtures, mocks, and utilities
used across multiple test files.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import tempfile
import os


# Event loop configuration for async tests
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def event_loop(event_loop_policy):
    """Create an event loop for async tests."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


# Mock HTTP client fixtures
@pytest.fixture
def mock_http_client():
    """Provide a mock HTTP client for testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def mock_http_response():
    """Provide a mock HTTP response."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json = MagicMock(return_value={})
    response.text = ""
    return response


# Configuration fixtures
@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "api_base_url": "https://api.test.alliancegenome.org",
        "timeout": 10,
        "max_retries": 2,
        "download_dir": "/tmp/agr_test_downloads",
        "cache_dir": "/tmp/agr_test_cache",
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


# File system fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_file_manager():
    """Provide a mock file manager."""
    manager = MagicMock()
    manager.save_file = MagicMock(return_value="/tmp/test_file.json")
    manager.save_stream = AsyncMock(return_value="/tmp/test_stream.json")
    manager.check_disk_space = MagicMock(return_value=True)
    manager.get_file_path = MagicMock(return_value="/tmp/test_file.json")
    return manager


# Sample data fixtures
@pytest.fixture
def sample_gene():
    """Provide sample gene data."""
    return {
        "id": "AGR:101000",
        "symbol": "BRCA2",
        "name": "breast cancer 2",
        "species": {
            "name": "Homo sapiens",
            "taxonId": "NCBITaxon:9606"
        },
        "chromosome": "13",
        "start": 32315474,
        "end": 32400266,
        "strand": "+",
        "soTermName": "protein_coding_gene",
        "synonyms": ["FANCD1", "BRCC2"],
        "crossReferences": [
            {"id": "HGNC:1101", "database": "HGNC"},
            {"id": "ENSG00000139618", "database": "Ensembl"}
        ]
    }


@pytest.fixture
def sample_search_results():
    """Provide sample search results."""
    return {
        "total": 3,
        "results": [
            {
                "id": "AGR:101000",
                "symbol": "BRCA2",
                "name": "breast cancer 2",
                "species": {"name": "Homo sapiens", "taxonId": "NCBITaxon:9606"}
            },
            {
                "id": "AGR:201000",
                "symbol": "Brca2",
                "name": "breast cancer 2",
                "species": {"name": "Mus musculus", "taxonId": "NCBITaxon:10090"}
            },
            {
                "id": "AGR:301000",
                "symbol": "brca2",
                "name": "breast cancer 2",
                "species": {"name": "Danio rerio", "taxonId": "NCBITaxon:7955"}
            }
        ]
    }


@pytest.fixture
def sample_ortholog_data():
    """Provide sample ortholog data."""
    return {
        "sourceGene": {
            "id": "AGR:101000",
            "symbol": "BRCA2",
            "species": "Homo sapiens"
        },
        "orthologs": [
            {
                "id": "AGR:201000",
                "symbol": "Brca2",
                "species": {"name": "Mus musculus", "taxonId": "NCBITaxon:10090"},
                "predictionMethodsMatched": ["DIOPT", "HGNC"],
                "confidence": "high"
            },
            {
                "id": "AGR:301000",
                "symbol": "brca2",
                "species": {"name": "Danio rerio", "taxonId": "NCBITaxon:7955"},
                "predictionMethodsMatched": ["DIOPT"],
                "confidence": "moderate"
            }
        ]
    }


@pytest.fixture
def sample_api_schema():
    """Provide sample API schema data."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Alliance Genome Resources API",
            "version": "7.0.0"
        },
        "paths": {
            "/gene/{id}": {
                "get": {
                    "summary": "Get gene details",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ]
                }
            },
            "/search": {
                "get": {
                    "summary": "Search for entities",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ]
                }
            }
        }
    }


# Error response fixtures
@pytest.fixture
def error_404_response():
    """Provide a 404 error response."""
    response = MagicMock()
    response.status_code = 404
    response.json.return_value = {
        "error": {
            "code": "NOT_FOUND",
            "message": "Resource not found"
        }
    }
    return response


@pytest.fixture
def error_500_response():
    """Provide a 500 error response."""
    response = MagicMock()
    response.status_code = 500
    response.json.return_value = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "Internal server error"
        }
    }
    return response


# Utility functions
def assert_error_response(response, expected_status, expected_message):
    """Helper to assert error response properties."""
    assert response.status_code == expected_status
    assert expected_message in str(response.json())


def create_mock_download_response(data_type, format_type, record_count=100):
    """Create a mock download response."""
    if format_type == "json":
        return {
            "data": [{"id": f"AGR:{i}", "type": data_type} for i in range(record_count)],
            "metadata": {
                "dataType": data_type,
                "recordCount": record_count,
                "version": "7.0.0"
            }
        }
    elif format_type == "tsv":
        headers = "id\ttype\n"
        rows = "\n".join([f"AGR:{i}\t{data_type}" for i in range(record_count)])
        return headers + rows
    elif format_type == "csv":
        headers = "id,type\n"
        rows = "\n".join([f"AGR:{i},{data_type}" for i in range(record_count)])
        return headers + rows


# Test markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
