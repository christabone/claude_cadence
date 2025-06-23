"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

import pytest
import pytest_asyncio
from httpx import AsyncClient, Response

from agr_mcp.client import AGRClient
from agr_mcp.server import AGRMCPServer
from agr_mcp.utils.cache import CacheManager


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest_asyncio.fixture
async def cache_manager(temp_cache_dir: Path) -> AsyncGenerator[CacheManager, None]:
    """Create cache manager with temporary directory."""
    manager = CacheManager(cache_dir=str(temp_cache_dir), ttl=60)
    yield manager
    await manager.clear()


@pytest.fixture
def mock_gene_data() -> Dict[str, Any]:
    """Sample gene data for testing."""
    return {
        "id": "HGNC:11998",
        "symbol": "TP53",
        "name": "tumor protein p53",
        "species": "Homo sapiens",
        "taxon_id": "9606",
        "gene_type": "protein coding",
        "synonyms": ["p53", "LFS1"],
        "chromosome": "17",
        "start_position": 7565097,
        "end_position": 7590856,
        "strand": "-",
        "cross_references": [
            {"db": "OMIM", "id": "191170"},
            {"db": "Ensembl", "id": "ENSG00000141510"},
        ],
    }


@pytest.fixture
def mock_disease_data() -> Dict[str, Any]:
    """Sample disease association data for testing."""
    return {
        "disease_id": "OMIM:151623",
        "disease_name": "Li-Fraumeni syndrome",
        "gene_id": "HGNC:11998",
        "gene_symbol": "TP53",
        "association_type": "causative",
        "evidence_codes": ["PCS", "TAS"],
        "publications": ["PMID:2259385", "PMID:20301488"],
        "confidence_level": "high",
        "inheritance_mode": "autosomal dominant",
    }


@pytest.fixture
def mock_expression_data() -> Dict[str, Any]:
    """Sample expression data for testing."""
    return {
        "gene_id": "HGNC:11998",
        "anatomy_term": "brain",
        "anatomy_id": "UBERON:0000955",
        "stage": "adult",
        "stage_id": "HsapDv:0000087",
        "expression_pattern": "widespread",
        "expression_level": "moderate",
        "assay_type": "RNA-seq",
        "publications": ["PMID:23539183"],
    }


@pytest.fixture
def mock_allele_data() -> Dict[str, Any]:
    """Sample allele data for testing."""
    return {
        "allele_id": "HGVS:NM_000546.5:c.524G>A",
        "allele_symbol": "TP53 R175H",
        "allele_name": "p.Arg175His",
        "gene_id": "HGNC:11998",
        "gene_symbol": "TP53",
        "allele_type": "missense",
        "molecular_mutation": "R175H",
        "generation_method": "spontaneous",
        "phenotypes": [
            {
                "phenotype_term": "increased tumor incidence",
                "phenotype_id": "MP:0002169",
                "evidence_code": "TAS",
            }
        ],
    }


@pytest_asyncio.fixture
async def mock_agr_client(
    cache_manager: CacheManager,
    mock_gene_data: Dict[str, Any],
) -> AsyncGenerator[AGRClient, None]:
    """Create mock AGR client for testing."""
    client = AGRClient(
        base_url="https://mock.alliancegenome.org/api",
        cache_dir=cache_manager.cache_dir,
    )

    # Mock the HTTP client
    async def mock_request(method, url, **kwargs):
        """Mock HTTP request handler."""
        response_data = {}

        if "/gene/" in url:
            response_data = mock_gene_data
        elif "/search/gene" in url:
            response_data = {"results": [mock_gene_data]}

        return Response(
            status_code=200,
            json=response_data,
            headers={"content-type": "application/json"},
        )

    # Replace the request method
    client._request = mock_request

    yield client
    await client.close()


@pytest.fixture
def mock_server_config() -> Dict[str, Any]:
    """Mock server configuration."""
    return {
        "api_url": "https://mock.alliancegenome.org/api",
        "cache_dir": "/tmp/test-cache",
        "cache_ttl": 60,
        "log_level": "DEBUG",
    }


@pytest_asyncio.fixture
async def agr_server(
    mock_server_config: Dict[str, Any],
    temp_cache_dir: Path,
) -> AsyncGenerator[AGRMCPServer, None]:
    """Create AGR MCP server for testing."""
    config = mock_server_config.copy()
    config["cache_dir"] = str(temp_cache_dir)

    server = AGRMCPServer(config=config)
    yield server
