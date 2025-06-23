"""Unit tests for AGR client.

This module tests the AGR API client functionality.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import httpx

from agr_mcp.client import AGRClient, AGRAPIError, AGRClientError
from agr_mcp.models import Gene, Disease, Expression, Allele


@pytest.mark.asyncio
class TestAGRClient:
    """Test AGR client functionality."""

    async def test_client_initialization(self, temp_cache_dir):
        """Test client initialization with custom parameters."""
        client = AGRClient(
            base_url="https://custom.api.com",
            cache_dir=str(temp_cache_dir),
            cache_ttl=1800,
            timeout=60.0,
        )

        assert client.base_url == "https://custom.api.com"
        assert client.cache_manager.cache_dir == temp_cache_dir
        assert client.cache_manager.ttl == 1800

        await client.close()

    async def test_context_manager(self):
        """Test client as async context manager."""
        async with AGRClient() as client:
            assert client.client is not None

        # Client should be closed after exiting context
        assert client.client.is_closed

    async def test_get_gene_success(self, mock_agr_client, mock_gene_data):
        """Test successful gene retrieval."""
        gene = await mock_agr_client.get_gene("HGNC:11998")

        assert isinstance(gene, Gene)
        assert gene.id == mock_gene_data["id"]
        assert gene.symbol == mock_gene_data["symbol"]

    async def test_get_gene_with_options(self, mock_agr_client):
        """Test gene retrieval with optional data."""
        gene = await mock_agr_client.get_gene(
            "HGNC:11998",
            include_orthologs=True,
            include_expression=True,
            include_disease=True,
        )

        assert isinstance(gene, Gene)

    @patch("agr_mcp.client.httpx.AsyncClient.request")
    async def test_api_error_handling(self, mock_request):
        """Test API error handling."""
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=AsyncMock(),
            response=mock_response,
        )
        mock_response.json.return_value = {"error": "Gene not found"}
        mock_request.return_value = mock_response

        client = AGRClient()

        with pytest.raises(AGRAPIError) as exc_info:
            await client.get_gene("INVALID:123")

        assert exc_info.value.status_code == 404
        await client.close()

    @patch("agr_mcp.client.httpx.AsyncClient.request")
    async def test_request_error_handling(self, mock_request):
        """Test network error handling."""
        # Mock network error
        mock_request.side_effect = httpx.RequestError("Connection failed")

        client = AGRClient()

        with pytest.raises(AGRClientError) as exc_info:
            await client.get_gene("HGNC:11998")

        assert "Request failed" in str(exc_info.value)
        await client.close()

    async def test_search_genes(self, mock_agr_client):
        """Test gene search functionality."""
        # Mock search to return list
        mock_agr_client._request = AsyncMock(
            return_value={"results": [mock_agr_client._request.__wrapped__()]}
        )

        genes = await mock_agr_client.search_genes(
            query="TP53",
            species="Homo sapiens",
            limit=10,
        )

        assert isinstance(genes, list)

    async def test_caching(self, mock_agr_client, cache_manager):
        """Test response caching."""
        # First request should hit API
        gene1 = await mock_agr_client.get_gene("HGNC:11998")

        # Second request should use cache
        gene2 = await mock_agr_client.get_gene("HGNC:11998")

        assert gene1.id == gene2.id

        # Check cache contains the data
        cache_key = cache_manager.make_key(
            "https://mock.alliancegenome.org/api/gene/HGNC:11998",
            {"includeOrthologs": False, "includeExpression": False, "includeDisease": False},
        )
        cached_data = await cache_manager.get(cache_key)
        assert cached_data is not None

    async def test_batch_get_genes(self, mock_agr_client):
        """Test batch gene retrieval."""
        # Mock POST request
        mock_agr_client._request = AsyncMock(
            return_value={"results": [{"id": "HGNC:11998", "symbol": "TP53", "species": "Homo sapiens"}]}
        )

        genes = await mock_agr_client.batch_get_genes(
            identifiers=["HGNC:11998", "HGNC:1100"],
            include_orthologs=False,
            format="json",
        )

        assert isinstance(genes, list)
        assert len(genes) == 1
        assert isinstance(genes[0], Gene)

    async def test_batch_get_genes_text_format(self, mock_agr_client):
        """Test batch gene retrieval with text format."""
        # Mock POST request
        mock_agr_client._request = AsyncMock(
            return_value={"content": "Gene\tSymbol\nHGNC:11998\tTP53"}
        )

        result = await mock_agr_client.batch_get_genes(
            identifiers=["HGNC:11998"],
            format="tsv",
        )

        assert isinstance(result, str)
        assert "TP53" in result
