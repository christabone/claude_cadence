"""Tests for gene query functionality in AGR MCP Server.

This module contains test cases for the gene query tools including
search_genes, get_gene_details, and find_orthologs functions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.gene_query import search_genes, get_gene_details, find_orthologs
from src.errors import ConnectionError, ResourceNotFoundError, ValidationError


class TestSearchGenes:
    """Test cases for the search_genes function."""

    @pytest.mark.asyncio
    async def test_search_genes_basic(self):
        """Test basic gene search functionality."""
        # Mock the HTTP client
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "total": 1,
                "results": [
                    {
                        "id": "AGR:101000",
                        "symbol": "BRCA2",
                        "name": "breast cancer 2",
                        "species": {"name": "Homo sapiens", "taxonId": "NCBITaxon:9606"},
                        "synonyms": ["FANCD1", "BRCC2"],
                        "chromosome": "13",
                        "soTermName": "protein_coding_gene"
                    }
                ]
            }
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test the search
            result = await search_genes("BRCA2", limit=10)

            # Verify results
            assert result["query"] == "BRCA2"
            assert result["total"] == 1
            assert len(result["results"]) == 1
            assert result["results"][0]["symbol"] == "BRCA2"

    @pytest.mark.asyncio
    async def test_search_genes_with_species_filter(self):
        """Test gene search with species filter."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"total": 0, "results": []}
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test with species filter
            result = await search_genes("pax6", species="Danio rerio", limit=5)

            # Verify the API was called with correct parameters
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args[1]["params"]["species"] == "Danio rerio"

    @pytest.mark.asyncio
    async def test_search_genes_empty_query(self):
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            await search_genes("")

        assert "Query string cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_genes_invalid_limit(self):
        """Test that invalid limit raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            await search_genes("BRCA2", limit=150)

        assert "Limit must be between 1 and 100" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_genes_api_error(self):
        """Test handling of API errors during search."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError) as exc_info:
                await search_genes("BRCA2")

            assert "Failed to search genes" in str(exc_info.value)


class TestGetGeneDetails:
    """Test cases for the get_gene_details function."""

    @pytest.mark.asyncio
    async def test_get_gene_details_success(self):
        """Test successful retrieval of gene details."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "AGR:101000",
                "symbol": "BRCA2",
                "name": "breast cancer 2",
                "automatedGeneSynopsis": "This gene encodes a protein...",
                "species": {"name": "Homo sapiens", "taxonId": "NCBITaxon:9606"},
                "chromosome": "13",
                "start": 32315474,
                "end": 32400266,
                "strand": "+",
                "assembly": "GRCh38",
                "soTermName": "protein_coding_gene",
                "synonyms": ["FANCD1"],
                "crossReferences": [],
                "diseases": [
                    {
                        "name": "Breast Cancer",
                        "id": "DOID:1612",
                        "associationType": "is_implicated_in"
                    }
                ],
                "expression": [],
                "interactions": {"total": 45},
                "phenotypes": []
            }
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test getting details
            result = await get_gene_details("AGR:101000")

            # Verify results
            assert result["id"] == "AGR:101000"
            assert result["symbol"] == "BRCA2"
            assert result["location"]["chromosome"] == "13"
            assert len(result["diseases"]) == 1
            assert result["interactions"] == 45

    @pytest.mark.asyncio
    async def test_get_gene_details_not_found(self):
        """Test handling of non-existent gene."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ResourceNotFoundError) as exc_info:
                await get_gene_details("AGR:INVALID")

            assert "Gene not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_gene_details_invalid_id(self):
        """Test validation of gene ID format."""
        with patch('src.tools.gene_query.validate_gene_id') as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid gene ID format")

            with pytest.raises(ValidationError):
                await get_gene_details("invalid-id")


class TestFindOrthologs:
    """Test cases for the find_orthologs function."""

    @pytest.mark.asyncio
    async def test_find_orthologs_success(self):
        """Test successful retrieval of orthologs."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()

            # Mock gene details response
            gene_response = MagicMock()
            gene_response.status_code = 200
            gene_response.json.return_value = {
                "id": "AGR:101000",
                "symbol": "BRCA2",
                "species": {"name": "Homo sapiens"}
            }

            # Mock orthologs response
            orthologs_response = MagicMock()
            orthologs_response.json.return_value = {
                "results": [
                    {
                        "id": "AGR:201000",
                        "symbol": "Brca2",
                        "species": {"name": "Mus musculus", "taxonId": "NCBITaxon:10090"},
                        "predictionMethodsMatched": ["DIOPT"],
                        "confidence": "high"
                    }
                ]
            }

            # Set up mock client
            mock_client.get.side_effect = [gene_response, orthologs_response]
            mock_get_client.return_value = mock_client

            # Test finding orthologs
            result = await find_orthologs("AGR:101000")

            # Verify results
            assert result["sourceGene"]["id"] == "AGR:101000"
            assert result["total"] == 1
            assert result["orthologs"][0]["symbol"] == "Brca2"
            assert result["orthologs"][0]["species"] == "Mus musculus"

    @pytest.mark.asyncio
    async def test_find_orthologs_with_target_species(self):
        """Test finding orthologs filtered by target species."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()

            # Mock responses
            gene_response = MagicMock()
            gene_response.status_code = 200
            gene_response.json.return_value = {
                "id": "AGR:101000",
                "symbol": "BRCA2",
                "species": {"name": "Homo sapiens"}
            }

            orthologs_response = MagicMock()
            orthologs_response.json.return_value = {
                "results": [
                    {
                        "id": "AGR:201000",
                        "symbol": "Brca2",
                        "species": {"name": "Mus musculus", "taxonId": "NCBITaxon:10090"},
                        "predictionMethodsMatched": ["DIOPT"],
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

            mock_client.get.side_effect = [gene_response, orthologs_response]
            mock_get_client.return_value = mock_client

            # Test with species filter
            result = await find_orthologs("AGR:101000", target_species="Mus musculus")

            # Verify only mouse ortholog is returned
            assert result["total"] == 1
            assert result["orthologs"][0]["species"] == "Mus musculus"

    @pytest.mark.asyncio
    async def test_find_orthologs_gene_not_found(self):
        """Test handling when source gene is not found."""
        with patch('src.tools.gene_query.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ResourceNotFoundError):
                await find_orthologs("AGR:INVALID")


# Test fixtures
@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    return {
        "api_base_url": "https://api.alliancegenome.org",
        "timeout": 30,
        "max_retries": 3
    }


@pytest.fixture
def sample_gene_data():
    """Provide sample gene data for tests."""
    return {
        "id": "AGR:101000",
        "symbol": "BRCA2",
        "name": "breast cancer 2",
        "species": {"name": "Homo sapiens", "taxonId": "NCBITaxon:9606"}
    }
