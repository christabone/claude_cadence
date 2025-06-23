"""Unit tests for tool handlers.

This module tests the tool registry and individual tool handlers.
"""

import json
from unittest.mock import AsyncMock

import pytest

from agr_mcp.tools import (
    get_tool_registry,
    handle_get_gene,
    handle_search_genes,
    handle_get_disease_associations,
    handle_get_expression,
    handle_get_alleles,
    handle_get_cross_references,
    handle_batch_get_genes,
)
from agr_mcp.models import Gene, Disease, Expression, Allele, CrossReference


@pytest.mark.asyncio
class TestToolRegistry:
    """Test tool registry functionality."""

    def test_tool_registry_structure(self):
        """Test that tool registry has expected structure."""
        registry = get_tool_registry()

        # Check expected tools exist
        expected_tools = [
            "get_gene",
            "search_genes",
            "get_disease_associations",
            "get_expression",
            "get_alleles",
            "get_cross_references",
            "get_genes_batch",
        ]

        for tool_name in expected_tools:
            assert tool_name in registry

            tool = registry[tool_name]
            assert "description" in tool
            assert "handler" in tool
            assert "input_schema" in tool

            # Check input schema structure
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_get_gene_schema(self):
        """Test get_gene tool schema."""
        registry = get_tool_registry()
        schema = registry["get_gene"]["input_schema"]

        assert "identifier" in schema["properties"]
        assert "identifier" in schema["required"]

        # Check optional parameters
        optional_params = [
            "include_orthologs",
            "include_expression",
            "include_disease",
            "format",
        ]
        for param in optional_params:
            assert param in schema["properties"]


@pytest.mark.asyncio
class TestToolHandlers:
    """Test individual tool handler functions."""

    async def test_handle_get_gene_json(self, mock_agr_client, mock_gene_data):
        """Test get_gene handler with JSON format."""
        result = await handle_get_gene(
            mock_agr_client,
            identifier="HGNC:11998",
            format="json",
        )

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["id"] == mock_gene_data["id"]
        assert parsed["symbol"] == mock_gene_data["symbol"]

    async def test_handle_get_gene_text(self, mock_agr_client):
        """Test get_gene handler with text format."""
        result = await handle_get_gene(
            mock_agr_client,
            identifier="HGNC:11998",
            format="text",
        )

        assert isinstance(result, str)
        assert "Gene: TP53 (HGNC:11998)" in result
        assert "Species: Homo sapiens" in result

    async def test_handle_search_genes(self, mock_agr_client):
        """Test search_genes handler."""
        # Mock the search method
        mock_genes = [
            Gene(id="HGNC:11998", symbol="TP53", species="Homo sapiens"),
            Gene(id="HGNC:11999", symbol="TP53BP1", species="Homo sapiens"),
        ]
        mock_agr_client.search_genes = AsyncMock(return_value=mock_genes)

        result = await handle_search_genes(
            mock_agr_client,
            query="TP53",
            species="Homo sapiens",
            limit=10,
            format="json",
        )

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["symbol"] == "TP53"

    async def test_handle_disease_associations(self, mock_agr_client):
        """Test get_disease_associations handler."""
        # Mock the method
        mock_diseases = [
            Disease(
                disease_id="OMIM:151623",
                disease_name="Li-Fraumeni syndrome",
                gene_id="HGNC:11998",
                gene_symbol="TP53",
                association_type="causative",
            )
        ]
        mock_agr_client.get_disease_associations = AsyncMock(return_value=mock_diseases)

        result = await handle_get_disease_associations(
            mock_agr_client,
            gene_id="HGNC:11998",
            format="text",
        )

        assert isinstance(result, str)
        assert "Li-Fraumeni syndrome" in result
        assert "TP53" in result

    async def test_handle_expression(self, mock_agr_client):
        """Test get_expression handler."""
        # Mock the method
        mock_expression = [
            Expression(
                gene_id="HGNC:11998",
                anatomy_term="brain",
                stage="adult",
                expression_pattern="widespread",
            )
        ]
        mock_agr_client.get_expression = AsyncMock(return_value=mock_expression)

        result = await handle_get_expression(
            mock_agr_client,
            gene_id="HGNC:11998",
            format="text",
        )

        assert isinstance(result, str)
        assert "Expression in: brain" in result
        assert "Stage: adult" in result

    async def test_handle_alleles(self, mock_agr_client):
        """Test get_alleles handler."""
        # Mock the method
        mock_alleles = [
            Allele(
                allele_id="test_allele",
                allele_symbol="TP53 R175H",
                gene_id="HGNC:11998",
                gene_symbol="TP53",
                allele_type="missense",
            )
        ]
        mock_agr_client.get_alleles = AsyncMock(return_value=mock_alleles)

        result = await handle_get_alleles(
            mock_agr_client,
            gene_id="HGNC:11998",
            format="json",
        )

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["allele_symbol"] == "TP53 R175H"

    async def test_handle_cross_references(self, mock_agr_client):
        """Test get_cross_references handler."""
        # Mock the method
        mock_xrefs = [
            CrossReference(
                source_id="HGNC:11998",
                source_db="HGNC",
                target_id="MGI:98834",
                target_db="MGI",
            )
        ]
        mock_agr_client.get_cross_references = AsyncMock(return_value=mock_xrefs)

        result = await handle_get_cross_references(
            mock_agr_client,
            identifier="HGNC:11998",
            format="text",
        )

        assert isinstance(result, str)
        assert "HGNC:11998 (HGNC) → MGI:98834 (MGI)" in result

    async def test_handle_batch_get_genes_json(self, mock_agr_client):
        """Test batch gene handler with JSON format."""
        # Mock the method
        mock_genes = [
            Gene(id="HGNC:11998", symbol="TP53", species="Homo sapiens"),
            Gene(id="HGNC:1100", symbol="AKT1", species="Homo sapiens"),
        ]
        mock_agr_client.batch_get_genes = AsyncMock(return_value=mock_genes)

        result = await handle_batch_get_genes(
            mock_agr_client,
            identifiers=["HGNC:11998", "HGNC:1100"],
            format="json",
        )

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["symbol"] == "TP53"
        assert parsed[1]["symbol"] == "AKT1"

    async def test_handle_batch_get_genes_tsv(self, mock_agr_client):
        """Test batch gene handler with TSV format."""
        # Mock the method to return TSV string
        tsv_content = "Gene\tSymbol\tSpecies\nHGNC:11998\tTP53\tHomo sapiens"
        mock_agr_client.batch_get_genes = AsyncMock(return_value=tsv_content)

        result = await handle_batch_get_genes(
            mock_agr_client,
            identifiers=["HGNC:11998"],
            format="tsv",
        )

        assert isinstance(result, str)
        assert result == tsv_content
