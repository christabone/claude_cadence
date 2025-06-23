"""Tool definitions and registry for AGR MCP server.

This module defines the available tools and their handlers for the MCP server.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from .client import AGRClient
from .formatters import format_gene, format_disease, format_expression, format_allele


# Tool handler type
ToolHandler = Callable[..., Any]


async def handle_get_gene(
    client: AGRClient,
    identifier: str,
    include_orthologs: bool = False,
    include_expression: bool = False,
    include_disease: bool = False,
    format: str = "json",
) -> str:
    """Handle get_gene tool invocation.

    Args:
        client: AGR API client
        identifier: Gene identifier
        include_orthologs: Include ortholog information
        include_expression: Include expression data
        include_disease: Include disease associations
        format: Output format (json, text)

    Returns:
        Formatted gene information
    """
    gene = await client.get_gene(
        identifier=identifier,
        include_orthologs=include_orthologs,
        include_expression=include_expression,
        include_disease=include_disease,
    )

    if format == "text":
        return format_gene(gene, format="text")
    else:
        return gene.model_dump_json(indent=2)


async def handle_search_genes(
    client: AGRClient,
    query: str,
    species: Optional[str] = None,
    limit: int = 20,
    format: str = "json",
) -> str:
    """Handle search_genes tool invocation.

    Args:
        client: AGR API client
        query: Search query
        species: Filter by species
        limit: Maximum results
        format: Output format

    Returns:
        Search results
    """
    genes = await client.search_genes(
        query=query,
        species=species,
        limit=limit,
    )

    if format == "text":
        results = []
        for gene in genes:
            results.append(format_gene(gene, format="summary"))
        return "\n\n".join(results)
    else:
        return json.dumps([g.model_dump() for g in genes], indent=2)


async def handle_get_disease_associations(
    client: AGRClient,
    gene_id: Optional[str] = None,
    disease_id: Optional[str] = None,
    include_orthologs: bool = False,
    format: str = "json",
) -> str:
    """Handle get_disease_associations tool invocation.

    Args:
        client: AGR API client
        gene_id: Filter by gene
        disease_id: Filter by disease
        include_orthologs: Include ortholog associations
        format: Output format

    Returns:
        Disease associations
    """
    associations = await client.get_disease_associations(
        gene_id=gene_id,
        disease_id=disease_id,
        include_orthologs=include_orthologs,
    )

    if format == "text":
        results = []
        for assoc in associations:
            results.append(format_disease(assoc, format="text"))
        return "\n\n".join(results)
    else:
        return json.dumps([a.model_dump() for a in associations], indent=2)


async def handle_get_expression(
    client: AGRClient,
    gene_id: str,
    stage: Optional[str] = None,
    anatomy_term: Optional[str] = None,
    format: str = "json",
) -> str:
    """Handle get_expression tool invocation.

    Args:
        client: AGR API client
        gene_id: Gene identifier
        stage: Filter by stage
        anatomy_term: Filter by anatomy
        format: Output format

    Returns:
        Expression data
    """
    expression_data = await client.get_expression(
        gene_id=gene_id,
        stage=stage,
        anatomy_term=anatomy_term,
    )

    if format == "text":
        results = []
        for expr in expression_data:
            results.append(format_expression(expr, format="text"))
        return "\n\n".join(results)
    else:
        return json.dumps([e.model_dump() for e in expression_data], indent=2)


async def handle_get_alleles(
    client: AGRClient,
    gene_id: str,
    allele_id: Optional[str] = None,
    include_phenotypes: bool = False,
    format: str = "json",
) -> str:
    """Handle get_alleles tool invocation.

    Args:
        client: AGR API client
        gene_id: Gene identifier
        allele_id: Specific allele
        include_phenotypes: Include phenotype data
        format: Output format

    Returns:
        Allele information
    """
    alleles = await client.get_alleles(
        gene_id=gene_id,
        allele_id=allele_id,
        include_phenotypes=include_phenotypes,
    )

    if format == "text":
        results = []
        for allele in alleles:
            results.append(format_allele(allele, format="text"))
        return "\n\n".join(results)
    else:
        return json.dumps([a.model_dump() for a in alleles], indent=2)


async def handle_get_cross_references(
    client: AGRClient,
    identifier: str,
    target_db: Optional[str] = None,
    format: str = "json",
) -> str:
    """Handle get_cross_references tool invocation.

    Args:
        client: AGR API client
        identifier: Source identifier
        target_db: Filter by target database
        format: Output format

    Returns:
        Cross-references
    """
    xrefs = await client.get_cross_references(
        identifier=identifier,
        target_db=target_db,
    )

    if format == "text":
        results = []
        for xref in xrefs:
            results.append(
                f"{xref.source_id} ({xref.source_db}) → "
                f"{xref.target_id} ({xref.target_db})"
            )
        return "\n".join(results)
    else:
        return json.dumps([x.model_dump() for x in xrefs], indent=2)


async def handle_batch_get_genes(
    client: AGRClient,
    identifiers: List[str],
    include_orthologs: bool = False,
    format: str = "json",
) -> str:
    """Handle batch gene retrieval.

    Args:
        client: AGR API client
        identifiers: List of gene identifiers
        include_orthologs: Include ortholog information
        format: Output format (json, tsv, text)

    Returns:
        Genes in requested format
    """
    result = await client.batch_get_genes(
        identifiers=identifiers,
        include_orthologs=include_orthologs,
        format=format,
    )

    if format == "json" and isinstance(result, list):
        return json.dumps([g.model_dump() for g in result], indent=2)
    else:
        return result


def get_tool_registry() -> Dict[str, Dict[str, Any]]:
    """Get the registry of available tools.

    Returns:
        Dictionary mapping tool names to their metadata and handlers
    """
    return {
        "get_gene": {
            "description": "Get detailed information about a gene",
            "handler": handle_get_gene,
            "input_schema": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Gene identifier (e.g., HGNC:11998, WB:WBGene00006789)",
                    },
                    "include_orthologs": {
                        "type": "boolean",
                        "description": "Include ortholog information",
                        "default": False,
                    },
                    "include_expression": {
                        "type": "boolean",
                        "description": "Include expression data",
                        "default": False,
                    },
                    "include_disease": {
                        "type": "boolean",
                        "description": "Include disease associations",
                        "default": False,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["identifier"],
            },
        },
        "search_genes": {
            "description": "Search for genes by name, symbol, or description",
            "handler": handle_search_genes,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "species": {
                        "type": "string",
                        "description": "Filter by species (e.g., 'Homo sapiens')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["query"],
            },
        },
        "get_disease_associations": {
            "description": "Get disease associations for genes",
            "handler": handle_get_disease_associations,
            "input_schema": {
                "type": "object",
                "properties": {
                    "gene_id": {
                        "type": "string",
                        "description": "Gene identifier",
                    },
                    "disease_id": {
                        "type": "string",
                        "description": "Disease identifier (e.g., OMIM:168600)",
                    },
                    "include_orthologs": {
                        "type": "boolean",
                        "description": "Include ortholog associations",
                        "default": False,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
            },
        },
        "get_expression": {
            "description": "Get gene expression data",
            "handler": handle_get_expression,
            "input_schema": {
                "type": "object",
                "properties": {
                    "gene_id": {
                        "type": "string",
                        "description": "Gene identifier",
                    },
                    "stage": {
                        "type": "string",
                        "description": "Developmental stage filter",
                    },
                    "anatomy_term": {
                        "type": "string",
                        "description": "Anatomical structure filter",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["gene_id"],
            },
        },
        "get_alleles": {
            "description": "Get allele information for a gene",
            "handler": handle_get_alleles,
            "input_schema": {
                "type": "object",
                "properties": {
                    "gene_id": {
                        "type": "string",
                        "description": "Gene identifier",
                    },
                    "allele_id": {
                        "type": "string",
                        "description": "Specific allele identifier",
                    },
                    "include_phenotypes": {
                        "type": "boolean",
                        "description": "Include phenotype information",
                        "default": False,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["gene_id"],
            },
        },
        "get_cross_references": {
            "description": "Map identifiers between databases",
            "handler": handle_get_cross_references,
            "input_schema": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Source identifier",
                    },
                    "target_db": {
                        "type": "string",
                        "description": "Target database filter",
                        "enum": ["HGNC", "MGI", "RGD", "ZFIN", "FB", "WB", "SGD", "XB"],
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["identifier"],
            },
        },
        "get_genes_batch": {
            "description": "Get multiple genes in a single request",
            "handler": handle_batch_get_genes,
            "input_schema": {
                "type": "object",
                "properties": {
                    "identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene identifiers",
                        "minItems": 1,
                        "maxItems": 100,
                    },
                    "include_orthologs": {
                        "type": "boolean",
                        "description": "Include ortholog information",
                        "default": False,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "tsv", "text"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["identifiers"],
            },
        },
    }
