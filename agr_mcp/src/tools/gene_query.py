"""Gene query tools for AGR MCP Server.

This module implements tools for searching and retrieving gene information
from the Alliance Genome Resources API.
"""

from typing import Any, Dict, List, Optional

from ..errors import ValidationError, ResourceNotFoundError, ToolExecutionError
from ..utils.http_client import http_client
from ..utils.validators import validate_gene_id
from ..utils.logging_config import get_logger

logger = get_logger('gene_query')


async def search_genes(
    query: str,
    species: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Search for genes by symbol, name, or identifier.

    Args:
        query: Gene symbol, name, or identifier to search for
        species: Optional species filter (name or taxon ID)
        limit: Maximum number of results to return

    Returns:
        Dictionary containing search results with gene information

    Raises:
        ValidationError: If input validation fails
        ToolExecutionError: If API request fails
    """
    if not query or not query.strip():
        raise ValidationError("Query string cannot be empty", field="query")

    if limit < 1 or limit > 100:
        raise ValidationError(
            "Limit must be between 1 and 100",
            field="limit",
            value=limit
        )

    # TODO: Add species validation when validate_species is implemented
    # if species:
    #     species = validate_species(species)

    logger.info(f"Searching for genes: query='{query}', species={species}, limit={limit}")

    # Build query parameters
    params = {
        "q": query,
        "limit": limit,
        "category": "gene"
    }

    if species:
        params["species"] = species

    # Make API request
    # Configure http_client with base URL if needed
    if not http_client.base_url:
        from ..config import Config
        http_client.base_url = Config.BASE_URL
    try:
        response = await http_client.get("/search", params=params)
        data = response.json()

        # Format results
        results = {
            "query": query,
            "total": data.get("total", 0),
            "results": []
        }

        for item in data.get("results", []):
            gene_info = {
                "id": item.get("id"),
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "species": item.get("species", {}).get("name"),
                "taxonId": item.get("species", {}).get("taxonId"),
                "synonyms": item.get("synonyms", []),
                "chromosome": item.get("chromosome"),
                "biotype": item.get("soTermName")
            }
            results["results"].append(gene_info)

        return results

    except Exception as e:
        logger.error(f"Gene search failed: {str(e)}")
        raise ToolExecutionError(
            message=f"Failed to search genes: {str(e)}",
            tool_name="gene_query",
            operation="search_genes"
        ) from e


async def get_gene_details(gene_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific gene.

    Args:
        gene_id: AGR gene identifier

    Returns:
        Dictionary containing detailed gene information

    Raises:
        ValidationError: If gene ID is invalid
        ResourceNotFoundError: If gene is not found
        ToolExecutionError: If API request fails
    """
    gene_id = validate_gene_id(gene_id)

    logger.info(f"Getting gene details for: {gene_id}")

    # Configure http_client with base URL if needed
    if not http_client.base_url:
        from ..config import Config
        http_client.base_url = Config.BASE_URL
    try:
        response = await http_client.get(f"/gene/{gene_id}")

        if response.status_code == 404:
            raise ResourceNotFoundError(
                f"Gene not found: {gene_id}",
                resource_type="gene",
                resource_id=gene_id
            )

        data = response.json()

        # Format detailed gene information
        gene_details = {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            "description": data.get("automatedGeneSynopsis"),
            "species": {
                "name": data.get("species", {}).get("name"),
                "taxonId": data.get("species", {}).get("taxonId")
            },
            "location": {
                "chromosome": data.get("chromosome"),
                "start": data.get("start"),
                "end": data.get("end"),
                "strand": data.get("strand"),
                "assembly": data.get("assembly")
            },
            "biotype": data.get("soTermName"),
            "synonyms": data.get("synonyms", []),
            "crossReferences": data.get("crossReferences", []),
            "diseases": [
                {
                    "name": disease.get("name"),
                    "id": disease.get("id"),
                    "associationType": disease.get("associationType")
                }
                for disease in data.get("diseases", [])
            ],
            "expression": data.get("expression", []),
            "interactions": data.get("interactions", {}).get("total", 0),
            "phenotypes": len(data.get("phenotypes", []))
        }

        return gene_details

    except ResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get gene details: {str(e)}")
        raise ToolExecutionError(
            message=f"Failed to get gene details: {str(e)}",
            tool_name="gene_query",
            operation="get_gene_details"
        ) from e


async def find_orthologs(
    gene_id: str,
    target_species: Optional[str] = None
) -> Dict[str, Any]:
    """Find orthologous genes across species.

    Args:
        gene_id: AGR gene identifier
        target_species: Optional target species filter

    Returns:
        Dictionary containing ortholog information

    Raises:
        ValidationError: If input validation fails
        ResourceNotFoundError: If gene is not found
        ToolExecutionError: If API request fails
    """
    gene_id = validate_gene_id(gene_id)

    # TODO: Add species validation when validate_species is implemented
    # if target_species:
    #     target_species = validate_species(target_species)

    logger.info(
        f"Finding orthologs for gene: {gene_id}, target_species={target_species}"
    )

    # Configure http_client with base URL if needed
    if not http_client.base_url:
        from ..config import Config
        http_client.base_url = Config.BASE_URL
    try:
        # First, get gene details to ensure it exists
        gene_response = await http_client.get(f"/gene/{gene_id}")
        if gene_response.status_code == 404:
            raise ResourceNotFoundError(
                f"Gene not found: {gene_id}",
                resource_type="gene",
                resource_id=gene_id
            )

        gene_data = gene_response.json()
        source_species = gene_data.get("species", {}).get("name", "Unknown")

        # Get orthologs
        response = await http_client.get(f"/gene/{gene_id}/orthologs")
        ortholog_data = response.json()

        # Format results
        results = {
            "sourceGene": {
                "id": gene_id,
                "symbol": gene_data.get("symbol"),
                "species": source_species
            },
            "orthologs": []
        }

        for ortholog in ortholog_data.get("results", []):
            ortholog_info = {
                "id": ortholog.get("id"),
                "symbol": ortholog.get("symbol"),
                "species": ortholog.get("species", {}).get("name"),
                "taxonId": ortholog.get("species", {}).get("taxonId"),
                "method": ortholog.get("predictionMethodsMatched", []),
                "confidence": ortholog.get("confidence", "N/A")
            }

            # Filter by target species if specified
            if target_species:
                species_match = (
                    ortholog_info["species"] == target_species or
                    ortholog_info["taxonId"] == target_species
                )
                if not species_match:
                    continue

            results["orthologs"].append(ortholog_info)

        results["total"] = len(results["orthologs"])

        return results

    except ResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to find orthologs: {str(e)}")
        raise ToolExecutionError(
            message=f"Failed to find orthologs: {str(e)}",
            tool_name="gene_query",
            operation="find_orthologs"
        ) from e


class GeneQueryTool:
    """Tool for querying gene information from the Alliance of Genome Resources API."""

    async def query_gene(
        self,
        gene_id: str,
        include_disease: bool = False,
        include_expression: bool = False
    ) -> Dict[str, Any]:
        """Query gene information from AGR API.

        Args:
            gene_id: Gene ID to query (e.g., MGI:123456)
            include_disease: Whether to include disease associations
            include_expression: Whether to include expression data

        Returns:
            Dictionary containing gene information

        Raises:
            ValidationError: If gene ID is invalid
            ToolExecutionError: If API request fails
            ResourceNotFoundError: If gene is not found
        """
        # Validate gene ID
        validated_gene_id = validate_gene_id(gene_id)

        # Build query parameters
        params = {}
        if include_disease:
            params['includeDisease'] = 'true'
        if include_expression:
            params['includeExpression'] = 'true'

        try:
            # Make API request
            logger.info(f"Querying gene information for {validated_gene_id}")

            # Configure http_client with base URL if needed
            if not http_client.base_url:
                from ..config import Config
                http_client.base_url = Config.BASE_URL

            response = await http_client.get(f"/api/gene/{validated_gene_id}", params=params)

            # Check for 404 - gene not found
            if response.status_code == 404:
                logger.warning(f"Gene not found: {validated_gene_id}")
                raise ResourceNotFoundError(
                    message=f"Gene not found: {validated_gene_id}",
                    resource_type="gene",
                    resource_id=validated_gene_id
                )

            # Parse JSON response
            try:
                response_data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse API response: {e}")
                raise ToolExecutionError(
                    message="Failed to parse gene API response",
                    tool_name="gene_query",
                    operation="parse_response",
                    error_output=str(e)
                )

            # Process response
            result = self._process_gene_response(response_data)
            logger.info(f"Successfully retrieved gene information for {validated_gene_id}")

            return result

        except (ValidationError, ResourceNotFoundError):
            # Re-raise these specific errors
            raise
        except Exception as e:
            logger.error(f"Error querying gene {validated_gene_id}: {str(e)}")
            raise ToolExecutionError(
                message=f"Failed to query gene information: {str(e)}",
                tool_name="gene_query",
                operation="query_gene",
                error_output=str(e)
            )

    def _process_gene_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure the gene API response.

        Args:
            response: Raw API response

        Returns:
            Processed and structured gene information
        """
        # Extract relevant fields
        result = {
            'id': response.get('id'),
            'symbol': response.get('symbol'),
            'name': response.get('name'),
            'synonyms': response.get('synonyms', []),
            'soTermName': response.get('soTermName'),
            'organism': None,
            'chromosomeLocation': None,
            'diseases': [],
            'expression': [],
        }

        # Extract organism information
        if 'organism' in response:
            result['organism'] = {
                'name': response['organism'].get('name'),
                'species': response['organism'].get('species'),
                'taxonId': response['organism'].get('taxonId'),
            }

        # Extract chromosome location
        if 'genomeLocations' in response and response['genomeLocations']:
            loc = response['genomeLocations'][0]
            result['chromosomeLocation'] = {
                'chromosome': loc.get('chromosome'),
                'startPosition': loc.get('start'),
                'endPosition': loc.get('end'),
                'assembly': loc.get('assembly'),
                'strand': loc.get('strand'),
            }

        # Extract disease associations if present
        if 'diseases' in response and response['diseases']:
            result['diseases'] = [
                {
                    'id': disease.get('id'),
                    'name': disease.get('name'),
                    'associationType': disease.get('associationType'),
                }
                for disease in response['diseases']
            ]

        # Extract expression data if present
        if 'expression' in response and response['expression']:
            result['expression'] = [
                {
                    'tissue': expr.get('tissue', {}).get('name'),
                    'stage': expr.get('stage', {}).get('name'),
                    'assay': expr.get('assay', {}).get('name'),
                }
                for expr in response['expression']
            ]

        return result


# Create singleton instance
gene_query_tool = GeneQueryTool()

# Export wrapper functions for compatibility
# Note: These are already defined as module-level functions above
