"""Metadata tools for AGR MCP Server.

This module implements tools for retrieving metadata information
from the Alliance Genome Resources API, including supported species.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import httpx
import json

from ..errors import ValidationError, ResourceNotFoundError, ToolExecutionError
from ..utils.http_client import http_client
from ..utils.logging_config import get_logger

logger = get_logger('metadata_tools')


class Species(BaseModel):
    """Pydantic model for Alliance Genome species data.

    This model represents species information from the Alliance Genome Resources API
    with proper field aliases for AGR API compatibility.
    """

    taxon_id: str = Field(..., alias="taxonId", description="NCBI Taxonomy ID")
    name: str = Field(..., description="Scientific name of the species")
    display_name: str = Field(..., alias="displayName", description="Display name for the species")

    class Config:
        """Pydantic model configuration."""
        allow_population_by_field_name = True
        extra = "ignore"  # Ignore extra fields from API response


async def get_supported_species() -> Dict[str, Any]:
    """Get list of supported species from Alliance Genome Resources API.

    Returns:
        Dictionary containing list of supported species with their information

    Raises:
        ToolExecutionError: If API request fails
    """
    logger.info("Fetching supported species from Alliance Genome Resources API")

    try:
        # Make API request to get supported species
        # Based on AGR API patterns, try species metadata endpoint
        response = await http_client.get("/api/meta-data/species")
        response.raise_for_status()  # Raise exception for HTTP error status codes
        data = response.json()

        # Parse response data into Species objects
        species_list = []

        if isinstance(data, list):
            # Direct list of species
            raw_species = data
        elif isinstance(data, dict) and "results" in data:
            # Paginated response with results array
            raw_species = data.get("results", [])
        elif isinstance(data, dict) and "species" in data:
            # Response with species array
            raw_species = data.get("species", [])
        else:
            # Fallback - treat entire response as single species if it has required fields
            if "taxonId" in data and "name" in data:
                raw_species = [data]
            else:
                raw_species = []

        # Convert raw data to Species objects
        for species_data in raw_species:
            try:
                species = Species(**species_data)
                species_list.append(species.dict(by_alias=True))
            except (ValidationError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse species data: {species_data}. Error: {e}")
                continue

        # Build response
        result = {
            "species": species_list,
            "total": len(species_list),
            "message": f"Successfully retrieved {len(species_list)} supported species"
        }

        logger.info(f"Successfully retrieved {len(species_list)} supported species")
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error when fetching species: {e.response.status_code} - {e.response.text}")
        raise ToolExecutionError(
            message=f"HTTP error when fetching species: {e.response.status_code}",
            tool_name="metadata_tools",
            operation="get_supported_species"
        ) from e
    except httpx.RequestError as e:
        logger.error(f"Network error when fetching species: {str(e)}")
        raise ToolExecutionError(
            message=f"Network error when fetching species: {str(e)}",
            tool_name="metadata_tools",
            operation="get_supported_species"
        ) from e
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response when fetching species: {str(e)}")
        raise ToolExecutionError(
            message=f"Invalid JSON response when fetching species: {str(e)}",
            tool_name="metadata_tools",
            operation="get_supported_species"
        ) from e
    except ValidationError as e:
        logger.error(f"Data validation error when parsing species: {str(e)}")
        raise ToolExecutionError(
            message=f"Data validation error when parsing species: {str(e)}",
            tool_name="metadata_tools",
            operation="get_supported_species"
        ) from e
