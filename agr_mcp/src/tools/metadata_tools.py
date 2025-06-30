"""Metadata tools for AGR MCP Server.

This module implements tools for retrieving metadata information
from the Alliance Genome Resources API, including supported species.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..errors import ValidationError, ResourceNotFoundError, ToolExecutionError
from ..utils.http_client import http_client
from ..utils.logging_config import get_logger

logger = get_logger('metadata_tools')


class Species(BaseModel):
    """Pydantic model for representing a species from the Alliance Genome Resources API.

    This model uses field aliases to map between the API response field names
    and more Pythonic attribute names.
    """

    id: str = Field(..., alias="primaryId", description="Primary identifier for the species")
    name: str = Field(..., alias="name", description="Species name")
    taxon_id: str = Field(..., alias="taxonId", description="NCBI Taxonomy ID")
    display_name: Optional[str] = Field(None, alias="displayName", description="Display name for the species")
    scientific_name: Optional[str] = Field(None, alias="scientificName", description="Scientific name")
    common_name: Optional[str] = Field(None, alias="commonName", description="Common name")
    abbreviation: Optional[str] = Field(None, alias="abbreviation", description="Species abbreviation")
    phylogenetic_order: Optional[int] = Field(None, alias="phylogeneticOrder", description="Phylogenetic ordering")

    class Config:
        """Pydantic configuration."""
        allow_population_by_field_name = True

    def __str__(self) -> str:
        """String representation of the species."""
        return f"Species(id={self.id}, name={self.name}, taxon_id={self.taxon_id})"


async def get_supported_species() -> Dict[str, Any]:
    """Get all supported species from the Alliance Genome Resources API.

    Returns:
        Dictionary containing list of supported species information

    Raises:
        ToolExecutionError: If API request fails
    """
    logger.info("Fetching supported species from Alliance API")

    # Configure http_client with base URL if needed
    if not http_client.base_url:
        from ..config import Config
        http_client.base_url = Config.BASE_URL

    try:
        # Make API request to get species data
        response = await http_client.get("/api/species")
        data = response.json()

        # Parse response into Species objects
        species_list = []
        for species_data in data.get("results", []):
            try:
                species = Species(**species_data)
                species_list.append(species.dict(by_alias=False))
            except Exception as e:
                logger.warning(f"Failed to parse species data: {species_data}, error: {e}")
                continue

        # Format results
        results = {
            "total": len(species_list),
            "species": species_list
        }

        logger.info(f"Successfully retrieved {len(species_list)} supported species")
        return results

    except Exception as e:
        logger.error(f"Failed to get supported species: {str(e)}")
        raise ToolExecutionError(
            message=f"Failed to get supported species: {str(e)}",
            tool_name="metadata_tools",
            operation="get_supported_species"
        ) from e


class MetadataTool:
    """Tool for querying metadata information from the Alliance of Genome Resources API."""

    async def get_species_list(self) -> Dict[str, Any]:
        """Get list of all supported species.

        Returns:
            Dictionary containing species information

        Raises:
            ToolExecutionError: If API request fails
        """
        return await get_supported_species()


# Create singleton instance
metadata_tool = MetadataTool()
