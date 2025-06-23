"""AGR API client for accessing Alliance Genome Resource data.

This module provides a high-level client for interacting with the Alliance
Genome Resource REST API.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import backoff
import httpx
from pydantic import BaseModel, Field

from .models import Gene, Disease, Expression, Allele, CrossReference
from .utils.cache import CacheManager

logger = logging.getLogger(__name__)


class AGRClientError(Exception):
    """Base exception for AGR client errors."""
    pass


class AGRAPIError(AGRClientError):
    """Exception raised for API errors."""

    def __init__(self, status_code: int, message: str, details: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(f"API Error {status_code}: {message}")


class AGRClient:
    """Client for accessing Alliance Genome Resource API.

    This client provides methods for querying various types of genomic data
    from the Alliance API, with built-in caching and retry logic.

    Attributes:
        base_url: Base URL for the AGR API
        cache_manager: Cache manager instance
        client: HTTP client instance
    """

    def __init__(
        self,
        base_url: str = "https://www.alliancegenome.org/api",
        cache_dir: Optional[str] = None,
        cache_ttl: int = 3600,
        timeout: float = 30.0,
    ):
        """Initialize the AGR client.

        Args:
            base_url: Base URL for the AGR API
            cache_dir: Directory for caching responses
            cache_ttl: Cache time-to-live in seconds
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.cache_manager = CacheManager(cache_dir=cache_dir, ttl=cache_ttl)
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "agr-mcp-server/0.1.0",
                "Accept": "application/json",
            },
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.HTTPStatusError),
        max_tries=3,
        max_time=60,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON request body
            use_cache: Whether to use caching

        Returns:
            API response data

        Raises:
            AGRAPIError: If the API returns an error
        """
        url = urljoin(self.base_url, endpoint.lstrip("/"))

        # Check cache for GET requests
        if method == "GET" and use_cache:
            cache_key = self.cache_manager.make_key(url, params)
            cached = await self.cache_manager.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {url}")
                return cached

        # Make request
        logger.debug(f"{method} {url} (params: {params})")

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                pass

            raise AGRAPIError(
                status_code=e.response.status_code,
                message=str(e),
                details=error_data,
            )

        except httpx.RequestError as e:
            raise AGRClientError(f"Request failed: {str(e)}")

        # Parse response
        data = response.json()

        # Cache successful GET responses
        if method == "GET" and use_cache:
            await self.cache_manager.set(cache_key, data)

        return data

    async def get_gene(
        self,
        identifier: str,
        include_orthologs: bool = False,
        include_expression: bool = False,
        include_disease: bool = False,
    ) -> Gene:
        """Get gene information by identifier.

        Args:
            identifier: Gene identifier (e.g., "HGNC:11998", "WB:WBGene00006789")
            include_orthologs: Include ortholog information
            include_expression: Include expression data
            include_disease: Include disease associations

        Returns:
            Gene object with requested information

        Raises:
            AGRAPIError: If gene not found or API error
        """
        endpoint = f"/gene/{identifier}"
        params = {
            "includeOrthologs": include_orthologs,
            "includeExpression": include_expression,
            "includeDisease": include_disease,
        }

        data = await self._request("GET", endpoint, params=params)
        return Gene(**data)

    async def search_genes(
        self,
        query: str,
        species: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Gene]:
        """Search for genes by query.

        Args:
            query: Search query
            species: Filter by species (e.g., "Homo sapiens", "Mus musculus")
            limit: Maximum number of results
            offset: Result offset for pagination

        Returns:
            List of matching genes
        """
        endpoint = "/search/gene"
        params = {
            "q": query,
            "limit": limit,
            "offset": offset,
        }

        if species:
            params["species"] = species

        data = await self._request("GET", endpoint, params=params)
        return [Gene(**item) for item in data.get("results", [])]

    async def get_disease_associations(
        self,
        gene_id: Optional[str] = None,
        disease_id: Optional[str] = None,
        include_orthologs: bool = False,
    ) -> List[Disease]:
        """Get disease associations.

        Args:
            gene_id: Filter by gene identifier
            disease_id: Filter by disease identifier
            include_orthologs: Include ortholog associations

        Returns:
            List of disease associations
        """
        endpoint = "/disease/associations"
        params = {}

        if gene_id:
            params["geneId"] = gene_id
        if disease_id:
            params["diseaseId"] = disease_id
        if include_orthologs:
            params["includeOrthologs"] = True

        data = await self._request("GET", endpoint, params=params)
        return [Disease(**item) for item in data.get("results", [])]

    async def get_expression(
        self,
        gene_id: str,
        stage: Optional[str] = None,
        anatomy_term: Optional[str] = None,
    ) -> List[Expression]:
        """Get gene expression data.

        Args:
            gene_id: Gene identifier
            stage: Filter by developmental stage
            anatomy_term: Filter by anatomy term

        Returns:
            List of expression records
        """
        endpoint = f"/expression/gene/{gene_id}"
        params = {}

        if stage:
            params["stage"] = stage
        if anatomy_term:
            params["anatomyTerm"] = anatomy_term

        data = await self._request("GET", endpoint, params=params)
        return [Expression(**item) for item in data.get("results", [])]

    async def get_alleles(
        self,
        gene_id: str,
        allele_id: Optional[str] = None,
        include_phenotypes: bool = False,
    ) -> List[Allele]:
        """Get allele information for a gene.

        Args:
            gene_id: Gene identifier
            allele_id: Specific allele identifier
            include_phenotypes: Include phenotype data

        Returns:
            List of alleles
        """
        endpoint = f"/allele/gene/{gene_id}"
        params = {"includePhenotypes": include_phenotypes}

        if allele_id:
            params["alleleId"] = allele_id

        data = await self._request("GET", endpoint, params=params)
        return [Allele(**item) for item in data.get("results", [])]

    async def get_cross_references(
        self,
        identifier: str,
        target_db: Optional[str] = None,
    ) -> List[CrossReference]:
        """Get cross-references for an identifier.

        Args:
            identifier: Source identifier
            target_db: Filter by target database

        Returns:
            List of cross-references
        """
        endpoint = f"/xref/{identifier}"
        params = {}

        if target_db:
            params["targetDb"] = target_db

        data = await self._request("GET", endpoint, params=params)
        return [CrossReference(**item) for item in data.get("results", [])]

    async def batch_get_genes(
        self,
        identifiers: List[str],
        include_orthologs: bool = False,
        format: str = "json",
    ) -> Union[List[Gene], str]:
        """Get multiple genes in a single request.

        Args:
            identifiers: List of gene identifiers
            include_orthologs: Include ortholog information
            format: Output format ("json", "tsv", "text")

        Returns:
            Genes in requested format
        """
        endpoint = "/gene/batch"
        json_data = {
            "identifiers": identifiers,
            "includeOrthologs": include_orthologs,
            "format": format,
        }

        data = await self._request("POST", endpoint, json_data=json_data)

        if format == "json":
            return [Gene(**item) for item in data.get("results", [])]
        else:
            return data.get("content", "")
