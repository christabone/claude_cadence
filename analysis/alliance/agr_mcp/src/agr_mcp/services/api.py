"""
REST API client service for Alliance API access.

This module provides functionality for interacting with the Alliance
REST API endpoints for retrieving genomic data.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import backoff
import httpx
from httpx import AsyncClient, Response

from ..core.config import APIConfig

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Represents an API response."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    elapsed: float

    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        """Check if response was an error."""
        return self.status_code >= 400


class APIService:
    """
    Service for Alliance REST API interactions.

    Provides methods for querying gene, disease, and other genomic
    data from the Alliance API with retry logic and error handling.
    """

    def __init__(self, config: APIConfig):
        """
        Initialize API service.

        Args:
            config: API configuration
        """
        self.config = config
        self._client: Optional[AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            headers = {
                "User-Agent": "AGR-MCP-Server/1.0",
                "Accept": "application/json",
            }

            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=20),
                follow_redirects=True
            )
            logger.info(f"Initialized API client for: {self.config.base_url}")

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _is_retryable_error(self, response: Response) -> bool:
        """Check if error is retryable."""
        if response.status_code in (429, 502, 503, 504):
            return True
        if response.status_code >= 500:
            return True
        return False

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPStatusError,
        max_tries=3,
        giveup=lambda e: not (500 <= e.response.status_code < 600 or e.response.status_code == 429)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body data

        Returns:
            APIResponse object
        """
        if self._client is None:
            await self.connect()

        try:
            response = await self._client.request(
                method=method,
                url=endpoint,
                params=params,
                json=json_data
            )

            # Raise for status codes that should be retried
            if self._is_retryable_error(response):
                response.raise_for_status()

            # Parse response
            data = None
            if response.content:
                try:
                    data = response.json()
                except Exception:
                    data = response.text

            return APIResponse(
                status_code=response.status_code,
                data=data,
                headers=dict(response.headers),
                elapsed=response.elapsed.total_seconds()
            )

        except httpx.HTTPStatusError as e:
            # Re-raise for backoff decorator
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            APIResponse object
        """
        return await self._make_request("GET", endpoint, params=params)

    async def post(
        self,
        endpoint: str,
        json_data: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint
            json_data: JSON body data
            params: Query parameters

        Returns:
            APIResponse object
        """
        return await self._make_request("POST", endpoint, params=params, json_data=json_data)

    # Alliance-specific API methods

    async def search_genes(
        self,
        query: str,
        species: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for genes across Alliance databases.

        Args:
            query: Search query
            species: Optional species filter (NCBITaxon ID)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of gene objects
        """
        params = {
            "q": query,
            "limit": limit,
            "offset": offset
        }

        if species:
            params["species"] = species

        response = await self.get("/genes/search", params=params)

        if response.is_success and isinstance(response.data, dict):
            return response.data.get("results", [])

        return []

    async def get_gene(self, gene_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific gene.

        Args:
            gene_id: Gene identifier

        Returns:
            Gene object
        """
        response = await self.get(f"/gene/{gene_id}")

        if response.is_success:
            return response.data
        elif response.status_code == 404:
            raise ValueError(f"Gene not found: {gene_id}")
        else:
            raise Exception(f"API error: {response.status_code}")

    async def get_gene_orthologs(self, gene_id: str) -> List[Dict[str, Any]]:
        """
        Get ortholog relationships for a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            List of ortholog objects
        """
        response = await self.get(f"/gene/{gene_id}/orthologs")

        if response.is_success and isinstance(response.data, list):
            return response.data

        return []

    async def get_disease_associations(
        self,
        gene_id: Optional[str] = None,
        disease_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get disease associations.

        Args:
            gene_id: Optional gene filter
            disease_id: Optional disease filter

        Returns:
            List of disease association objects
        """
        params = {}
        if gene_id:
            params["geneID"] = gene_id
        if disease_id:
            params["diseaseID"] = disease_id

        response = await self.get("/disease-associations", params=params)

        if response.is_success and isinstance(response.data, dict):
            return response.data.get("results", [])

        return []

    async def get_alleles(
        self,
        gene_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alleles for a gene.

        Args:
            gene_id: Gene identifier
            limit: Maximum results

        Returns:
            List of allele objects
        """
        params = {"limit": limit}
        response = await self.get(f"/gene/{gene_id}/alleles", params=params)

        if response.is_success and isinstance(response.data, dict):
            return response.data.get("results", [])

        return []

    async def batch_genes(self, gene_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get information for multiple genes in a single request.

        Args:
            gene_ids: List of gene identifiers

        Returns:
            List of gene objects
        """
        response = await self.post(
            "/genes/batch",
            json_data={"ids": gene_ids}
        )

        if response.is_success and isinstance(response.data, list):
            return response.data

        return []
