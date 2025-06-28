"""Module-level configuration initialization for AGR MCP.

This module provides performance-optimized module-level configuration
that initializes once at import time, reducing redundant configuration calls.

The module creates a global configuration instance that can be accessed
without creating new ConfigManager instances for each operation.
"""

import os
import logging
from typing import Optional, Dict, Any
from .config import ConfigManager

logger = logging.getLogger(__name__)

# Global configuration instance - initialized once at module import
_global_config: Optional[ConfigManager] = None
_initialization_lock = False

# Module-level configuration values - cached for performance
BASE_URL: str = ""
API_TIMEOUT: int = 30
MAX_REQUESTS_PER_SECOND: int = 10
MAX_RETRIES: int = 3
RETRY_BACKOFF_FACTOR: float = 0.5
ENABLE_CACHING: bool = True  # Enable request deduplication caching
CACHE_TTL: int = 300  # 5 minutes for request deduplication

# API endpoint configurations - pre-built for performance
API_ENDPOINTS = {
    "search": "/search",
    "gene_details": "/api/gene/{gene_id}",
    "orthologs": "/api/gene/{gene_id}/orthologs",
    "disease": "/api/disease",
    "expression": "/api/expression",
    "alleles": "/api/allele"
}


def _initialize_module_config() -> None:
    """Initialize module-level configuration.

    This function is called once at module import time to set up
    global configuration values that can be reused throughout
    the application lifecycle.
    """
    global _global_config, _initialization_lock
    global BASE_URL, API_TIMEOUT, MAX_REQUESTS_PER_SECOND
    global MAX_RETRIES, RETRY_BACKOFF_FACTOR, ENABLE_CACHING, CACHE_TTL

    if _initialization_lock:
        return

    _initialization_lock = True

    try:
        # Create a minimal ConfigManager instance for module-level use
        _global_config = ConfigManager.create('minimal', validate_on_init=False)

        # Cache configuration values at module level for fast access
        BASE_URL = _global_config.base_url
        API_TIMEOUT = _global_config.api_timeout
        MAX_REQUESTS_PER_SECOND = _global_config.max_requests_per_second
        MAX_RETRIES = _global_config.max_retries
        RETRY_BACKOFF_FACTOR = _global_config.retry_backoff_factor
        ENABLE_CACHING = _global_config.enable_caching
        CACHE_TTL = _global_config.cache_ttl

        logger.info(f"Module-level configuration initialized with base URL: {BASE_URL}")

    except Exception as e:
        logger.warning(f"Failed to initialize module-level config: {e}")
        # Set fallback values
        BASE_URL = os.environ.get('AGR_BASE_URL', 'https://www.alliancegenome.org')
        API_TIMEOUT = int(os.environ.get('AGR_API_TIMEOUT', '30'))
        MAX_REQUESTS_PER_SECOND = int(os.environ.get('AGR_MAX_RPS', '10'))
        MAX_RETRIES = int(os.environ.get('AGR_MAX_RETRIES', '3'))
        RETRY_BACKOFF_FACTOR = float(os.environ.get('AGR_RETRY_BACKOFF', '0.5'))
        ENABLE_CACHING = os.environ.get('AGR_ENABLE_CACHING', 'False').lower() == 'true'
        CACHE_TTL = int(os.environ.get('AGR_CACHE_TTL', '3600'))


def get_global_config() -> Optional[ConfigManager]:
    """Get the global ConfigManager instance.

    Returns:
        The global ConfigManager instance, or None if initialization failed
    """
    return _global_config


def get_base_url() -> str:
    """Get the pre-configured base URL.

    Returns:
        The AGR API base URL string
    """
    return BASE_URL


def get_api_endpoint(endpoint_name: str, **kwargs) -> str:
    """Get a pre-configured API endpoint URL.

    Args:
        endpoint_name: Name of the endpoint (e.g., 'search', 'gene_details')
        **kwargs: Format parameters for the endpoint (e.g., gene_id)

    Returns:
        Full URL for the API endpoint

    Raises:
        KeyError: If endpoint_name is not recognized
        ValueError: If required format parameters are missing
    """
    if endpoint_name not in API_ENDPOINTS:
        raise KeyError(f"Unknown endpoint: {endpoint_name}")

    endpoint_path = API_ENDPOINTS[endpoint_name]

    try:
        # Format the endpoint path with provided parameters
        formatted_path = endpoint_path.format(**kwargs)
        return BASE_URL + formatted_path
    except KeyError as e:
        raise ValueError(f"Missing required parameter for endpoint {endpoint_name}: {e}")


def get_config_values() -> Dict[str, Any]:
    """Get all cached configuration values as a dictionary.

    Returns:
        Dictionary containing all module-level configuration values
    """
    return {
        'base_url': BASE_URL,
        'api_timeout': API_TIMEOUT,
        'max_requests_per_second': MAX_REQUESTS_PER_SECOND,
        'max_retries': MAX_RETRIES,
        'retry_backoff_factor': RETRY_BACKOFF_FACTOR,
        'enable_caching': ENABLE_CACHING,
        'cache_ttl': CACHE_TTL
    }


def reload_config() -> None:
    """Reload the module-level configuration.

    This can be used to refresh configuration if environment
    variables or configuration files have changed.
    """
    global _initialization_lock
    _initialization_lock = False
    _initialize_module_config()


# Initialize configuration when module is imported
_initialize_module_config()
