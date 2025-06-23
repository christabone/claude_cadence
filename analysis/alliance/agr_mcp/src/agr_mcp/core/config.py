"""
Configuration management for the AGR MCP server.

This module handles loading and managing configuration from environment
variables, configuration files, and command-line arguments.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "alliance_db"
    user: str = ""
    password: str = ""

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables."""
        return cls(
            host=os.getenv("AGR_DB_HOST", "localhost"),
            port=int(os.getenv("AGR_DB_PORT", "5432")),
            name=os.getenv("AGR_DB_NAME", "alliance_db"),
            user=os.getenv("AGR_DB_USER", ""),
            password=os.getenv("AGR_DB_PASSWORD", "")
        )


@dataclass
class S3Config:
    """AWS S3 configuration."""
    bucket: str = ""
    prefix: str = "data/"
    region: str = "us-east-1"
    profile: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'S3Config':
        """Create S3 config from environment variables."""
        return cls(
            bucket=os.getenv("AGR_S3_BUCKET", ""),
            prefix=os.getenv("AGR_S3_PREFIX", "data/"),
            region=os.getenv("AWS_REGION", "us-east-1"),
            profile=os.getenv("AWS_PROFILE")
        )


@dataclass
class APIConfig:
    """Alliance API configuration."""
    base_url: str = "https://www.alliancegenome.org/api"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create API config from environment variables."""
        return cls(
            base_url=os.getenv("AGR_API_BASE_URL", "https://www.alliancegenome.org/api"),
            api_key=os.getenv("AGR_API_KEY"),
            timeout=int(os.getenv("AGR_API_TIMEOUT", "30")),
            max_retries=int(os.getenv("AGR_API_MAX_RETRIES", "3"))
        )


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    directory: Path = Path.home() / ".cache" / "agr-mcp"
    ttl: int = 3600  # 1 hour default
    max_size: int = 1000  # Maximum number of cached items

    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create cache config from environment variables."""
        cache_dir = os.getenv("AGR_CACHE_DIR", str(Path.home() / ".cache" / "agr-mcp"))
        return cls(
            enabled=os.getenv("AGR_CACHE_ENABLED", "true").lower() == "true",
            directory=Path(cache_dir),
            ttl=int(os.getenv("AGR_CACHE_TTL", "3600")),
            max_size=int(os.getenv("AGR_CACHE_MAX_SIZE", "1000"))
        )


@dataclass
class Config:
    """Main configuration class for the AGR MCP server."""
    # Server settings
    port: int = 3000
    host: str = "localhost"
    log_level: str = "INFO"

    # Service configurations
    database: DatabaseConfig = None
    s3: S3Config = None
    api: APIConfig = None
    cache: CacheConfig = None

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.database is None:
            self.database = DatabaseConfig.from_env()
        if self.s3 is None:
            self.s3 = S3Config.from_env()
        if self.api is None:
            self.api = APIConfig.from_env()
        if self.cache is None:
            self.cache = CacheConfig.from_env()

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            port=int(os.getenv("AGR_MCP_PORT", "3000")),
            host=os.getenv("AGR_MCP_HOST", "localhost"),
            log_level=os.getenv("AGR_MCP_LOG_LEVEL", "INFO"),
            database=DatabaseConfig.from_env(),
            s3=S3Config.from_env(),
            api=APIConfig.from_env(),
            cache=CacheConfig.from_env()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "port": self.port,
            "host": self.host,
            "log_level": self.log_level,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "user": self.database.user,
                "password": "***" if self.database.password else ""
            },
            "s3": {
                "bucket": self.s3.bucket,
                "prefix": self.s3.prefix,
                "region": self.s3.region,
                "profile": self.s3.profile
            },
            "api": {
                "base_url": self.api.base_url,
                "api_key": "***" if self.api.api_key else None,
                "timeout": self.api.timeout,
                "max_retries": self.api.max_retries
            },
            "cache": {
                "enabled": self.cache.enabled,
                "directory": str(self.cache.directory),
                "ttl": self.cache.ttl,
                "max_size": self.cache.max_size
            }
        }


def load_config(config_file: Optional[Path] = None) -> Config:
    """
    Load configuration from environment and optional config file.

    Args:
        config_file: Optional path to configuration file

    Returns:
        Loaded configuration object
    """
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    # Load from environment
    config = Config.from_env()

    # TODO: Load from config file if provided
    if config_file and config_file.exists():
        # Implementation for loading from JSON/YAML config file
        pass

    return config
