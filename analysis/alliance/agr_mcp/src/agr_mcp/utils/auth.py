"""
Authentication utilities for the AGR MCP server.

This module provides credential management and authentication helpers
for various services including databases and AWS.
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Credentials:
    """Container for service credentials."""
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    token: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if credentials contain any authentication info."""
        return any([self.username, self.api_key, self.token])


class AuthManager:
    """
    Manages authentication credentials for various services.

    Handles loading credentials from environment variables, files,
    and secure credential stores.
    """

    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize authentication manager.

        Args:
            env_file: Optional path to .env file
        """
        self._credentials: Dict[str, Credentials] = {}

        # Load environment variables
        if env_file and env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from: {env_file}")
        else:
            # Try to load from default locations
            for env_path in [Path(".env"), Path.home() / ".agr-mcp" / ".env"]:
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from: {env_path}")
                    break

    def get_database_credentials(self) -> Credentials:
        """
        Get database credentials from environment.

        Returns:
            Credentials object with database auth info
        """
        if "database" not in self._credentials:
            self._credentials["database"] = Credentials(
                username=os.getenv("AGR_DB_USER"),
                password=os.getenv("AGR_DB_PASSWORD")
            )

        return self._credentials["database"]

    def get_api_credentials(self) -> Credentials:
        """
        Get API credentials from environment.

        Returns:
            Credentials object with API auth info
        """
        if "api" not in self._credentials:
            self._credentials["api"] = Credentials(
                api_key=os.getenv("AGR_API_KEY"),
                token=os.getenv("AGR_API_TOKEN")
            )

        return self._credentials["api"]

    def get_aws_credentials(self) -> Dict[str, str]:
        """
        Get AWS credentials from environment.

        Returns:
            Dictionary with AWS credential info
        """
        return {
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
            "aws_profile": os.getenv("AWS_PROFILE"),
            "aws_region": os.getenv("AWS_REGION", "us-east-1")
        }

    def validate_credentials(self, service: str) -> bool:
        """
        Validate that required credentials are available.

        Args:
            service: Service name (database, api, aws)

        Returns:
            True if valid credentials exist
        """
        if service == "database":
            creds = self.get_database_credentials()
            return bool(creds.username and creds.password)

        elif service == "api":
            creds = self.get_api_credentials()
            return bool(creds.api_key or creds.token)

        elif service == "aws":
            aws_creds = self.get_aws_credentials()
            # Either profile or key/secret required
            has_profile = bool(aws_creds.get("aws_profile"))
            has_keys = bool(
                aws_creds.get("aws_access_key_id") and
                aws_creds.get("aws_secret_access_key")
            )
            return has_profile or has_keys

        else:
            logger.warning(f"Unknown service for credential validation: {service}")
            return False

    def mask_credential(self, value: str, visible_chars: int = 4) -> str:
        """
        Mask sensitive credential for logging.

        Args:
            value: Credential value to mask
            visible_chars: Number of characters to leave visible

        Returns:
            Masked credential string
        """
        if not value or len(value) <= visible_chars:
            return "***"

        return value[:visible_chars] + "***"

    def get_credential_summary(self) -> Dict[str, Any]:
        """
        Get summary of available credentials for debugging.

        Returns:
            Dictionary with credential availability info
        """
        db_creds = self.get_database_credentials()
        api_creds = self.get_api_credentials()
        aws_creds = self.get_aws_credentials()

        return {
            "database": {
                "username": self.mask_credential(db_creds.username) if db_creds.username else None,
                "password": "***" if db_creds.password else None,
                "valid": self.validate_credentials("database")
            },
            "api": {
                "api_key": self.mask_credential(api_creds.api_key) if api_creds.api_key else None,
                "token": self.mask_credential(api_creds.token) if api_creds.token else None,
                "valid": self.validate_credentials("api")
            },
            "aws": {
                "profile": aws_creds.get("aws_profile"),
                "access_key": self.mask_credential(aws_creds.get("aws_access_key_id", "")) if aws_creds.get("aws_access_key_id") else None,
                "region": aws_creds.get("aws_region"),
                "valid": self.validate_credentials("aws")
            }
        }
