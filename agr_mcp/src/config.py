"""Alliance Genome Resource MCP Server Configuration Module.

This module provides centralized configuration management for the AGR MCP server,
including environment variable loading, validation, and file-based configuration support.

Configuration Precedence:
------------------------
1. Environment variables (highest priority)
2. Configuration files (medium priority)
3. Default values (lowest priority)

Usage Examples:
--------------

Basic usage with defaults:
    ```python
    from agr_mcp.src.config import Config

    # Access configuration values
    print(f"API URL: {Config.BASE_URL}")
    print(f"Timeout: {Config.API_TIMEOUT} seconds")

    # Validate configuration
    try:
        warnings = Config.validate()
        if warnings:
            print("Configuration warnings:", warnings)
    except ValidationError as e:
        print(f"Configuration error: {e}")
    ```

Loading from configuration file:
    ```python
    # Load from JSON file
    Config.from_file('config.json')

    # Load from YAML file with explicit format
    Config.from_file('settings.yaml', format='yaml')

    # Load from INI file without validation
    Config.from_file('app.ini', validate_after_load=False)
    ```

Environment variable override:
    ```bash
    # Set environment variables (bash)
    export AGR_BASE_URL="https://api.alliancegenome.org"
    export AGR_API_TIMEOUT=60
    export AGR_LOG_LEVEL=DEBUG
    export AGR_ENABLE_CACHING=true
    ```

Getting configuration as dictionary:
    ```python
    config_dict = Config.as_dict()
    print(json.dumps(config_dict, indent=2))
    ```

Ensuring directories exist:
    ```python
    # Create download and log directories if they don't exist
    Config.ensure_directories()
    ```

Custom configuration extension:
    ```python
    class MyConfig(Config):
        # Add custom configuration values
        CUSTOM_FEATURE_ENABLED = os.environ.get('AGR_CUSTOM_FEATURE', 'False').lower() == 'true'
        CUSTOM_THRESHOLD = int(os.environ.get('AGR_CUSTOM_THRESHOLD', '100'))
    ```
"""

from typing import Dict, Any, List, Optional
import os
import re
import json
import configparser
from pathlib import Path
from urllib.parse import urlparse

from .errors import ValidationError, ResourceNotFoundError


class Config:
    """Configuration class for AGR MCP Server.

    All configuration values can be overridden using environment variables
    with the AGR_ prefix. Environment variables take precedence over defaults.

    Attributes:
        BASE_URL (str): Base URL for AGR API endpoints
        API_TIMEOUT (int): Timeout for API requests in seconds
        MAX_REQUESTS_PER_SECOND (int): Rate limit for API requests
        MAX_RETRIES (int): Maximum number of retry attempts for failed requests
        RETRY_BACKOFF_FACTOR (float): Exponential backoff multiplier between retries
        DEFAULT_DOWNLOAD_DIR (str): Directory for downloaded files
        MAX_DOWNLOAD_SIZE (int): Maximum allowed file download size in bytes
        LOG_LEVEL (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_DIR (str): Directory for log files
        LOG_FILE_MAX_SIZE (int): Maximum size of a single log file in bytes
        LOG_FILE_BACKUP_COUNT (int): Number of rotated log files to keep
        ENABLE_CACHING (bool): Whether to enable response caching
        CACHE_TTL (int): Cache time-to-live in seconds
    """

    # API Configuration
    BASE_URL: str = os.environ.get('AGR_BASE_URL', 'https://www.alliancegenome.org')
    """Base URL for Alliance Genome Resource API.

    Environment variable: AGR_BASE_URL
    Default: https://www.alliancegenome.org
    Constraints: Must be a valid HTTP/HTTPS URL with scheme and host
    """

    API_TIMEOUT: int = int(os.environ.get('AGR_API_TIMEOUT', '30'))
    """Timeout for API requests in seconds.

    Environment variable: AGR_API_TIMEOUT
    Default: 30 seconds
    Constraints: Must be positive, warning if > 300 seconds
    """

    # Rate Limiting
    MAX_REQUESTS_PER_SECOND: int = int(os.environ.get('AGR_MAX_RPS', '10'))
    """Maximum number of API requests allowed per second.

    Environment variable: AGR_MAX_RPS
    Default: 10 requests/second
    Constraints: Must be positive, warning if > 100
    """

    # Retry Configuration
    MAX_RETRIES: int = int(os.environ.get('AGR_MAX_RETRIES', '3'))
    """Maximum number of retry attempts for failed requests.

    Environment variable: AGR_MAX_RETRIES
    Default: 3 attempts
    Constraints: Cannot be negative, warning if > 10
    """

    RETRY_BACKOFF_FACTOR: float = float(os.environ.get('AGR_RETRY_BACKOFF', '0.5'))
    """Exponential backoff multiplier between retry attempts.

    Environment variable: AGR_RETRY_BACKOFF
    Default: 0.5
    Constraints: Cannot be negative, warning if > 10
    Example: With factor 0.5, delays are 0.5s, 1s, 2s, 4s...
    """

    # File Download Configuration
    DEFAULT_DOWNLOAD_DIR: str = os.environ.get('AGR_DOWNLOAD_DIR', os.path.join(os.getcwd(), 'downloads'))
    """Default directory for downloaded files.

    Environment variable: AGR_DOWNLOAD_DIR
    Default: <current_directory>/downloads
    Note: Directory will be created if it doesn't exist
    """

    MAX_DOWNLOAD_SIZE: int = int(os.environ.get('AGR_MAX_DOWNLOAD_SIZE', str(1024 * 1024 * 1024)))
    """Maximum allowed file download size in bytes.

    Environment variable: AGR_MAX_DOWNLOAD_SIZE
    Default: 1GB (1073741824 bytes)
    Constraints: Must be positive, warning if > 10GB
    """

    # Logging Configuration
    LOG_LEVEL: str = os.environ.get('AGR_LOG_LEVEL', 'INFO')
    """Logging level for the application.

    Environment variable: AGR_LOG_LEVEL
    Default: INFO
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """

    LOG_DIR: str = os.environ.get('AGR_LOG_DIR', os.path.join(os.getcwd(), 'logs'))
    """Directory for log files.

    Environment variable: AGR_LOG_DIR
    Default: <current_directory>/logs
    Note: Directory will be created if it doesn't exist
    """

    LOG_FILE_MAX_SIZE: int = int(os.environ.get('AGR_LOG_FILE_MAX_SIZE', str(100 * 1024 * 1024)))
    """Maximum size of a single log file in bytes before rotation.

    Environment variable: AGR_LOG_FILE_MAX_SIZE
    Default: 100MB (104857600 bytes)
    Constraints: Must be positive
    """

    LOG_FILE_BACKUP_COUNT: int = int(os.environ.get('AGR_LOG_FILE_BACKUP_COUNT', '5'))
    """Number of rotated log files to keep.

    Environment variable: AGR_LOG_FILE_BACKUP_COUNT
    Default: 5 files
    Constraints: Cannot be negative
    """

    # Caching Configuration
    ENABLE_CACHING: bool = os.environ.get('AGR_ENABLE_CACHING', 'False').lower() == 'true'
    """Whether to enable response caching.

    Environment variable: AGR_ENABLE_CACHING
    Default: False
    Valid values: true, false (case-insensitive)
    """

    CACHE_TTL: int = int(os.environ.get('AGR_CACHE_TTL', '3600'))
    """Cache time-to-live in seconds.

    Environment variable: AGR_CACHE_TTL
    Default: 3600 seconds (1 hour)
    Constraints: Must be positive when caching is enabled
    """

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return configuration as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration values
        """
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('__') and not callable(getattr(cls, k))}

    @classmethod
    def validate(cls) -> List[str]:
        """Validate all configuration values.

        Returns:
            List[str]: List of validation warnings (non-critical issues)

        Raises:
            ValidationError: If any critical validation errors are found
        """
        errors = []
        warnings = []

        # Validate BASE_URL
        try:
            parsed_url = urlparse(cls.BASE_URL)
            if not parsed_url.scheme:
                errors.append("BASE_URL must include a scheme (http:// or https://)")
            elif parsed_url.scheme not in ['http', 'https']:
                errors.append(f"BASE_URL scheme must be http or https, got: {parsed_url.scheme}")
            if not parsed_url.netloc:
                errors.append("BASE_URL must include a valid host")
        except Exception as e:
            errors.append(f"BASE_URL is not a valid URL: {str(e)}")

        # Validate numeric values
        if cls.API_TIMEOUT <= 0:
            errors.append("API_TIMEOUT must be positive")
        elif cls.API_TIMEOUT > 300:
            warnings.append(f"API_TIMEOUT of {cls.API_TIMEOUT}s is very high, consider reducing")

        if cls.MAX_REQUESTS_PER_SECOND <= 0:
            errors.append("MAX_REQUESTS_PER_SECOND must be positive")
        elif cls.MAX_REQUESTS_PER_SECOND > 100:
            warnings.append(f"MAX_REQUESTS_PER_SECOND of {cls.MAX_REQUESTS_PER_SECOND} is very high")

        if cls.MAX_RETRIES < 0:
            errors.append("MAX_RETRIES cannot be negative")
        elif cls.MAX_RETRIES > 10:
            warnings.append(f"MAX_RETRIES of {cls.MAX_RETRIES} is very high")

        if cls.RETRY_BACKOFF_FACTOR < 0:
            errors.append("RETRY_BACKOFF_FACTOR cannot be negative")
        elif cls.RETRY_BACKOFF_FACTOR > 10:
            warnings.append(f"RETRY_BACKOFF_FACTOR of {cls.RETRY_BACKOFF_FACTOR} is very high")

        # Validate file download settings
        if cls.MAX_DOWNLOAD_SIZE <= 0:
            errors.append("MAX_DOWNLOAD_SIZE must be positive")
        elif cls.MAX_DOWNLOAD_SIZE > 10 * 1024 * 1024 * 1024:  # 10GB
            warnings.append(f"MAX_DOWNLOAD_SIZE of {cls.MAX_DOWNLOAD_SIZE} bytes is very large")

        # Validate logging settings
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of {valid_log_levels}, got: {cls.LOG_LEVEL}")

        if cls.LOG_FILE_MAX_SIZE <= 0:
            errors.append("LOG_FILE_MAX_SIZE must be positive")

        if cls.LOG_FILE_BACKUP_COUNT < 0:
            errors.append("LOG_FILE_BACKUP_COUNT cannot be negative")

        # Validate cache settings
        if cls.ENABLE_CACHING and cls.CACHE_TTL <= 0:
            errors.append("CACHE_TTL must be positive when caching is enabled")

        # Raise exception if critical errors found
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValidationError(
                error_msg,
                field="config",
                constraints={"errors": errors}
            )

        return warnings

    @classmethod
    def validate_directory(cls, path: str, name: str, create_if_missing: bool = True) -> None:
        """Validate and optionally create a directory.

        Args:
            path: Directory path to validate
            name: Name of the directory (for error messages)
            create_if_missing: Whether to create the directory if it doesn't exist

        Raises:
            ValidationError: If directory validation fails
        """
        if not path:
            raise ValidationError(f"{name} cannot be empty", field=name)

        abs_path = os.path.abspath(path)

        if os.path.exists(abs_path):
            if not os.path.isdir(abs_path):
                raise ValidationError(
                    f"{name} exists but is not a directory: {abs_path}",
                    field=name,
                    value=abs_path
                )
            if not os.access(abs_path, os.W_OK):
                raise ValidationError(
                    f"{name} is not writable: {abs_path}",
                    field=name,
                    value=abs_path
                )
        elif create_if_missing:
            try:
                os.makedirs(abs_path, exist_ok=True)
            except Exception as e:
                raise ValidationError(
                    f"Failed to create {name}: {str(e)}",
                    field=name,
                    value=abs_path
                )

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist and are writable.

        Raises:
            ValidationError: If any directory cannot be created or accessed
        """
        cls.validate_directory(cls.DEFAULT_DOWNLOAD_DIR, "DEFAULT_DOWNLOAD_DIR")
        cls.validate_directory(cls.LOG_DIR, "LOG_DIR")

    @classmethod
    def from_file(cls,
                  file_path: str,
                  format: Optional[str] = None,
                  validate_after_load: bool = True) -> None:
        """Load configuration from a file.

        File values are loaded first, then overridden by any environment variables.
        This ensures environment variables always take precedence.

        Args:
            file_path: Path to the configuration file
            format: File format ('json', 'yaml', 'ini'). If None, inferred from extension
            validate_after_load: Whether to validate configuration after loading

        Raises:
            ResourceNotFoundError: If the file doesn't exist
            ValidationError: If the file format is invalid or validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ResourceNotFoundError(
                f"Configuration file not found: {file_path}",
                resource_type="file",
                resource_id=str(file_path)
            )

        # Infer format from extension if not provided
        if format is None:
            extension = file_path.suffix.lower()
            format_map = {
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.ini': 'ini',
                '.cfg': 'ini',
                '.conf': 'ini'
            }
            format = format_map.get(extension)
            if not format:
                raise ValidationError(
                    f"Cannot infer format from extension: {extension}",
                    field="format",
                    value=extension,
                    constraints={"valid_extensions": list(format_map.keys())}
                )

        # Load the configuration based on format
        config_data = {}
        try:
            if format == 'json':
                config_data = cls._load_json(file_path)
            elif format == 'yaml':
                config_data = cls._load_yaml(file_path)
            elif format == 'ini':
                config_data = cls._load_ini(file_path)
            else:
                raise ValidationError(
                    f"Unsupported format: {format}",
                    field="format",
                    value=format,
                    constraints={"valid_formats": ['json', 'yaml', 'ini']}
                )
        except Exception as e:
            if isinstance(e, (ValidationError, ResourceNotFoundError)):
                raise
            raise ValidationError(
                f"Failed to parse {format} file: {str(e)}",
                field="file_path",
                value=str(file_path)
            )

        # Apply loaded configuration (environment variables still take precedence)
        cls._apply_config(config_data)

        # Validate if requested
        if validate_after_load:
            warnings = cls.validate()
            if warnings:
                print(f"Configuration warnings:\n" + "\n".join(f"  - {w}" for w in warnings))

    @classmethod
    def _load_json(cls, file_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @classmethod
    def _load_yaml(cls, file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ValidationError(
                "PyYAML is required for YAML configuration files. Install with: pip install pyyaml",
                field="format",
                value="yaml"
            )

        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    @classmethod
    def _load_ini(cls, file_path: Path) -> Dict[str, Any]:
        """Load configuration from INI file."""
        parser = configparser.ConfigParser()
        parser.read(file_path)

        # Flatten INI sections into a single dict
        config_data = {}
        for section in parser.sections():
            for key, value in parser.items(section):
                # Use section.key format for namespacing
                full_key = f"{section}.{key}" if section != 'DEFAULT' else key
                config_data[full_key] = value

        # Also include DEFAULT section
        for key, value in parser.defaults().items():
            if key not in config_data:
                config_data[key] = value

        return config_data

    @classmethod
    def _apply_config(cls, config_data: Dict[str, Any]) -> None:
        """Apply configuration data to class attributes.

        Only updates attributes if they haven't been set by environment variables.

        Args:
            config_data: Dictionary of configuration values
        """
        # Map of config keys to class attributes
        key_mapping = {
            'base_url': 'BASE_URL',
            'api_timeout': 'API_TIMEOUT',
            'max_requests_per_second': 'MAX_REQUESTS_PER_SECOND',
            'max_rps': 'MAX_REQUESTS_PER_SECOND',  # alias
            'max_retries': 'MAX_RETRIES',
            'retry_backoff_factor': 'RETRY_BACKOFF_FACTOR',
            'retry_backoff': 'RETRY_BACKOFF_FACTOR',  # alias
            'default_download_dir': 'DEFAULT_DOWNLOAD_DIR',
            'download_dir': 'DEFAULT_DOWNLOAD_DIR',  # alias
            'max_download_size': 'MAX_DOWNLOAD_SIZE',
            'log_level': 'LOG_LEVEL',
            'log_dir': 'LOG_DIR',
            'log_file_max_size': 'LOG_FILE_MAX_SIZE',
            'log_file_backup_count': 'LOG_FILE_BACKUP_COUNT',
            'enable_caching': 'ENABLE_CACHING',
            'cache_ttl': 'CACHE_TTL',
        }

        for key, value in config_data.items():
            # Normalize key: remove section prefix, convert to lowercase
            normalized_key = key.lower()
            if '.' in normalized_key:
                # Handle INI-style section.key format
                section, key_part = normalized_key.rsplit('.', 1)
                if section in ['agr', 'config', 'settings']:
                    normalized_key = key_part

            # Check if this is a known configuration key
            if normalized_key in key_mapping:
                attr_name = key_mapping[normalized_key]
                env_var_name = f'AGR_{attr_name}'

                # Only apply if not already set by environment variable
                if env_var_name not in os.environ:
                    # Type conversion based on current attribute type
                    current_value = getattr(cls, attr_name)
                    try:
                        if isinstance(current_value, bool):
                            # Handle boolean conversion
                            if isinstance(value, bool):
                                converted_value = value
                            else:
                                converted_value = str(value).lower() in ('true', '1', 'yes', 'on')
                        elif isinstance(current_value, int):
                            converted_value = int(value)
                        elif isinstance(current_value, float):
                            converted_value = float(value)
                        else:
                            converted_value = str(value)

                        setattr(cls, attr_name, converted_value)
                    except (ValueError, TypeError) as e:
                        raise ValidationError(
                            f"Invalid value for {attr_name}: {value}",
                            field=attr_name,
                            value=value
                        )
