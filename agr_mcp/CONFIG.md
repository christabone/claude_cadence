# AGR MCP Server Configuration Guide

The AGR MCP Server provides flexible configuration management through environment variables, configuration files, and sensible defaults.

## Table of Contents

- [Configuration Precedence](#configuration-precedence)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Configuration Options](#configuration-options)
- [Validation](#validation)
- [Usage Examples](#usage-examples)
- [Extending Configuration](#extending-configuration)

## Configuration Precedence

Configuration values are resolved in the following order (highest to lowest priority):

1. **Environment Variables** - Always take precedence
2. **Configuration Files** - Loaded via `Config.from_file()`
3. **Default Values** - Built-in defaults

## Environment Variables

All configuration options can be set via environment variables with the `AGR_` prefix.

### Setting Environment Variables

**Linux/Mac (bash/zsh):**
```bash
export AGR_BASE_URL="https://api.alliancegenome.org"
export AGR_API_TIMEOUT=60
export AGR_LOG_LEVEL=DEBUG
```

**Windows (PowerShell):**
```powershell
$env:AGR_BASE_URL = "https://api.alliancegenome.org"
$env:AGR_API_TIMEOUT = 60
$env:AGR_LOG_LEVEL = "DEBUG"
```

**Windows (Command Prompt):**
```cmd
set AGR_BASE_URL=https://api.alliancegenome.org
set AGR_API_TIMEOUT=60
set AGR_LOG_LEVEL=DEBUG
```

### Using .env Files

For development, you can use a `.env` file with python-dotenv:

```bash
# .env file
AGR_BASE_URL=https://api.alliancegenome.org
AGR_API_TIMEOUT=60
AGR_LOG_LEVEL=DEBUG
AGR_ENABLE_CACHING=true
```

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file

from agr_mcp.src.config import Config
```

## Configuration Files

The server supports loading configuration from JSON, YAML, and INI files.

### JSON Configuration

```json
{
  "base_url": "https://api.alliancegenome.org",
  "api_timeout": 60,
  "max_requests_per_second": 20,
  "max_retries": 5,
  "retry_backoff_factor": 1.0,
  "default_download_dir": "/data/agr/downloads",
  "max_download_size": 5368709120,
  "log_level": "INFO",
  "log_dir": "/var/log/agr-mcp",
  "log_file_max_size": 52428800,
  "log_file_backup_count": 10,
  "enable_caching": true,
  "cache_ttl": 7200
}
```

### YAML Configuration

```yaml
# config.yaml
base_url: https://api.alliancegenome.org
api_timeout: 60
max_requests_per_second: 20
max_retries: 5
retry_backoff_factor: 1.0

# File handling
default_download_dir: /data/agr/downloads
max_download_size: 5368709120  # 5GB

# Logging
log_level: INFO
log_dir: /var/log/agr-mcp
log_file_max_size: 52428800  # 50MB
log_file_backup_count: 10

# Caching
enable_caching: true
cache_ttl: 7200  # 2 hours
```

### INI Configuration

```ini
# config.ini
[agr]
base_url = https://api.alliancegenome.org
api_timeout = 60
max_requests_per_second = 20

[retry]
max_retries = 5
retry_backoff_factor = 1.0

[files]
default_download_dir = /data/agr/downloads
max_download_size = 5368709120

[logging]
log_level = INFO
log_dir = /var/log/agr-mcp
log_file_max_size = 52428800
log_file_backup_count = 10

[cache]
enable_caching = true
cache_ttl = 7200
```

### Loading Configuration Files

```python
from agr_mcp.src.config import Config

# Auto-detect format from extension
Config.from_file('config.json')
Config.from_file('settings.yaml')
Config.from_file('app.ini')

# Explicit format specification
Config.from_file('custom.conf', format='ini')

# Skip validation during load
Config.from_file('config.json', validate_after_load=False)
```

## Configuration Options

### API Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `BASE_URL` | `AGR_BASE_URL` | `https://www.alliancegenome.org` | Base URL for AGR API endpoints |
| `API_TIMEOUT` | `AGR_API_TIMEOUT` | `30` | Request timeout in seconds |

### Rate Limiting

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `MAX_REQUESTS_PER_SECOND` | `AGR_MAX_RPS` | `10` | Maximum API requests per second |

### Retry Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `MAX_RETRIES` | `AGR_MAX_RETRIES` | `3` | Maximum retry attempts |
| `RETRY_BACKOFF_FACTOR` | `AGR_RETRY_BACKOFF` | `0.5` | Exponential backoff multiplier |

### File Download

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `DEFAULT_DOWNLOAD_DIR` | `AGR_DOWNLOAD_DIR` | `./downloads` | Download directory path |
| `MAX_DOWNLOAD_SIZE` | `AGR_MAX_DOWNLOAD_SIZE` | `1073741824` | Max download size in bytes (1GB) |

### Logging

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `LOG_LEVEL` | `AGR_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_DIR` | `AGR_LOG_DIR` | `./logs` | Log file directory |
| `LOG_FILE_MAX_SIZE` | `AGR_LOG_FILE_MAX_SIZE` | `104857600` | Max log file size in bytes (100MB) |
| `LOG_FILE_BACKUP_COUNT` | `AGR_LOG_FILE_BACKUP_COUNT` | `5` | Number of rotated log files to keep |

### Caching

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `ENABLE_CACHING` | `AGR_ENABLE_CACHING` | `false` | Enable response caching |
| `CACHE_TTL` | `AGR_CACHE_TTL` | `3600` | Cache time-to-live in seconds |

## Validation

The configuration module includes comprehensive validation to ensure settings are within acceptable ranges.

### Running Validation

```python
from agr_mcp.src.config import Config
from agr_mcp.src.errors import ValidationError

try:
    warnings = Config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
except ValidationError as e:
    print(f"Configuration error: {e}")
    # Handle invalid configuration
```

### Validation Rules

- **URLs**: Must include scheme (http/https) and valid host
- **Timeouts**: Must be positive, warning if > 300 seconds
- **Rate Limits**: Must be positive, warning if > 100 requests/second
- **Retries**: Cannot be negative, warning if > 10
- **File Sizes**: Must be positive, warning if > 10GB
- **Log Levels**: Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Directories**: Created automatically if they don't exist

### Ensuring Directories

```python
# Create required directories if they don't exist
Config.ensure_directories()
```

## Usage Examples

### Basic Usage

```python
from agr_mcp.src.config import Config

# Access configuration values
api_url = Config.BASE_URL
timeout = Config.API_TIMEOUT

# Print all configuration
config_dict = Config.as_dict()
for key, value in config_dict.items():
    print(f"{key}: {value}")
```

### Production Setup

```python
import os
from agr_mcp.src.config import Config

# Load base configuration from file
Config.from_file('/etc/agr-mcp/config.json')

# Ensure directories exist
Config.ensure_directories()

# Validate configuration
warnings = Config.validate()
if warnings:
    logger.warning("Configuration warnings: %s", warnings)
```

### Development Setup

```python
from agr_mcp.src.config import Config

# Load development configuration
Config.from_file('config/development.yaml')

# Override with local settings
if os.path.exists('config/local.yaml'):
    Config.from_file('config/local.yaml')

# Enable debug logging for development
Config.LOG_LEVEL = 'DEBUG'
```

## Extending Configuration

You can extend the Config class for custom settings:

```python
import os
from agr_mcp.src.config import Config

class CustomConfig(Config):
    """Extended configuration with custom settings."""

    # Custom API endpoints
    GENE_API_ENDPOINT = os.environ.get('AGR_GENE_API_ENDPOINT', '/api/gene')
    ALLELE_API_ENDPOINT = os.environ.get('AGR_ALLELE_API_ENDPOINT', '/api/allele')

    # Custom features
    ENABLE_METRICS = os.environ.get('AGR_ENABLE_METRICS', 'False').lower() == 'true'
    METRICS_PORT = int(os.environ.get('AGR_METRICS_PORT', '9090'))

    # Custom validation
    @classmethod
    def validate(cls):
        warnings = super().validate()

        # Add custom validation
        if cls.ENABLE_METRICS and cls.METRICS_PORT <= 0:
            raise ValidationError("METRICS_PORT must be positive when metrics are enabled")

        return warnings
```

### Using Custom Configuration

```python
# Use the extended configuration
from myapp.config import CustomConfig

if CustomConfig.ENABLE_METRICS:
    start_metrics_server(CustomConfig.METRICS_PORT)
```

## Best Practices

1. **Use Environment Variables in Production**: Keep sensitive data out of config files
2. **Validate Early**: Call `Config.validate()` during application startup
3. **Create Directories**: Use `Config.ensure_directories()` during initialization
4. **Document Custom Settings**: Add docstrings for any custom configuration
5. **Use Type Hints**: Maintain type annotations for better IDE support
6. **Handle Validation Errors**: Gracefully handle configuration errors at startup

## Troubleshooting

### Common Issues

**Issue**: Configuration file not found
```python
try:
    Config.from_file('config.json')
except ResourceNotFoundError as e:
    print(f"Config file not found: {e}")
    # Use defaults or exit
```

**Issue**: Invalid configuration values
```python
try:
    Config.validate()
except ValidationError as e:
    print(f"Invalid configuration: {e}")
    # Log detailed errors
    for error in e.details.get('errors', []):
        print(f"  - {error}")
```

**Issue**: Missing required directories
```python
try:
    Config.ensure_directories()
except ValidationError as e:
    print(f"Cannot create directory: {e}")
    # Check permissions
```

### Debug Configuration Loading

```python
import json
from agr_mcp.src.config import Config

# Load and validate
Config.from_file('config.json')

# Print loaded configuration
print("Loaded configuration:")
print(json.dumps(Config.as_dict(), indent=2))

# Check specific values
print(f"\nAPI URL: {Config.BASE_URL}")
print(f"Timeout: {Config.API_TIMEOUT}")
print(f"Caching: {'Enabled' if Config.ENABLE_CACHING else 'Disabled'}")
```
