"""Alliance Genome Resource MCP Server Error Classes.

This module provides a comprehensive error handling system for the AGR MCP server,
including base exceptions, specific error types, and utilities for error serialization
and response formatting.

Error Handling Philosophy:
-------------------------
1. **Fail Fast**: Detect and report errors as early as possible
2. **Be Specific**: Use the most specific error type available
3. **Provide Context**: Include relevant details to help diagnose issues
4. **Secure by Default**: Sanitize sensitive information in error messages
5. **User-Friendly**: Provide clear, actionable error messages

Usage Examples:
--------------
Basic error handling:
    ```python
    from agr_mcp.src.errors import (
        ValidationError,
        create_error_response,
        format_error_for_logging
    )

    # Raise a validation error
    try:
        if not gene_id:
            raise ValidationError(
                "Gene ID is required",
                field="gene_id",
                constraints={"required": True}
            )
    except ValidationError as e:
        # Create API response
        response = create_error_response(e)
        # Log the error
        logger.error(format_error_for_logging(e, context={"endpoint": "/gene"}))
        return response
    ```

Using helper functions:
    ```python
    from agr_mcp.src.errors import create_validation_error, create_resource_error

    # Create standardized errors
    if not is_valid_gene_id(gene_id):
        raise create_validation_error(
            field="gene_id",
            message="Invalid gene ID format",
            value=gene_id,
            constraints={"pattern": "^[A-Z]+\\d+$"}
        )

    # Resource not found
    gene = database.get_gene(gene_id)
    if not gene:
        raise create_resource_error("gene", gene_id)
    ```

Error handling with fallback:
    ```python
    from agr_mcp.src.errors import handle_error_with_fallback, ConnectionError

    result = handle_error_with_fallback(
        primary_action=lambda: fetch_from_primary_api(gene_id),
        fallback_action=lambda: fetch_from_cache(gene_id),
        error_types=(ConnectionError, TimeoutError),
        logger=logger
    )
    ```

Best Practices:
--------------
- Always catch specific exceptions rather than broad Exception class
- Include relevant context in error details
- Use appropriate HTTP status codes via get_http_status_code()
- Log errors with full context for debugging
- Sanitize sensitive data before including in error messages
- Use helper functions for consistency across the codebase
"""

import json
from typing import Any, Dict, Optional, Union
from datetime import datetime


class AGRMCPError(Exception):
    """Base exception for all AGR MCP server errors.

    This is the root exception class from which all other AGR MCP exceptions inherit.
    It provides common functionality for error tracking, serialization, and logging.

    Attributes:
        message (str): Human-readable error message
        error_code (str): Machine-readable error code for programmatic handling
        details (Dict[str, Any]): Additional context about the error
        timestamp (datetime): When the error occurred
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the AGR MCP error.

        Args:
            message: Human-readable error description
            error_code: Optional machine-readable error code
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary for serialization.

        Returns:
            Dictionary containing error information
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "type": self.__class__.__name__
        }

    def to_json(self) -> str:
        """Convert the exception to a JSON string.

        Returns:
            JSON string representation of the error
        """
        return json.dumps(self.to_dict(), default=str)


class ConfigurationError(AGRMCPError):
    """Raised when there's an issue with server configuration.

    This includes missing configuration files, invalid settings,
    or incompatible configuration values.

    Usage Example:
        ```python
        # Missing required configuration
        if 'api_key' not in config:
            raise ConfigurationError(
                "API key not found in configuration",
                config_key="api_key"
            )

        # Invalid configuration value
        if config.get('timeout') <= 0:
            raise ConfigurationError(
                f"Invalid timeout value: {config['timeout']}",
                config_key="timeout"
            )
        ```
    """

    def __init__(self, message: str, config_key: Optional[str] = None):
        """Initialize configuration error.

        Args:
            message: Error description
            config_key: The specific configuration key that caused the error
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class AuthenticationError(AGRMCPError):
    """Raised when authentication fails.

    This includes invalid API keys, expired tokens, or missing credentials.

    Usage Example:
        ```python
        # Invalid API key
        if not validate_api_key(api_key):
            raise AuthenticationError(
                "Invalid API key provided",
                auth_type="api_key"
            )

        # Expired token
        if token_is_expired(token):
            raise AuthenticationError(
                "Authentication token has expired",
                auth_type="token"
            )

        # Missing credentials
        if not request.headers.get('Authorization'):
            raise AuthenticationError(
                "Authorization header required",
                auth_type="header"
            )
        ```
    """

    def __init__(self, message: str, auth_type: Optional[str] = None):
        """Initialize authentication error.

        Args:
            message: Error description
            auth_type: The type of authentication that failed (e.g., "api_key", "token")
        """
        details = {}
        if auth_type:
            details["auth_type"] = auth_type
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class ValidationError(AGRMCPError):
    """Raised when input validation fails.

    This includes invalid parameters, malformed data, or constraint violations.

    Usage Example:
        ```python
        # Invalid parameter type
        if not isinstance(limit, int):
            raise ValidationError(
                "Limit must be an integer",
                field="limit",
                value=limit,
                constraints={"type": "integer"}
            )

        # Value out of range
        if limit < 1 or limit > 1000:
            raise ValidationError(
                "Limit must be between 1 and 1000",
                field="limit",
                value=limit,
                constraints={"min": 1, "max": 1000}
            )

        # Pattern mismatch
        if not re.match(r'^[A-Z]+\d+$', gene_id):
            raise ValidationError(
                "Invalid gene ID format",
                field="gene_id",
                value=gene_id,
                constraints={"pattern": "^[A-Z]+\\d+$"}
            )
        ```
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        """Initialize validation error.

        Args:
            message: Error description
            field: The field that failed validation
            value: The invalid value (sanitized if necessary)
            constraints: The constraints that were violated
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            # Sanitize sensitive values
            details["value"] = str(value)[:100] if len(str(value)) > 100 else value
        if constraints:
            details["constraints"] = constraints
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class ResourceNotFoundError(AGRMCPError):
    """Raised when a requested resource cannot be found.

    This includes missing files, non-existent database records, or invalid URLs.

    Usage Example:
        ```python
        # Database record not found
        gene = db.query_gene(gene_id)
        if not gene:
            raise ResourceNotFoundError(
                f"Gene {gene_id} not found in database",
                resource_type="gene",
                resource_id=gene_id
            )

        # File not found
        if not os.path.exists(file_path):
            raise ResourceNotFoundError(
                f"File not found: {file_path}",
                resource_type="file",
                resource_id=file_path
            )

        # URL returns 404
        response = requests.get(url)
        if response.status_code == 404:
            raise ResourceNotFoundError(
                f"Resource not found at URL: {url}",
                resource_type="url",
                resource_id=url
            )
        ```
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ):
        """Initialize resource not found error.

        Args:
            message: Error description
            resource_type: The type of resource (e.g., "gene", "allele", "file")
            resource_id: The identifier of the missing resource
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details=details
        )


class PermissionError(AGRMCPError):
    """Raised when the user lacks necessary permissions.

    This includes insufficient privileges for an operation or access to a resource.

    Usage Example:
        ```python
        # Insufficient role
        if user.role != 'admin':
            raise PermissionError(
                "Admin access required for this operation",
                required_permission="admin",
                resource="user_management"
            )

        # Resource access denied
        if not user.can_access_project(project_id):
            raise PermissionError(
                f"Access denied to project {project_id}",
                required_permission="project_read",
                resource=f"project/{project_id}"
            )

        # Write permission required
        if user.readonly:
            raise PermissionError(
                "Write access required",
                required_permission="write",
                resource="database"
            )
        ```
    """

    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        resource: Optional[str] = None
    ):
        """Initialize permission error.

        Args:
            message: Error description
            required_permission: The permission that was required
            resource: The resource that access was denied to
        """
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        if resource:
            details["resource"] = resource
        super().__init__(
            message=message,
            error_code="PERMISSION_ERROR",
            details=details
        )


class ConnectionError(AGRMCPError):
    """Raised when network or database connections fail.

    This includes timeout errors, connection refused, or network unreachable.

    Usage Example:
        ```python
        # Database connection failed
        try:
            db = connect_to_database(host, port)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to database: {str(e)}",
                host=host,
                port=port,
                service="postgresql"
            )

        # API connection refused
        try:
            response = requests.get(api_url, timeout=30)
        except requests.ConnectionError:
            raise ConnectionError(
                "API server is not responding",
                host="api.alliancegenome.org",
                port=443,
                service="agr_api"
            )

        # Network unreachable
        if not ping_host(host):
            raise ConnectionError(
                f"Network unreachable: {host}",
                host=host,
                service="network"
            )
        ```
    """

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        service: Optional[str] = None
    ):
        """Initialize connection error.

        Args:
            message: Error description
            host: The host that couldn't be connected to
            port: The port number
            service: The service name (e.g., "database", "api")
        """
        details = {}
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        if service:
            details["service"] = service
        super().__init__(
            message=message,
            error_code="CONNECTION_ERROR",
            details=details
        )


class DataIntegrityError(AGRMCPError):
    """Raised when data integrity issues are detected.

    This includes corrupted data, checksum failures, or inconsistent state.

    Usage Example:
        ```python
        # Checksum mismatch
        if calculated_checksum != expected_checksum:
            raise DataIntegrityError(
                "File checksum verification failed",
                data_type="file",
                expected=expected_checksum,
                actual=calculated_checksum
            )

        # Data consistency check
        if gene.chromosome not in valid_chromosomes:
            raise DataIntegrityError(
                f"Invalid chromosome reference: {gene.chromosome}",
                data_type="gene_record",
                expected=valid_chromosomes,
                actual=gene.chromosome
            )

        # Foreign key violation
        if allele.gene_id and not gene_exists(allele.gene_id):
            raise DataIntegrityError(
                f"Allele references non-existent gene: {allele.gene_id}",
                data_type="allele_record",
                expected="valid_gene_id",
                actual=allele.gene_id
            )
        ```
    """

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None
    ):
        """Initialize data integrity error.

        Args:
            message: Error description
            data_type: The type of data with integrity issues
            expected: The expected value or state
            actual: The actual value or state found
        """
        details = {}
        if data_type:
            details["data_type"] = data_type
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        super().__init__(
            message=message,
            error_code="DATA_INTEGRITY_ERROR",
            details=details
        )


class RateLimitError(AGRMCPError):
    """Raised when rate limits are exceeded.

    This includes API rate limits, query limits, or resource usage limits.

    Usage Example:
        ```python
        # API rate limit exceeded
        if request_count > rate_limit:
            raise RateLimitError(
                "API rate limit exceeded",
                limit=100,
                window="1h",
                retry_after=3600 - elapsed_seconds
            )

        # Query size limit
        if query_size > max_query_size:
            raise RateLimitError(
                f"Query size {query_size} exceeds limit",
                limit=max_query_size,
                window="per_request"
            )

        # Concurrent connection limit
        if active_connections >= max_connections:
            raise RateLimitError(
                "Maximum concurrent connections reached",
                limit=max_connections,
                window="concurrent",
                retry_after=60
            )
        ```
    """

    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        """Initialize rate limit error.

        Args:
            message: Error description
            limit: The rate limit that was exceeded
            window: The time window for the limit (e.g., "1h", "24h")
            retry_after: Seconds until the limit resets
        """
        details = {}
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = window
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details
        )


class ToolExecutionError(AGRMCPError):
    """Raised when a tool fails to execute properly.

    This includes tool crashes, invalid tool output, or tool-specific errors.

    Usage Example:
        ```python
        # Tool execution failed
        result = run_tool('gene_analyzer', gene_id)
        if result.exit_code != 0:
            raise ToolExecutionError(
                f"Gene analyzer failed with exit code {result.exit_code}",
                tool_name="gene_analyzer",
                operation="analyze_gene",
                error_output=result.stderr
            )

        # Invalid tool output
        try:
            data = json.loads(tool_output)
        except json.JSONDecodeError as e:
            raise ToolExecutionError(
                "Tool returned invalid JSON",
                tool_name="data_exporter",
                operation="export_json",
                error_output=str(e)
            )

        # Tool not found
        if not shutil.which(tool_name):
            raise ToolExecutionError(
                f"Required tool '{tool_name}' not found in PATH",
                tool_name=tool_name,
                operation="tool_check"
            )
        ```
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        error_output: Optional[str] = None
    ):
        """Initialize tool execution error.

        Args:
            message: Error description
            tool_name: The name of the tool that failed
            operation: The operation being performed
            error_output: Error output from the tool
        """
        details = {}
        if tool_name:
            details["tool_name"] = tool_name
        if operation:
            details["operation"] = operation
        if error_output:
            details["error_output"] = error_output[:500]  # Limit output length
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details=details
        )


class TimeoutError(AGRMCPError):
    """Raised when operations exceed time limits.

    This includes query timeouts, connection timeouts, or operation deadlines.

    Usage Example:
        ```python
        # Database query timeout
        try:
            results = db.query(complex_query, timeout=30)
        except QueryTimeout:
            raise TimeoutError(
                "Database query exceeded time limit",
                timeout_seconds=30,
                operation="database_query"
            )

        # API request timeout
        try:
            response = requests.get(url, timeout=60)
        except requests.Timeout:
            raise TimeoutError(
                f"Request to {url} timed out",
                timeout_seconds=60,
                operation="http_request"
            )

        # Long-running operation timeout
        start_time = time.time()
        while processing:
            if time.time() - start_time > max_duration:
                raise TimeoutError(
                    "Processing exceeded maximum duration",
                    timeout_seconds=max_duration,
                    operation="data_processing"
                )
        ```
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None
    ):
        """Initialize timeout error.

        Args:
            message: Error description
            timeout_seconds: The timeout value in seconds
            operation: The operation that timed out
        """
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=details
        )


# Error Response Helper Functions

def get_http_status_code(error: AGRMCPError) -> int:
    """Map error types to appropriate HTTP status codes.

    Args:
        error: The AGRMCPError instance

    Returns:
        HTTP status code appropriate for the error type
    """
    error_status_mapping = {
        "CONFIGURATION_ERROR": 500,
        "AUTHENTICATION_ERROR": 401,
        "VALIDATION_ERROR": 400,
        "RESOURCE_NOT_FOUND": 404,
        "PERMISSION_ERROR": 403,
        "CONNECTION_ERROR": 503,
        "DATA_INTEGRITY_ERROR": 500,
        "RATE_LIMIT_ERROR": 429,
        "TOOL_EXECUTION_ERROR": 500,
        "TIMEOUT_ERROR": 504,
    }
    return error_status_mapping.get(error.error_code, 500)


def create_error_response(
    exception: Union[AGRMCPError, Exception],
    include_timestamp: bool = True,
    include_type: bool = True
) -> Dict[str, Any]:
    """Create a standardized error response from an exception.

    Args:
        exception: The exception to convert to a response
        include_timestamp: Whether to include timestamp in response
        include_type: Whether to include exception type in response

    Returns:
        Dictionary formatted for API response
    """
    if isinstance(exception, AGRMCPError):
        response = {
            "success": False,
            "error": {
                "code": exception.error_code,
                "message": exception.message,
                "details": exception.details
            },
            "status_code": get_http_status_code(exception)
        }

        if include_timestamp:
            response["error"]["timestamp"] = exception.timestamp.isoformat()

        if include_type:
            response["error"]["type"] = exception.__class__.__name__
    else:
        # Handle non-AGRMCPError exceptions
        response = {
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exception),
                "details": {}
            },
            "status_code": 500
        }

        if include_timestamp:
            response["error"]["timestamp"] = datetime.utcnow().isoformat()

        if include_type:
            response["error"]["type"] = exception.__class__.__name__

    return response


def format_error_for_logging(
    exception: Union[AGRMCPError, Exception],
    include_traceback: bool = False,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Format an error for consistent log entries.

    Args:
        exception: The exception to format
        include_traceback: Whether to include stack trace
        context: Additional context to include in log

    Returns:
        Formatted string for logging
    """
    import traceback

    parts = []

    if isinstance(exception, AGRMCPError):
        parts.append(f"[{exception.error_code}] {exception.message}")
        parts.append(f"Timestamp: {exception.timestamp.isoformat()}")
        if exception.details:
            parts.append(f"Details: {json.dumps(exception.details, default=str)}")
    else:
        parts.append(f"[{exception.__class__.__name__}] {str(exception)}")
        parts.append(f"Timestamp: {datetime.utcnow().isoformat()}")

    if context:
        parts.append(f"Context: {json.dumps(context, default=str)}")

    if include_traceback:
        parts.append("Traceback:")
        parts.append(traceback.format_exc())

    return "\n".join(parts)


def create_validation_error(
    field: str,
    message: str,
    value: Optional[Any] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> ValidationError:
    """Create a validation error with standard formatting.

    Args:
        field: The field that failed validation
        message: Error message
        value: The invalid value
        constraints: Validation constraints that were violated

    Returns:
        ValidationError instance
    """
    full_message = f"Validation failed for field '{field}': {message}"
    return ValidationError(
        message=full_message,
        field=field,
        value=value,
        constraints=constraints
    )


def create_resource_error(
    resource_type: str,
    resource_id: str,
    message: Optional[str] = None
) -> ResourceNotFoundError:
    """Create a resource not found error with standard formatting.

    Args:
        resource_type: Type of resource (e.g., "gene", "allele")
        resource_id: Identifier of the missing resource
        message: Optional custom message

    Returns:
        ResourceNotFoundError instance
    """
    if not message:
        message = f"{resource_type.capitalize()} with ID '{resource_id}' not found"

    return ResourceNotFoundError(
        message=message,
        resource_type=resource_type,
        resource_id=resource_id
    )


def create_connection_error(
    service: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    message: Optional[str] = None
) -> ConnectionError:
    """Create a connection error with standard formatting.

    Args:
        service: The service that couldn't be connected to
        host: Optional hostname
        port: Optional port number
        message: Optional custom message

    Returns:
        ConnectionError instance
    """
    if not message:
        if host and port:
            message = f"Failed to connect to {service} at {host}:{port}"
        elif host:
            message = f"Failed to connect to {service} at {host}"
        else:
            message = f"Failed to connect to {service}"

    return ConnectionError(
        message=message,
        host=host,
        port=port,
        service=service
    )


def create_rate_limit_error(
    limit: int,
    window: str,
    retry_after: Optional[int] = None,
    message: Optional[str] = None
) -> RateLimitError:
    """Create a rate limit error with standard formatting.

    Args:
        limit: The rate limit that was exceeded
        window: Time window (e.g., "1h", "24h")
        retry_after: Seconds until limit resets
        message: Optional custom message

    Returns:
        RateLimitError instance
    """
    if not message:
        message = f"Rate limit of {limit} requests per {window} exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

    return RateLimitError(
        message=message,
        limit=limit,
        window=window,
        retry_after=retry_after
    )


def handle_error_with_fallback(
    primary_action: callable,
    fallback_action: Optional[callable] = None,
    error_types: tuple = (AGRMCPError,),
    logger: Optional[Any] = None
) -> Any:
    """Execute an action with error handling and optional fallback.

    Args:
        primary_action: The primary function to execute
        fallback_action: Optional fallback function if primary fails
        error_types: Tuple of error types to catch
        logger: Optional logger instance

    Returns:
        Result from primary or fallback action

    Raises:
        Re-raises exception if no fallback or fallback also fails
    """
    try:
        return primary_action()
    except error_types as e:
        if logger:
            logger.error(format_error_for_logging(e))

        if fallback_action:
            try:
                return fallback_action()
            except Exception as fallback_error:
                if logger:
                    logger.error(f"Fallback action failed: {fallback_error}")
                raise
        else:
            raise
