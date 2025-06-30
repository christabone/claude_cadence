# Code Style and Conventions

## Code Formatting
- **Black**: Used for code formatting (line length: 88)
- **isort**: Import sorting with Black profile
- **MyPy**: Static type checking with strict settings
- **Flake8**: Linting

## Typing and Documentation
- **Type Hints**: Required for all functions (`disallow_untyped_defs: true`)
- **Docstrings**: Google-style docstrings with Args, Returns, Raises sections
- **Pydantic Models**: Used for data validation and serialization

## Error Handling
- Custom exception hierarchy in `src/errors.py`
- Structured error responses with HTTP status codes
- Comprehensive error logging

## Async Patterns
- All API calls use `httpx.AsyncClient`
- Functions returning data are async
- Proper context manager usage

## Logging
- Structured logging with request IDs
- Logger per module: `logger = get_logger('module_name')`
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## File Structure Patterns
- Tools in `src/tools/` directory
- Utilities in `src/utils/` directory
- Each tool has corresponding tests in `tests/tools/`
- Configuration centralized in `src/config.py`

## Import Patterns
From existing code:
```python
from ..errors import ValidationError, ResourceNotFoundError, ToolExecutionError
from ..utils.http_client import http_client
from ..utils.validators import validate_gene_id
from ..utils.logging_config import get_logger
```
