# Suggested Commands for AGR MCP Development

## Development Workflow Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=agr_mcp tests/

# Run specific test file
pytest tests/tools/test_gene_query.py

# Run tests with verbose output
pytest -v

# Run only failed tests
pytest --lf
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with Flake8
flake8 src/ tests/

# Type checking with MyPy
mypy src/

# Run all quality checks
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/ && mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install requirements
pip install -r requirements.txt
```

### Running the Server
```bash
# Run MCP server
python -m agr_mcp.server

# Run with CLI
agr-mcp serve

# Run with specific config
python -m agr_mcp.server --config config/config.yml
```

### Project Management
```bash
# Check project structure
find src/ -name "*.py" | head -10

# View logs
tail -f logs/agr_mcp.log

# Clean cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
```

## System Commands (Linux)
```bash
# File operations
ls -la              # List files with details
find . -name "*.py" # Find Python files
grep -r "pattern"   # Search in files
tree src/           # View directory structure

# Git operations
git status
git add .
git commit -m "message"
git push origin main

# Process management
ps aux | grep python
kill -9 <pid>
```
