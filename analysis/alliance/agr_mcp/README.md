# AGR MCP Server

A Model Context Protocol (MCP) server for the Alliance of Genome Resources (AGR), providing structured access to Alliance data and tools through a standardized interface.

## Quick Start

```python
# Start the server
from agr_mcp import start_server
await start_server()

# Or use the command line
agr-mcp-server --api-url https://www.alliancegenome.org/api
```

## Overview

The AGR MCP Server enables AI assistants and other MCP-compatible clients to interact with Alliance of Genome Resources data, including:

- Gene and allele information across model organisms
- Disease associations and phenotype data
- Expression patterns and anatomical data
- Orthology relationships and evolutionary conservation
- Publication and reference data
- Variant and sequence information

## Features

- **Multi-species Support**: Access data from all Alliance member databases (WormBase, FlyBase, MGI, RGD, SGD, ZFIN, XenBase)
- **Standardized Data Access**: Consistent interface for querying heterogeneous data sources
- **Rich Data Formatting**: Multiple output formats (JSON, TSV, human-readable text)
- **Batch Operations**: Efficient bulk data retrieval
- **Cross-reference Resolution**: Automatic ID mapping between Alliance members
- **Caching**: Intelligent caching for improved performance
- **Error Handling**: Robust error handling with informative messages

## Installation

### From PyPI (Recommended)

```bash
pip install agr-mcp-server
```

### From Source

```bash
git clone https://github.com/alliance-genome/agr-mcp-server.git
cd agr-mcp-server
pip install -e .
```

## Configuration

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop `config.json`):

```json
{
  "mcpServers": {
    "agr": {
      "command": "agr-mcp",
      "args": ["--api-url", "https://www.alliancegenome.org/api"],
      "env": {
        "AGR_CACHE_DIR": "/path/to/cache"
      }
    }
  }
}
```

### Environment Variables

- `AGR_API_URL`: Alliance API endpoint (default: https://www.alliancegenome.org/api)
- `AGR_CACHE_DIR`: Directory for caching responses (default: `~/.cache/agr-mcp`)
- `AGR_CACHE_TTL`: Cache time-to-live in seconds (default: 3600)
- `AGR_LOG_LEVEL`: Logging level (default: INFO)

## Usage

### Available Tools

#### Gene Information

```python
# Get gene information
{
  "tool": "get_gene",
  "arguments": {
    "identifier": "HGNC:11998",  # or WB:WBGene00006789, FB:FBgn0000490, etc.
    "include_orthologs": true,
    "include_expression": true,
    "include_disease": true
  }
}
```

#### Disease Associations

```python
# Get disease associations for a gene
{
  "tool": "get_disease_associations",
  "arguments": {
    "gene_id": "HGNC:11998",
    "disease_id": "OMIM:168600",  # optional
    "include_orthologs": true
  }
}
```

#### Expression Data

```python
# Get expression data
{
  "tool": "get_expression",
  "arguments": {
    "gene_id": "WB:WBGene00006789",
    "stage": "adult",  # optional
    "anatomy_term": "nervous system"  # optional
  }
}
```

#### Allele Information

```python
# Get allele data
{
  "tool": "get_alleles",
  "arguments": {
    "gene_id": "FB:FBgn0000490",
    "allele_id": "FB:FBal0123456",  # optional
    "include_phenotypes": true
  }
}
```

#### Cross-references

```python
# Map identifiers between databases
{
  "tool": "get_cross_references",
  "arguments": {
    "identifier": "HGNC:11998",
    "target_db": "all"  # or specific: "WB", "FB", "MGI", etc.
  }
}
```

#### Batch Operations

```python
# Get multiple genes
{
  "tool": "get_genes_batch",
  "arguments": {
    "identifiers": ["HGNC:11998", "WB:WBGene00006789", "FB:FBgn0000490"],
    "include_orthologs": true,
    "format": "json"  # or "tsv", "text"
  }
}
```

#### File Downloads

```python
# Download files from AGR S3 repository
{
  "tool": "download_file",
  "arguments": {
    "key": "path/to/file.gff3",
    "destination": "/local/path/to/save/file.gff3"  # optional
  }
}
```

#### API Schema Information

```python
# Get JSON schema for AGR data types
{
  "tool": "get_api_schema",
  "arguments": {
    "schema_type": "gene"  # options: "gene", "allele", "disease", "expression"
  }
}
```

### Output Formats

The server supports multiple output formats:

- **JSON**: Structured data for programmatic use
- **TSV**: Tab-separated values for spreadsheet import
- **Text**: Human-readable formatted text

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/alliance-genome/agr-mcp-server.git
cd agr-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=agr_mcp --cov-report=html
```

### Project Structure

```
agr_mcp/
├── src/agr_mcp/
│   ├── __init__.py          # Package initialization
│   ├── server.py            # Main MCP server implementation
│   ├── tools/               # Tool implementations
│   │   ├── __init__.py
│   │   ├── gene.py          # Gene-related tools
│   │   ├── disease.py       # Disease association tools
│   │   ├── expression.py    # Expression data tools
│   │   ├── allele.py        # Allele and phenotype tools
│   │   └── xref.py          # Cross-reference tools
│   ├── parsers/             # Data parsers
│   │   ├── __init__.py
│   │   ├── alliance.py      # Alliance API response parser
│   │   └── identifier.py    # ID parsing and validation
│   ├── formatters/          # Output formatters
│   │   ├── __init__.py
│   │   ├── json.py          # JSON formatting
│   │   ├── tsv.py           # TSV formatting
│   │   └── text.py          # Human-readable text
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── cache.py         # Caching implementation
│       ├── api.py           # API client
│       └── validation.py    # Input validation
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── pyproject.toml         # Project configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_gene_tools.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=agr_mcp --cov-report=html
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linters
flake8 src tests
mypy src
```

## API Reference

### Tool Specifications

Each tool follows the MCP tool specification format:

```python
{
  "name": "tool_name",
  "description": "Tool description",
  "inputSchema": {
    "type": "object",
    "properties": {
      "param1": {"type": "string", "description": "Parameter description"},
      "param2": {"type": "boolean", "default": false}
    },
    "required": ["param1"]
  }
}
```

### Error Responses

Errors follow the MCP error format:

```json
{
  "error": {
    "code": "INVALID_GENE_ID",
    "message": "Gene identifier not found: HGNC:99999",
    "details": {
      "identifier": "HGNC:99999",
      "suggestion": "Check the identifier format and try again"
    }
  }
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/alliance-genome/agr-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alliance-genome/agr-mcp-server/discussions)
- **Alliance Help**: [help@alliancegenome.org](mailto:help@alliancegenome.org)

## Acknowledgments

This project is part of the Alliance of Genome Resources, a consortium of model organism databases working together to provide integrated genomic information.

Member databases:
- [WormBase](https://wormbase.org)
- [FlyBase](https://flybase.org)
- [MGI](http://www.informatics.jax.org)
- [RGD](https://rgd.mcw.edu)
- [SGD](https://www.yeastgenome.org)
- [ZFIN](https://zfin.org)
- [XenBase](https://www.xenbase.org)
