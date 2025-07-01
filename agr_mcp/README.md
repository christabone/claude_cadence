# Alliance Genome Resources MCP Server

An MCP (Model Context Protocol) server that provides access to Alliance Genome Resources data and functionality.

## Overview

The Alliance Genome Resources (AGR) MCP Server enables AI assistants to interact with AGR data, including gene information, orthology relationships, disease associations, and more. This server implements the Model Context Protocol to provide standardized access to AGR's comprehensive genomic data.

## Features

- **Gene Information Retrieval**: Access detailed gene data across multiple model organisms
- **Orthology Queries**: Find orthologs and paralogs across species
- **Disease Associations**: Query gene-disease relationships
- **Expression Data**: Access gene expression patterns
- **Allele Information**: Retrieve allele and variant data
- **Publication Search**: Find relevant scientific publications
- **Cross-References**: Access external database identifiers

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agr-mcp-server.git
cd agr-mcp-server

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Configuration

1. Copy the example configuration file:
```bash
cp config/config.example.yml config/config.yml
```

2. Edit `config/config.yml` with your AGR API credentials and settings.

## Usage

### Starting the Server

```bash
# Run the MCP server
python -m agr_mcp.server

# Or use the command-line interface
agr-mcp serve
```

### Available Tools

The server provides the following tools:

- `search_genes`: Search for genes by symbol, name, or identifier
- `get_gene_details`: Retrieve detailed information about a specific gene
- `find_orthologs`: Find orthologous genes across species
- `download_data`: Download AGR data files (genes, alleles, disease, expression, orthology)
- `get_api_schema`: Get AGR API schema information
- `get_supported_species`: Get list of supported species from Alliance Genome Resources with their names and taxonomic IDs

### Example Usage

```python
# Example of using the AGR MCP client
from agr_mcp.client import AGRMCPClient

client = AGRMCPClient()

# Search for a gene
results = client.search_genes("BRCA1", species="Homo sapiens")

# Get gene details
gene_info = client.get_gene_details("HGNC:1100")

# Find orthologs
orthologs = client.find_orthologs("HGNC:1100", target_species="Mus musculus")
```

## Development

### Project Structure

```
agr_mcp/
├── src/
│   ├── __init__.py
│   ├── server.py          # Main MCP server implementation
│   ├── client.py          # Client library for the server
│   ├── config.py          # Configuration management
│   ├── tools/             # Individual tool implementations
│   │   ├── __init__.py
│   │   ├── gene_tools.py
│   │   ├── orthology_tools.py
│   │   ├── disease_tools.py
│   │   └── allele_tools.py
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── api_client.py
│       └── validators.py
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_server.py
│   ├── test_client.py
│   └── test_tools.py
├── logs/                  # Log files directory
├── config/               # Configuration files
├── docs/                 # Documentation
└── examples/             # Example scripts
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agr_mcp tests/

# Run specific test file
pytest tests/test_server.py
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## API Reference

### Server Endpoints

The MCP server exposes the following endpoints:

- `POST /tools/list` - List available tools
- `POST /tools/call` - Execute a tool
- `GET /health` - Health check endpoint

### Tool Specifications

Each tool follows the MCP tool specification format:

```json
{
  "name": "tool_name",
  "description": "Tool description",
  "inputSchema": {
    "type": "object",
    "properties": {
      "param1": {"type": "string", "description": "Parameter description"}
    },
    "required": ["param1"]
  }
}
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure the AGR API is accessible and your credentials are valid
2. **Rate Limiting**: The server implements rate limiting to comply with AGR API limits
3. **Data Format Issues**: Check that your queries match the expected format

### Logging

Logs are written to the `logs/` directory. Set the log level in your configuration:

```yaml
logging:
  level: INFO  # Options: DEBUG, INFO, WARNING, ERROR
  file: logs/agr_mcp.log
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alliance of Genome Resources for providing the data API
- Model Context Protocol specification contributors

## Contact

For questions or support, please contact:
- Email: support@example.com
- GitHub Issues: https://github.com/your-org/agr-mcp-server/issues
