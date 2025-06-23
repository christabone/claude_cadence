# Alliance of Genome Resources (AGR) API Documentation

## Overview

The Alliance of Genome Resources (AGR) provides a comprehensive REST API for accessing genomic data across multiple model organisms. This documentation provides a detailed analysis of the API structure, endpoints, and usage patterns.

## Base API Information

### Base URL
```
https://www.alliancegenome.org/api
```

### Configuration Details
- **Default Timeout**: 30 seconds
- **Max Retries**: 3
- **Rate Limit**: 10 requests per second
- **User Agent**: AGR-MCP-Server/1.0

### Authentication
- **No authentication required** for read operations
- Public API with open access to genomic data

### Response Format
- Primary format: JSON
- Additional formats supported: TSV, CSV (for specific endpoints)
- All responses include standard HTTP status codes

### Rate Limiting
- **Requests per second**: 10
- **Requests per minute**: 100
- Rate limit headers returned: `Retry-After` on 429 responses

## API Structure Analysis

### Date: 2024-06-22
### Author: AGR MCP Documentation Team

---

## 1. API Base Structure Exploration

The AGR API follows RESTful principles and is organized around resource types. The main entry points are focused on genomic data types rather than a traditional API root endpoint.

### Key Findings:
- The base API URL (`/api`) returns HTML, not JSON - it's not a traditional REST API root
- API endpoints are resource-specific and directly accessible
- No API discovery endpoint or OpenAPI/Swagger spec available at standard locations
- The API is designed for direct resource access rather than hypermedia navigation

---

## 2. Gene-Related API Endpoints

The AGR API provides comprehensive endpoints for querying gene data across multiple model organisms.

### 2.1 Gene Search Endpoint

**Endpoint**: `GET /api/search`

**Description**: Search for genes (and other entities) across the AGR database.

**Parameters**:
- `q` (required): Query string for searching genes by symbol, name, or identifier
- `category`: Filter by entity type (e.g., "gene", "allele", "disease")
- `species`: Filter by species name or taxon ID
- `limit`: Maximum number of results (default: 10, max: 100)
- `offset`: Pagination offset for results

**Example Request**:
```bash
curl -X GET "https://www.alliancegenome.org/api/search?q=BRCA1&category=gene&limit=2" \
  -H "Accept: application/json"
```

**Response Structure**:
```json
{
  "total": 108,
  "results": [
    {
      "id": "HGNC:1100",
      "symbol": "BRCA1",
      "name": "BRCA1 DNA repair associated",
      "species": "Homo sapiens",
      "synonyms": ["RNF53", "FANCS", ...],
      "diseases": ["breast cancer", "ovarian cancer", ...],
      "soTermName": "protein_coding_gene",
      "category": "gene"
    }
  ],
  "aggregations": [
    {
      "key": "species",
      "values": [{"key": "Homo sapiens", "total": 29}, ...]
    }
  ]
}
```

### 2.2 Gene Details Endpoint

**Endpoint**: `GET /api/gene/{id}`

**Description**: Retrieve comprehensive information about a specific gene.

**Parameters**:
- `id` (required): AGR gene identifier (e.g., "HGNC:1100")

**Example Request**:
```bash
curl -X GET "https://www.alliancegenome.org/api/gene/HGNC:1100" \
  -H "Accept: application/json"
```

**Response Structure**:
```json
{
  "id": "HGNC:1100",
  "symbol": "BRCA1",
  "name": "BRCA1 DNA repair associated",
  "species": {
    "name": "Homo sapiens",
    "taxonId": "NCBITaxon:9606"
  },
  "geneSynopsis": "This gene encodes a 190 kD nuclear phosphoprotein...",
  "automatedGeneSynopsis": "Enables several functions...",
  "synonyms": ["RNF53", "FANCS", ...],
  "genomeLocations": [
    {
      "chromosome": "17",
      "start": 43044295,
      "end": 43170327,
      "assembly": "GRCh38",
      "strand": "-"
    }
  ],
  "crossReferenceMap": {
    "other": [
      {
        "name": "ENSEMBL:ENSG00000012048",
        "crossRefCompleteUrl": "http://www.ensembl.org/id/ENSG00000012048"
      }
    ]
  }
}
```

### 2.3 Gene Orthologs Endpoint

**Endpoint**: `GET /api/gene/{id}/orthologs`

**Description**: Find orthologous genes across different species.

**Parameters**:
- `id` (required): AGR gene identifier

**Example Request**:
```bash
curl -X GET "https://www.alliancegenome.org/api/gene/HGNC:1100/orthologs" \
  -H "Accept: application/json"
```

**Response Structure**:
```json
{
  "results": [
    {
      "category": "gene_to_gene_orthology",
      "stringencyFilter": "stringent",
      "geneAnnotations": [
        {
          "geneIdentifier": "HGNC:1100",
          "hasDiseaseAnnotations": true,
          "hasExpressionAnnotations": false
        },
        {
          "geneIdentifier": "Xenbase:XB-GENE-490624",
          "hasDiseaseAnnotations": false,
          "hasExpressionAnnotations": false
        }
      ]
    }
  ]
}
```

### 2.4 Gene Alleles Endpoint

**Endpoint**: `GET /api/gene/{id}/alleles`

**Description**: Retrieve alleles and variants associated with a specific gene.

**Parameters**:
- `id` (required): AGR gene identifier
- `limit`: Maximum number of results
- `offset`: Pagination offset

**Example Request**:
```bash
curl -X GET "https://www.alliancegenome.org/api/gene/HGNC:1100/alleles?limit=2" \
  -H "Accept: application/json"
```

**Response Structure**:
```json
{
  "results": [
    {
      "id": "rs1332532415",
      "symbol": "NC_000017.11:g.43044315T>A",
      "hasDisease": false,
      "hasPhenotype": false,
      "category": "variant",
      "type": "allele",
      "variants": [
        {
          "variantType": {"id": "SNP", "name": "SNP"},
          "genomicReferenceSequence": "T",
          "genomicVariantSequence": "A",
          "location": {
            "chromosome": "17",
            "start": 43044315,
            "end": 43044315
          }
        }
      ]
    }
  ]
}
```

### 2.5 Additional Gene Endpoints (Not Currently Active)

Based on the codebase analysis, these endpoints are defined but may not be currently active:

- `GET /api/gene/{id}/expression` - Gene expression data
- `GET /api/gene/{id}/disease-associations` - Disease associations
- `GET /api/gene/{id}/phenotypes` - Associated phenotypes
- `GET /api/gene/{id}/interactions` - Gene interactions

---

## 3. AGR Downloads Structure

The Alliance of Genome Resources provides downloadable data files through an S3 bucket accessible at:

### Base Download URL
```
https://download.alliancegenome.org/
```

### File Organization

The downloads are organized in a hierarchical structure:

```
{version}/{data_type}/{species}/{version}_{data_type}_{species}_{iteration}.{format}.gz
```

#### Key Components:

1. **Version**: Release version (e.g., `6.0.0`, `1.0.2.3`)
2. **Data Types**:
   - `BGI` - Basic Gene Information
   - `ALLELE` - Allele/variant data
   - `DAF` - Disease Annotation File
   - `DISEASE-ALLIANCE-JSON` - Disease associations
   - `ORTHO` - Orthology data
   - `EXPRESSION` - Expression data
   - `GFF` - Gene Feature Format files
   - `FASTA` - Sequence data
   - `AGM` - Affected Genomic Model
   - `CONSTRUCT` - Genetic constructs
   - `BIOGRID-ORCS` - BioGRID CRISPR screens

3. **Species/MODs**:
   - `FB` - FlyBase (Drosophila)
   - `MGI` - Mouse Genome Informatics
   - `RGD` - Rat Genome Database
   - `SGD` - Saccharomyces Genome Database
   - `WB` - WormBase
   - `ZFIN` - Zebrafish Information Network
   - `HUMAN` - Human data
   - `XBXL` - Xenopus laevis
   - `XBXT` - Xenopus tropicalis

4. **File Formats**:
   - `.json.gz` - Compressed JSON
   - `.gff.gz` - Compressed GFF3
   - `.tsv.gz` - Compressed TSV
   - `.tar.gz` - Compressed archives
   - `.obo` - Ontology files

### Example Download URLs

```bash
# Basic Gene Information for Mouse
https://download.alliancegenome.org/6.0.0/BGI/MGI/1.0.2.3_BGI_MGI_0.json.gz

# Disease annotations for Human
https://download.alliancegenome.org/6.0.0/DAF/HUMAN/1.0.2.3_DAF_HUMAN_0.json.gz

# Allele data for Zebrafish
https://download.alliancegenome.org/6.0.0/ALLELE/ZFIN/1.0.2.3_ALLELE_ZFIN_0.json.gz
```

### File Naming Patterns

- Files use consistent naming: `{schema_version}_{type}_{species}_{iteration}.{format}.gz`
- Schema version (e.g., `1.0.2.3`) indicates the data schema version
- Iteration number (e.g., `_0`, `_1`) indicates file parts for large datasets
- All files are gzip compressed

### Storage Classes

Files are stored in different AWS S3 storage classes:
- `STANDARD` - Frequently accessed data
- `GLACIER_IR` - Archived data with retrieval delays

### Update Frequency

Based on the LastModified timestamps, data appears to be updated:
- Major releases: Quarterly or semi-annually
- Incremental updates: Monthly or as needed
- Different data types may have different update schedules

### Download API Integration

The MCP server's file download tool (`file_download.py`) attempts to use API endpoints for downloads:
- **Note**: The documented download URLs (`/downloads/*`) appear to return HTML, not data files
- The actual data files are hosted on the S3 bucket directly
- Direct S3 access is more reliable than the API endpoints for bulk data downloads

---

## 4. Error Handling and Response Codes

### Standard HTTP Status Codes

The AGR API uses standard HTTP status codes to indicate the success or failure of requests:

#### Success Codes
- `200 OK` - Request succeeded, data returned
- `201 Created` - Resource successfully created (not currently used in read-only API)
- `204 No Content` - Request succeeded, no data to return

#### Client Error Codes
- `400 Bad Request` - Invalid request parameters or malformed request
- `404 Not Found` - Resource not found (gene ID, allele ID, etc.)
- `405 Method Not Allowed` - HTTP method not supported for endpoint
- `429 Too Many Requests` - Rate limit exceeded

#### Server Error Codes
- `500 Internal Server Error` - Server-side error
- `502 Bad Gateway` - Upstream service unavailable
- `503 Service Unavailable` - Service temporarily unavailable
- `504 Gateway Timeout` - Request timed out

### Error Response Format

Error responses typically include a JSON body with error details:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Gene with ID 'INVALID:123' not found",
    "details": {
      "resource_type": "gene",
      "resource_id": "INVALID:123"
    }
  },
  "timestamp": "2024-06-22T10:30:00Z",
  "path": "/api/gene/INVALID:123"
}
```

### Common Error Scenarios

1. **Invalid Gene ID Format**
   - Error: 400 Bad Request
   - Cause: Gene ID doesn't match expected pattern (e.g., "HGNC:12345")
   - Solution: Validate gene ID format before making request

2. **Rate Limit Exceeded**
   - Error: 429 Too Many Requests
   - Headers: `Retry-After: 60` (seconds)
   - Solution: Implement exponential backoff and respect rate limits

3. **Timeout Errors**
   - Error: 504 Gateway Timeout
   - Cause: Large result sets or complex queries
   - Solution: Use pagination parameters (`limit`, `offset`)

4. **Missing Required Parameters**
   - Error: 400 Bad Request
   - Cause: Required query parameter not provided
   - Solution: Ensure all required parameters are included

---

## 5. Implementation Guidance

### Best Practices

#### 1. Rate Limiting
```python
# Implement rate limiting (10 requests/second)
from asyncio import sleep
from time import time

class RateLimiter:
    def __init__(self, rate=10):
        self.rate = rate
        self.last_request = 0

    async def wait(self):
        elapsed = time() - self.last_request
        if elapsed < 1.0 / self.rate:
            await sleep(1.0 / self.rate - elapsed)
        self.last_request = time()
```

#### 2. Error Handling and Retries
```python
# Implement retry logic with exponential backoff
async def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                await sleep(retry_after)
                continue
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await sleep(2 ** attempt)  # Exponential backoff
```

#### 3. Pagination for Large Result Sets
```python
# Handle paginated results
async def get_all_results(base_url, params):
    all_results = []
    offset = 0
    limit = 100  # Maximum allowed

    while True:
        params['offset'] = offset
        params['limit'] = limit

        response = await client.get(base_url, params=params)
        data = response.json()

        all_results.extend(data.get('results', []))

        if len(data.get('results', [])) < limit:
            break

        offset += limit

    return all_results
```

### Integration Examples

#### 1. Search for Genes Across Species
```python
# Search for orthologs of a human gene
async def find_cross_species_orthologs(human_gene_symbol):
    # First, find the human gene
    search_results = await search_genes(
        query=human_gene_symbol,
        species="Homo sapiens",
        limit=1
    )

    if not search_results['results']:
        return None

    gene_id = search_results['results'][0]['id']

    # Get orthologs
    orthologs = await find_orthologs(gene_id)

    return orthologs
```

#### 2. Bulk Data Download
```python
# Download and process bulk data files
async def download_species_data(species_code, data_type):
    base_url = "https://download.alliancegenome.org"
    version = "6.0.0"

    # Construct S3 URL directly
    file_url = f"{base_url}/{version}/{data_type}/{species_code}/"

    # List available files (requires S3 client)
    # Download compressed file
    # Process data incrementally to handle large files
```

#### 3. Gene Information Pipeline
```python
# Complete gene information retrieval
async def get_complete_gene_info(gene_id):
    # Parallel requests for all gene data
    tasks = [
        get_gene_details(gene_id),
        find_orthologs(gene_id),
        client.get(f"/gene/{gene_id}/alleles"),
        client.get(f"/gene/{gene_id}/expression")
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    gene_info = {
        'details': results[0] if not isinstance(results[0], Exception) else None,
        'orthologs': results[1] if not isinstance(results[1], Exception) else None,
        'alleles': results[2].json() if not isinstance(results[2], Exception) else None,
        'expression': results[3].json() if not isinstance(results[3], Exception) else None
    }

    return gene_info
```

### Performance Optimization

1. **Use Specific Endpoints**: Rather than searching, use direct gene IDs when known
2. **Batch Requests**: Group related requests and execute in parallel
3. **Cache Results**: Implement caching for frequently accessed data
4. **Compress Transfers**: Use gzip encoding for large responses
5. **Filter Fields**: Request only needed fields when endpoints support it

### Security Considerations

1. **API Keys**: While not required for public data, future versions may require authentication
2. **Input Validation**: Always validate and sanitize user inputs before API calls
3. **Rate Limiting**: Respect rate limits to avoid IP blocking
4. **HTTPS Only**: Always use HTTPS for API requests
5. **Error Messages**: Don't expose internal errors to end users

### Monitoring and Logging

Implement comprehensive logging for API interactions:

```python
import logging

logger = logging.getLogger('agr_api')

# Log all API requests
logger.info(f"API Request: {method} {url}", extra={
    'method': method,
    'url': url,
    'params': params,
    'timestamp': datetime.now().isoformat()
})

# Log errors with context
logger.error(f"API Error: {status_code}", extra={
    'status_code': response.status_code,
    'error_body': response.text,
    'url': url,
    'retry_attempt': attempt
})
```

---

## 6. Additional Resources

### Official Documentation
- AGR Website: https://www.alliancegenome.org/
- API Documentation: https://www.alliancegenome.org/api-docs
- Swagger UI: https://www.alliancegenome.org/swagger-ui/

### Data Downloads
- S3 Bucket Browser: https://download.alliancegenome.org/
- Release Notes: Available in version-specific directories

### Support
- Email: support@alliancegenome.org
- GitHub: https://github.com/alliance-genome

### Related Tools
- AGR JBrowse: Interactive genome browser
- AGR File Management System: For MOD submissions
- AGRKB: Knowledge base for genomic data

---

## Appendix: Gene ID Formats by Species

Different model organism databases (MODs) use specific ID formats:

- **Human (HGNC)**: `HGNC:12345`
- **Mouse (MGI)**: `MGI:12345`
- **Rat (RGD)**: `RGD:12345`
- **Zebrafish (ZFIN)**: `ZFIN:ZDB-GENE-12345`
- **Fly (FlyBase)**: `FB:FBgn0012345`
- **Worm (WormBase)**: `WB:WBGene00012345`
- **Yeast (SGD)**: `SGD:S000012345`
- **Xenopus**: `Xenbase:XB-GENE-12345`

Always use the full prefixed ID format when making API requests.
