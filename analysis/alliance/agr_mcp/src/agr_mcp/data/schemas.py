"""
JSON schemas and validation utilities for AGR data.

This module provides JSON schema definitions and validation functions
for Alliance Genome Resource data structures.
"""

import re
from typing import Dict, Any, Optional, Tuple

from jsonschema import validate, ValidationError


# Identifier patterns for different databases
IDENTIFIER_PATTERNS = {
    "HGNC": re.compile(r"^HGNC:\d+$"),
    "MGI": re.compile(r"^MGI:\d+$"),
    "RGD": re.compile(r"^RGD:\d+$"),
    "ZFIN": re.compile(r"^ZFIN:ZDB-GENE-\d{6}-\d+$"),
    "FB": re.compile(r"^FB:FBgn\d{7}$"),
    "WB": re.compile(r"^WB:WBGene\d{8}$"),
    "SGD": re.compile(r"^SGD:S\d{9}$"),
    "XENBASE": re.compile(r"^XENBASE:XB-GENE-\d+$"),
}


class GeneSchema:
    """JSON schema for gene data validation."""

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "symbol": {"type": "string", "minLength": 1},
            "name": {"type": ["string", "null"]},
            "species": {"type": "string", "pattern": "^NCBITaxon:\\d+$"},
            "database": {
                "type": "string",
                "enum": ["HGNC", "MGI", "RGD", "ZFIN", "FB", "WB", "SGD", "XENBASE"]
            },
            "synonyms": {
                "type": "array",
                "items": {"type": "string"}
            },
            "description": {"type": ["string", "null"]},
            "geneType": {"type": ["string", "null"]},
            "location": {
                "type": ["object", "null"],
                "properties": {
                    "chromosome": {"type": "string"},
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                    "strand": {"type": ["string", "null"], "pattern": "^[+-]?$"},
                    "assembly": {"type": ["string", "null"]}
                },
                "required": ["chromosome", "start", "end"]
            },
            "crossReferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "identifier": {"type": "string"},
                        "url": {"type": ["string", "null"]}
                    },
                    "required": ["database", "identifier"]
                }
            },
            "orthologs": {
                "type": "array",
                "items": {"type": "string"}
            },
            "paralogs": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["id", "symbol", "species", "database"],
        "additionalProperties": True
    }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> None:
        """Validate gene data against schema."""
        validate(instance=data, schema=cls.schema)


class AlleleSchema:
    """JSON schema for allele data validation."""

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "symbol": {"type": "string", "minLength": 1},
            "geneId": {"type": "string", "minLength": 1},
            "name": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
            "alleleType": {"type": ["string", "null"]},
            "synonyms": {
                "type": "array",
                "items": {"type": "string"}
            },
            "molecularConsequence": {"type": ["string", "null"]},
            "mutationType": {"type": ["string", "null"]},
            "phenotypes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "diseases": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["id", "symbol", "geneId"],
        "additionalProperties": True
    }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> None:
        """Validate allele data against schema."""
        validate(instance=data, schema=cls.schema)


class DiseaseSchema:
    """JSON schema for disease data validation."""

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "name": {"type": "string", "minLength": 1},
            "synonyms": {
                "type": "array",
                "items": {"type": "string"}
            },
            "definition": {"type": ["string", "null"]},
            "associatedGenes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "associatedAlleles": {
                "type": "array",
                "items": {"type": "string"}
            },
            "crossReferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "identifier": {"type": "string"},
                        "url": {"type": ["string", "null"]}
                    },
                    "required": ["database", "identifier"]
                }
            }
        },
        "required": ["id", "name"],
        "additionalProperties": True
    }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> None:
        """Validate disease data against schema."""
        validate(instance=data, schema=cls.schema)


class ExpressionSchema:
    """JSON schema for expression data validation."""

    schema = {
        "type": "object",
        "properties": {
            "geneId": {"type": "string", "minLength": 1},
            "anatomicalStructure": {"type": ["string", "null"]},
            "developmentalStage": {"type": ["string", "null"]},
            "cellType": {"type": ["string", "null"]},
            "expressionLevel": {"type": ["string", "null"]},
            "pattern": {"type": ["string", "null"]},
            "assay": {"type": ["string", "null"]},
            "evidenceCode": {"type": ["string", "null"]},
            "qualifiers": {
                "type": "array",
                "items": {"type": "string"}
            },
            "publications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pubmedId": {"type": ["string", "null"]},
                        "doi": {"type": ["string", "null"]},
                        "title": {"type": ["string", "null"]}
                    }
                }
            }
        },
        "required": ["geneId"],
        "additionalProperties": True
    }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> None:
        """Validate expression data against schema."""
        validate(instance=data, schema=cls.schema)


def validate_identifier(identifier: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an Alliance identifier format.

    Args:
        identifier: The identifier to validate

    Returns:
        Tuple of (is_valid, database_name)
    """
    if not identifier or not isinstance(identifier, str):
        return False, None

    # Check each database pattern
    for db_name, pattern in IDENTIFIER_PATTERNS.items():
        if pattern.match(identifier):
            return True, db_name

    return False, None


def parse_identifier(identifier: str) -> Dict[str, str]:
    """
    Parse an Alliance identifier into its components.

    Args:
        identifier: The identifier to parse

    Returns:
        Dictionary with 'database' and 'local_id' keys

    Raises:
        ValueError: If identifier format is invalid
    """
    is_valid, database = validate_identifier(identifier)

    if not is_valid:
        raise ValueError(f"Invalid identifier format: {identifier}")

    # Split on first colon
    parts = identifier.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid identifier format: {identifier}")

    return {
        "database": database,
        "prefix": parts[0],
        "local_id": parts[1],
        "full_id": identifier
    }
