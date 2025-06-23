"""
Data models and schemas for the AGR MCP server.

This module contains Pydantic models and JSON schemas for validating
and serializing Alliance Genome Resource data.
"""

from .models import (
    Gene,
    Allele,
    Disease,
    Expression,
    Phenotype,
    CrossReference,
    Publication,
    Species,
    Location,
    Sequence
)
from .schemas import (
    GeneSchema,
    AlleleSchema,
    DiseaseSchema,
    ExpressionSchema,
    validate_identifier,
    parse_identifier
)

__all__ = [
    # Models
    "Gene",
    "Allele",
    "Disease",
    "Expression",
    "Phenotype",
    "CrossReference",
    "Publication",
    "Species",
    "Location",
    "Sequence",
    # Schemas and validators
    "GeneSchema",
    "AlleleSchema",
    "DiseaseSchema",
    "ExpressionSchema",
    "validate_identifier",
    "parse_identifier",
]
