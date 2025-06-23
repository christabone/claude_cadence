"""
Input validation utilities for the AGR MCP server.

This module provides functions for validating gene identifiers,
species codes, and other Alliance-specific data formats.
"""

import re
from typing import Tuple, Optional, List

from ..data.models import Species, Database


class ValidationError(ValueError):
    """Raised when validation fails."""
    pass


# Identifier patterns for each database
IDENTIFIER_PATTERNS = {
    Database.HGNC: re.compile(r"^HGNC:\d+$"),
    Database.MGI: re.compile(r"^MGI:\d+$"),
    Database.RGD: re.compile(r"^RGD:\d+$"),
    Database.ZFIN: re.compile(r"^ZFIN:ZDB-GENE-\d{6}-\d+$"),
    Database.FB: re.compile(r"^FB:FBgn\d{7}$"),
    Database.WB: re.compile(r"^WB:WBGene\d{8}$"),
    Database.SGD: re.compile(r"^SGD:S\d{9}$"),
    Database.XENBASE: re.compile(r"^XENBASE:XB-GENE-\d+$"),
}

# Valid species taxonomy IDs
VALID_SPECIES = {
    species.value: species for species in Species
}

# Map database to species
DATABASE_SPECIES_MAP = {
    Database.HGNC: Species.HUMAN,
    Database.MGI: Species.MOUSE,
    Database.RGD: Species.RAT,
    Database.ZFIN: Species.ZEBRAFISH,
    Database.FB: Species.FLY,
    Database.WB: Species.WORM,
    Database.SGD: Species.YEAST,
    Database.XENBASE: Species.XENOPUS,
}


def validate_gene_id(gene_id: str) -> Tuple[bool, Optional[Database], Optional[str]]:
    """
    Validate a gene identifier format.

    Args:
        gene_id: Gene identifier to validate

    Returns:
        Tuple of (is_valid, database, error_message)

    Examples:
        >>> validate_gene_id("HGNC:11998")
        (True, Database.HGNC, None)

        >>> validate_gene_id("invalid:123")
        (False, None, "Unknown database prefix: invalid")
    """
    if not gene_id or not isinstance(gene_id, str):
        return False, None, "Gene ID must be a non-empty string"

    # Check if it contains a colon
    if ":" not in gene_id:
        return False, None, "Gene ID must contain a colon separator"

    # Split on first colon
    prefix, local_id = gene_id.split(":", 1)

    # Check if prefix matches any known database
    for db, pattern in IDENTIFIER_PATTERNS.items():
        if pattern.match(gene_id):
            return True, db, None

    # If no match, try to provide helpful error
    known_prefixes = [db.value for db in Database]
    if prefix not in known_prefixes:
        return False, None, f"Unknown database prefix: {prefix}. Valid prefixes: {', '.join(known_prefixes)}"

    # Prefix is valid but format is wrong
    return False, None, f"Invalid {prefix} identifier format"


def validate_species(species_id: str) -> Tuple[bool, Optional[Species], Optional[str]]:
    """
    Validate a species taxonomy ID.

    Args:
        species_id: NCBITaxon ID to validate

    Returns:
        Tuple of (is_valid, species, error_message)

    Examples:
        >>> validate_species("NCBITaxon:9606")
        (True, Species.HUMAN, None)

        >>> validate_species("9606")
        (False, None, "Species ID must start with 'NCBITaxon:'")
    """
    if not species_id or not isinstance(species_id, str):
        return False, None, "Species ID must be a non-empty string"

    if not species_id.startswith("NCBITaxon:"):
        return False, None, "Species ID must start with 'NCBITaxon:'"

    if species_id in VALID_SPECIES:
        return True, VALID_SPECIES[species_id], None

    # Provide list of valid species
    valid_list = [f"{s.value} ({s.name.lower()})" for s in Species]
    return False, None, f"Unknown species ID. Valid species: {', '.join(valid_list)}"


def validate_database(database: str) -> Tuple[bool, Optional[Database], Optional[str]]:
    """
    Validate a database name.

    Args:
        database: Database name to validate

    Returns:
        Tuple of (is_valid, database_enum, error_message)

    Examples:
        >>> validate_database("HGNC")
        (True, Database.HGNC, None)

        >>> validate_database("InvalidDB")
        (False, None, "Unknown database: InvalidDB")
    """
    if not database or not isinstance(database, str):
        return False, None, "Database must be a non-empty string"

    # Try exact match
    try:
        db_enum = Database(database.upper())
        return True, db_enum, None
    except ValueError:
        pass

    # Try case-insensitive match
    for db in Database:
        if db.value.upper() == database.upper():
            return True, db, None

    # Provide list of valid databases
    valid_list = [db.value for db in Database]
    return False, None, f"Unknown database: {database}. Valid databases: {', '.join(valid_list)}"


def validate_identifier_consistency(gene_id: str, species_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that a gene identifier is consistent with the species.

    Args:
        gene_id: Gene identifier
        species_id: Optional species taxonomy ID

    Returns:
        Tuple of (is_consistent, error_message)
    """
    # First validate the gene ID
    is_valid, database, error = validate_gene_id(gene_id)
    if not is_valid:
        return False, error

    # If no species specified, always consistent
    if not species_id:
        return True, None

    # Validate species
    is_valid, species, error = validate_species(species_id)
    if not is_valid:
        return False, error

    # Check consistency
    expected_species = DATABASE_SPECIES_MAP.get(database)
    if expected_species and species != expected_species:
        return False, f"{database.value} identifiers are for {expected_species.name.lower()}, not {species.name.lower()}"

    return True, None


def validate_batch_identifiers(identifiers: List[str]) -> Tuple[bool, List[Tuple[str, Optional[str]]]]:
    """
    Validate a batch of identifiers.

    Args:
        identifiers: List of identifiers to validate

    Returns:
        Tuple of (all_valid, list of (identifier, error_message) tuples)
    """
    if not identifiers:
        return False, [("", "No identifiers provided")]

    if not isinstance(identifiers, list):
        return False, [("", "Identifiers must be provided as a list")]

    results = []
    all_valid = True

    for identifier in identifiers:
        is_valid, _, error = validate_gene_id(identifier)
        if not is_valid:
            all_valid = False
            results.append((identifier, error))
        else:
            results.append((identifier, None))

    return all_valid, results


def sanitize_query_string(query: str, max_length: int = 200) -> str:
    """
    Sanitize a search query string.

    Args:
        query: Query string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized query string

    Raises:
        ValidationError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValidationError("Query must be a non-empty string")

    # Strip whitespace
    query = query.strip()

    # Check length
    if len(query) > max_length:
        raise ValidationError(f"Query too long (max {max_length} characters)")

    # Remove potentially harmful characters
    # Allow alphanumeric, spaces, hyphens, underscores, and common punctuation
    allowed_pattern = re.compile(r"^[a-zA-Z0-9\s\-_.,;:()'\"/]+$")
    if not allowed_pattern.match(query):
        raise ValidationError("Query contains invalid characters")

    return query
