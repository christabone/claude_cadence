import re
import os
from typing import List, Optional, Union, Dict, Any

from ..errors import ValidationError
from .logging_config import get_logger

logger = get_logger('validators')

# Regular expressions for validation
GENE_ID_PATTERN = re.compile(r'^([A-Za-z]+):([\w\d]+)$')  # e.g., MGI:123456, FB:FBgn0000001
FILE_TYPE_PATTERN = re.compile(r'^[\w\-\.]+$')  # Alphanumeric with dash, underscore, dot

# Valid model organism prefixes
VALID_ORGANISM_PREFIXES = [
    'MGI',  # Mouse
    'RGD',  # Rat
    'SGD',  # Yeast
    'WB',   # C. elegans (worm)
    'ZFIN', # Zebrafish
    'XB',   # Xenopus (frog)
    'FB',   # Fly
]

# Valid file types
VALID_FILE_TYPES = [
    'gff3',
    'tab',
    'csv',
    'json',
    'fasta',
    'obo',
    'owl',
    'txt',
    'gz',
    'zip',
]

def validate_gene_id(gene_id: str) -> str:
    """Validate a gene ID format.

    Args:
        gene_id: Gene ID to validate (e.g., MGI:123456)

    Returns:
        The validated gene ID

    Raises:
        ValidationError: If gene ID format is invalid
    """
    if not gene_id or not isinstance(gene_id, str):
        raise ValidationError(
            message="Gene ID must be a non-empty string",
            details={"provided": gene_id}
        )

    match = GENE_ID_PATTERN.match(gene_id)
    if not match:
        raise ValidationError(
            message="Invalid gene ID format. Expected format: PREFIX:ID (e.g., MGI:123456)",
            details={"provided": gene_id, "pattern": GENE_ID_PATTERN.pattern}
        )

    prefix = match.group(1)
    if prefix not in VALID_ORGANISM_PREFIXES:
        raise ValidationError(
            message=f"Invalid organism prefix: {prefix}",
            details={
                "provided": prefix,
                "valid_prefixes": VALID_ORGANISM_PREFIXES
            }
        )

    return gene_id

def validate_file_type(file_type: str) -> str:
    """Validate a file type.

    Args:
        file_type: File type to validate (e.g., gff3, tab)

    Returns:
        The validated file type

    Raises:
        ValidationError: If file type is invalid
    """
    if not file_type or not isinstance(file_type, str):
        raise ValidationError(
            message="File type must be a non-empty string",
            details={"provided": file_type}
        )

    if not FILE_TYPE_PATTERN.match(file_type):
        raise ValidationError(
            message="Invalid file type format",
            details={"provided": file_type, "pattern": FILE_TYPE_PATTERN.pattern}
        )

    file_type = file_type.lower()
    if file_type not in VALID_FILE_TYPES:
        raise ValidationError(
            message=f"Unsupported file type: {file_type}",
            details={
                "provided": file_type,
                "valid_types": VALID_FILE_TYPES
            }
        )

    return file_type

def validate_output_directory(directory: Optional[str] = None) -> str:
    """Validate and create output directory if it doesn't exist.

    Args:
        directory: Directory path to validate and create

    Returns:
        The validated directory path

    Raises:
        ValidationError: If directory is invalid or cannot be created
    """
    from ..config import Config

    if not directory:
        directory = Config.DEFAULT_DOWNLOAD_DIR

    try:
        os.makedirs(directory, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise ValidationError(
            message=f"Cannot create output directory: {str(e)}",
            details={"directory": directory, "error": str(e)}
        )

    if not os.access(directory, os.W_OK):
        raise ValidationError(
            message="Output directory is not writable",
            details={"directory": directory}
        )

    return directory

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Replace potentially dangerous characters
    filename = re.sub(r'[^\w\-\. ]', '_', filename)

    # Ensure filename is not empty
    if not filename:
        filename = "download"

    return filename
