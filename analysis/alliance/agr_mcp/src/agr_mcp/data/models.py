"""
Data models for Alliance Genome Resource entities.

This module defines Pydantic models representing various biological
entities and their relationships in the Alliance data model.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class Species(str, Enum):
    """Supported model organism species."""
    HUMAN = "NCBITaxon:9606"
    MOUSE = "NCBITaxon:10090"
    RAT = "NCBITaxon:10116"
    ZEBRAFISH = "NCBITaxon:7955"
    FLY = "NCBITaxon:7227"
    WORM = "NCBITaxon:6239"
    YEAST = "NCBITaxon:559292"
    XENOPUS = "NCBITaxon:8364"


class Database(str, Enum):
    """Alliance member databases."""
    HGNC = "HGNC"
    MGI = "MGI"
    RGD = "RGD"
    ZFIN = "ZFIN"
    FB = "FB"
    WB = "WB"
    SGD = "SGD"
    XENBASE = "XENBASE"


class Location(BaseModel):
    """Genomic location information."""
    model_config = ConfigDict(populate_by_name=True)

    chromosome: str = Field(..., description="Chromosome name")
    start: int = Field(..., description="Start position (1-based)")
    end: int = Field(..., description="End position (1-based)")
    strand: Optional[str] = Field(None, pattern="^[+-]?$", description="Strand orientation")
    assembly: Optional[str] = Field(None, description="Genome assembly version")


class CrossReference(BaseModel):
    """Cross-reference to external database."""
    model_config = ConfigDict(populate_by_name=True)

    database: str = Field(..., description="Database name")
    identifier: str = Field(..., description="Database-specific identifier")
    url: Optional[str] = Field(None, description="URL to external resource")


class Publication(BaseModel):
    """Publication reference."""
    model_config = ConfigDict(populate_by_name=True)

    pubmed_id: Optional[str] = Field(None, alias="pubmedId")
    doi: Optional[str] = Field(None)
    title: Optional[str] = Field(None)
    authors: Optional[List[str]] = Field(default_factory=list)
    year: Optional[int] = Field(None)
    journal: Optional[str] = Field(None)


class Gene(BaseModel):
    """Gene entity model."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Primary gene identifier")
    symbol: str = Field(..., description="Gene symbol")
    name: Optional[str] = Field(None, description="Full gene name")
    species: Species = Field(..., description="Species taxonomy ID")
    database: Database = Field(..., description="Source database")

    synonyms: List[str] = Field(default_factory=list, description="Alternative symbols")
    description: Optional[str] = Field(None, description="Gene description")
    gene_type: Optional[str] = Field(None, alias="geneType", description="Gene biotype")

    location: Optional[Location] = Field(None, description="Genomic location")
    cross_references: List[CrossReference] = Field(
        default_factory=list,
        alias="crossReferences",
        description="External database references"
    )

    # Relationships
    orthologs: List[str] = Field(default_factory=list, description="Orthologous gene IDs")
    paralogs: List[str] = Field(default_factory=list, description="Paralogous gene IDs")

    # Metadata
    created_date: Optional[datetime] = Field(None, alias="createdDate")
    updated_date: Optional[datetime] = Field(None, alias="updatedDate")


class Allele(BaseModel):
    """Allele entity model."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Primary allele identifier")
    symbol: str = Field(..., description="Allele symbol")
    gene_id: str = Field(..., alias="geneId", description="Associated gene ID")

    name: Optional[str] = Field(None, description="Full allele name")
    description: Optional[str] = Field(None, description="Allele description")
    allele_type: Optional[str] = Field(None, alias="alleleType")

    synonyms: List[str] = Field(default_factory=list)
    cross_references: List[CrossReference] = Field(
        default_factory=list,
        alias="crossReferences"
    )

    # Molecular details
    molecular_consequence: Optional[str] = Field(None, alias="molecularConsequence")
    mutation_type: Optional[str] = Field(None, alias="mutationType")

    # Associated data
    phenotypes: List[str] = Field(default_factory=list, description="Associated phenotype IDs")
    diseases: List[str] = Field(default_factory=list, description="Associated disease IDs")
    publications: List[Publication] = Field(default_factory=list)


class Disease(BaseModel):
    """Disease entity model."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Disease identifier (e.g., OMIM ID)")
    name: str = Field(..., description="Disease name")

    synonyms: List[str] = Field(default_factory=list)
    definition: Optional[str] = Field(None, description="Disease definition")

    # Associations
    associated_genes: List[str] = Field(
        default_factory=list,
        alias="associatedGenes",
        description="Gene IDs associated with disease"
    )
    associated_alleles: List[str] = Field(
        default_factory=list,
        alias="associatedAlleles",
        description="Allele IDs associated with disease"
    )

    cross_references: List[CrossReference] = Field(
        default_factory=list,
        alias="crossReferences"
    )
    publications: List[Publication] = Field(default_factory=list)


class Expression(BaseModel):
    """Gene expression data model."""
    model_config = ConfigDict(populate_by_name=True)

    gene_id: str = Field(..., alias="geneId", description="Gene identifier")

    # Expression location
    anatomical_structure: Optional[str] = Field(
        None,
        alias="anatomicalStructure",
        description="Anatomical structure where expressed"
    )
    developmental_stage: Optional[str] = Field(
        None,
        alias="developmentalStage",
        description="Developmental stage of expression"
    )
    cell_type: Optional[str] = Field(None, alias="cellType")

    # Expression details
    expression_level: Optional[str] = Field(None, alias="expressionLevel")
    pattern: Optional[str] = Field(None, description="Expression pattern")

    # Evidence
    assay: Optional[str] = Field(None, description="Assay type used")
    publications: List[Publication] = Field(default_factory=list)

    # Metadata
    evidence_code: Optional[str] = Field(None, alias="evidenceCode")
    qualifiers: List[str] = Field(default_factory=list)


class Phenotype(BaseModel):
    """Phenotype entity model."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Phenotype term ID")
    name: str = Field(..., description="Phenotype name")

    definition: Optional[str] = Field(None)
    synonyms: List[str] = Field(default_factory=list)

    # Associations
    associated_genes: List[str] = Field(
        default_factory=list,
        alias="associatedGenes"
    )
    associated_alleles: List[str] = Field(
        default_factory=list,
        alias="associatedAlleles"
    )

    # Ontology information
    namespace: Optional[str] = Field(None, description="Ontology namespace")
    parents: List[str] = Field(default_factory=list, description="Parent term IDs")
    children: List[str] = Field(default_factory=list, description="Child term IDs")


class Sequence(BaseModel):
    """Sequence information model."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Sequence identifier")
    sequence_type: str = Field(..., alias="sequenceType")

    length: Optional[int] = Field(None, description="Sequence length")
    md5_checksum: Optional[str] = Field(None, alias="md5Checksum")

    # For smaller sequences
    sequence: Optional[str] = Field(None, description="Actual sequence (if small)")

    # For larger sequences
    download_url: Optional[str] = Field(None, alias="downloadUrl")
    file_format: Optional[str] = Field(None, alias="fileFormat")
