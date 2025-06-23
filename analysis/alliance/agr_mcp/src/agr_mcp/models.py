"""Data models for AGR MCP server.

This module defines Pydantic models for various Alliance Genome Resource
data types, ensuring type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class Species(str, Enum):
    """Supported model organism species."""

    HUMAN = "Homo sapiens"
    MOUSE = "Mus musculus"
    RAT = "Rattus norvegicus"
    ZEBRAFISH = "Danio rerio"
    FLY = "Drosophila melanogaster"
    WORM = "Caenorhabditis elegans"
    YEAST = "Saccharomyces cerevisiae"
    XENOPUS = "Xenopus laevis"


class DataProvider(str, Enum):
    """Alliance member database providers."""

    HGNC = "HGNC"  # Human
    MGI = "MGI"    # Mouse
    RGD = "RGD"    # Rat
    ZFIN = "ZFIN"  # Zebrafish
    FB = "FB"      # FlyBase
    WB = "WB"      # WormBase
    SGD = "SGD"    # Yeast
    XB = "XB"      # XenBase


class Gene(BaseModel):
    """Gene data model.

    Represents a gene with its basic information and optional
    related data like orthologs, expression, and disease associations.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique gene identifier")
    symbol: str = Field(..., description="Gene symbol")
    name: Optional[str] = Field(None, description="Gene name/description")
    species: str = Field(..., description="Species name")
    taxon_id: Optional[str] = Field(None, description="NCBI taxonomy ID")

    # Basic information
    gene_type: Optional[str] = Field(None, description="Type of gene")
    synonyms: List[str] = Field(default_factory=list, description="Alternative symbols")

    # Location information
    chromosome: Optional[str] = Field(None, description="Chromosome location")
    start_position: Optional[int] = Field(None, description="Start position")
    end_position: Optional[int] = Field(None, description="End position")
    strand: Optional[str] = Field(None, description="DNA strand (+/-)")

    # Cross-references
    cross_references: List[Dict[str, str]] = Field(
        default_factory=list,
        description="External database references"
    )

    # Optional related data
    orthologs: Optional[List["Ortholog"]] = Field(None, description="Orthologous genes")
    expression_data: Optional[List["Expression"]] = Field(None, description="Expression data")
    disease_associations: Optional[List["Disease"]] = Field(None, description="Disease associations")

    # Metadata
    date_created: Optional[datetime] = Field(None, description="Creation date")
    date_updated: Optional[datetime] = Field(None, description="Last update date")


class Ortholog(BaseModel):
    """Ortholog relationship model."""

    gene_id: str = Field(..., description="Ortholog gene identifier")
    gene_symbol: str = Field(..., description="Ortholog gene symbol")
    species: str = Field(..., description="Ortholog species")

    # Orthology information
    orthology_type: Optional[str] = Field(None, description="Type of orthology")
    confidence: Optional[str] = Field(None, description="Confidence level")
    methods: List[str] = Field(default_factory=list, description="Detection methods")


class Disease(BaseModel):
    """Disease association model."""

    model_config = ConfigDict(extra="allow")

    disease_id: str = Field(..., description="Disease identifier (e.g., OMIM:168600)")
    disease_name: str = Field(..., description="Disease name")

    # Association details
    gene_id: Optional[str] = Field(None, description="Associated gene identifier")
    gene_symbol: Optional[str] = Field(None, description="Associated gene symbol")
    association_type: str = Field(..., description="Type of association")
    evidence_codes: List[str] = Field(default_factory=list, description="Evidence codes")

    # Evidence
    publications: List[str] = Field(default_factory=list, description="Supporting publications")
    confidence_level: Optional[str] = Field(None, description="Confidence level")

    # Additional information
    phenotypes: List[str] = Field(default_factory=list, description="Associated phenotypes")
    inheritance_mode: Optional[str] = Field(None, description="Mode of inheritance")


class Expression(BaseModel):
    """Gene expression data model."""

    model_config = ConfigDict(extra="allow")

    gene_id: str = Field(..., description="Gene identifier")

    # Expression location
    anatomy_term: str = Field(..., description="Anatomical structure")
    anatomy_id: Optional[str] = Field(None, description="Anatomy ontology ID")

    # Expression timing
    stage: Optional[str] = Field(None, description="Developmental stage")
    stage_id: Optional[str] = Field(None, description="Stage ontology ID")

    # Expression details
    expression_pattern: Optional[str] = Field(None, description="Expression pattern description")
    expression_level: Optional[str] = Field(None, description="Expression level")
    assay_type: Optional[str] = Field(None, description="Assay type used")

    # Evidence
    publications: List[str] = Field(default_factory=list, description="Supporting publications")
    images: List[Dict[str, str]] = Field(default_factory=list, description="Expression images")


class Allele(BaseModel):
    """Allele/variant data model."""

    model_config = ConfigDict(extra="allow")

    allele_id: str = Field(..., description="Allele identifier")
    allele_symbol: str = Field(..., description="Allele symbol")
    allele_name: Optional[str] = Field(None, description="Allele name/description")

    # Associated gene
    gene_id: str = Field(..., description="Associated gene identifier")
    gene_symbol: str = Field(..., description="Associated gene symbol")

    # Allele details
    allele_type: Optional[str] = Field(None, description="Type of allele")
    molecular_mutation: Optional[str] = Field(None, description="Molecular change")

    # Phenotype information
    phenotypes: List["Phenotype"] = Field(default_factory=list, description="Associated phenotypes")

    # Generation method
    generation_method: Optional[str] = Field(None, description="How allele was generated")

    # References
    publications: List[str] = Field(default_factory=list, description="Supporting publications")


class Phenotype(BaseModel):
    """Phenotype data model."""

    phenotype_term: str = Field(..., description="Phenotype term")
    phenotype_id: Optional[str] = Field(None, description="Phenotype ontology ID")

    # Phenotype details
    phenotype_statement: Optional[str] = Field(None, description="Full phenotype description")
    evidence_code: Optional[str] = Field(None, description="Evidence code")

    # Conditions
    conditions: List[str] = Field(default_factory=list, description="Experimental conditions")

    # References
    publications: List[str] = Field(default_factory=list, description="Supporting publications")


class CrossReference(BaseModel):
    """Cross-reference/identifier mapping model."""

    source_id: str = Field(..., description="Source identifier")
    source_db: str = Field(..., description="Source database")

    target_id: str = Field(..., description="Target identifier")
    target_db: str = Field(..., description="Target database")

    # Mapping details
    relationship_type: Optional[str] = Field("equivalent", description="Type of relationship")
    confidence: Optional[float] = Field(None, description="Mapping confidence score")


class Publication(BaseModel):
    """Publication reference model."""

    pubmed_id: Optional[str] = Field(None, description="PubMed ID")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")

    # Publication details
    title: str = Field(..., description="Publication title")
    authors: List[str] = Field(default_factory=list, description="Author list")
    journal: Optional[str] = Field(None, description="Journal name")
    year: Optional[int] = Field(None, description="Publication year")

    # Abstract
    abstract: Optional[str] = Field(None, description="Publication abstract")

    # Related entities
    genes: List[str] = Field(default_factory=list, description="Associated genes")
    alleles: List[str] = Field(default_factory=list, description="Associated alleles")
    diseases: List[str] = Field(default_factory=list, description="Associated diseases")


# Update forward references
Gene.model_rebuild()
Allele.model_rebuild()
