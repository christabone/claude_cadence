"""Unit tests for data models.

This module tests the Pydantic models used for representing AGR data.
"""

import pytest
from pydantic import ValidationError

from agr_mcp.models import (
    Gene,
    Disease,
    Expression,
    Allele,
    Phenotype,
    CrossReference,
    Species,
    DataProvider,
)


class TestGeneModel:
    """Test Gene model validation and serialization."""

    def test_gene_minimal(self):
        """Test creating gene with minimal required fields."""
        gene = Gene(
            id="HGNC:11998",
            symbol="TP53",
            species="Homo sapiens",
        )

        assert gene.id == "HGNC:11998"
        assert gene.symbol == "TP53"
        assert gene.species == "Homo sapiens"
        assert gene.name is None
        assert gene.synonyms == []

    def test_gene_full(self, mock_gene_data):
        """Test creating gene with all fields."""
        gene = Gene(**mock_gene_data)

        assert gene.id == mock_gene_data["id"]
        assert gene.symbol == mock_gene_data["symbol"]
        assert gene.name == mock_gene_data["name"]
        assert gene.chromosome == mock_gene_data["chromosome"]
        assert len(gene.cross_references) == 2

    def test_gene_validation_error(self):
        """Test gene validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            Gene(symbol="TP53", species="Homo sapiens")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)

    def test_gene_extra_fields(self):
        """Test gene allows extra fields."""
        gene = Gene(
            id="HGNC:11998",
            symbol="TP53",
            species="Homo sapiens",
            custom_field="custom_value",
        )

        assert hasattr(gene, "custom_field")
        assert gene.custom_field == "custom_value"


class TestDiseaseModel:
    """Test Disease model validation."""

    def test_disease_creation(self, mock_disease_data):
        """Test creating disease association."""
        disease = Disease(**mock_disease_data)

        assert disease.disease_id == mock_disease_data["disease_id"]
        assert disease.disease_name == mock_disease_data["disease_name"]
        assert disease.association_type == mock_disease_data["association_type"]
        assert len(disease.evidence_codes) == 2

    def test_disease_required_fields(self):
        """Test disease required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Disease(disease_id="OMIM:123456")

        errors = exc_info.value.errors()
        required_fields = {"disease_name", "association_type"}
        error_fields = {e["loc"][0] for e in errors}
        assert required_fields.issubset(error_fields)


class TestExpressionModel:
    """Test Expression model validation."""

    def test_expression_creation(self, mock_expression_data):
        """Test creating expression data."""
        expression = Expression(**mock_expression_data)

        assert expression.gene_id == mock_expression_data["gene_id"]
        assert expression.anatomy_term == mock_expression_data["anatomy_term"]
        assert expression.stage == mock_expression_data["stage"]
        assert expression.assay_type == mock_expression_data["assay_type"]

    def test_expression_minimal(self):
        """Test expression with minimal fields."""
        expression = Expression(
            gene_id="HGNC:11998",
            anatomy_term="brain",
        )

        assert expression.gene_id == "HGNC:11998"
        assert expression.anatomy_term == "brain"
        assert expression.stage is None


class TestAlleleModel:
    """Test Allele model validation."""

    def test_allele_creation(self, mock_allele_data):
        """Test creating allele data."""
        allele = Allele(**mock_allele_data)

        assert allele.allele_id == mock_allele_data["allele_id"]
        assert allele.allele_symbol == mock_allele_data["allele_symbol"]
        assert allele.gene_symbol == mock_allele_data["gene_symbol"]
        assert len(allele.phenotypes) == 1

    def test_allele_with_phenotypes(self):
        """Test allele with multiple phenotypes."""
        allele = Allele(
            allele_id="test_allele",
            allele_symbol="test_symbol",
            gene_id="HGNC:11998",
            gene_symbol="TP53",
            phenotypes=[
                Phenotype(
                    phenotype_term="increased tumor incidence",
                    phenotype_id="MP:0002169",
                ),
                Phenotype(
                    phenotype_term="decreased body weight",
                    phenotype_id="MP:0001262",
                ),
            ],
        )

        assert len(allele.phenotypes) == 2
        assert allele.phenotypes[0].phenotype_term == "increased tumor incidence"


class TestEnums:
    """Test enum validations."""

    def test_species_enum(self):
        """Test Species enum values."""
        assert Species.HUMAN.value == "Homo sapiens"
        assert Species.MOUSE.value == "Mus musculus"
        assert Species.FLY.value == "Drosophila melanogaster"

    def test_data_provider_enum(self):
        """Test DataProvider enum values."""
        assert DataProvider.HGNC.value == "HGNC"
        assert DataProvider.MGI.value == "MGI"
        assert DataProvider.FB.value == "FB"


class TestCrossReference:
    """Test CrossReference model."""

    def test_cross_reference_creation(self):
        """Test creating cross-reference."""
        xref = CrossReference(
            source_id="HGNC:11998",
            source_db="HGNC",
            target_id="MGI:98834",
            target_db="MGI",
            relationship_type="ortholog",
            confidence=0.95,
        )

        assert xref.source_id == "HGNC:11998"
        assert xref.target_db == "MGI"
        assert xref.confidence == 0.95

    def test_cross_reference_defaults(self):
        """Test cross-reference default values."""
        xref = CrossReference(
            source_id="HGNC:11998",
            source_db="HGNC",
            target_id="MGI:98834",
            target_db="MGI",
        )

        assert xref.relationship_type == "equivalent"
        assert xref.confidence is None
