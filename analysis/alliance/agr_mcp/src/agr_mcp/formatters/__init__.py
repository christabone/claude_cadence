"""Output formatters for AGR data.

This package provides functions to format AGR data models into
various output formats (JSON, text, TSV, etc.).
"""

from typing import Any, List, Union

from ..models import Gene, Disease, Expression, Allele


def format_gene(gene: Gene, format: str = "text") -> str:
    """Format gene data for output.

    Args:
        gene: Gene object
        format: Output format (text, summary)

    Returns:
        Formatted gene data
    """
    if format == "summary":
        return (
            f"{gene.symbol} ({gene.id}) - {gene.name or 'No description'}\n"
            f"Species: {gene.species}"
        )

    # Full text format
    lines = [
        f"Gene: {gene.symbol} ({gene.id})",
        f"Name: {gene.name or 'N/A'}",
        f"Species: {gene.species}",
        f"Type: {gene.gene_type or 'N/A'}",
    ]

    if gene.chromosome:
        lines.append(
            f"Location: {gene.chromosome}:{gene.start_position}-{gene.end_position}"
        )

    if gene.synonyms:
        lines.append(f"Synonyms: {', '.join(gene.synonyms)}")

    if gene.orthologs:
        lines.append(f"\nOrthologs ({len(gene.orthologs)}):")
        for orth in gene.orthologs[:5]:  # Show first 5
            lines.append(
                f"  - {orth.gene_symbol} ({orth.species}): {orth.orthology_type or 'ortholog'}"
            )
        if len(gene.orthologs) > 5:
            lines.append(f"  ... and {len(gene.orthologs) - 5} more")

    if gene.disease_associations:
        lines.append(f"\nDisease Associations ({len(gene.disease_associations)}):")
        for disease in gene.disease_associations[:5]:
            lines.append(f"  - {disease.disease_name} ({disease.disease_id})")
        if len(gene.disease_associations) > 5:
            lines.append(f"  ... and {len(gene.disease_associations) - 5} more")

    return "\n".join(lines)


def format_disease(disease: Disease, format: str = "text") -> str:
    """Format disease association data.

    Args:
        disease: Disease object
        format: Output format

    Returns:
        Formatted disease data
    """
    lines = [
        f"Disease: {disease.disease_name} ({disease.disease_id})",
    ]

    if disease.gene_symbol:
        lines.append(f"Gene: {disease.gene_symbol} ({disease.gene_id})")

    lines.append(f"Association Type: {disease.association_type}")

    if disease.evidence_codes:
        lines.append(f"Evidence: {', '.join(disease.evidence_codes)}")

    if disease.inheritance_mode:
        lines.append(f"Inheritance: {disease.inheritance_mode}")

    if disease.phenotypes:
        lines.append(f"Phenotypes: {', '.join(disease.phenotypes[:3])}")
        if len(disease.phenotypes) > 3:
            lines.append(f"  ... and {len(disease.phenotypes) - 3} more")

    return "\n".join(lines)


def format_expression(expression: Expression, format: str = "text") -> str:
    """Format expression data.

    Args:
        expression: Expression object
        format: Output format

    Returns:
        Formatted expression data
    """
    lines = [
        f"Expression in: {expression.anatomy_term}",
    ]

    if expression.stage:
        lines.append(f"Stage: {expression.stage}")

    if expression.expression_pattern:
        lines.append(f"Pattern: {expression.expression_pattern}")

    if expression.expression_level:
        lines.append(f"Level: {expression.expression_level}")

    if expression.assay_type:
        lines.append(f"Assay: {expression.assay_type}")

    return "\n".join(lines)


def format_allele(allele: Allele, format: str = "text") -> str:
    """Format allele data.

    Args:
        allele: Allele object
        format: Output format

    Returns:
        Formatted allele data
    """
    lines = [
        f"Allele: {allele.allele_symbol} ({allele.allele_id})",
        f"Gene: {allele.gene_symbol} ({allele.gene_id})",
    ]

    if allele.allele_name:
        lines.append(f"Name: {allele.allele_name}")

    if allele.allele_type:
        lines.append(f"Type: {allele.allele_type}")

    if allele.molecular_mutation:
        lines.append(f"Mutation: {allele.molecular_mutation}")

    if allele.generation_method:
        lines.append(f"Generation: {allele.generation_method}")

    if allele.phenotypes:
        lines.append(f"\nPhenotypes ({len(allele.phenotypes)}):")
        for pheno in allele.phenotypes[:3]:
            lines.append(f"  - {pheno.phenotype_term}")
        if len(allele.phenotypes) > 3:
            lines.append(f"  ... and {len(allele.phenotypes) - 3} more")

    return "\n".join(lines)
