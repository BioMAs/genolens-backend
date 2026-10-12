"""
Claim referential helpers used by the CosmOmics nova: pathway-id normalization and
canonical claim derivation from the SciLicium workbooks.
"""

from app.services.cosmetics_taxonomy import (
    derive_canonical_claims,
    normalize_pathway_id,
)


def test_normalize_pathway_id_reactome_unification():
    assert normalize_pathway_id("R-HSA-69306") == "HSA-69306"
    assert normalize_pathway_id("HSA-111447") == "HSA-111447"
    assert normalize_pathway_id("REACTOME:HSA-123") == "HSA-123"


def test_normalize_pathway_id_go_and_cl_untouched():
    assert normalize_pathway_id("go:0006084") == "GO:0006084"
    assert normalize_pathway_id("CL:9273") == "CL:9273"
    assert normalize_pathway_id(None) == ""


def test_normalize_pathway_id_extracts_embedded_accession():
    # Enrichr/GSEA-style labels with the accession embedded
    assert normalize_pathway_id("protein binding (GO:0005515)") == "GO:0005515"
    assert normalize_pathway_id("ISG15-protein conjugation (GO:0032020)") == "GO:0032020"
    assert normalize_pathway_id("Some pathway (R-HSA-69306)") == "HSA-69306"
    # Gene-set names without an accession fall back (won't false-match referential)
    assert normalize_pathway_id("INTERFERON_SIGNALING") == "INTERFERON_SIGNALING"
    assert "GO:" not in normalize_pathway_id("hsa00100 Steroid biosynthesis")


def test_derive_canonical_claims_from_text():
    assert "anti-ageing" in derive_canonical_claims("Anti-ageing, Energising", None)
    assert "firming" in derive_canonical_claims("Firming, Elasticity", "ECM_CYTO")
    assert derive_canonical_claims("Anti-wrinkles", None) == ["anti-wrinkles"]


def test_derive_canonical_claims_category_fallback():
    # "Unknown" text -> falls back to category mapping
    assert derive_canonical_claims("Unknown", "BARRIER_LIPID") == ["moisturising"]
    assert derive_canonical_claims(None, "REDOX_ANTIOX") == ["antioxidant"]
