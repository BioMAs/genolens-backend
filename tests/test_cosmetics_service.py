"""
Unit tests for the Cosmetics module scoring + taxonomy helpers.
No real database required (db.execute is mocked).
"""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.models import ClaimDirection, EvidenceLevel
from app.services.cosmetics_service import score_claims
from app.services.cosmetics_taxonomy import (
    derive_canonical_claims,
    normalize_pathway_id,
)


# ── Pure helpers ─────────────────────────────────────────────────────────────

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


# ── score_claims with a mocked DB ────────────────────────────────────────────

def _enr(pathway_id, regulation, padj=1e-4, category="ECM_CYTO", genes=None, name="P",
         comparison_name="cmp", pvalue=1e-4):
    row = MagicMock()
    row.pathway_id = pathway_id
    row.regulation = regulation
    row.padj = padj
    row.pvalue = pvalue
    row.category = category
    row.genes = genes or ["COL1A1", "ELN"]
    row.pathway_name = name
    row.comparison_name = comparison_name
    return row


def _mapping(term_id, direction, claims, evidence=EvidenceLevel.HIGH, category="ECM_CYTO"):
    m = MagicMock()
    m.term_id = term_id
    m.term_id_normalized = normalize_pathway_id(term_id)
    m.updated_direction = direction
    m.evidence_level = evidence
    m.canonical_claims = claims
    m.category = category
    m.ref_cat = None
    m.caveats = None
    m.rationale = None
    return m


class _Result:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        s = MagicMock()
        s.all.return_value = self._items
        return s


@pytest.mark.asyncio
async def test_score_claims_favorable_match():
    dataset_id = uuid4()
    enr_rows = [
        _enr("GO:0030198", "UP", name="ECM organization"),
        _enr("GO:0030199", "UP", name="Collagen fibril organization"),
    ]
    mappings = [
        _mapping("GO:0030198", ClaimDirection.UP, ["firming"]),
        _mapping("GO:0030199", ClaimDirection.UP, ["firming"]),
    ]

    db = MagicMock()

    async def fake_execute(stmt):
        # 1st call: enrichment, 2nd: mappings, 3rd: references (no ref_cats -> skipped)
        fake_execute.calls += 1
        if fake_execute.calls == 1:
            return _Result(enr_rows)
        return _Result(mappings)

    fake_execute.calls = 0
    db.execute = fake_execute

    out = await score_claims(db, dataset_id, "Treated_vs_Control")

    firming = next(c for c in out["claims"] if c["slug"] == "firming")
    assert firming["n_supporting"] == 2
    assert firming["n_contradicting"] == 0
    assert firming["direction"] == "favorable"
    assert firming["score"] > 0
    assert firming["confidence"] == "HIGH"
    assert out["coverage"]["n_matched"] == 2


@pytest.mark.asyncio
async def test_score_claims_direction_from_comparison_suffix():
    """annoDB ENRICHMENT pipeline encodes direction in the comparison suffix."""
    dataset_id = uuid4()
    enr_rows = [
        _enr("GO:0030198", "ALL", category="ECM_CYTO", name="ECM organization",
             comparison_name="UVB_cream_vs_UVB_ctrl (up)"),
    ]
    mappings = [_mapping("GO:0030198", ClaimDirection.UP, ["firming"])]

    db = MagicMock()

    async def fake_execute(stmt):
        fake_execute.calls += 1
        return _Result(enr_rows if fake_execute.calls == 1 else mappings)

    fake_execute.calls = 0
    db.execute = fake_execute

    out = await score_claims(db, dataset_id, "UVB_cream_vs_UVB_ctrl")
    firming = next(c for c in out["claims"] if c["slug"] == "firming")
    assert firming["n_supporting"] == 1
    assert firming["score"] > 0
    assert out["coverage"]["significance_basis"] == "fdr"


@pytest.mark.asyncio
async def test_score_claims_contradiction_lowers_score():
    dataset_id = uuid4()
    # Pathway up-regulated but claim expects DOWN -> contradiction
    enr_rows = [_enr("GO:0006954", "UP", category="IMMUNE_INFLAM", name="Inflammatory response")]
    mappings = [
        _mapping("GO:0006954", ClaimDirection.DOWN, ["soothing"], category="IMMUNE_INFLAM")
    ]

    db = MagicMock()

    async def fake_execute(stmt):
        fake_execute.calls += 1
        return _Result(enr_rows if fake_execute.calls == 1 else mappings)

    fake_execute.calls = 0
    db.execute = fake_execute

    out = await score_claims(db, dataset_id, "cmp")
    soothing = next(c for c in out["claims"] if c["slug"] == "soothing")
    assert soothing["n_contradicting"] == 1
    assert soothing["n_supporting"] == 0
    assert soothing["score"] == 0.0
