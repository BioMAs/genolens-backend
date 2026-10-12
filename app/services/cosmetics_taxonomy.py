"""
Cosmetic claims — referential helpers shared by the CosmOmics nova.

  - the pathway-id normalization used to join the claim referential against
    EnrichmentPathway rows,
  - derivation of canonical claim slugs from the free-text `Original_claims` of the
    SciLicium workbooks (with the referential `Category` as fallback).

The slugs map to the claim names shown on the nova in `cosmomics_nova.CLAIM_NAMES`.
"""

from __future__ import annotations

import re
from typing import Optional

# Ontology accessions can be embedded anywhere in an enrichment pathway id, e.g.
# Enrichr-style "protein binding (GO:0005515)" or "name (R-HSA-69306)".
_GO_RE = re.compile(r"GO:(\d{5,7})", re.IGNORECASE)
_REACTOME_RE = re.compile(r"R-HSA-(\d+)", re.IGNORECASE)
_HSA_RE = re.compile(r"(?<![A-Za-z0-9])HSA-(\d+)", re.IGNORECASE)
_CL_RE = re.compile(r"(?<![A-Za-z0-9])CL:(\d+)", re.IGNORECASE)

# Referential Category -> fallback canonical claim, used when Original_claims is
# empty/"Unknown" so a curated row still contributes.
CATEGORY_TO_CLAIM: dict[str, str] = {
    "BARRIER_LIPID": "moisturising",
    "ECM_CYTO": "firming",
    "BARRIER_ECM": "firming",
    "IMMUNE_INFLAM": "soothing",
    "EICOSANOID": "soothing",
    "REDOX_ANTIOX": "antioxidant",
    "ENERGY_METAB": "energising",
    "PROTEOSTASIS": "anti-ageing",
    "GENOME_STABILITY": "anti-ageing",
    "AGING": "anti-ageing",
    "CELL_DEATH": "anti-ageing",
    "TRAFFICKING": "regenerating",
    "GENE_EXPRESSION": "regenerating",
    "CELL_CYCLE": "regenerating",
    "GROWTH_SIGNAL": "regenerating",
    "MTOR_SIGNAL": "regenerating",
}

# Ordered keyword -> canonical slug. First matching keyword wins per fragment.
# Order matters (most specific first).
_CLAIM_KEYWORDS: list[tuple[str, str]] = [
    ("anti-wrinkle", "anti-wrinkles"),
    ("wrinkle", "anti-wrinkles"),
    ("anti-ageing", "anti-ageing"),
    ("anti-aging", "anti-ageing"),
    ("ageing", "anti-ageing"),
    ("aging", "anti-ageing"),
    ("anti-oxidant", "antioxidant"),
    ("antioxidant", "antioxidant"),
    ("detox", "antioxidant"),
    ("redox", "antioxidant"),
    ("moistur", "moisturising"),
    ("hydrat", "moisturising"),
    ("barrier", "moisturising"),
    ("anti-redness", "soothing"),
    ("redness", "soothing"),
    ("anti-inflammator", "soothing"),
    ("inflammat", "soothing"),
    ("anti-irritation", "soothing"),
    ("irritation", "soothing"),
    ("sooth", "soothing"),
    ("calm", "soothing"),
    ("firm", "firming"),
    ("elasticity", "firming"),
    ("tissue integrity", "firming"),
    ("tissue remodel", "firming"),
    ("remodel", "firming"),
    ("heal", "healing"),
    ("regenerat", "regenerating"),
    ("renew", "regenerating"),
    ("turnover", "regenerating"),
    ("energ", "energising"),
]


def normalize_pathway_id(pid: Optional[str]) -> str:
    """Normalize a pathway/term id so the claim referential joins robustly
    against the enrichment results.

    Enrichment ids are often Enrichr/GSEA-style with the accession embedded in
    the label, e.g. "protein binding (GO:0005515)" or "name (R-HSA-69306)". The
    referential stores clean ids ("GO:0006084", "HSA-111447", "CL:9273"). This
    extracts the accession from anywhere in the string and unifies Reactome
    "R-HSA-123" -> "HSA-123" so both sides match.
    """
    if not pid:
        return ""
    s = str(pid).strip()

    m = _GO_RE.search(s)
    if m:
        return "GO:" + m.group(1)
    m = _REACTOME_RE.search(s)
    if m:
        return "HSA-" + m.group(1)
    m = _HSA_RE.search(s)
    if m:
        return "HSA-" + m.group(1)
    m = _CL_RE.search(s)
    if m:
        return "CL:" + m.group(1)

    # Fallback: uppercase and unify any leftover Reactome prefix.
    u = s.upper().replace("REACTOME:", "")
    if u.startswith("R-HSA-"):
        u = u[2:]
    return u


def derive_canonical_claims(original_claims: Optional[str], category: Optional[str]) -> list[str]:
    """Reduce the free-text Original_claims (and Category as fallback) to a
    de-duplicated list of canonical claim slugs."""
    found: list[str] = []

    text = (original_claims or "").lower()
    if text and text.strip() not in ("unknown", "context-dependent", "none"):
        for keyword, slug in _CLAIM_KEYWORDS:
            if keyword in text and slug not in found:
                found.append(slug)

    if not found and category:
        fallback = CATEGORY_TO_CLAIM.get(category.strip().upper())
        if fallback:
            found.append(fallback)

    return found
