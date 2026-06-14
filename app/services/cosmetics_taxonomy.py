"""
Cosmetics module — shared taxonomy & helpers.

Single source of truth for:
  - the canonical (display-facing) cosmetic claim taxonomy (English),
  - the skin-zone taxonomy used by the annotated skin schematic,
  - mapping of referential `Category` -> skin zone and fallback claim,
  - the pathway-id normalization used to join the claim referential against
    EnrichmentPathway rows,
  - derivation of canonical claim slugs from the free-text `Original_claims`.

Imported by both the import service (to populate canonical_claims / seed the
taxonomy) and the scoring service (to aggregate scores per claim and per zone).
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Canonical claim taxonomy (English). Seeded into the cosmetic_claims table.
# ---------------------------------------------------------------------------
# Each entry: slug, label, skin_zone, color (hex), icon, description, order
CANONICAL_CLAIMS: list[dict] = [
    {
        "slug": "anti-ageing",
        "label": "Anti-ageing",
        "skin_zone": "cellular",
        "color": "#8b5cf6",
        "icon": "sparkles",
        "description": "Supports cellular resilience, proteostasis and DNA integrity associated with younger-looking skin.",
        "display_order": 1,
    },
    {
        "slug": "anti-wrinkles",
        "label": "Anti-wrinkles",
        "skin_zone": "dermis",
        "color": "#7c3aed",
        "icon": "waves",
        "description": "Targets the dermal matrix mechanics linked to fine lines and wrinkles.",
        "display_order": 2,
    },
    {
        "slug": "firming",
        "label": "Firming & Elasticity",
        "skin_zone": "dermis",
        "color": "#db2777",
        "icon": "activity",
        "description": "Reinforces extracellular matrix, adhesion and cytoskeleton for firmer, more elastic skin.",
        "display_order": 3,
    },
    {
        "slug": "regenerating",
        "label": "Regenerating & Renewal",
        "skin_zone": "epidermis",
        "color": "#16a34a",
        "icon": "refresh-cw",
        "description": "Stimulates turnover, protein synthesis and cellular trafficking for skin renewal.",
        "display_order": 4,
    },
    {
        "slug": "moisturising",
        "label": "Moisturising & Barrier",
        "skin_zone": "stratum_corneum",
        "color": "#0ea5e9",
        "icon": "droplet",
        "description": "Supports barrier lipid synthesis and organization for better hydration and TEWL control.",
        "display_order": 5,
    },
    {
        "slug": "soothing",
        "label": "Soothing & Anti-redness",
        "skin_zone": "inflammation",
        "color": "#10b981",
        "icon": "shield",
        "description": "Modulates inflammatory and eicosanoid signaling to calm redness and irritation.",
        "display_order": 6,
    },
    {
        "slug": "antioxidant",
        "label": "Antioxidant & Detox",
        "skin_zone": "antioxidant",
        "color": "#f59e0b",
        "icon": "zap-off",
        "description": "Reinforces redox defenses and detoxification against oxidative stress.",
        "display_order": 7,
    },
    {
        "slug": "energising",
        "label": "Energising",
        "skin_zone": "energy",
        "color": "#ef4444",
        "icon": "battery-charging",
        "description": "Boosts mitochondrial and metabolic activity for energized skin cells.",
        "display_order": 8,
    },
    {
        "slug": "healing",
        "label": "Healing & Repair",
        "skin_zone": "epidermis",
        "color": "#14b8a6",
        "icon": "heart-pulse",
        "description": "Promotes wound-healing and tissue repair processes.",
        "display_order": 9,
    },
]

CANONICAL_SLUGS = {c["slug"] for c in CANONICAL_CLAIMS}

# ---------------------------------------------------------------------------
# Skin zones used by the annotated schematic.
# ---------------------------------------------------------------------------
SKIN_ZONES: list[dict] = [
    {"slug": "stratum_corneum", "label": "Skin barrier (stratum corneum)"},
    {"slug": "epidermis", "label": "Epidermis (renewal & repair)"},
    {"slug": "dermis", "label": "Dermis (collagen & elasticity)"},
    {"slug": "inflammation", "label": "Inflammation & redness"},
    {"slug": "antioxidant", "label": "Antioxidant defense"},
    {"slug": "energy", "label": "Cellular energy"},
    {"slug": "cellular", "label": "Cellular resilience & ageing"},
]
SKIN_ZONE_SLUGS = {z["slug"] for z in SKIN_ZONES}

# Referential Category -> skin zone (for the schematic aggregation)
CATEGORY_TO_ZONE: dict[str, str] = {
    "BARRIER_LIPID": "stratum_corneum",
    "ECM_CYTO": "dermis",
    "BARRIER_ECM": "dermis",
    "IMMUNE_INFLAM": "inflammation",
    "EICOSANOID": "inflammation",
    "REDOX_ANTIOX": "antioxidant",
    "ENERGY_METAB": "energy",
    "PROTEOSTASIS": "cellular",
    "GENOME_STABILITY": "cellular",
    "AGING": "cellular",
    "CELL_DEATH": "cellular",
    "TRAFFICKING": "epidermis",
    "GENE_EXPRESSION": "epidermis",
    "CELL_CYCLE": "epidermis",
    "GROWTH_SIGNAL": "epidermis",
    "MTOR_SIGNAL": "epidermis",
}

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
    against EnrichmentPathway.pathway_id.

    - uppercases and trims
    - unifies Reactome "R-HSA-123" / "REACTOME:HSA-123" -> "HSA-123"
    - leaves GO:xxxx and CL:xxxx untouched
    """
    if not pid:
        return ""
    s = str(pid).strip().upper()
    s = s.replace("REACTOME:", "")
    if s.startswith("R-HSA-"):
        s = s[2:]  # drop leading "R-"
    return s


def derive_canonical_claims(
    original_claims: Optional[str], category: Optional[str]
) -> list[str]:
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


def category_to_zone(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    return CATEGORY_TO_ZONE.get(category.strip().upper())


# Evidence-level numeric weights used by the scoring service.
EVIDENCE_WEIGHTS: dict[str, float] = {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.3}
