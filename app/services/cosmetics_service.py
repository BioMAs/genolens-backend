"""
Cosmetics module — scoring service.

Translates the enriched (modulated) pathways of a comparison into cosmetic claim
scores and skin-zone activity, by joining EnrichmentPathway rows against the
admin-curated claim referential (ClaimPathwayMapping).

Two complementary signals are produced:
  - claim scores (radar / claim cards): directional — does the observed pathway
    direction (UP/DOWN regulation) match the direction that supports each claim?
  - skin-zone activity (annotated schematic): how strongly each skin compartment
    is transcriptionally engaged, regardless of direction.
"""
from __future__ import annotations

import math
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    ClaimDirection,
    ClaimPathwayMapping,
    ClaimReference,
    Dataset,
    DatasetType,
    EnrichmentPathway,
)
from app.services.cosmetics_taxonomy import (
    CANONICAL_CLAIMS,
    EVIDENCE_WEIGHTS,
    SKIN_ZONES,
    category_to_zone,
    normalize_pathway_id,
)

# Saturation constant: accumulated weighted evidence at which "strength" ~63%.
_STRENGTH_TAU = 3.0
# Cap for the -log10(padj) significance multiplier.
_SIG_CAP = 10.0


def _significance_weight(padj: float) -> float:
    """Map padj -> ~[0.5, 1.5] multiplier (more significant => higher)."""
    if padj is None or padj <= 0:
        return 1.5
    s = min(-math.log10(padj), _SIG_CAP) / _SIG_CAP  # 0..1
    return 0.5 + s


def _confidence(evidence_levels: list[str], n_supporting: int) -> str:
    n_high = sum(1 for e in evidence_levels if e == "HIGH")
    if (n_high >= 2) or (n_high >= 1 and n_supporting >= 5):
        return "HIGH"
    if n_supporting >= 2:
        return "MODERATE"
    return "LOW"


def _observed_direction(row) -> str | None:
    """Observed regulation of an enrichment row.

    Some pipelines store it in the `regulation` column (UP/DOWN/ALL); the annoDB
    ENRICHMENT-dataset pipeline encodes it in the comparison name suffix instead
    (e.g. "X (up)" / "X (down)").
    """
    reg = (row.regulation or "").upper()
    if reg in ("UP", "DOWN"):
        return reg
    cn = (row.comparison_name or "").lower().strip()
    if cn.endswith("(up)"):
        return "UP"
    if cn.endswith("(down)"):
        return "DOWN"
    return None


async def _resolve_enrichment_rows(
    db: AsyncSession, dataset_id: UUID, comparison_name: str, max_padj: float
) -> tuple[list, str]:
    """Find the enrichment rows for a comparison and pick a significance basis.

    Enrichment may be stored either directly under the given (DEG) dataset id, or
    under a separate ENRICHMENT-type dataset (annoDB pipeline). Direction may be
    split into "(up)"/"(down)" comparison variants. Returns (rows, basis) where
    basis is one of: "fdr", "nominal", "exploratory", "none".
    """
    variants = [
        comparison_name,
        f"{comparison_name} (up)",
        f"{comparison_name} (down)",
        f"{comparison_name} (UP)",
        f"{comparison_name} (DOWN)",
    ]

    def _query(ids):
        return select(EnrichmentPathway).where(
            EnrichmentPathway.dataset_id.in_(ids),
            or_(
                EnrichmentPathway.comparison_name.in_(variants),
                EnrichmentPathway.comparison_name.like(f"{comparison_name}%"),
            ),
        )

    # 1. Directly under the given dataset id.
    rows = (await db.execute(_query([dataset_id]))).scalars().all()

    # 2. Fall back to ENRICHMENT-type datasets in the same project.
    if not rows:
        deg = (
            await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        ).scalar_one_or_none()
        if deg is not None:
            enr_ds = (
                await db.execute(
                    select(Dataset).where(
                        Dataset.project_id == deg.project_id,
                        Dataset.type == DatasetType.ENRICHMENT,
                    )
                )
            ).scalars().all()
            matched_ids = []
            for d in enr_ds:
                meta = d.dataset_metadata or {}
                cn = meta.get("comparison_name")
                comps = meta.get("enrichment_comparisons") or meta.get("comparisons") or []
                if (
                    cn == comparison_name
                    or comparison_name in comps
                    or any(str(c).startswith(comparison_name) for c in comps)
                    or (d.name or "").startswith(comparison_name)
                ):
                    matched_ids.append(d.id)
            if matched_ids:
                rows = (await db.execute(_query(matched_ids))).scalars().all()

    if not rows:
        return [], "none"

    # 3. Significance basis: prefer FDR, then nominal p, then exploratory top-N.
    sig = [r for r in rows if r.padj is not None and r.padj <= max_padj]
    if sig:
        return sig, "fdr"
    nom = [r for r in rows if r.pvalue is not None and r.pvalue <= 0.05]
    if nom:
        return nom, "nominal"
    rows_sorted = sorted(rows, key=lambda r: (r.padj if r.padj is not None else 1.0))
    return rows_sorted[:150], "exploratory"


async def score_claims(
    db: AsyncSession,
    dataset_id: UUID,
    comparison_name: str,
    max_padj: float = 0.05,
) -> dict:
    """Compute cosmetic claim scores and skin-zone activity for one comparison."""

    # 1. Resolve enrichment rows (DEG dataset or ENRICHMENT dataset) + basis.
    enr_rows, significance_basis = await _resolve_enrichment_rows(
        db, dataset_id, comparison_name, max_padj
    )

    # 2. Active claim referential, indexed by normalized term id.
    mappings = (
        await db.execute(
            select(ClaimPathwayMapping).where(ClaimPathwayMapping.is_active.is_(True))
        )
    ).scalars().all()
    mapping_by_norm: dict[str, ClaimPathwayMapping] = {
        m.term_id_normalized: m for m in mappings
    }

    # Accumulators
    claim_meta = {c["slug"]: c for c in CANONICAL_CLAIMS}
    claim_pos: dict[str, float] = {s: 0.0 for s in claim_meta}
    claim_neg: dict[str, float] = {s: 0.0 for s in claim_meta}
    claim_n_support: dict[str, int] = {s: 0 for s in claim_meta}
    claim_n_against: dict[str, int] = {s: 0 for s in claim_meta}
    claim_evidence: dict[str, list[str]] = {s: [] for s in claim_meta}
    claim_pathways: dict[str, list[dict]] = {s: [] for s in claim_meta}
    claim_genes: dict[str, list[str]] = {s: [] for s in claim_meta}

    zone_activity: dict[str, float] = {z["slug"]: 0.0 for z in SKIN_ZONES}
    zone_up: dict[str, float] = {z["slug"]: 0.0 for z in SKIN_ZONES}
    zone_down: dict[str, float] = {z["slug"]: 0.0 for z in SKIN_ZONES}
    zone_n: dict[str, int] = {z["slug"]: 0 for z in SKIN_ZONES}

    caveats: list[dict] = []
    ref_cats: set[str] = set()
    n_matched = 0
    matched_terms: set[str] = set()

    for row in enr_rows:
        mapping = mapping_by_norm.get(normalize_pathway_id(row.pathway_id))
        if mapping is None:
            continue
        n_matched += 1
        matched_terms.add(mapping.term_id)
        if mapping.ref_cat:
            ref_cats.add(mapping.ref_cat)

        evidence = (
            mapping.evidence_level.value
            if hasattr(mapping.evidence_level, "value")
            else str(mapping.evidence_level)
        )
        base_w = EVIDENCE_WEIGHTS.get(evidence, 0.3) * _significance_weight(row.padj)
        zone = category_to_zone(row.category) or category_to_zone(mapping.category)
        obs = _observed_direction(row)

        # --- skin-zone activity (any regulation engages the compartment) ---
        if zone:
            zone_activity[zone] += base_w
            zone_n[zone] += 1
            if obs == "UP":
                zone_up[zone] += base_w
            elif obs == "DOWN":
                zone_down[zone] += base_w

        direction = mapping.updated_direction

        # --- caveats for flagged mappings ---
        if direction in (ClaimDirection.AVOID, ClaimDirection.UNKNOWN):
            caveats.append(
                {
                    "term_id": mapping.term_id,
                    "pathway_name": row.pathway_name,
                    "flag": direction.value,
                    "note": mapping.caveats or mapping.rationale,
                }
            )
            continue
        if direction == ClaimDirection.BOTH:
            if mapping.caveats:
                caveats.append(
                    {
                        "term_id": mapping.term_id,
                        "pathway_name": row.pathway_name,
                        "flag": "CONTEXT",
                        "note": mapping.caveats,
                    }
                )
            continue

        # --- directional claim scoring (needs UP/DOWN observed direction) ---
        if obs not in ("UP", "DOWN"):
            continue
        observed = ClaimDirection(obs)
        match = 1 if observed == direction else -1

        for slug in (mapping.canonical_claims or []):
            if slug not in claim_meta:
                continue
            if match > 0:
                claim_pos[slug] += base_w
                claim_n_support[slug] += 1
                claim_evidence[slug].append(evidence)
                claim_pathways[slug].append(
                    {
                        "term_id": mapping.term_id,
                        "pathway_name": row.pathway_name,
                        "direction": observed.value,
                        "evidence_level": evidence,
                        "padj": row.padj,
                        "category": row.category,
                        "weight": round(base_w, 3),
                    }
                )
                for g in (row.genes or [])[:10]:
                    if g not in claim_genes[slug]:
                        claim_genes[slug].append(g)
            else:
                claim_neg[slug] += base_w
                claim_n_against[slug] += 1

    # 3. Build claim results.
    claims_out: list[dict] = []
    for slug, meta in claim_meta.items():
        pos = claim_pos[slug]
        neg = claim_neg[slug]
        total = pos + neg
        consistency = (pos / total) if total > 0 else 0.0
        strength = 1.0 - math.exp(-total / _STRENGTH_TAU) if total > 0 else 0.0
        score = round(100.0 * consistency * strength, 1)
        net = pos - neg
        pathways = sorted(claim_pathways[slug], key=lambda p: p["weight"], reverse=True)[:8]
        claims_out.append(
            {
                "slug": slug,
                "label": meta["label"],
                "color": meta["color"],
                "icon": meta["icon"],
                "skin_zone": meta["skin_zone"],
                "description": meta["description"],
                "score": score,
                "direction": "favorable" if net >= 0 else "unfavorable",
                "n_supporting": claim_n_support[slug],
                "n_contradicting": claim_n_against[slug],
                "confidence": _confidence(claim_evidence[slug], claim_n_support[slug]),
                "evidence_pathways": pathways,
                "top_genes": claim_genes[slug][:15],
            }
        )
    claims_out.sort(key=lambda c: c["score"], reverse=True)

    # 4. Build skin-zone results.
    max_zone = max(zone_activity.values()) if zone_activity else 0.0
    zones_out: list[dict] = []
    for z in SKIN_ZONES:
        slug = z["slug"]
        activity = zone_activity[slug]
        norm = round(100.0 * activity / max_zone, 1) if max_zone > 0 else 0.0
        if zone_up[slug] == 0 and zone_down[slug] == 0:
            dominant = "neutral"
        else:
            dominant = "up" if zone_up[slug] >= zone_down[slug] else "down"
        zones_out.append(
            {
                "slug": slug,
                "label": z["label"],
                "activity": norm,
                "dominant_direction": dominant,
                "n_pathways": zone_n[slug],
            }
        )

    # 5. References for the categories actually involved.
    references: list[dict] = []
    if ref_cats:
        ref_rows = (
            await db.execute(
                select(ClaimReference).where(ClaimReference.ref_cat.in_(ref_cats))
            )
        ).scalars().all()
        references = [
            {"ref_cat": r.ref_cat, "source_summary": r.source_summary, "url": r.url}
            for r in ref_rows
        ]

    n_significant = len(enr_rows)
    return {
        "dataset_id": str(dataset_id),
        "comparison_name": comparison_name,
        "claims": claims_out,
        "skin_zones": zones_out,
        "caveats": caveats[:30],
        "references": references,
        "coverage": {
            "n_significant": n_significant,
            "n_matched": n_matched,
            "n_terms_matched": len(matched_terms),
            "match_rate": round(n_matched / n_significant, 3) if n_significant else 0.0,
            "significance_basis": significance_basis,
        },
    }
