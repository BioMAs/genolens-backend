"""
CosmOmics Pathway Nova — which pathways support or contradict each cosmetic claim.

Successor of the Skin claims scoring (claim scores, Skin Stack). For one comparison it reads
the over-representation results already stored in `enrichment_pathways`, joins them to the
versioned claim referential `app/data/claims/claim_subgroups.yaml` (one row per claim × term,
grouped into pathway sub-groups) and returns the payload drawn by the frontend nova.

Definitions (reference: docs/reference/cosmomics/cosmomics_pathway_nova.py):

  term       nl = -log10(p). effect = +1 when the observed direction matches the direction the
             claim expects for that term ("pro"), -1 otherwise ("anti"), 0 when not linked.
  sub-group  p = unsigned Stouffer over its terms: z_i = Φ⁻¹(1 − p_i), Z = Σ z_i / √n,
             p = 1 − Φ(Z), bounded at 1e-300. S_pro / S_anti = Σ nl of its pro / anti terms,
             delta = S_pro − S_anti, balance = delta / (S_pro + S_anti) (0 when both are 0),
             obs = "up" when Σ nl(up) ≥ Σ nl(down), else "down".
  claim      S_pro / S_anti = sums over its sub-groups, importance = S_pro + S_anti,
             coherence = S_pro / importance, n_for = #sub-groups with delta > 0,
             n_against = #sub-groups with delta ≤ 0. Claims are independent: one pathway may
             sit in several claims. Only claims with at least one term at p < alpha are kept;
             they are ordered by S_pro, strongest first.
  not linked significant sub-groups (Stouffer p < alpha) attached to no claim.

Choices made for the data GenoLens actually stores:

  * p is the Benjamini-Hochberg `padj` of each term, as in the Skin claims module (which also
    filtered on padj); the raw `pvalue` is only used when a row carries no padj.
  * The observed direction is the `regulation` of the row: enrichment is run separately on the
    up- and on the down-regulated genes. A term enriched in both lists yields two entries, one
    per direction, so it counts both for and against. Rows of the undirected "ALL" list are
    ignored — they say nothing about direction.
  * Producers only store significant terms (annoDB keeps p.adjust < 0.05), so evidence against a
    claim and non-significant terms are rare by construction.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import yaml  # type: ignore[import-untyped]
from scipy.stats import norm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Dataset, EnrichmentPathway
from app.services.cosmetics_taxonomy import normalize_pathway_id
from app.services.cosmomics_grouping import clean_label, label_match_key, subgroup_name
from app.services.enrichment_source import resolve_pathway_dataset_id

logger = logging.getLogger(__name__)

Direction = Literal["up", "down"]

P_FLOOR = 1e-300
# Φ⁻¹(1 − p) diverges at p = 1; a padj of exactly 1 is treated as "no evidence at all".
_P_CEIL = 1.0 - 1e-15

DEFAULT_ALPHA = 0.05
# The "not linked" sector is a fixed 36° slice: beyond a dozen sub-groups it turns into a smear.
# The payload keeps the strongest ones and the total count.
UNLINKED_LIMIT = 12

CLAIM_SUBGROUPS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "claims" / "claim_subgroups.yaml"
)

# Canonical claim slug (Skin claims taxonomy) -> name shown on the nova.
CLAIM_NAMES: dict[str, str] = {
    "firming": "Firming",
    "moisturising": "Barrier",
    "regenerating": "Regeneration",
    "anti-ageing": "Anti-ageing",
    "soothing": "Soothing",
    "antioxidant": "Antioxidant",
    "energising": "Energising",
    "anti-wrinkles": "Anti-wrinkles",
    "healing": "Healing",
}

_ACCESSION_RE = re.compile(r"^(?:GO:\d+|HSA-\d+|CL:\d+)$")
_KEGG_RE = re.compile(r"\b(hsa\d{5})\b", re.IGNORECASE)
_WP_RE = re.compile(r"\((WP\d+)\)")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TermHit:
    """One enriched term of the comparison, in one direction."""

    term_id: str
    label: str
    p: float
    obs: Direction
    category: Optional[str] = None


@dataclass(frozen=True)
class SubgroupTerm:
    """One row of claim_subgroups.yaml. `claim` None = a curated "not linked" sub-group."""

    claim: Optional[str]
    subgroup: str
    term_id: str
    label: str
    expected: Optional[Direction]


@dataclass
class ClaimSubgroupMap:
    """The claim referential, indexed for joining enrichment rows."""

    rows: list[SubgroupTerm]
    by_id: dict[str, list[SubgroupTerm]] = field(default_factory=dict)
    by_label: dict[str, list[SubgroupTerm]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[tuple[Optional[str], str]] = set()
        for row in self.rows:
            key = normalize_pathway_id(row.term_id)
            if (row.claim, key) in seen:
                raise ValueError(
                    f"Duplicate claim/term in referential: {row.claim} / {row.term_id}"
                )
            seen.add((row.claim, key))
            self.by_id.setdefault(key, []).append(row)
            label_key = label_match_key(row.label)
            if label_key:
                self.by_label.setdefault(label_key, []).append(row)

    def lookup(self, hit: TermHit) -> list[SubgroupTerm]:
        """Referential rows for an enrichment term: by accession, else by label.

        The label join only applies to names without an accession (MSigDB "REACTOME_…" sets),
        so a GO term absent from the referential is never mapped through a homonym.
        """
        key = normalize_pathway_id(hit.term_id)
        if key in self.by_id:
            return self.by_id[key]
        if _ACCESSION_RE.match(key):
            return []
        for candidate in (hit.label, hit.term_id):
            rows = self.by_label.get(label_match_key(candidate))
            if rows:
                # A GO term and a Reactome pathway may share a label: one row per claim.
                first_per_claim: dict[Optional[str], SubgroupTerm] = {}
                for row in rows:
                    first_per_claim.setdefault(row.claim, row)
                return list(first_per_claim.values())
        return []


def load_claim_subgroups(path: Path | str = CLAIM_SUBGROUPS_PATH) -> ClaimSubgroupMap:
    """Parse and validate claim_subgroups.yaml."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    known = set(CLAIM_NAMES.values())
    rows: list[SubgroupTerm] = []
    for i, raw in enumerate(doc.get("terms") or []):
        claim = raw.get("claim") or None
        direction = (raw.get("expected_direction") or "").strip().lower() or None
        if claim is not None and claim not in known:
            raise ValueError(f"claim_subgroups row {i}: unknown claim {claim!r}")
        if claim is not None and direction not in ("up", "down"):
            raise ValueError(f"claim_subgroups row {i}: expected_direction must be UP or DOWN")
        if not raw.get("term_id") or not raw.get("subgroup"):
            raise ValueError(f"claim_subgroups row {i}: term_id and subgroup are required")
        rows.append(
            SubgroupTerm(
                claim=claim,
                subgroup=str(raw["subgroup"]),
                term_id=str(raw["term_id"]),
                label=str(raw.get("label") or raw["term_id"]),
                expected=direction if claim is not None else None,  # type: ignore[arg-type]
            )
        )
    return ClaimSubgroupMap(rows)


@lru_cache(maxsize=1)
def default_claim_subgroups() -> ClaimSubgroupMap:
    return load_claim_subgroups()


def subgroup_rows_from_referential(pathway_rows: Iterable[dict]) -> list[dict[str, str]]:
    """Pre-fill claim_subgroups.yaml from the Skin claims workbooks (one row per claim × term).

    `pathway_rows` are the parsed "Pathways" sheets (claim_workbooks.parse_workbook),
    in workbook order: a later row for the same Term_ID wins.
    Terms whose curated direction is BOTH / UNKNOWN / AVOID carry no expected direction and
    are left out — when significant they show as "not linked".
    """
    from app.services.cosmetics_taxonomy import derive_canonical_claims

    by_term: dict[str, dict] = {}
    for row in pathway_rows:
        if row.get("term_id"):
            by_term[str(row["term_id"]).strip()] = row

    claim_order = {name: i for i, name in enumerate(CLAIM_NAMES.values())}
    rows: list[dict[str, str]] = []
    for term_id, row in by_term.items():
        direction = (row.get("updated_direction") or "").strip().upper()
        if direction not in ("UP", "DOWN"):
            continue
        label = clean_label(row.get("description")) or term_id
        group = subgroup_name(label, row.get("category"))
        for slug in derive_canonical_claims(row.get("original_claims"), row.get("category")):
            rows.append(
                {
                    "claim": CLAIM_NAMES[slug],
                    "subgroup": group,
                    "term_id": term_id,
                    "label": label,
                    "expected_direction": direction,
                }
            )
    rows.sort(key=lambda r: (claim_order[r["claim"]], r["subgroup"], r["term_id"]))
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def clamp_p(p: Optional[float]) -> float:
    if p is None or math.isnan(p):
        return 1.0
    return min(max(float(p), P_FLOOR), 1.0)


def nl_of(p: Optional[float]) -> float:
    """-log10(p), with p bounded at 1e-300."""
    return -math.log10(clamp_p(p))


def stouffer_p(ps: Sequence[float]) -> float:
    """Unsigned Stouffer combination: p = 1 − Φ(Σ Φ⁻¹(1 − p_i) / √n), bounded at 1e-300."""
    if not ps:
        return 1.0
    z = norm.isf([min(clamp_p(p), _P_CEIL) for p in ps])
    combined = float(norm.sf(float(z.sum()) / math.sqrt(len(ps))))
    return max(combined, P_FLOOR)


def balance_of(s_pro: float, s_anti: float) -> float:
    total = s_pro + s_anti
    return (s_pro - s_anti) / total if total else 0.0


def coherence_of(s_pro: float, s_anti: float) -> float:
    total = s_pro + s_anti
    return s_pro / total if total else 0.0


# ---------------------------------------------------------------------------
# Nova
# ---------------------------------------------------------------------------


@dataclass
class _Term:
    id: str
    label: str
    p: float
    effect: int
    obs: Direction

    @property
    def nl(self) -> float:
        return nl_of(self.p)


def _display_id(term_id: str) -> str:
    """Accession shown to the user: "R-HSA-…" for Reactome, KEGG/WikiPathways ids when found."""
    key = normalize_pathway_id(term_id)
    if key.startswith("HSA-"):
        return "R-" + key
    if _ACCESSION_RE.match(key):
        return key
    kegg = _KEGG_RE.search(term_id)
    if kegg:
        return kegg.group(1).lower()
    wp = _WP_RE.search(term_id)
    if wp:
        return wp.group(1)
    return term_id


def _subgroup_payload(name: str, terms: list[_Term]) -> dict[str, Any]:
    terms = sorted(terms, key=lambda t: (-t.nl, t.label))
    p = stouffer_p([t.p for t in terms])
    s_pro = sum(t.nl for t in terms if t.effect > 0)
    s_anti = sum(t.nl for t in terms if t.effect < 0)
    w_up = sum(t.nl for t in terms if t.obs == "up")
    w_down = sum(t.nl for t in terms if t.obs == "down")
    return {
        "name": name,
        "nl": nl_of(p),
        "s_pro": s_pro,
        "s_anti": s_anti,
        "delta": s_pro - s_anti,
        "balance": balance_of(s_pro, s_anti),
        "obs": "up" if w_up >= w_down else "down",
        "terms": [
            {"id": t.id, "label": t.label, "nl": t.nl, "effect": t.effect, "obs": t.obs}
            for t in terms
        ],
    }


def _dedupe(hits: Iterable[TermHit]) -> list[TermHit]:
    """Keep the most significant row per (term, direction) — producers may repeat a term."""
    best: dict[tuple[str, str], TermHit] = {}
    for hit in hits:
        key = (normalize_pathway_id(hit.term_id) or hit.label, hit.obs)
        if key not in best or clamp_p(hit.p) < clamp_p(best[key].p):
            best[key] = hit
    return list(best.values())


def build_nova(
    hits: Iterable[TermHit],
    mapping: ClaimSubgroupMap,
    *,
    alpha: float = DEFAULT_ALPHA,
    contrast: str = "",
    condition: str = "Active",
    unlinked_limit: int = UNLINKED_LIMIT,
) -> dict[str, Any]:
    """Assemble the nova payload from the comparison's enriched terms. Pure function."""
    unique_hits = _dedupe(hits)
    claim_groups: dict[str, dict[str, list[_Term]]] = defaultdict(lambda: defaultdict(list))
    unlinked_groups: dict[str, list[_Term]] = defaultdict(list)
    n_matched = 0

    for hit in unique_hits:
        rows = mapping.lookup(hit)
        label = clean_label(hit.label) or hit.term_id
        if rows:
            n_matched += 1
        linked = [r for r in rows if r.claim is not None]
        for row in linked:
            effect = 1 if hit.obs == row.expected else -1
            claim_groups[row.claim][row.subgroup].append(  # type: ignore[index]
                _Term(_display_id(row.term_id), row.label or label, hit.p, effect, hit.obs)
            )
        if not linked:
            name = rows[0].subgroup if rows else subgroup_name(label, hit.category)
            unlinked_groups[name].append(_Term(_display_id(hit.term_id), label, hit.p, 0, hit.obs))

    claims: list[dict[str, Any]] = []
    for claim, groups in claim_groups.items():
        if not any(t.p < alpha for terms in groups.values() for t in terms):
            continue
        subgroups = sorted(
            (_subgroup_payload(name, terms) for name, terms in groups.items()),
            key=lambda s: (-s["nl"], s["name"]),
        )
        s_pro = sum(s["s_pro"] for s in subgroups)
        s_anti = sum(s["s_anti"] for s in subgroups)
        best = subgroups[0]
        claims.append(
            {
                "name": claim,
                "importance": s_pro + s_anti,
                "s_pro": s_pro,
                "s_anti": s_anti,
                "coherence": coherence_of(s_pro, s_anti),
                "n_for": sum(1 for s in subgroups if s["delta"] > 0),
                "n_against": sum(1 for s in subgroups if s["delta"] <= 0),
                "best_subgroup": best["name"],
                "best_nl": best["nl"],
                "subgroups": subgroups,
            }
        )
    claims.sort(key=lambda c: (-c["s_pro"], c["name"]))

    sig_nl = nl_of(alpha)
    unlinked = sorted(
        (
            s
            for s in (_subgroup_payload(name, terms) for name, terms in unlinked_groups.items())
            if s["nl"] > sig_nl
        ),
        key=lambda s: (-s["nl"], s["name"]),
    )

    return {
        "contrast": contrast,
        "condition": condition,
        "alpha": alpha,
        "p_source": "padj (Benjamini-Hochberg, over-representation on up / down gene lists)",
        "claims": claims,
        "unlinked": unlinked[:unlinked_limit],
        "unlinked_total": len(unlinked),
        "coverage": {
            "n_terms": len(unique_hits),
            "n_terms_significant": sum(1 for h in unique_hits if h.p < alpha),
            "n_terms_matched": n_matched,
        },
    }


def condition_label(comparison_name: str, default: str = "Active", max_len: int = 8) -> str:
    """Name for the central disc: the numerator of "A_vs_B" when short enough, else `default`."""
    numerator = re.split(r"[_\s]+vs[_\s]+", comparison_name or "", maxsplit=1, flags=re.I)[0]
    numerator = numerator.strip()
    return numerator if 0 < len(numerator) <= max_len else default


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _row_direction(regulation: Optional[str], comparison_name: str) -> Optional[Direction]:
    """UP/DOWN from the row's `regulation`, else from a legacy " (up)" / " (down)" suffix."""
    reg = (regulation or "").strip().upper()
    if reg == "UP":
        return "up"
    if reg == "DOWN":
        return "down"
    name = (comparison_name or "").strip().lower()
    if name.endswith("(up)"):
        return "up"
    if name.endswith("(down)"):
        return "down"
    return None


async def fetch_term_hits(
    db: AsyncSession, deg_dataset: Dataset, comparison_name: str
) -> list[TermHit]:
    """The comparison's directional enrichment rows, read where the enrichment panel reads them.

    Uses the shared resolver (ENRICHMENT dataset of the same analysis when there is one, else
    the DEG dataset) and an exact comparison-name match — never a prefix match, so
    "A_vs_B" cannot pick up the rows of "A_vs_B_24h".
    """
    source_id = await resolve_pathway_dataset_id(db, deg_dataset, comparison_name)
    names = [comparison_name] + [
        f"{comparison_name} {suffix}" for suffix in ("(up)", "(down)", "(UP)", "(DOWN)")
    ]
    result = await db.execute(
        select(
            EnrichmentPathway.pathway_id,
            EnrichmentPathway.pathway_name,
            EnrichmentPathway.category,
            EnrichmentPathway.pvalue,
            EnrichmentPathway.padj,
            EnrichmentPathway.regulation,
            EnrichmentPathway.comparison_name,
        ).where(
            EnrichmentPathway.dataset_id == source_id,
            EnrichmentPathway.comparison_name.in_(names),
        )
    )
    hits: list[TermHit] = []
    for row in result.all():
        obs = _row_direction(row.regulation, row.comparison_name)
        if obs is None:
            continue
        p = row.padj if row.padj is not None else row.pvalue
        hits.append(
            TermHit(
                term_id=str(row.pathway_id),
                label=str(row.pathway_name or row.pathway_id),
                p=float(p) if p is not None else 1.0,
                obs=obs,
                category=row.category,
            )
        )
    return hits
