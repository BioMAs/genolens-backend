"""
CosmOmics Pathway Nova — which pathways support or contradict each cosmetic claim.

Successor of the Skin claims scoring (claim scores, Skin Stack). For one comparison it reads
the over-representation results already stored in `enrichment_pathways`, joins them to the
versioned claim referential `app/data/claims/claim_subgroups.yaml` (one row per claim × term,
grouped into pathway sub-groups) and returns the payload drawn by the frontend nova. The payload
also says what it was computed from (`provenance`, `p_source`, `coverage`).

Definitions (reference: docs/reference/cosmomics/cosmomics_pathway_nova.py):

  term       nl = -log10(p). effect = +1 when the observed direction matches the direction the
             claim expects for that term ("pro"), -1 otherwise ("anti"), 0 when not linked.
             `significant` is p < alpha; `both_directions` says the term is enriched in the up
             list and in the down list (it then has one entry per direction).
  sub-group  p = unsigned Stouffer over its terms: z_i = Φ⁻¹(1 − p_i), Z = Σ z_i / √n,
             p = 1 − Φ(Z), bounded at 1e-300. S_pro / S_anti = Σ nl of its pro / anti terms,
             delta = S_pro − S_anti, balance = delta / (S_pro + S_anti) (0 when both are 0),
             obs = "up" when Σ nl(up) ≥ Σ nl(down), else "down". A sub-group with delta = 0 is
             balanced: it goes neither for nor against the claim.
  claim      S_pro / S_anti = sums over its sub-groups, importance = S_pro + S_anti,
             coherence = S_pro / importance, n_for = #sub-groups with delta > 0,
             n_against = #sub-groups with delta < 0, n_balanced = the others. `state` is
             "for_only" (S_anti = 0), "against_only" (S_pro = 0) or "both_ways".
             best_subgroup / best_nl: the most significant sub-group, whatever its direction;
             best_for_subgroup / best_for_nl: the most significant one among those with
             delta > 0 (None without one) — the figure the OmicSkin score is read from.
             Claims are independent: one pathway may sit in several claims. Only claims with at
             least one term at p < alpha are kept, ordered by S_pro, strongest first; the other
             claims of the referential are listed in `not_evaluated`.
  not linked significant sub-groups (Stouffer p < alpha) attached to no claim.

What the p-values are, and what is stored (the payload's `provenance` and `p_source` say which):

  * p is the `padj` of each term. From an analysis run in GenoLens it is the Benjamini-Hochberg
    figure of the annoDB over-representation, and only the terms below the cut-off the run was
    given are stored (`provenance.enrichment_fdr`: 0.05 unless the wizard chose another, and
    recorded as `padj_cutoff` on the ENRICHMENT dataset); the enrichment GenoLens runs on a DEG
    upload stores its Benjamini-Hochberg terms at fdr <= 0.05. For tables uploaded to GenoLens it
    is the column the parser read as adjusted p (`provenance.padj_columns`), and the tables are
    stored as deposited, not filtered by anyone. A row without an adjusted p is never stored as if it
    had one, and a term without a padj is left out of the nova and counted
    (`coverage.n_terms_without_padj`).
  * `provenance.enrichment_source` says what kind of enrichment the rows are, once, for every
    reader (the p-value label, the poster header, the editorial and LaTeX methods, the AI prompt):
    "annodb" (an ENRICHMENT dataset GenoLens registered, `source` "annodb"), "uploaded" (any other
    ENRICHMENT dataset: tables somebody deposited), "legacy" (rows stored under the DEG dataset
    by an earlier GenoLens method, no ENRICHMENT dataset) and "none" (no enrichment result is
    stored: the cut-off, the databases and the columns are then not facts of any run and stay
    empty). `provenance.enrichment_categories` is what the annoDB run recorded about itself (the
    categories it enriched), None when it recorded none.
  * `alpha` marks each term `significant` and keeps a claim only if one of its terms is. A stored
    term at p >= alpha is not dropped: it weighs in the sums like any other and is counted in
    `coverage.n_terms_not_significant`. Such terms are rare with GenoLens's own enrichment at its
    defaults (`enrichment_fdr` = `alpha` = 0.05); they appear when the analysis kept terms above
    `alpha`, and in uploaded tables.
  * The observed direction is the `regulation` of the row: enrichment is run separately on the
    up- and on the down-regulated genes. A term enriched in both lists yields two entries, one
    per direction, so it counts both for and against. Rows of the undirected "ALL" list are
    ignored — they say nothing about direction — so a comparison with none of the directional
    rows has `status` "no_enrichment".
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence
from uuid import UUID

import yaml  # type: ignore[import-untyped]
from scipy.stats import norm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import Dataset, EnrichmentPathway, SelfServiceAnalysis
from app.services.cosmetics_taxonomy import normalize_pathway_id
from app.services.cosmomics_grouping import clean_label, label_match_key, subgroup_name
from app.services.enrichment_source import find_enrichment_dataset

logger = logging.getLogger(__name__)

Direction = Literal["up", "down"]
Side = Literal["for", "against", "balanced"]

P_FLOOR = 1e-300
# Φ⁻¹(1 − p) diverges at p = 1; a padj of exactly 1 is treated as "no evidence at all".
_P_CEIL = 1.0 - 1e-15

DEFAULT_ALPHA = 0.05
# The R enrichment script's own term cut-off: what an analysis ran at when no other was passed.
DEFAULT_ENRICHMENT_FDR = 0.05
# Sub-groups whose delta is within this of zero are balanced: float sums of equal evidence need
# not cancel exactly.
_TIE_EPS = 1e-9
# `cosmetic_interpretations` holds one row per comparison, shared with the legacy Skin claims
# narrative. The nova's rows carry this prefix in `model`, so a legacy text is never served as
# a nova interpretation.
NOVA_MODEL_TAG = "cosmomics-nova:"
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
    """One enriched term of the comparison, in one direction. `p` None = no adjusted p-value."""

    term_id: str
    label: str
    p: Optional[float]
    obs: Direction
    category: Optional[str] = None
    gene_count: Optional[int] = None
    # Genes of the enriched list that hit the term (EnrichmentPathway.genes); feeds the report's
    # "key genes" chips, never the nova statistics.
    genes: tuple[str, ...] = ()


ENRICHMENT_SOURCES = ("annodb", "uploaded", "legacy", "none")


@dataclass(frozen=True)
class NovaSource:
    """What the nova's enrichment rows came from: the facts `provenance` and `p_source` state.

    `enrichment_source` is the kind of enrichment (`ENRICHMENT_SOURCES`). As read from the
    database it is "annodb" or "uploaded" when an ENRICHMENT dataset exists and "none" when it does
    not; `build_nova` settles it: no row at all is "none", and rows with no ENRICHMENT dataset are
    "legacy" (the earlier GenoLens method stores them under the DEG dataset). `padj_columns` are
    the headers the parser read as adjusted p of an uploaded table. `enrichment_databases` is the
    analysis's own pick (an empty tuple: it picked none, so all the categories ran; None: not
    known, e.g. the analysis was deleted) and `enrichment_categories` what the run recorded about
    itself. All fields default to "unknown", so a nova built from hits alone still has a
    provenance block.
    """

    dataset_id: Optional[str] = None
    pathway_dataset_id: Optional[UUID] = None
    enrichment_dataset_ids: tuple[str, ...] = ()
    analysis_id: Optional[str] = None
    enrichment_fdr: Optional[float] = None
    enrichment_databases: Optional[tuple[str, ...]] = None
    enrichment_categories: Optional[tuple[str, ...]] = None
    enrichment_source: str = "none"
    padj_columns: tuple[str, ...] = ()
    source_versions: tuple[str, ...] = ()
    demo: bool = False


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
    # `version` of claim_subgroups.yaml — the only part of the referential a deliverable may print.
    version: Optional[str] = None

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
    version = doc.get("version")
    return ClaimSubgroupMap(rows, version=str(version) if version is not None else None)


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


def _has_p(hit: TermHit) -> bool:
    return hit.p is not None and not math.isnan(hit.p)


@dataclass
class _Term:
    id: str
    label: str
    p: float
    effect: int
    obs: Direction
    both_directions: bool = False
    significant: bool = False

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
        # One term enriched in both lists has two entries; it is one term here.
        "n_terms_both_directions": len({t.id for t in terms if t.both_directions}),
        "terms": [
            {
                "id": t.id,
                "label": t.label,
                "nl": t.nl,
                "effect": t.effect,
                "obs": t.obs,
                "both_directions": t.both_directions,
                "significant": t.significant,
            }
            for t in terms
        ],
    }


def _term_key(hit: TermHit) -> str:
    return normalize_pathway_id(hit.term_id) or hit.label


def _significance_rank(hit: TermHit) -> float:
    """Smaller is more significant; a row without an adjusted p ranks after any row with one."""
    return clamp_p(hit.p) if _has_p(hit) else 2.0


def _dedupe(hits: Iterable[TermHit]) -> list[TermHit]:
    """Keep the most significant row per (term, direction) — producers may repeat a term."""
    best: dict[tuple[str, str], TermHit] = {}
    for hit in hits:
        key = (_term_key(hit), hit.obs)
        if key not in best or _significance_rank(hit) < _significance_rank(best[key]):
            best[key] = hit
    return list(best.values())


def subgroup_side(subgroup: dict[str, Any]) -> Side:
    """Which side of its claim a sub-group of the payload stands on.

    "for" when delta > 0, "against" when delta < 0, "balanced" when the evidence for and against
    cancels (delta within `_TIE_EPS` of 0). The one rule behind `n_for` / `n_against` /
    `n_balanced`, the wheel, the editorial sentences and the AI prompt: a balanced sub-group is
    never presented as contrary.
    """
    delta = subgroup.get("delta") or 0.0
    if delta > _TIE_EPS:
        return "for"
    if delta < -_TIE_EPS:
        return "against"
    return "balanced"


def best_supporting_subgroup(subgroups: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The most significant sub-group whose evidence goes the expected way (delta > 0), if any."""
    supporting = [s for s in subgroups if subgroup_side(s) == "for"]
    return min(supporting, key=lambda s: (-s["nl"], s["name"]), default=None)


# Said wherever a claim has no contrary term at all (state "for_only"): on the wheel, on the
# editorial pages and in what the AI is asked to write. The absence of a contrary term is not
# evidence for; and it is said of the enrichment results, whatever their origin (an uploaded table
# is not filtered, GenoLens's own enrichment is stored under its cut-off).
ZERO_AGAINST_NOTE = (
    "Only terms present in the enrichment results can count against a claim; "
    "no contrary term is not a confirmation."
)


def claim_state(s_pro: float, s_anti: float) -> str:
    """"for_only" without evidence against, "against_only" without evidence for, else "both_ways"."""
    if s_anti == 0:
        return "for_only"
    if s_pro == 0:
        return "against_only"
    return "both_ways"


def claim_state_of(claim: dict[str, Any]) -> str:
    """The state of a payload claim: its `state`, or (a payload that predates the field) the one
    its sums give."""
    return claim.get("state") or claim_state(
        float(claim.get("s_pro") or 0.0), float(claim.get("s_anti") or 0.0)
    )


def _referential_version(mapping: ClaimSubgroupMap) -> Optional[int]:
    if mapping.version is None:
        return None
    try:
        return int(mapping.version)
    except ValueError:
        return None


# What the p-values are, by the kind of enrichment (`NovaSource.enrichment_source`). One function
# for the payload's `p_source`, the poster header and the AI prompt's label, so they cannot differ.
# P_SOURCE_UPLOADED is how the uploaded wording begins; nothing reads the kind from it any more.
P_SOURCE_ANNODB = "padj (Benjamini–Hochberg, GenoLens enrichment)"
P_SOURCE_UPLOADED = "adjusted p from uploaded tables"
P_SOURCE_LEGACY = "adjusted p from an earlier GenoLens enrichment (test and correction not recorded)"
P_SOURCE_NONE = "no enrichment result is stored"


def p_source_of(kind: Optional[str], padj_columns: Iterable[str] = ()) -> Optional[str]:
    """What the adjusted p-values are, in words, for an enrichment source; None for a kind that is
    not one (a payload that predates the field: its own `p_source` stands)."""
    if kind == "annodb":
        return P_SOURCE_ANNODB
    if kind == "uploaded":
        columns = [str(c) for c in padj_columns or ()]
        noun = "columns" if len(columns) > 1 else "column"
        return f"{P_SOURCE_UPLOADED} ({noun} {', '.join(columns) or 'not recorded'})"
    if kind == "legacy":
        return P_SOURCE_LEGACY
    if kind == "none":
        return P_SOURCE_NONE
    return None


def _settled(source: NovaSource, has_rows: bool) -> NovaSource:
    """The source with its kind settled by what was found: no enrichment row at all is "none"
    (nothing is stored, so no cut-off, database or column is a fact of any run); rows with no
    ENRICHMENT dataset behind them are "legacy"."""
    if not has_rows:
        return replace(
            source,
            enrichment_source="none",
            enrichment_fdr=None,
            enrichment_databases=None,
            enrichment_categories=None,
            padj_columns=(),
        )
    if source.enrichment_source == "none":
        return replace(source, enrichment_source="legacy")
    return source


def _provenance(
    source: NovaSource, comparison: str, alpha: float, referential_version: Optional[int]
) -> dict[str, Any]:
    return {
        "dataset_id": source.dataset_id,
        "comparison": comparison,
        "enrichment_dataset_ids": list(source.enrichment_dataset_ids),
        "analysis_id": source.analysis_id,
        "enrichment_source": source.enrichment_source,
        "enrichment_fdr": source.enrichment_fdr,
        "enrichment_databases": (
            list(source.enrichment_databases) if source.enrichment_databases is not None else None
        ),
        "enrichment_categories": (
            list(source.enrichment_categories) if source.enrichment_categories is not None else None
        ),
        "padj_columns": list(source.padj_columns),
        "source_versions": list(source.source_versions),
        "referential_version": referential_version,
        "alpha": alpha,
        "app_version": settings.APP_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "demo": source.demo,
    }


def build_nova(
    hits: Iterable[TermHit],
    mapping: ClaimSubgroupMap,
    *,
    alpha: float = DEFAULT_ALPHA,
    contrast: str = "",
    unlinked_limit: int = UNLINKED_LIMIT,
    source: Optional[NovaSource] = None,
) -> dict[str, Any]:
    """Assemble the nova payload from the comparison's enriched terms.

    Pure but for `provenance.generated_at`, which is the clock. The condition shown on the
    central disc is the numerator of `contrast`.
    """
    unique_hits = _dedupe(hits)
    source = _settled(source or NovaSource(), bool(unique_hits))
    # (term, its adjusted p): a term without one is left out of everything but the counts.
    scored = [(h, h.p) for h in unique_hits if h.p is not None and not math.isnan(h.p)]
    n_without_padj = len(unique_hits) - len(scored)

    # A term enriched in the up list and in the down list is flagged on both of its entries.
    directions: dict[str, set[str]] = defaultdict(set)
    for hit, _p in scored:
        directions[_term_key(hit)].add(hit.obs)

    claim_groups: dict[str, dict[str, list[_Term]]] = defaultdict(lambda: defaultdict(list))
    unlinked_groups: dict[str, list[_Term]] = defaultdict(list)
    n_matched = 0

    for hit, p in scored:
        rows = mapping.lookup(hit)
        label = clean_label(hit.label) or hit.term_id
        both = len(directions[_term_key(hit)]) > 1
        significant = p < alpha
        if rows:
            n_matched += 1
        linked = [r for r in rows if r.claim is not None]
        for row in linked:
            effect = 1 if hit.obs == row.expected else -1
            claim_groups[row.claim][row.subgroup].append(  # type: ignore[index]
                _Term(
                    _display_id(row.term_id),
                    row.label or label,
                    p,
                    effect,
                    hit.obs,
                    both,
                    significant,
                )
            )
        if not linked:
            name = rows[0].subgroup if rows else subgroup_name(label, hit.category)
            unlinked_groups[name].append(
                _Term(_display_id(hit.term_id), label, p, 0, hit.obs, both, significant)
            )

    claims: list[dict[str, Any]] = []
    for claim, groups in claim_groups.items():
        if not any(t.significant for terms in groups.values() for t in terms):
            continue
        subgroups = sorted(
            (_subgroup_payload(name, terms) for name, terms in groups.items()),
            key=lambda s: (-s["nl"], s["name"]),
        )
        s_pro = sum(s["s_pro"] for s in subgroups)
        s_anti = sum(s["s_anti"] for s in subgroups)
        best = subgroups[0]
        best_for = best_supporting_subgroup(subgroups)
        sides = [subgroup_side(s) for s in subgroups]
        n_for, n_against = sides.count("for"), sides.count("against")
        claims.append(
            {
                "name": claim,
                "state": claim_state(s_pro, s_anti),
                "importance": s_pro + s_anti,
                "s_pro": s_pro,
                "s_anti": s_anti,
                "coherence": coherence_of(s_pro, s_anti),
                "n_for": n_for,
                "n_against": n_against,
                "n_balanced": sides.count("balanced"),
                "best_subgroup": best["name"],
                "best_nl": best["nl"],
                "best_for_subgroup": best_for["name"] if best_for else None,
                "best_for_nl": best_for["nl"] if best_for else None,
                "subgroups": subgroups,
            }
        )
    claims.sort(key=lambda c: (-c["s_pro"], c["name"]))

    kept = {c["name"] for c in claims}
    in_referential = {row.claim for row in mapping.rows if row.claim is not None}
    not_evaluated = [
        name for name in CLAIM_NAMES.values() if name in in_referential and name not in kept
    ]

    sig_nl = nl_of(alpha)
    unlinked = sorted(
        (
            s
            for s in (_subgroup_payload(name, terms) for name, terms in unlinked_groups.items())
            if s["nl"] > sig_nl
        ),
        key=lambda s: (-s["nl"], s["name"]),
    )

    n_significant = sum(1 for _hit, p in scored if p < alpha)
    return {
        "status": "ok" if unique_hits else "no_enrichment",
        "contrast": contrast,
        "condition": condition_label(contrast),
        "alpha": alpha,
        "p_source": p_source_of(source.enrichment_source, source.padj_columns),
        "claims": claims,
        "not_evaluated": not_evaluated,
        "unlinked": unlinked[:unlinked_limit],
        "unlinked_total": len(unlinked),
        "coverage": {
            "n_terms": len(unique_hits),
            "n_terms_significant": n_significant,
            "n_terms_not_significant": len(scored) - n_significant,
            "n_terms_without_padj": n_without_padj,
            "n_terms_matched": n_matched,
        },
        "provenance": _provenance(source, contrast, alpha, _referential_version(mapping)),
    }


def condition_label(comparison_name: str) -> str:
    """Name for the central disc: the numerator of "A_vs_B", whatever its length."""
    numerator = re.split(r"[_\s]+vs[_\s]+", comparison_name or "", maxsplit=1, flags=re.I)[0]
    return numerator.strip()


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


async def _analysis_params(db: AsyncSession, analysis_id: Any) -> Optional[dict]:
    """The parameters the self-service analysis ran with, None when there is no such analysis."""
    try:
        key = UUID(str(analysis_id))
    except ValueError:
        return None
    params = await db.scalar(select(SelfServiceAnalysis.params).where(SelfServiceAnalysis.id == key))
    return params if isinstance(params, dict) else None


def _enrichment_fdr(recorded: Any, params: Optional[dict]) -> Optional[float]:
    """The term cut-off an annoDB enrichment ran at, None when it cannot be told.

    The worker records the `--padj-cutoff` it passes to the script (`padj_cutoff` on the
    ENRICHMENT dataset): that is the answer. A dataset that predates the record falls back on the
    analysis parameters, which are only trustworthy when they ask for 0.05 (or for nothing): until
    the cut-off was passed to the script (2026-09-29) it ran at 0.05 whatever the parameters
    said, and since then it runs at what they say, so any other value may be either.
    """
    if isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
        return float(recorded)
    if params is None:
        return None
    asked = params.get("enrichment_fdr")
    if asked is None or asked == DEFAULT_ENRICHMENT_FDR:
        return DEFAULT_ENRICHMENT_FDR
    return None


def _distinct(values: Iterable[Any]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        if isinstance(value, str) and value.strip():
            seen.setdefault(value.strip())
    return tuple(seen)


def _recorded_categories(raw: Any) -> Optional[tuple[str, ...]]:
    """The categories an annoDB run recorded about itself; None for no record or a malformed one."""
    if isinstance(raw, list) and raw and all(isinstance(c, str) and c.strip() for c in raw):
        return tuple(c.strip() for c in raw)
    return None


async def resolve_nova_source(
    db: AsyncSession, deg_dataset: Dataset, comparison_name: str
) -> NovaSource:
    """Where the comparison's enrichment rows are read, and what is known of how they were made.

    The rows are read where the enrichment panel reads them: the ENRICHMENT dataset of the same
    analysis when there is one, else the DEG dataset (the legacy Python enrichment stores its rows
    there). An enrichment GenoLens ran itself (`source` "annodb") is described by the term cut-off
    it ran at (`_enrichment_fdr`), the analysis's `enrichment_databases` pick and the categories the
    run recorded; a deposited table by the column read as adjusted p. With no ENRICHMENT dataset
    nothing says an enrichment ran, so the analysis's parameters are not read as if one had: the
    kind is "none" here, which `build_nova` settles to "legacy" when rows exist under the DEG
    dataset.
    """
    enrichment = await find_enrichment_dataset(db, deg_dataset, comparison_name)
    deg_meta = deg_dataset.dataset_metadata or {}
    enr_meta = (enrichment.dataset_metadata or {}) if enrichment is not None else {}
    if enrichment is None:
        kind = "none"
    elif enr_meta.get("source") == "annodb":
        kind = "annodb"
    else:
        kind = "uploaded"

    analysis_id = deg_meta.get("analysis_id") or enr_meta.get("analysis_id")
    fdr: Optional[float] = None
    databases: Optional[tuple[str, ...]] = None
    categories: Optional[tuple[str, ...]] = None
    if kind == "annodb":
        params = await _analysis_params(db, analysis_id) if analysis_id else None
        fdr = _enrichment_fdr(enr_meta.get("padj_cutoff"), params)
        if params is not None:
            raw = params.get("enrichment_databases")
            databases = tuple(d for d in raw if d) if isinstance(raw, list) else ()
        categories = _recorded_categories(enr_meta.get("categories"))

    return NovaSource(
        dataset_id=str(deg_dataset.id),
        pathway_dataset_id=enrichment.id if enrichment is not None else deg_dataset.id,
        enrichment_dataset_ids=(str(enrichment.id),) if enrichment is not None else (),
        analysis_id=str(analysis_id) if analysis_id else None,
        enrichment_fdr=fdr,
        enrichment_databases=databases,
        enrichment_categories=categories,
        enrichment_source=kind,
        padj_columns=_distinct([enr_meta.get("padj_column")]) if kind == "uploaded" else (),
        source_versions=_distinct([deg_meta.get("source_version"), enr_meta.get("source_version")]),
        demo=bool(deg_meta.get("demo") or enr_meta.get("demo")),
    )


async def fetch_term_hits(
    db: AsyncSession, source: NovaSource, comparison_name: str
) -> list[TermHit]:
    """The comparison's directional enrichment rows, read where `source` says they are.

    An exact comparison-name match — never a prefix match, so "A_vs_B" cannot pick up the rows of
    "A_vs_B_24h". A row with no adjusted p-value comes back with `p` None: the raw p-value is
    never read in its place.
    """
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
            EnrichmentPathway.gene_count,
            EnrichmentPathway.genes,
        ).where(
            EnrichmentPathway.dataset_id == source.pathway_dataset_id,
            EnrichmentPathway.comparison_name.in_(names),
        )
    )
    hits: list[TermHit] = []
    for row in result.all():
        obs = _row_direction(row.regulation, row.comparison_name)
        if obs is None:
            continue
        hits.append(
            TermHit(
                term_id=str(row.pathway_id),
                label=str(row.pathway_name or row.pathway_id),
                p=float(row.padj) if row.padj is not None else None,
                obs=obs,
                category=row.category,
                gene_count=row.gene_count,
                genes=tuple(str(g) for g in (row.genes or []) if g),
            )
        )
    return hits


def nova_from_hits(
    hits: Iterable[TermHit],
    comparison_name: str,
    alpha: float = DEFAULT_ALPHA,
    source: Optional[NovaSource] = None,
) -> dict[str, Any]:
    """The nova of one comparison from its enriched terms, joined to the default referential.

    Shared by the `cosmomics-nova` endpoints and the OmicSkin editorial report.
    """
    return build_nova(
        hits, default_claim_subgroups(), alpha=alpha, contrast=comparison_name, source=source
    )


async def compute_nova(
    db: AsyncSession, dataset: Dataset, comparison_name: str, alpha: float = DEFAULT_ALPHA
) -> tuple[dict[str, Any], list[TermHit]]:
    """The nova of one comparison as served by the API, plus the enriched terms behind it."""
    source = await resolve_nova_source(db, dataset, comparison_name)
    hits = await fetch_term_hits(db, source, comparison_name)
    return nova_from_hits(hits, comparison_name, alpha, source), hits


def term_gene_map(
    hits: Iterable[TermHit], mapping: ClaimSubgroupMap
) -> dict[tuple[str, str], set[str]]:
    """Genes per referential term and direction, joined exactly as `build_nova` joins terms.

    Keys are `(normalize_pathway_id(term["id"]), term["obs"])` for the terms of the nova payload:
    a term enriched in both lists has two entries, and each takes the genes of its own list only.
    A report can list the genes behind a claim without re-reading the referential.
    """
    genes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for hit in hits:
        if not hit.genes or not _has_p(hit):
            continue
        for row in mapping.lookup(hit):
            genes[(normalize_pathway_id(row.term_id), hit.obs)].update(hit.genes)
    return dict(genes)
