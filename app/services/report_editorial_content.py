"""
Content of the OmicSkin editorial report: pure functions turning the collected data (comparison,
DEGs, enrichment, nova, OmicSkin scores, AI texts) into the list of pages the template renders.

Every sentence is built from the data — no hand-written study text, no LLM: claim statements,
the verdict headline and the lede are deterministic. Page layout follows
docs/omicskin-report-editorial/build_D.py:

  cover · verdict · nova [· nova AI reading] · one page per strongly supported claim (≥ 75) ·
  "with reservations" pages (25–74, two per page, then claims < 25 and unscored) ·
  science I–IV · method and traceability

No Skin Stack page: the claim referential has no sub-group → compartment allocation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from markupsafe import Markup, escape

from app.services.cosmetics_taxonomy import normalize_pathway_id
from app.services.cosmomics_grouping import clean_label
from app.services.cosmomics_nova import (
    DEFAULT_ALPHA,
    ZERO_AGAINST_NOTE,
    best_supporting_subgroup,
    claim_state_of,
    subgroup_side,
)
from app.services.cosmomics_nova_svg import claim_color
from app.services.omicskin_score import ALL_CLAIMS, BAND_LABELS, ClaimScore, claim_score
from app.services.report_common import ai_label, ai_text_record
from app.services.report_editorial_figures import volcano_sampling
from app.services.report_editorial_svg import (
    clean_inline_svg,
    dose_ladder,
    gene_chips,
    ring,
    subgroup_bars,
)

NAVY, INK, GREY, LINE = "#0B2A3E", "#1B2733", "#7D8B9C", "#E3E8ED"
GREEN, BLUE, CORAL = "#34B596", "#33A8E3", "#FF5F5F"

DEFAULT_INSTITUTE = "SciLicium"
DEFAULT_ADDRESS = "35 boulevard du Portugal, 35200 Rennes, France"
DEFAULT_EMAIL = "contact@scilicium.com"

CLAIM_TITLES = {
    "Barrier": "Barrier reinforcement",
    "Regeneration": "Epidermal regeneration",
}

# AI texts are capped so a fixed-height page never overflows (pages clip their content).
NOVA_READING_INLINE_CHARS = 1400
NOVA_READING_PAGE_CHARS = 6500
COMPARISON_AI_CHARS = 3200
COMPARISON_AI_WIDE_CHARS = 6000

TERMS_PER_CLAIM_PAGE = 10
RESERVATIONS_PER_PAGE = 2

_NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
# What a claim's state says, for the verdict: the direction of the terms found for it. A claim
# without a significant term is "not evaluated" (n.e.): it has no state of its own.
_STATE_PHRASES = {
    "for_only": "with evidence for the claim only",
    "both_ways": "with evidence for and against",
    "against_only": "with evidence against the claim only",
}
_LADDER_LABELS = {  # short band names of the verdict ladder
    "strong": "STRONG",
    "moderate": "MODERATE",
    "weak": "WEAK",
    "not_supported": "WEAK EVIDENCE",
}
NOT_EVALUATED = "n.e."


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def number_word(n: int) -> str:
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def thousands(n: Optional[float]) -> str:
    """1459 → "1 459" (narrow no-break space), as in the reference."""
    if n is None:
        return "—"
    return f"{int(round(n)):,}".replace(",", " ")


def fmt_p(p: Optional[float]) -> Markup:
    """p-value as m.m×10⁻ᵉ (HTML superscript); plain decimals above 0.001."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return Markup("—")
    p = max(float(p), 1e-300)
    if p >= 1e-3:
        return Markup(f"{p:.3f}")
    e = math.floor(math.log10(p))
    m = p / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return Markup(f"{m:.1f}×10<sup>{e}</sup>")


def fmt_p_from_nl(nl: float) -> Markup:
    return fmt_p(10 ** -float(nl))


def _plain_p(p: float) -> str:
    if p >= 1e-3:
        return f"{p:.3f}"
    e = math.floor(math.log10(max(p, 1e-300)))
    m = p / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f}×10{sup}"


_WORD = re.compile(r"[a-zà-ÿœæ]+", re.I)
_EN = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "is",
    "are",
    "with",
    "that",
    "this",
    "for",
    "which",
    "genes",
    "expression",
    "suggests",
    "these",
    "was",
    "were",
    "by",
    "an",
    "as",
    "on",
}
_FR = {
    "le",
    "la",
    "les",
    "des",
    "et",
    "est",
    "sont",
    "dans",
    "une",
    "du",
    "pour",
    "avec",
    "que",
    "qui",
    "gènes",
    "cette",
    "ces",
    "au",
    "aux",
    "par",
    "sur",
    "pas",
    "plus",
}


def looks_english(text: Optional[str]) -> bool:
    """Stop-word vote: True when the text reads as English rather than French."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    if not words:
        return False
    en = sum(w in _EN for w in words)
    fr = sum(w in _FR for w in words)
    return en >= 3 and en > 1.5 * fr


def cap_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Cut at the last paragraph (else sentence) boundary under max_chars."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    for sep in ("\n\n", "\n", ". "):
        i = cut.rfind(sep)
        if i > max_chars * 0.5:
            return cut[: i + (1 if sep == ". " else 0)].rstrip() + " …", True
    return cut.rstrip() + " …", True


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])[*_](?!\s)(.+?)(?<!\s)[*_](?![\w*])")
_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")


def _inline(text: str) -> str:
    s = str(escape(text))
    s = _BOLD.sub(r"<b>\1</b>", s)
    s = _ITAL.sub(r"<i>\1</i>", s)
    s = _CODE.sub(r"\1", s)
    return s


def md_to_html(text: Optional[str]) -> Markup:
    """Minimal, escape-first markdown → HTML for LLM texts: headings, lists, paragraphs, bold,
    italic. Raw HTML in the text is escaped, never interpreted."""
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + " ".join(_inline(x) for x in para) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
            items.clear()

    for line in (text or "").splitlines():
        if not line.strip():
            flush()
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            out.append(f"<h4>{_inline(heading.group(1).strip().strip('*'))}</h4>")
            continue
        bullet = _BULLET.match(line)
        if bullet:
            if para:
                flush()
            items.append(bullet.group(1))
            continue
        if items:
            flush()
        para.append(line.strip())
    flush()
    return Markup("".join(out))


def sanitize_condition(name: Optional[str]) -> str:
    s = str(name or "").replace("_", " ")
    return " ".join(s.split())


def _relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def on_navy(hex_color: str) -> str:
    """Claim colour made legible on the navy cover (dark claim colours are lightened)."""
    lum = _relative_luminance(hex_color)
    if lum >= 0.25:
        return hex_color
    mix = 0.55 if lum < 0.08 else 0.35
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    return "#%02X%02X%02X" % tuple(round(c + (255 - c) * mix) for c in rgb)


# ---------------------------------------------------------------------------
# Claim content
# ---------------------------------------------------------------------------


def claim_title(name: str) -> str:
    return CLAIM_TITLES.get(name, name)


def supporting_subgroup(claim: dict) -> Optional[tuple[str, float]]:
    """The claim's most significant sub-group among those that go in the direction of the claim,
    with its −log10 p; None when no sub-group does. Read from the payload's `best_for_subgroup`;
    a payload that predates the field is searched."""
    if "best_for_subgroup" in claim:
        name = claim["best_for_subgroup"]
        return (str(name), float(claim.get("best_for_nl") or 0.0)) if name else None
    best = best_supporting_subgroup(claim.get("subgroups") or [])
    return (str(best["name"]), float(best["nl"])) if best else None


def state_of(claim: dict) -> str:
    """for_only / both_ways / against_only (see `cosmomics_nova.claim_state_of`)."""
    return claim_state_of(claim)


def _n_balanced(claim: dict) -> int:
    if claim.get("n_balanced") is not None:
        return int(claim["n_balanced"])
    return sum(1 for s in claim.get("subgroups") or [] if subgroup_side(s) == "balanced")


def _sub_groups(n: int) -> str:
    return f"{number_word(n)} sub-group{'s' if n != 1 else ''}"


def claim_statement(cs: ClaimScore, test: str, control: str) -> str:
    """Deterministic statement of a scored claim, built from its nova entry only.

    A claim is "led by" its most significant sub-group among those that go in the direction of the
    claim (`best_for_subgroup`), not by the most significant one whatever its direction; a claim
    with no such sub-group says so."""
    c = cs.claim or {}
    subs = c.get("subgroups") or []
    n_for, n_against = int(c.get("n_for") or 0), int(c.get("n_against") or 0)
    n_balanced = _n_balanced(c)
    coherence = round(100 * float(c.get("coherence") or 0))
    versus = f" in {test} versus {control}" if control else f" in {test}"
    n_sub = len(subs)
    lead = supporting_subgroup(c)
    if lead is None:
        s = f"No pathway sub-group supports the {cs.name} claim{versus}"
    else:
        if n_sub == 1 and n_for == 1:
            who = "The only pathway sub-group supports"
        elif n_for == n_sub:
            who = f"All {number_word(n_sub)} pathway sub-groups support"
        else:
            who = (
                f"{number_word(n_for).capitalize()} of the {number_word(n_sub)} pathway "
                f"sub-groups support{'s' if n_for == 1 else ''}"
            )
        s = f"{who} the {cs.name} claim{versus}, led by {lead[0]} (p = {_plain_p(10 ** -lead[1])})"
    s += f". {coherence} % of the evidence goes in the direction of the claim"
    if n_against:
        against = [x.get("name") for x in subs if subgroup_side(x) == "against"][:2]
        verb = "goes" if n_against == 1 else "go"
        s += (
            f"; {_sub_groups(n_against)} "
            f"({', '.join(str(a) for a in against)}{', …' if n_against > 2 else ''}) {verb} the "
            f"other way"
        )
    if n_balanced:
        # A balanced sub-group (evidence for and against cancel) is neither for nor against.
        s += (
            f"; {_sub_groups(n_balanced)} {'is' if n_balanced == 1 else 'are'} balanced "
            f"(evidence for and against cancel)"
        )
    return s + "."


def key_genes(
    claim: dict, term_genes: dict[tuple[str, str], set[str]], degs: Sequence[dict], limit: int = 7
) -> list[tuple[str, float]]:
    """DEGs behind the claim's supporting terms, largest |log2 FC| first."""
    genes: set[str] = set()
    for sg in claim.get("subgroups") or []:
        for t in sg.get("terms") or []:
            if (t.get("effect") or 0) > 0:
                found = term_genes.get((normalize_pathway_id(t.get("id")), t.get("obs")), ())
                genes |= {g.upper() for g in found}
    if not genes:
        return []
    picked: dict[str, tuple[str, float]] = {}
    for g in degs:
        if (g.get("regulation") or "").upper() not in ("UP", "DOWN") or g.get("log_fc") is None:
            continue
        keys = {str(g.get("gene_name") or "").upper(), str(g.get("gene_id") or "").upper()}
        if keys & genes:
            name = str(g.get("gene_name") or g.get("gene_id"))
            if name not in picked or abs(g["log_fc"]) > abs(picked[name][1]):
                picked[name] = (name, float(g["log_fc"]))
    return sorted(picked.values(), key=lambda x: -abs(x[1]))[:limit]


def terms_rows(claim: dict, limit: int = TERMS_PER_CLAIM_PAGE) -> tuple[list[dict], int]:
    """Rows of the ontology-term table (sub-groups in decreasing significance) and the total."""
    rows: list[dict] = []
    n = 0
    for sg in sorted(claim.get("subgroups") or [], key=lambda s: -(s.get("nl") or 0)):
        for j, t in enumerate(sorted(sg.get("terms") or [], key=lambda t: -(t.get("nl") or 0))):
            n += 1
            if n > limit:
                continue
            rows.append(
                {
                    "subgroup": sg.get("name") if j == 0 else "",
                    "label": t.get("label") or t.get("id"),
                    "id": t.get("id") or "",
                    "nl": float(t.get("nl") or 0),
                    "up": t.get("obs") == "up",
                    "pro": (t.get("effect") or 0) > 0,
                    # Enriched in both lists: it has one entry per direction, one for and one
                    # against, each marked.
                    "both": bool(t.get("both_directions")),
                    # Below alpha it still weighs in the sums; a payload without the flag is
                    # read as significant.
                    "significant": t.get("significant", True) is not False,
                }
            )
    return rows, n


def dose_points(
    name: str, dose_series: Optional[Sequence[tuple[str, dict]]]
) -> list[tuple[str, int]]:
    """OmicSkin score of one claim at each dose (0 when the claim has no evidence there)."""
    points = []
    for label, payload in dose_series or []:
        claim = next((c for c in payload.get("claims") or [] if c.get("name") == name), None)
        value = claim_score(claim) if claim else 0
        points.append((label, value))
    return points


def _dose_caption(points: list[tuple[str, int]]) -> str:
    values = [v for _, v in points]
    monotonic = all(b >= a for a, b in zip(values, values[1:]))
    trend = "Monotonic increase with dose" if monotonic else "Non-monotonic response across doses"
    return f"{trend}; {points[-1][0]} reaches {values[-1]}."


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class EditorialInputs:
    """Everything the editorial report needs, already collected (no DB access below)."""

    comparison_name: str
    test: str
    control: str
    project_name: str
    degs: list[dict]
    n_up: int
    n_down: int
    enr_up: list[dict]
    enr_down: list[dict]
    qc_rows: list[dict]
    sample_conditions: dict[str, str]
    params: dict[str, Any]
    has_analysis: bool
    analysis_label: Optional[str]
    nova: dict[str, Any]
    scores: list[ClaimScore]
    term_genes: dict[tuple[str, str], set[str]]
    figures: dict[str, Optional[str]]  # pca, volcano, dot_up, dot_down, heatmap (raw SVG)
    nova_light: str
    nova_dark: str
    # The AI texts: as stored / generated, with the model and the moment of generation the label
    # names. `*_stale`: a stored text was left out because it was written from other data than
    # these (the traceability page says so).
    comparison_ai: Optional[str] = None
    comparison_ai_model: Optional[str] = None
    comparison_ai_generated_at: Optional[datetime] = None
    comparison_ai_stale: bool = False
    nova_ai: Optional[str] = None
    nova_ai_model: Optional[str] = None
    nova_ai_generated_at: Optional[datetime] = None
    nova_ai_stale: bool = False
    cover_info: dict[str, Any] = field(default_factory=dict)
    branding: Optional[dict[str, Any]] = None
    logo_data_uri: Optional[str] = None
    referential_version: Optional[str] = None
    app_version: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    demo_watermark: bool = False
    # False: a report without AI — no AI text, box, notice or label is printed.
    use_ai: bool = True
    illustrative: bool = False
    dose_series: Optional[list[tuple[str, dict]]] = None
    # How the differentially expressed genes were called, as the DEG dataset records it: the method
    # that decided the calls (`primary_method`: DESeq2, edgeR or limma) and the methods the
    # analysis asked for (`de_method`: all, deseq2, edger or limma). None = not recorded.
    de_primary_method: Optional[str] = None
    de_method_run: Optional[str] = None
    # The DEG dataset's metadata names the analysis that produced it, and that analysis cannot be
    # found any more (deleted): the results are orphaned, not imported (the LaTeX report says the
    # same, `report_latex_methods.not_described_tex`).
    analysis_gone: bool = False


def nova_alpha(nova: dict) -> float:
    """The significance threshold of the nova (a payload that predates the field: the default)."""
    return float(nova.get("alpha") or DEFAULT_ALPHA)


def _headline(scores: list[ClaimScore], no_enrichment: bool = False) -> str:
    """The verdict, descriptive: how many claims have evidence for, against or both ways, and how
    many were not evaluated. It does not say that a claim is supported."""
    if not any(s.scored for s in scores):
        if no_enrichment:
            return "No enrichment result is stored for this comparison; no claim is evaluated."
        return "No claim has a significant term in this comparison."
    counts: dict[str, int] = {}
    for s in scores:
        if s.scored:
            state = state_of(s.claim or {})
            counts[state] = counts.get(state, 0) + 1
    parts = []
    for state in ("for_only", "both_ways", "against_only"):
        n = counts.get(state, 0)
        if not n:
            continue
        lead = (
            f"{number_word(n).capitalize()} claim{'s' if n != 1 else ''}"
            if not parts
            else number_word(n)
        )
        parts.append(f"{lead} {_STATE_PHRASES[state]}")
    n_ne = sum(1 for s in scores if not s.scored)
    if n_ne:
        parts.append(f"{number_word(n_ne)} not evaluated ({NOT_EVALUATED})")
    return ", ".join(parts) + "."


def _samples_by_condition(qc_rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in qc_rows:
        cond = str(r.get("condition") or "Unknown")
        out[cond] = out.get(cond, 0) + 1
    return out


def _qc_sentence(qc_rows: list[dict]) -> str:
    if not qc_rows:
        return "Per-sample quality-control metrics are not available for this dataset."
    libs = [r.get("library_size") or 0 for r in qc_rows]
    det = [r.get("detected_genes") or 0 for r in qc_rows]
    lib = (
        f"library sizes ranged from {reads(min(libs))} to {reads(max(libs))} assigned reads"
        if min(libs) != max(libs)
        else f"every library holds {reads(min(libs))} assigned reads"
    )
    genes = (
        f"{thousands(min(det))}–{thousands(max(det))} genes were detected per sample"
        if min(det) != max(det)
        else f"{thousands(min(det))} genes were detected per sample"
    )
    return f"Across the {len(qc_rows)} samples of the analysis, {lib} and {genes}."


def reads(n: float) -> str:
    """Library size: 4.3 M, 48.2 k, or the count itself below a thousand."""
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} k"
    return thousands(n)


def analysis_thresholds(params: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    def num(*keys):
        for k in keys:
            v = params.get(k)
            try:
                if v is not None:
                    return float(v)
            except (TypeError, ValueError):
                continue
        return None

    return num("fdr", "fdr_threshold", "padj_cutoff"), num("min_log2fc")


# ---------------------------------------------------------------------------
# Methods: what is said about the enrichment and the calls comes from what the data record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentFacts:
    """What the nova's provenance says about the enrichment behind the claims.

    The kind comes from `provenance.enrichment_source` and from nothing else: not from the analysis
    id, the enrichment dataset ids or the start of `p_source`. kind "genolens" (source "annodb"):
    the over-representation GenoLens ran (hypergeometric test, Benjamini–Hochberg correction),
    kept below `fdr` (None: the cut-off of the run was not recorded); `categories` are what the run
    recorded about itself, else `databases` is the analysis's pick (an empty tuple: it picked none,
    so all the categories ran; None: not known, the analysis is gone). kind "deposited" (source
    "uploaded"): tables somebody uploaded, kept as deposited, whose adjusted p is the column(s)
    read as such (`columns`, empty when not recorded); there is no cut-off and no database to
    name. kind "legacy": rows stored by an earlier GenoLens enrichment (no enrichment dataset): its
    test, correction, cut-off and databases are not recorded. kind "none": no enrichment result is
    stored (source "none", or a payload whose status says so, whatever else it carries): nothing
    more than that is said. kind "unknown": a payload without provenance, or whose provenance
    names no source.
    """

    kind: str
    columns: tuple[str, ...] = ()
    fdr: Optional[float] = None
    databases: Optional[tuple[str, ...]] = None
    categories: Optional[tuple[str, ...]] = None


_FACT_KIND_OF_SOURCE = {
    "annodb": "genolens",
    "uploaded": "deposited",
    "legacy": "legacy",
    "none": "none",
}


def enrichment_facts(nova: dict) -> EnrichmentFacts:
    if nova.get("status") == "no_enrichment":
        return EnrichmentFacts("none")
    prov = nova.get("provenance")
    if not isinstance(prov, dict):
        return EnrichmentFacts("unknown")
    kind = _FACT_KIND_OF_SOURCE.get(str(prov.get("enrichment_source")), "unknown")
    if kind == "deposited":
        return EnrichmentFacts("deposited", tuple(str(c) for c in prov.get("padj_columns") or []))
    if kind != "genolens":
        return EnrichmentFacts(kind)
    fdr = prov.get("enrichment_fdr")
    databases = prov.get("enrichment_databases")
    categories = prov.get("enrichment_categories")
    return EnrichmentFacts(
        "genolens",
        fdr=float(fdr) if isinstance(fdr, (int, float)) and not isinstance(fdr, bool) else None,
        databases=tuple(str(d) for d in databases) if isinstance(databases, list) else None,
        categories=(
            tuple(str(c) for c in categories)
            if isinstance(categories, list) and categories
            else None
        ),
    )


def _column_phrase(facts: EnrichmentFacts) -> str:
    if not facts.columns:
        return "column not recorded"
    return f"{'columns' if len(facts.columns) > 1 else 'column'} {', '.join(facts.columns)}"


def _cutoff_phrase(facts: EnrichmentFacts) -> str:
    """The term cut-off the enrichment ran at: the recorded one, never a default."""
    if facts.fdr is None:
        return "the cut-off of the run (cut-off not recorded)"
    return f"adjusted p < {facts.fdr:g}"


def _databases_phrase(facts: EnrichmentFacts) -> str:
    """What the enrichment GenoLens ran was run over: the categories the run recorded, else the
    analysis's pick, "all categories" only for an analysis that is known and picked none, and
    "not recorded" when nothing is known (a deleted analysis). Never a database that is not in the
    list."""
    if facts.categories:
        return ", ".join(facts.categories)
    if facts.databases:
        return ", ".join(facts.databases)
    if facts.databases is not None:
        return "all categories of the annotation database"
    return "not recorded"


def enrichment_method(facts: EnrichmentFacts) -> str:
    """One sentence on how the enrichment p-values were obtained (plain text)."""
    if facts.kind == "genolens":
        return (
            "Over-representation analysis of the up- and down-regulated genes was run by GenoLens "
            "(hypergeometric test, Benjamini–Hochberg correction)"
        )
    if facts.kind == "deposited":
        return (
            "The enrichment tables were deposited with the dataset; their adjusted p-value is "
            f"read from {_column_phrase(facts)}, and the test that produced it is not described here"
        )
    if facts.kind == "legacy":
        return (
            "The enrichment results come from an earlier GenoLens method whose parameters (test, "
            "correction, cut-off, databases) are not recorded"
        )
    if facts.kind == "none":
        return "No enrichment result is stored for this comparison"
    return (
        "The enrichment results are those stored with the dataset; their test and correction "
        "are not recorded"
    )


def enrichment_text(facts: EnrichmentFacts) -> str:
    """The sentence above the enrichment figures (plain text)."""
    if facts.kind == "genolens":
        text = f"{enrichment_method(facts)}; terms kept at {_cutoff_phrase(facts)}. "
        text += f"Databases: {_databases_phrase(facts)}. "
    elif facts.kind == "deposited":
        text = (
            f"{enrichment_method(facts)}. The terms are listed as deposited: GenoLens applied "
            f"no cut-off. "
        )
    elif facts.kind == "none":
        return f"{enrichment_method(facts)}."  # no result, so no figure to explain
    else:
        text = f"{enrichment_method(facts)}. "
    return text + "Dot size = number of genes, colour = significance."


_DE_NAMES = {"deseq2": "DESeq2", "edger": "edgeR", "limma": "limma-voom"}


def _de_name(method: str) -> str:
    return _DE_NAMES.get(str(method).lower(), str(method))


def _de_thresholds(params: dict[str, Any]) -> list[str]:
    fdr, min_lfc = analysis_thresholds(params)
    thr = []
    if fdr is not None:
        thr.append(f"adjusted p &lt; {fdr:g}")
    if min_lfc is not None:
        thr.append(f"|log<sub>2</sub> FC| ≥ {min_lfc:g}")
    return thr


def de_sentence(inp: EditorialInputs) -> Markup:
    """How the differentially expressed genes were called, from what the DEG dataset records: the
    method that decided the calls, and the others only when the analysis ran them. Imported
    results name no method, and results whose analysis was deleted say so (`analysis_gone`)."""
    primary = inp.de_primary_method
    thr = _de_thresholds(inp.params)
    cut = f" ({', '.join(thr)})" if thr else ""
    if inp.analysis_gone:
        # Whatever the dataset still records: the analysis is not there to describe, as in the
        # LaTeX report.
        s = (
            "The analysis that produced these results is no longer available; its methods are "
            "not described here."
        )
    elif primary:
        name = _de_name(primary)
        s = f"Differentially expressed genes were called with {escape(name)}{cut}"
        if str(inp.de_method_run or "").lower() == "all":
            others = [m for m in ("DESeq2", "edgeR", "limma-voom") if m != name]
            s += (
                f"; {' and '.join(others)} were run on the same counts to report how many "
                f"methods support each call, and do not change the DEG list."
            )
        else:
            s += "."
    elif inp.has_analysis:
        s = (
            "Differentially expressed genes come from the analysis run in GenoLens; the method "
            f"that decided the calls is not recorded for this comparison{cut}."
        )
    else:
        s = (
            "Differential expression results were imported with the dataset (imported results); "
            "how they were produced is not described here."
        )
    return Markup(s)


def _de_row(inp: EditorialInputs) -> str:
    if inp.analysis_gone:
        return "analysis no longer available (method not described)"
    if inp.de_primary_method:
        name = _de_name(inp.de_primary_method)
        if str(inp.de_method_run or "").lower() == "all":
            others = " and ".join(m for m in ("DESeq2", "edgeR", "limma-voom") if m != name)
            return f"{name} (decides the calls); {others} (concordance)"
        return f"{name} (decides the calls)"
    if inp.has_analysis:
        return "not recorded"
    return "imported results (method not described)"


def _design_text(inp: EditorialInputs, n_by_cond: dict[str, int]) -> str:
    n_test, n_ctrl = n_by_cond.get(inp.test), n_by_cond.get(inp.control)
    who = f"{escape(inp.test)}{f' (n = {n_test})' if n_test else ''}"
    ref = f"{escape(inp.control)}{f' (n = {n_ctrl})' if n_ctrl else ''}"
    s = f"{who} was compared with {ref}" if inp.control else f"{who} was analysed"
    s += f" ({escape(inp.project_name)})." if inp.project_name else "."
    s += f" {de_sentence(inp)}"
    s += f" {escape(enrichment_method(enrichment_facts(inp.nova)))}."
    return s


def _versions(inp: EditorialInputs, version: Optional[str]) -> list[tuple[str, str]]:
    fdr, min_lfc = analysis_thresholds(inp.params)
    facts = enrichment_facts(inp.nova)
    prov = inp.nova.get("provenance") or {}
    alpha = nova_alpha(inp.nova)
    rows: list[tuple[str, str]] = [("Analysis engine", f"GenoLens {inp.app_version}".strip())]
    if inp.analysis_label:
        rows.append(("Analysis", inp.analysis_label))
    rows.append(
        (
            "Comparison",
            (
                f"{inp.test} vs {inp.control} ({inp.comparison_name})"
                if inp.control
                else inp.comparison_name
            ),
        )
    )
    if inp.params.get("species"):
        rows.append(("Organism", str(inp.params["species"])))
    rows.append(("Differential expression", _de_row(inp)))
    thr = []
    if fdr is not None:
        thr.append(f"adjusted p < {fdr:g}")
    if min_lfc is not None:
        thr.append(f"|log2 FC| ≥ {min_lfc:g}")
    if inp.params.get("min_count") is not None:
        thr.append(f"min. {inp.params['min_count']} counts")
    if thr:
        rows.append(("DEG thresholds", "; ".join(thr)))
    enrichment = enrichment_method(facts)
    if facts.kind == "genolens":
        enrichment += f"; terms kept at {_cutoff_phrase(facts)}"
        rows.append(("Enrichment", enrichment))
        rows.append(("Enrichment databases", _databases_phrase(facts)))
    else:
        rows.append(("Enrichment", enrichment))
    sources = [str(v) for v in prov.get("source_versions") or []]
    rows.append(("Data source version", ", ".join(sources) if sources else "not recorded"))
    rows.append(
        (
            "Claim referential",
            (
                f"SciLicium claim_subgroups v{version}, {len(ALL_CLAIMS)} claims"
                if version
                else f"SciLicium claim_subgroups, {len(ALL_CLAIMS)} claims"
            ),
        )
    )
    if facts.kind != "none":
        # With no enrichment result no term is tested against α and no claim is scored.
        rows.append(
            (
                "Significance threshold",
                f"α = {alpha:g} (a term is significant when its adjusted p is below α)",
            )
        )
        rows.append(
            (
                "OmicSkin score",
                "100 × coherence × min(1, −log10 p of the most significant supporting "
                "sub-group / 8)",
            )
        )
    models = [m for m in (inp.comparison_ai_model, inp.nova_ai_model) if m]
    if models and inp.use_ai:
        rows.append(("Interpretation model", ", ".join(dict.fromkeys(models))))
    if inp.use_ai:
        # A stored text written from other data is left out, and the report says so.
        for stale, name in (
            (inp.nova_ai_stale and not inp.nova_ai, "Nova AI text"),
            (inp.comparison_ai_stale and not inp.comparison_ai, "Comparison AI text"),
        ):
            if stale:
                rows.append((name, "not included (generated on other data)"))
    rows.append(("Report generated", inp.generated_at.strftime("%d %B %Y, %H:%M UTC")))
    return rows


def method_nova(inp: EditorialInputs) -> Markup:
    """How the nova is built, in terms of what this report's data are (HTML)."""
    facts = enrichment_facts(inp.nova)
    if facts.kind == "none":
        # Nothing was joined, scored or combined: the paragraph says what is, and no more.
        return Markup(
            "No enrichment result is stored for this comparison, so no term is linked to the "
            "SciLicium claim referential and no claim is evaluated. The referential is "
            "versioned; its claim-to-pathway table is SciLicium know-how and is not disclosed."
        )
    alpha = nova_alpha(inp.nova)
    if facts.kind == "genolens":
        p_values = (
            "Benjamini–Hochberg adjusted p-values of the over-representation analysis run by "
            "GenoLens, on up- and down-regulated genes separately"
        )
    elif facts.kind == "deposited":
        p_values = f"adjusted p-values read from {_column_phrase(facts)} of the deposited tables"
    elif facts.kind == "legacy":
        p_values = (
            "adjusted p-values stored by an earlier GenoLens enrichment method, whose parameters "
            "are not recorded"
        )
    else:
        p_values = "adjusted p-values as stored with the enrichment results"
    return Markup(
        f"For each comparison, the enrichment results ({escape(p_values)}) are joined to the "
        f"SciLicium claim referential, a curated table that assigns ontology terms to "
        f"{number_word(len(ALL_CLAIMS))} cosmetic claims and groups them into pathway "
        "sub-groups, each with the expected direction of regulation. An enriched term counts "
        "toward a claim when the curated referential links it to that claim — <i>for</i> it "
        "when its observed direction matches the expected one, <i>against</i> otherwise; a term "
        "enriched in both lists (↑↓) counts once each way. "
        f"A term is significant when its adjusted p is below α (α = {alpha:g}); a claim with no "
        "significant term is n.e. (not evaluated), and terms not significant at α still "
        "contribute with their weight (marked n.s.). Within each sub-group the term p-values are "
        "combined by an unsigned Stouffer test: terms are not weighted, their direction is "
        "ignored, and they are treated as independent although they often share genes, so the "
        "combined p is indicative, not a calibrated probability. Balance = (S<sub>pro</sub> − "
        "S<sub>anti</sub>) / (S<sub>pro</sub> + S<sub>anti</sub>), S = Σ −log<sub>10</sub> p "
        "of the terms for or against. The referential is versioned; its claim-to-pathway table "
        "is SciLicium know-how and is not disclosed."
    )


def method_score(alpha: float) -> Markup:
    return Markup(
        "OmicSkin = 100 × coherence × min(1, s / 8), where coherence = S<sub>pro</sub> / "
        "(S<sub>pro</sub> + S<sub>anti</sub>) is the share of the claim's evidence that goes in "
        "its direction and s is the −log<sub>10</sub> p of its most significant sub-group among "
        "those that go that way (S<sub>pro</sub> &gt; S<sub>anti</sub>); the factor saturates at "
        "p = 10<sup>−8</sup>, and a claim with no such sub-group scores 0. Bands: ≥ 75 strong "
        "support, 50–74 moderate, 25–49 weak, &lt; 25 weak evidence. A claim with no term at "
        f"adjusted p &lt; α (α = {alpha:g}) is n.e. and has no score."
    )


METHOD_DOSE = " Scores are computed for each dose and checked for monotony across doses."
SCOPE = (
    "This report describes gene-expression changes and the pathways they enrich in the "
    "biological model tested. It does not by itself establish a clinical effect: a claim made "
    "to consumers needs its own supporting file (Regulation (EC) 1223/2009 and Regulation (EU) "
    "655/2013)."
)


def _weak_evidence_reason(claim: dict) -> str:
    """Why a scored claim stays below 25: little evidence, or evidence going the other way."""
    coh = round(100 * float(claim.get("coherence") or 0))
    lead = supporting_subgroup(claim)
    if lead is None:
        return f"No pathway sub-group goes in the direction of the claim (coherence {coh} %)."
    if coh >= 50:
        return (
            f"The evidence goes in the direction of the claim (coherence {coh} %) but stays "
            f"weak: most significant supporting sub-group p = {_plain_p(10 ** -lead[1])}."
        )
    return f"Most of the pathway evidence goes against the claim (coherence {coh} %)."


def _not_evaluated_text(nova: dict) -> str:
    """Why a claim of the referential is n.e. (not evaluated): no term of it is significant in the
    enrichment results. Nothing is said of how many terms were stored: a deposited table is kept
    as it was given, an enrichment GenoLens ran keeps only the terms under its cut-off."""
    if nova.get("status") == "no_enrichment":
        return "This comparison has no enrichment result, so no claim is evaluated."
    return (
        f"No term of this claim is significant at α (α = {nova_alpha(nova):g}); "
        "the claim is not scored."
    )


def _claim_block(
    cs: ClaimScore,
    inp: EditorialInputs,
    index: int,
    total: int,
    reviewer_text: str,
    compact: bool = False,
) -> dict[str, Any]:
    c = cs.claim or {}
    fill, text = claim_color(cs.name)
    genes = key_genes(c, inp.term_genes, inp.degs)
    points = dose_points(cs.name, inp.dose_series) if inp.dose_series else []
    rows, n_terms = terms_rows(c)
    if any(r["both"] or not r["significant"] for r in rows):
        # The table's footnote takes the room of two rows (the page clips what overflows).
        rows, n_terms = terms_rows(c, TERMS_PER_CLAIM_PAGE - 2)
    subs = c.get("subgroups") or []
    lead = supporting_subgroup(c)
    if compact:
        bars = subgroup_bars(subs, 300, 20, 12, INK, "#EEF2F5", 9, limit=6)
        ring_svg = ring(cs.score or 0, fill, 110, 10, "#EEF2F5", INK)
        dose_svg = dose_ladder(points, fill, 160, 60) if points else None
    else:
        bars = subgroup_bars(subs, 330, 24, 12, INK, limit=9)
        ring_svg = ring(cs.score or 0, fill, 150, 12, "#EEF2F5", INK)
        dose_svg = dose_ladder(points, fill, 200, 80) if points else None
    return {
        "name": cs.name,
        "title": claim_title(cs.name),
        "score": cs.score,
        "band": cs.band,
        "band_label": cs.label,
        "color": fill,
        "text_color": text,
        "ring": ring_svg,
        "kicker": f"Claim {index} of {total} · {claim_title(cs.name)}",
        "statement": claim_statement(cs, inp.test, inp.control),
        "n_for": int(c.get("n_for") or 0),
        "n_against": int(c.get("n_against") or 0),
        "n_balanced": _n_balanced(c),
        "coherence": round(100 * float(c.get("coherence") or 0)),
        "n_subgroups": len(subs),
        "bars": bars,
        "genes": gene_chips(genes) if genes else None,
        "dose": dose_svg,
        "dose_caption": _dose_caption(points) if points else None,
        # The most significant sub-group that goes in the direction of the claim; None without one.
        "lead": {"name": lead[0], "p": fmt_p_from_nl(lead[1])} if lead else None,
        "terms": rows,
        "n_terms": n_terms,
        "terms_truncated": n_terms > len(rows),
        # A claim with no contrary term at all (state for_only): no contrary term is not a
        # confirmation. A claim whose sub-groups are all for it but that has a contrary term does
        # not need the note.
        "zero_against_note": ZERO_AGAINST_NOTE if state_of(c) == "for_only" else None,
        "has_not_significant": any(not r["significant"] for r in rows),
        "has_both": any(r["both"] for r in rows),
        "review": reviewer_text,
    }


def build_editorial_context(inp: EditorialInputs) -> dict[str, Any]:
    """Pages and shared values for templates/report_editorial/report.html.j2."""
    scores = inp.scores
    scored = [s for s in scores if s.scored]
    strong = [s for s in scored if (s.score or 0) >= 75]
    reserved = [s for s in scored if 25 <= (s.score or 0) < 75]
    strip_claims = [s for s in scores if not s.scored or (s.score or 0) < 25]
    n_evidence = len(scored)

    branding = inp.branding or {}
    institute = branding.get("institute_name") or DEFAULT_INSTITUTE
    cover = inp.cover_info or {}
    checked_by = cover.get("checked_by")
    review_card = f"Report checked by {checked_by}." if checked_by else "Not yet reviewed."
    n_by_cond = _samples_by_condition(inp.qc_rows)
    n_test, n_ctrl = n_by_cond.get(inp.test, 0), n_by_cond.get(inp.control, 0)
    date_str = f"{inp.generated_at.day} {inp.generated_at.strftime('%B %Y')}"
    illustrative = " · Illustrative data" if inp.illustrative else ""
    alpha = nova_alpha(inp.nova)
    facts = enrichment_facts(inp.nova)
    no_enrichment = inp.nova.get("status") == "no_enrichment"

    pages: list[dict[str, Any]] = []

    # 1. cover
    title = inp.test
    pages.append(
        {
            "kind": "cover",
            "title": title,
            "title_size": 64 if len(title) <= 7 else 46 if len(title) <= 12 else 32,
            "subtitle": [
                s for s in (f"versus {inp.control}" if inp.control else "", inp.project_name) if s
            ],
            "strip": [
                {"score": s.score, "name": s.name, "color": on_navy(claim_color(s.name)[0])}
                for s in scored[:6]
            ],
            "nova": clean_inline_svg(inp.nova_dark, "cover-nova", "width:100%;height:auto"),
            "footer_left": " · ".join(
                x
                for x in (
                    f"Study {cover['client_ref']}" if cover.get("client_ref") else "",
                    f"Sponsor: {cover['sponsor_name']}" if cover.get("sponsor_name") else "",
                )
                if x
            )
            or inp.comparison_name,
            # The engine that produced the report, not a document version nobody maintains.
            "footer_right": (f"GenoLens {inp.app_version} · " if inp.app_version else "")
            + f"{date_str}{illustrative}",
        }
    )

    # 2. verdict
    n_deg = inp.n_up + inp.n_down
    lede = (
        f"{inp.test} was compared with {inp.control}" if inp.control else f"{inp.test} was analysed"
    )
    if n_test and n_ctrl:
        lede += f" ({n_test} versus {n_ctrl} samples)"
    if n_evidence == 0:
        evidence = f"none of the {len(ALL_CLAIMS)} claims of the referential has"
    elif n_evidence == 1:
        evidence = f"one of the {len(ALL_CLAIMS)} claims of the referential has"
    else:
        evidence = (
            f"{number_word(n_evidence)} of the {len(ALL_CLAIMS)} claims of the referential have"
        )
    lede += (
        f". {thousands(n_deg)} genes are differentially expressed "
        f"({thousands(inp.n_up)} up, {thousands(inp.n_down)} down)"
    )
    if no_enrichment:
        lede += "; no enrichment result is stored for this comparison."
    else:
        lede += f" and {evidence} a significant enriched term."
    # By state: which claims have terms for them only, for and against, or against only.
    for state, label in (
        ("for_only", "Evidence for the claim only"),
        ("both_ways", "Evidence for and against"),
        ("against_only", "Evidence against the claim only"),
    ):
        names = [s.name for s in scored if state_of(s.claim or {}) == state]
        if names:
            lede += f" {label}: {', '.join(names)}."
    stats = [{"value": thousands(n_deg), "label": "differentially expressed genes", "color": None}]
    if n_test and n_ctrl:
        stats.append(
            {
                "value": str(n_test + n_ctrl),
                "label": f"samples · {inp.test} vs {inp.control}",
                "color": None,
            }
        )
    else:
        stats.append({"value": thousands(inp.n_up), "label": "up-regulated genes", "color": None})
    stats.append(
        {
            "value": f"{n_evidence} / {len(ALL_CLAIMS)}",
            "label": f"claims evaluated ({NOT_EVALUATED} otherwise)",
            "color": None,
        }
    )
    stats.append(
        {"value": str(len(strong)), "label": "claims with strong support (≥ 75)", "color": GREEN}
    )
    ladder = []
    for s in scores:
        ladder.append(
            {
                "name": s.name,
                "score": s.score,
                "scored": s.scored,
                "band_short": _LADDER_LABELS[s.band] if s.scored else NOT_EVALUATED,
                "color": claim_color(s.name)[0],
            }
        )
    n_unscored = sum(1 for s in scores if not s.scored)
    unscored_note = ""
    if n_unscored and not no_enrichment:
        unscored_note = (
            f" {number_word(n_unscored).capitalize()} claim"
            + (" is" if n_unscored == 1 else "s are")
            + f" not evaluated ({NOT_EVALUATED}): "
            + ("none of its terms is" if n_unscored == 1 else "none of their terms is")
            + f" significant at α (α = {alpha:g})."
        )
    pages.append(
        {
            "kind": "verdict",
            "headline": _headline(scores, no_enrichment),
            "lede": lede,
            "stats": stats,
            "ladder": ladder,
            "ladder_title": (
                f"OmicSkin score, {inp.test} vs {inp.control}" if inp.control else "OmicSkin score"
            ),
            "caption": (
                "Score 0–100 = coherence × significance of the most significant sub-group that "
                "goes in the direction of the claim. Coherence = share of the claim's evidence "
                "that goes in its direction. "
                "Bands: ≥ 75 strong · 50–74 moderate · 25–49 weak · < 25 weak evidence."
                + unscored_note
            ),
            "guide": [],  # filled once page numbers are known
        }
    )

    # 3. nova (+ optional AI reading page)
    nova_text = (inp.nova_ai or "").strip() if inp.use_ai else ""
    nova_label = ai_label(inp.nova_ai_model or "", inp.nova_ai_generated_at)
    nova_inline = bool(nova_text) and len(nova_text) <= NOVA_READING_INLINE_CHARS
    ai_texts: list[dict[str, Any]] = []
    if nova_text:
        ai_texts.append(ai_text_record("nova", inp.nova_ai_model, inp.nova_ai_generated_at))
    pages.append(
        {
            "kind": "nova",
            "svg": clean_inline_svg(inp.nova_light, "page-nova", "width:100%;height:auto"),
            "version": inp.referential_version,
            "no_enrichment": no_enrichment,
            "ai_html": md_to_html(nova_text) if nova_inline else None,
            "ai_label": nova_label,
            "unlinked_total": int(
                inp.nova.get("unlinked_total") or len(inp.nova.get("unlinked") or [])
            ),
        }
    )
    if nova_text and not nova_inline:
        capped, truncated = cap_text(nova_text, NOVA_READING_PAGE_CHARS)
        pages.append(
            {
                "kind": "nova_reading",
                "ai_html": md_to_html(capped),
                "truncated": truncated,
                "ai_label": nova_label,
            }
        )

    # 4. one page per strongly supported claim
    n_claim_pages = len(strong) + len(reserved)
    claim_index = 0
    first_claim_page = len(pages) + 1
    for s in strong:
        claim_index += 1
        pages.append(
            {
                "kind": "claim",
                "claim": _claim_block(s, inp, claim_index, n_claim_pages, review_card),
            }
        )

    # 5. with reservations (+ the claims without support)
    chunks = [
        reserved[i : i + RESERVATIONS_PER_PAGE]
        for i in range(0, len(reserved), RESERVATIONS_PER_PAGE)
    ]
    if not chunks and strip_claims:
        chunks = [[]]
    for i, chunk in enumerate(chunks):
        blocks = []
        for s in chunk:
            claim_index += 1
            blocks.append(
                _claim_block(s, inp, claim_index, n_claim_pages, review_card, compact=True)
            )
        strip = []
        if i == len(chunks) - 1:
            for s in strip_claims:
                if s.scored:
                    strip.append(
                        {
                            "name": s.name,
                            "title": f"Score {s.score} · {BAND_LABELS[s.band].lower()}.",
                            "text": _weak_evidence_reason(s.claim or {}),
                        }
                    )
                else:
                    strip.append(
                        {
                            "name": s.name,
                            "title": (
                                f"{NOT_EVALUATED} — no enrichment result for this comparison."
                                if no_enrichment
                                else f"{NOT_EVALUATED} — no significant term for this claim in "
                                "the enrichment results."
                            ),
                            "text": _not_evaluated_text(inp.nova),
                        }
                    )
        if chunk:
            numbers = [str(claim_index - len(blocks) + k + 1) for k in range(len(blocks))]
            kicker = (
                f"Claim{'s' if len(numbers) > 1 else ''} {' and '.join(numbers)}"
                f" · with reservations"
            )
            headline = (
                "Supported, but with sub-groups going the other way."
                if any(b["n_against"] for b in blocks)
                else "Moderately or weakly supported."
            )
        else:
            kicker = "Claims with weak evidence or not evaluated"
            headline = (
                "No claim is evaluated in this comparison."
                if not scored
                else "The remaining claims have weak evidence or are not evaluated."
            )
        pages.append(
            {
                "kind": "reservations",
                "kicker": kicker,
                "headline": headline,
                "blocks": blocks,
                "strip": strip,
            }
        )
    last_claim_page = len(pages)

    # 6–9. science
    first_science_page = len(pages) + 1
    figures = {
        k: (clean_inline_svg(v, f"fig-{k}", "width:100%;height:auto") if v else None)
        for k, v in inp.figures.items()
    }
    design_rows = []
    order = [c for c in (inp.control, inp.test) if c] + sorted(
        c for c in n_by_cond if c not in (inp.control, inp.test)
    )
    for cond in order:
        design_rows.append(
            {
                "condition": cond,
                "n": n_by_cond.get(cond),
                "up": thousands(inp.n_up) if cond == inp.test else "—",
                "down": thousands(inp.n_down) if cond == inp.test else "—",
            }
        )
    pages.append(
        {
            "kind": "science_design",
            "design_text": Markup(_design_text(inp, n_by_cond)),
            "qc_sentence": _qc_sentence(inp.qc_rows),
            "design_rows": design_rows,
            "pca": figures.get("pca"),
            "n_samples": len(inp.qc_rows),
        }
    )

    def top_degs(direction: str, n: int) -> list[dict]:
        sel = [g for g in inp.degs if (g.get("regulation") or "").upper() == direction]
        sel.sort(
            key=lambda g: (
                g.get("padj") if g.get("padj") is not None else 1.0,
                -abs(g.get("log_fc") or 0),
            )
        )
        return sel[:n]

    deg_rows = [
        {
            "gene": g.get("gene_name") or g.get("gene_id"),
            "lfc": float(g.get("log_fc") or 0),
            "padj": fmt_p(g.get("padj")),
        }
        for g in top_degs("UP", 10) + top_degs("DOWN", 10)
    ]
    sampling = volcano_sampling(inp.degs)
    n_drawn_degs = sum(1 for g in inp.degs if (g.get("regulation") or "").upper() in ("UP", "DOWN"))
    pages.append(
        {
            "kind": "science_deg",
            "title": f"{inp.test} versus {inp.control}" if inp.control else inp.test,
            "volcano": figures.get("volcano"),
            "n_up": thousands(inp.n_up),
            "n_down": thousands(inp.n_down),
            "deg_rows": deg_rows,
            "de_text": de_sentence(inp),
            # The volcano draws every up- and down-regulated gene and, past a budget of points, a
            # sample of the others: its caption says so.
            "volcano_note": (
                f"All {thousands(n_drawn_degs)} differentially expressed genes are drawn; the "
                f"others are a sample of {thousands(sampling[0])} of {thousands(sampling[1])} genes."
                if sampling
                else None
            ),
        }
    )

    def enr_rows(rows: list[dict]) -> list[dict]:
        top = sorted((r for r in rows if r.get("padj") is not None), key=lambda r: r["padj"])[:8]
        return [
            {
                "term": clean_label(r.get("pathway_name")) or r.get("pathway_id"),
                "db": r.get("category") or "—",
                "genes": r.get("gene_count"),
                "padj": fmt_p(r.get("padj")),
            }
            for r in top
        ]

    if inp.enr_up or inp.enr_down:
        pages.append(
            {
                "kind": "science_enrichment",
                "dot_up": figures.get("dot_up"),
                "dot_down": figures.get("dot_down"),
                "rows_up": enr_rows(inp.enr_up),
                "rows_down": enr_rows(inp.enr_down),
                # From the provenance of the enrichment, never from a constant.
                "text": enrichment_text(facts),
            }
        )

    cmp_text = (inp.comparison_ai or "").strip() if inp.use_ai else ""
    english = looks_english(cmp_text) if cmp_text else False
    if figures.get("heatmap") and not inp.use_ai:
        # The heatmap alone: a report without AI has no interpretation box, not even empty.
        pages.append({"kind": "science_heatmap", "heatmap": figures["heatmap"], "ai": False})
    elif figures.get("heatmap") or cmp_text:
        wide = not figures.get("heatmap")
        capped, truncated = cap_text(
            cmp_text, COMPARISON_AI_WIDE_CHARS if wide else COMPARISON_AI_CHARS
        )
        if not cmp_text and inp.comparison_ai_stale:
            notice = (
                "The stored AI interpretation of this comparison was generated on other data "
                "and is not included."
            )
        elif not cmp_text:
            notice = "No AI interpretation has been generated for this comparison."
        elif not english:
            notice = "No English AI interpretation is available for this comparison."
        else:
            notice = None
        pages.append(
            {
                "kind": "science_heatmap",
                "heatmap": figures.get("heatmap"),
                "ai": True,
                "ai_html": md_to_html(capped) if english else None,
                "ai_notice": notice,
                "ai_label": ai_label(inp.comparison_ai_model or "", inp.comparison_ai_generated_at),
                "truncated": truncated and english,
            }
        )
        if cmp_text and english:
            ai_texts.append(
                ai_text_record(
                    "comparison", inp.comparison_ai_model, inp.comparison_ai_generated_at
                )
            )
    last_science_page = len(pages)

    # 10. method + versions + contact
    if branding.get("institute_name"):
        contact_lines = [
            x
            for x in (
                branding.get("institute_address"),
                cover.get("contact_email"),
                cover.get("contact_phone"),
            )
            if x
        ]
    else:
        contact_lines = [DEFAULT_ADDRESS, cover.get("contact_email") or DEFAULT_EMAIL]
        if cover.get("contact_phone"):
            contact_lines.append(cover["contact_phone"])
    pages.append(
        {
            "kind": "method",
            "method_nova": method_nova(inp),
            # None when nothing was scored (no enrichment result): the page has no such section.
            "method_score": (
                None
                if facts.kind == "none"
                else Markup(str(method_score(alpha)) + (METHOD_DOSE if inp.dose_series else ""))
            ),
            "scope": (
                "All AI-generated texts in this report are labelled with the model and the date "
                "of their generation. " + SCOPE
                if inp.use_ai
                else SCOPE
            ),
            "versions": _versions(inp, inp.referential_version),
            "signoff": [
                (k, cover.get(f))
                for k, f in (
                    ("Prepared by", "prepared_by"),
                    ("Checked by", "checked_by"),
                    ("Approved by", "approved_by"),
                )
                if cover.get(f)
            ],
            "institute": institute,
            "contact": " · ".join(str(x) for x in contact_lines),
        }
    )

    # page numbers + reading guide
    total = len(pages)
    for i, page in enumerate(pages, start=1):
        page["n"] = i
    guide = []
    if n_claim_pages or strip_claims:
        single = first_claim_page == last_claim_page
        span = (
            f"Page {first_claim_page}" if single else f"Pages {first_claim_page}–{last_claim_page}"
        )
        verb = "gives" if single else "give"
        n_weak = sum(1 for s in strip_claims if s.scored)
        reading = "the claims with weak evidence or not evaluated (n.e.)"
        if strong or reserved:
            lead = (
                "one page per strongly supported claim — the score, a statement built from the "
                "data, the pathways and genes behind it"
                if strong
                else "the claims supported with reservations — score, statement, pathways and genes"
            )
            # What follows, by what is there: the claims with reservations (when the first pages
            # were the strong ones), then those with weak evidence or not evaluated.
            then = (["the claims supported with reservations"] if strong and reserved else []) + (
                [reading] if strip_claims else []
            )
            text = f"{span} {verb} {lead}" + (f" — then {' and '.join(then)}" if then else "") + "."
        else:
            lists = "lists" if single else "list"
            if no_enrichment:
                what = (
                    f"the {len(ALL_CLAIMS)} claims and why none is evaluated: this comparison "
                    "has no enrichment result"
                )
            elif n_weak and n_weak < len(strip_claims):
                what = (
                    "the claims with weak evidence (score below 25) and the claims not "
                    f"evaluated ({NOT_EVALUATED})"
                )
            elif n_weak:
                what = "the claims, all with weak evidence (score below 25)"
            else:
                what = (
                    f"the {len(ALL_CLAIMS)} claims and why none of them is evaluated "
                    f"({NOT_EVALUATED}): no term of theirs is significant"
                )
            text = f"{span} {lists} {what}."
        guide.append(("How to read this review.", text))
    guide.append(
        (
            "Where the evidence comes from.",
            f"Pages {first_science_page}–{last_science_page} give the scientific results: "
            + (
                "study design, differential expression, functional enrichment and the AI "
                "interpretation, labelled with its model and date."
                if inp.use_ai
                else "study design, differential expression and functional enrichment."
            ),
        )
    )
    guide.append(
        (
            "Traceability.",
            f"Page {total} lists every version behind the score (analysis engine, "
            + (
                "thresholds, claim referential, interpretation model)."
                if inp.use_ai
                else "thresholds, claim referential)."
            )
            + (" Illustrative data." if inp.illustrative else ""),
        )
    )
    pages[1]["guide"] = guide

    footer = f"OmicSkin PLUS · {inp.test} · {institute}{illustrative}"
    return {
        "pages": pages,
        "total": total,
        "ai_texts": ai_texts,  # the AI texts printed, as recorded on the report job
        "footer": footer,
        "title": f"OmicSkin PLUS — {inp.test}",
        "logo": inp.logo_data_uri,
        "demo_watermark": inp.demo_watermark,
        "colors": {
            "navy": NAVY,
            "ink": INK,
            "grey": GREY,
            "line": LINE,
            "green": GREEN,
            "blue": BLUE,
            "coral": CORAL,
        },
    }
