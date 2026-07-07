"""
Server-side Plotly figure builders for the agentic chat assistant.

The LLM only ever returns a small, validated spec (``chart_type`` + a few options —
see ``GenerateChartParams``). This module expands that spec into a full Plotly figure
dict ``{"data": [...traces...], "layout": {...}}`` by injecting the *real* data of the
currently selected DEG comparison. The model can therefore never emit an invalid or
data-hallucinating figure — it chooses the shape, the server owns the numbers.

Each builder returns ``(figure_dict | None, summary_for_model)`` where the summary is a
compact dict (counts / top-N / stats) fed back to the model — never the full data.

Data sources (reused read paths):
  - per-gene rows      → ``DegGene``          (log_fc, padj, pvalue, base_mean, regulation)
  - enrichment rows    → ``EnrichmentPathway`` (pathway_name, category, padj, gene_count)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import DegGene, EnrichmentPathway

# ── palette (matches the app + frontend chartPalettes) ────────────────────────

UP_COLOR = "#ef4444"
DOWN_COLOR = "#00BFA5"
NS_COLOR = "#d1d5db"

_PALETTES: Dict[str, Dict[str, str]] = {
    "standard": {"up": UP_COLOR, "down": DOWN_COLOR, "ns": NS_COLOR},
    "colorblind": {"up": "#D55E00", "down": "#0072B2", "ns": "#999999"},
}

# Numeric per-gene fields the LLM may reference for histogram / scatter.
NUMERIC_FIELDS = ("log_fc", "padj", "pvalue", "base_mean")

_FIELD_LABELS = {
    "log_fc": "log2 fold-change",
    "padj": "adjusted p-value",
    "pvalue": "p-value",
    "base_mean": "base mean",
}

# Cap on scatter-type point counts streamed to the browser (significant kept first).
MAX_POINTS = 6000


# ── data loading ──────────────────────────────────────────────────────────────

async def _load_deg_rows(
    db: AsyncSession, dataset_id: UUID, comparison_name: Optional[str]
) -> List[Dict[str, Any]]:
    """Load per-gene DEG rows for the selected comparison as plain dicts."""
    stmt = (
        select(
            DegGene.gene_id,
            DegGene.gene_name,
            DegGene.base_mean,
            DegGene.log_fc,
            DegGene.pvalue,
            DegGene.padj,
            DegGene.regulation,
        )
        .where(DegGene.dataset_id == dataset_id)
        .where(DegGene.comparison_name == comparison_name)
    )
    rows = (await db.execute(stmt)).all()
    return [
        {
            "gene": r.gene_name or r.gene_id,
            "gene_id": r.gene_id,
            "base_mean": r.base_mean,
            "log_fc": r.log_fc,
            "pvalue": r.pvalue,
            "padj": r.padj,
            "regulation": (r.regulation or "NS").upper(),
        }
        for r in rows
    ]


async def _load_pathways(
    db: AsyncSession, dataset_id: UUID, comparison_name: Optional[str], top_n: int
) -> List[Dict[str, Any]]:
    stmt = (
        select(EnrichmentPathway)
        .where(EnrichmentPathway.dataset_id == dataset_id)
        .where(EnrichmentPathway.comparison_name == comparison_name)
        .order_by(EnrichmentPathway.padj.asc())
        .limit(top_n)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "pathway_name": p.pathway_name,
            "category": p.category,
            "padj": p.padj,
            "gene_count": p.gene_count,
            "regulation": p.regulation,
        }
        for p in rows
    ]


# ── helpers ─────────────────────────────────────────────────────────────────--

def _neg_log10(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    # Clamp zero / tiny p-values so -log10 stays finite.
    v = max(float(value), 1e-300)
    return -math.log10(v)


def _resolve_palette(palette: Optional[str]) -> Dict[str, str]:
    if palette and palette.lower() in _PALETTES:
        return _PALETTES[palette.lower()]
    return _PALETTES["standard"]


def _classify(log_fc: Optional[float], padj: Optional[float],
              padj_thr: float, logfc_thr: float) -> str:
    if log_fc is None or padj is None:
        return "ns"
    if padj < padj_thr and abs(log_fc) > logfc_thr:
        return "up" if log_fc > 0 else "down"
    return "ns"


def _downsample(rows: List[Dict[str, Any]], keep: List[bool]) -> List[Dict[str, Any]]:
    """Keep every 'significant' row; sub-sample the rest to stay under MAX_POINTS."""
    if len(rows) <= MAX_POINTS:
        return rows
    significant = [r for r, k in zip(rows, keep) if k]
    others = [r for r, k in zip(rows, keep) if not k]
    budget = max(0, MAX_POINTS - len(significant))
    if budget and others:
        step = max(1, len(others) // budget)
        others = others[::step][:budget]
    return significant + others


def _layout(title: str, x_title: str, y_title: str, **extra: Any) -> Dict[str, Any]:
    layout: Dict[str, Any] = {
        "title": {"text": title},
        "xaxis": {"title": {"text": x_title}},
        "yaxis": {"title": {"text": y_title}},
        "plot_bgcolor": "#f9fafb",
        "paper_bgcolor": "#ffffff",
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }
    layout.update(extra)
    return layout


# ── builders (one per chart_type) ─────────────────────────────────────────────

def _build_volcano(rows, opts) -> Tuple[Optional[Dict], Dict]:
    padj_thr = opts.get("padj_threshold") or 0.05
    logfc_thr = opts.get("logfc_threshold") if opts.get("logfc_threshold") is not None else 0.58
    palette = _resolve_palette(opts.get("palette"))

    pts = [r for r in rows if r["log_fc"] is not None and r["padj"] is not None]
    classes = [_classify(r["log_fc"], r["padj"], padj_thr, logfc_thr) for r in pts]
    keep = [c != "ns" for c in classes]
    # Downsample jointly to keep classes aligned with points.
    paired = list(zip(pts, classes))
    if len(paired) > MAX_POINTS:
        sig = [p for p, k in zip(paired, keep) if k]
        rest = [p for p, k in zip(paired, keep) if not k]
        budget = max(0, MAX_POINTS - len(sig))
        if budget and rest:
            step = max(1, len(rest) // budget)
            rest = rest[::step][:budget]
        paired = sig + rest

    x = [p["log_fc"] for p, _ in paired]
    y = [_neg_log10(p["padj"]) for p, _ in paired]
    colors = [palette[c] for _, c in paired]
    text = [p["gene"] for p, _ in paired]
    single = opts.get("color")  # e.g. "make it blue" overrides all points
    marker_color = single if single else colors

    up = sum(1 for _, c in paired if c == "up")
    down = sum(1 for _, c in paired if c == "down")

    figure = {
        "data": [{
            "type": "scattergl",
            "mode": "markers",
            "x": x,
            "y": y,
            "text": text,
            "marker": {"color": marker_color, "size": 5, "opacity": 0.7},
            "hovertemplate": "<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10(padj): %{y:.2f}<extra></extra>",
            "name": "genes",
        }],
        "layout": _layout(
            opts.get("title") or "Volcano plot",
            "log2 fold-change", "-log10(adjusted p-value)",
            shapes=[
                {"type": "line", "x0": logfc_thr, "x1": logfc_thr, "yref": "paper",
                 "y0": 0, "y1": 1, "line": {"color": "#9ca3af", "width": 1, "dash": "dot"}},
                {"type": "line", "x0": -logfc_thr, "x1": -logfc_thr, "yref": "paper",
                 "y0": 0, "y1": 1, "line": {"color": "#9ca3af", "width": 1, "dash": "dot"}},
            ],
        ),
    }
    summary = {
        "chart_type": "volcano", "total_genes": len(pts),
        "significant": up + down, "up": up, "down": down,
        "thresholds": {"padj": padj_thr, "logfc": logfc_thr},
    }
    return figure, summary


def _build_histogram(rows, opts) -> Tuple[Optional[Dict], Dict]:
    field = opts.get("field") or "log_fc"
    if field not in NUMERIC_FIELDS:
        field = "log_fc"
    bins = opts.get("bins") or 40
    color = opts.get("color") or _resolve_palette(opts.get("palette"))["up"]
    values = [r[field] for r in rows if r.get(field) is not None]

    figure = {
        "data": [{
            "type": "histogram",
            "x": values,
            "nbinsx": int(bins),
            "marker": {"color": color},
            "name": field,
        }],
        "layout": _layout(
            opts.get("title") or f"Distribution of {_FIELD_LABELS.get(field, field)}",
            _FIELD_LABELS.get(field, field), "count",
        ),
    }
    summary = {"chart_type": "histogram", "field": field, "bins": int(bins), "n": len(values)}
    return figure, summary


def _build_ma_plot(rows, opts) -> Tuple[Optional[Dict], Dict]:
    palette = _resolve_palette(opts.get("palette"))
    reg_map = {"UP": "up", "DOWN": "down"}
    pts = [r for r in rows if r["base_mean"] and r["base_mean"] > 0 and r["log_fc"] is not None]
    if not pts:
        return None, {"chart_type": "ma_plot", "n": 0,
                      "note": "No base-mean values are available for this comparison, "
                              "so an MA plot cannot be drawn."}
    keep = [r["regulation"] in ("UP", "DOWN") for r in pts]
    pts = _downsample(pts, keep)

    x = [math.log10(r["base_mean"]) for r in pts]
    y = [r["log_fc"] for r in pts]
    colors = [palette[reg_map.get(r["regulation"], "ns")] for r in pts]
    text = [r["gene"] for r in pts]
    single = opts.get("color")

    figure = {
        "data": [{
            "type": "scattergl",
            "mode": "markers",
            "x": x,
            "y": y,
            "text": text,
            "marker": {"color": single if single else colors, "size": 5, "opacity": 0.7},
            "hovertemplate": "<b>%{text}</b><br>log10(baseMean): %{x:.2f}<br>log2FC: %{y:.2f}<extra></extra>",
            "name": "genes",
        }],
        "layout": _layout(
            opts.get("title") or "MA plot",
            "log10(base mean)", "log2 fold-change",
            shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                     "line": {"color": "#9ca3af", "width": 1, "dash": "dot"}}],
        ),
    }
    summary = {"chart_type": "ma_plot", "n": len(pts)}
    return figure, summary


def _build_bar_genes(rows, opts) -> Tuple[Optional[Dict], Dict]:
    palette = _resolve_palette(opts.get("palette"))
    top_n = opts.get("top_n") or 15
    ranked = sorted(
        (r for r in rows if r["log_fc"] is not None),
        key=lambda r: abs(r["log_fc"]), reverse=True,
    )[:int(top_n)]
    # Order for a horizontal bar: strongest at the top.
    ranked = list(reversed(ranked))
    genes = [r["gene"] for r in ranked]
    values = [r["log_fc"] for r in ranked]
    single = opts.get("color")
    colors = [(single if single else (palette["up"] if v >= 0 else palette["down"])) for v in values]

    figure = {
        "data": [{
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": genes,
            "marker": {"color": colors},
            "hovertemplate": "<b>%{y}</b><br>log2FC: %{x:.2f}<extra></extra>",
        }],
        "layout": _layout(
            opts.get("title") or f"Top {len(ranked)} genes by |log2FC|",
            "log2 fold-change", "",
            yaxis={"title": {"text": ""}, "automargin": True},
        ),
    }
    summary = {
        "chart_type": "bar_genes", "top_n": len(ranked),
        "genes": [{"gene": r["gene"], "log_fc": r["log_fc"]} for r in reversed(ranked)],
    }
    return figure, summary


def _build_bar_regulation(rows, opts) -> Tuple[Optional[Dict], Dict]:
    palette = _resolve_palette(opts.get("palette"))
    up = sum(1 for r in rows if r["regulation"] == "UP")
    down = sum(1 for r in rows if r["regulation"] == "DOWN")

    figure = {
        "data": [{
            "type": "bar",
            "x": ["Up-regulated", "Down-regulated"],
            "y": [up, down],
            "marker": {"color": [palette["up"], palette["down"]]},
            "hovertemplate": "%{x}: %{y}<extra></extra>",
        }],
        "layout": _layout(
            opts.get("title") or "Regulated gene counts", "regulation", "gene count",
        ),
    }
    summary = {"chart_type": "bar_regulation", "up": up, "down": down}
    return figure, summary


def _build_enrichment_bar(pathways, opts) -> Tuple[Optional[Dict], Dict]:
    if not pathways:
        return None, {"chart_type": "enrichment_bar", "returned": 0,
                      "note": "No enrichment pathways available for this comparison."}
    color = opts.get("color") or "#7C3AED"
    ordered = list(reversed(pathways))  # most significant at the top
    names = [p["pathway_name"] for p in ordered]
    values = [_neg_log10(p["padj"]) for p in ordered]

    figure = {
        "data": [{
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": names,
            "marker": {"color": color},
            "customdata": [[p["category"], p["gene_count"], p["padj"]] for p in ordered],
            "hovertemplate": ("<b>%{y}</b><br>-log10(padj): %{x:.2f}"
                              "<br>category: %{customdata[0]}<br>genes: %{customdata[1]}<extra></extra>"),
        }],
        "layout": _layout(
            opts.get("title") or f"Top {len(pathways)} enriched pathways",
            "-log10(adjusted p-value)", "",
            yaxis={"title": {"text": ""}, "automargin": True},
        ),
    }
    summary = {
        "chart_type": "enrichment_bar", "returned": len(pathways),
        "pathways": [{"pathway_name": p["pathway_name"], "category": p["category"],
                      "padj": p["padj"], "gene_count": p["gene_count"]} for p in pathways],
    }
    return figure, summary


def _build_scatter(rows, opts) -> Tuple[Optional[Dict], Dict]:
    x_field = opts.get("x_field") or "base_mean"
    y_field = opts.get("y_field") or "log_fc"
    if x_field not in NUMERIC_FIELDS:
        x_field = "base_mean"
    if y_field not in NUMERIC_FIELDS:
        y_field = "log_fc"
    color = opts.get("color") or "#2A2E5B"

    pts = [r for r in rows if r.get(x_field) is not None and r.get(y_field) is not None]
    if not pts:
        return None, {"chart_type": "scatter", "x_field": x_field, "y_field": y_field, "n": 0,
                      "note": f"No paired values for '{x_field}' and '{y_field}' in this comparison."}
    keep = [r["regulation"] in ("UP", "DOWN") for r in pts]
    pts = _downsample(pts, keep)

    figure = {
        "data": [{
            "type": "scattergl",
            "mode": "markers",
            "x": [r[x_field] for r in pts],
            "y": [r[y_field] for r in pts],
            "text": [r["gene"] for r in pts],
            "marker": {"color": color, "size": 5, "opacity": 0.7},
            "hovertemplate": "<b>%{text}</b><br>%{x:.3g}, %{y:.3g}<extra></extra>",
        }],
        "layout": _layout(
            opts.get("title") or f"{_FIELD_LABELS.get(y_field, y_field)} vs {_FIELD_LABELS.get(x_field, x_field)}",
            _FIELD_LABELS.get(x_field, x_field), _FIELD_LABELS.get(y_field, y_field),
        ),
    }
    summary = {"chart_type": "scatter", "x_field": x_field, "y_field": y_field, "n": len(pts)}
    return figure, summary


# ── public entry point ─────────────────────────────────────────────────────────

_GENE_BUILDERS = {
    "volcano": _build_volcano,
    "histogram": _build_histogram,
    "ma_plot": _build_ma_plot,
    "bar_genes": _build_bar_genes,
    "bar_regulation": _build_bar_regulation,
    "scatter": _build_scatter,
}


async def build_chart(
    db: AsyncSession,
    dataset_id: UUID,
    comparison_name: Optional[str],
    chart_type: str,
    options: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a Plotly figure dict for ``chart_type`` using the comparison's real data.

    Returns ``(figure | None, summary_for_model)``. ``figure`` is ``None`` only when a
    chart has no data to draw (e.g. enrichment_bar with no pathways) — the caller then
    emits a text-only answer.
    """
    if chart_type == "enrichment_bar":
        top_n = options.get("top_n") or 15
        pathways = await _load_pathways(db, dataset_id, comparison_name, int(top_n))
        return _build_enrichment_bar(pathways, options)

    builder = _GENE_BUILDERS.get(chart_type)
    if builder is None:
        raise ValueError(f"Unknown chart_type '{chart_type}'")

    rows = await _load_deg_rows(db, dataset_id, comparison_name)
    if not rows:
        return None, {"chart_type": chart_type, "n": 0,
                      "note": "No DEG rows available for this comparison."}
    return builder(rows, options)
