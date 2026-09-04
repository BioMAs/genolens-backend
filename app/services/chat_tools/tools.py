"""
Concrete chat-mode tools.

Every tool is keyed on the selected context (DEG dataset + comparison) injected via
ToolContext. Tools reuse the existing analysis endpoint coroutines directly (in-process,
no HTTP round-trip) so their output is byte-identical to the REST API — the frontend can
render the same plot components from a `figure` SSE event that it would from a fetch.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.models.models import DegGene, EnrichmentPathway
from app.services import chart_builder
from app.services.chat_tools.base import BaseTool, ToolContext, ToolResult
from app.services.enrichment_source import resolve_pathway_dataset_id

# ── Parameter schemas ────────────────────────────────────────────────────────

class EmptyParams(BaseModel):
    """No parameters."""


class DegListParams(BaseModel):
    regulation: Optional[str] = Field(
        None, description="Filter by regulation direction: 'UP' or 'DOWN'. Omit for both."
    )
    padj_max: Optional[float] = Field(
        None, description="Keep only genes with adjusted p-value below this (e.g. 0.05)."
    )
    logfc_min: Optional[float] = Field(
        None, description="Keep only genes with |log2 fold-change| above this (e.g. 1.0)."
    )
    top_n: int = Field(20, ge=1, le=50, description="How many genes to return (max 50).")
    sort_by: str = Field(
        "padj", description="Sort field: 'padj' (most significant) or 'log_fc' (strongest change)."
    )


class GenerateChartParams(BaseModel):
    """Constrained figure spec — the server expands it into a full Plotly figure."""

    chart_type: Literal[
        "volcano", "histogram", "ma_plot", "bar_genes",
        "bar_regulation", "enrichment_bar", "scatter",
    ] = Field(..., description="Which chart to draw (required).")
    title: Optional[str] = Field(None, description="Optional chart title override.")
    color: Optional[str] = Field(
        None, description="Named colour for single-series charts, e.g. 'blue', 'red'."
    )
    palette: Optional[str] = Field(
        None, description="Palette key: 'standard' or 'colorblind'."
    )
    top_n: Optional[int] = Field(
        None, ge=1, le=50,
        description="How many items for bar_genes (default 15) / enrichment_bar (default 15).",
    )
    field: Optional[Literal["log_fc", "padj", "pvalue", "base_mean"]] = Field(
        None, description="Numeric field for histogram (default 'log_fc')."
    )
    bins: Optional[int] = Field(None, ge=2, le=200, description="Histogram bin count (default 40).")
    x_field: Optional[Literal["log_fc", "padj", "pvalue", "base_mean"]] = Field(
        None, description="X axis field for a generic scatter."
    )
    y_field: Optional[Literal["log_fc", "padj", "pvalue", "base_mean"]] = Field(
        None, description="Y axis field for a generic scatter."
    )
    padj_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Volcano significance threshold on adjusted p-value."
    )
    logfc_threshold: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Volcano significance threshold on |log2 fold-change|."
    )


class PathwaysParams(BaseModel):
    category: Optional[str] = Field(
        None,
        description=(
            "Filter by category. The available categories depend on the dataset — call "
            "get_dataset_summary to read available_pathway_categories rather than guessing. "
            "annoDB datasets use names like 'biological_process', 'molecular_function', "
            "'cellular_component', 'matrisome'; older ones use 'GO:BP', 'GO:MF', 'GO:CC'. "
            "Either spelling of the three GO namespaces is accepted, case-insensitively."
        ),
    )
    regulation: Optional[str] = Field(
        None, description="Filter by regulation: 'ALL', 'UP' or 'DOWN'."
    )
    top_n: int = Field(15, ge=1, le=50, description="How many pathways to return (max 50).")


# ── Pathway category vocabulary ──────────────────────────────────────────────
#
# The two enrichment sources name the GO namespaces differently: the legacy Python path writes
# "GO:BP"/"GO:MF"/"GO:CC", annoDB writes "biological_process"/… (plus families of its own, like
# "matrisome" or "senescence_signatures"). Now that the tools read whichever source actually holds
# the comparison's enrichment, a model asking for "GO:BP" must not silently get nothing back.

_GO_CATEGORY_ALIASES = {
    "go:bp": "biological_process",
    "go:mf": "molecular_function",
    "go:cc": "cellular_component",
    "biological_process": "go:bp",
    "molecular_function": "go:mf",
    "cellular_component": "go:cc",
}


def _match_category(requested: str, available: List[str]) -> Optional[str]:
    """
    The stored category the request means, or None if the dataset has no such category.

    Tries the literal value, then a case-insensitive match, then the other vocabulary's spelling
    of the same GO namespace.
    """
    if requested in available:
        return requested

    by_lower = {c.lower(): c for c in available if c}
    wanted = requested.lower()
    if wanted in by_lower:
        return by_lower[wanted]

    alias = _GO_CATEGORY_ALIASES.get(wanted)
    if alias and alias in by_lower:
        return by_lower[alias]
    return None


# ── Tools ────────────────────────────────────────────────────────────────────

class GetDatasetSummaryTool(BaseTool):
    name = "get_dataset_summary"
    description = (
        "Get an overview of the currently selected comparison: number of up- and "
        "down-regulated genes, total DEGs, and the list of available comparisons in "
        "this dataset. Call this to answer questions about scale or to orient yourself."
    )
    params_model = EmptyParams
    figure_type = None

    async def execute(self, ctx: ToolContext, params: BaseModel) -> ToolResult:
        db = ctx.db
        up = await db.scalar(
            select(func.count(DegGene.id))
            .where(DegGene.dataset_id == ctx.dataset_id)
            .where(DegGene.comparison_name == ctx.comparison_name)
            .where(DegGene.regulation == "UP")
        ) or 0
        down = await db.scalar(
            select(func.count(DegGene.id))
            .where(DegGene.dataset_id == ctx.dataset_id)
            .where(DegGene.comparison_name == ctx.comparison_name)
            .where(DegGene.regulation == "DOWN")
        ) or 0
        comparisons = (await db.execute(
            select(DegGene.comparison_name)
            .where(DegGene.dataset_id == ctx.dataset_id)
            .distinct()
        )).scalars().all()

        # The pathway categories this comparison actually has, so get_enrichment_pathways can be
        # filtered against real values instead of a guessed vocabulary — annoDB and the legacy
        # Python path name the GO namespaces differently.
        pathway_categories: List[str] = []
        if ctx.dataset is not None and ctx.comparison_name:
            pathways_dataset_id = await resolve_pathway_dataset_id(
                db, ctx.dataset, ctx.comparison_name
            )
            pathway_categories = sorted(
                c for c in (await db.execute(
                    select(EnrichmentPathway.category)
                    .where(EnrichmentPathway.dataset_id == pathways_dataset_id)
                    .where(EnrichmentPathway.comparison_name == ctx.comparison_name)
                    .distinct()
                )).scalars().all() if c
            )

        summary = {
            "dataset_name": getattr(ctx.dataset, "name", None),
            "comparison_name": ctx.comparison_name,
            "deg_up": int(up),
            "deg_down": int(down),
            "deg_total": int(up) + int(down),
            "available_comparisons": list(comparisons),
            "available_pathway_categories": pathway_categories,
        }
        return ToolResult(summary_for_model=summary, params={})


class ListDegGenesTool(BaseTool):
    name = "list_deg_genes"
    description = (
        "List the top differentially expressed genes for the selected comparison, "
        "optionally filtered by regulation direction, adjusted p-value or fold-change. "
        "Use to name specific genes."
    )
    params_model = DegListParams
    figure_type = None

    async def execute(self, ctx: ToolContext, params: DegListParams) -> ToolResult:
        from app.api.endpoints.datasets import get_deg_genes

        regulation = params.regulation.upper() if params.regulation else None
        sort_by = params.sort_by if params.sort_by in ("padj", "log_fc", "gene_id") else "padj"
        result = await get_deg_genes(
            dataset_id=ctx.dataset_id,
            comparison_name=ctx.comparison_name,
            db=ctx.db,
            current_user=ctx.current_user,
            regulation=regulation,
            padj_max=params.padj_max,
            logfc_min=params.logfc_min,
            page=1,
            page_size=params.top_n,
            sort_by=sort_by,
            sort_order="desc" if sort_by == "log_fc" else "asc",
        )
        genes = result.get("genes", [])
        summary = {
            "returned": len(genes),
            "total_up": result.get("total_up"),
            "total_down": result.get("total_down"),
            "genes": [
                {
                    "gene": g.get("gene_name") or g.get("gene_id"),
                    "log_fc": g.get("log_fc"),
                    "padj": g.get("padj"),
                    "regulation": g.get("regulation"),
                }
                for g in genes
            ],
        }
        return ToolResult(summary_for_model=summary, params=params.model_dump(exclude_none=True))


class GenerateChartTool(BaseTool):
    name = "generate_chart"
    description = (
        "Generate a figure for the selected comparison. Call this whenever the user asks "
        "to see, draw, plot or visualise something. Choose one 'chart_type':\n"
        "  - volcano: log2 fold-change vs -log10(padj), coloured by significance.\n"
        "  - histogram: distribution of one field (default log_fc); set 'field' and 'bins'.\n"
        "  - ma_plot: log10(base mean) vs log2 fold-change, coloured by regulation.\n"
        "  - bar_genes: top 'top_n' genes by |log2FC| (default 15), coloured up/down.\n"
        "  - bar_regulation: counts of up- vs down-regulated genes.\n"
        "  - enrichment_bar: top 'top_n' enriched pathways by significance.\n"
        "  - scatter: generic scatter of any two numeric fields ('x_field','y_field').\n"
        "Numeric fields available per gene: log_fc, padj, pvalue, base_mean. Optional "
        "'title', 'color' (e.g. 'blue') and 'palette' ('standard'|'colorblind') tune the look. "
        "Only pass parameters relevant to the chosen chart_type."
    )
    params_model = GenerateChartParams
    figure_type = "plotly"

    async def execute(self, ctx: ToolContext, params: GenerateChartParams) -> ToolResult:
        options = params.model_dump(exclude_none=True)
        figure, summary = await chart_builder.build_chart(
            db=ctx.db,
            dataset_id=ctx.dataset_id,
            comparison_name=ctx.comparison_name,
            chart_type=params.chart_type,
            options=options,
        )
        # figure is None when there is no data to draw (e.g. enrichment with no pathways):
        # return a text-only summary so the model narrates instead of showing an empty plot.
        return ToolResult(
            summary_for_model=summary,
            figure_type=self.figure_type if figure is not None else None,
            figure_payload=figure,
            params=options,
        )


class GetEnrichmentPathwaysTool(BaseTool):
    name = "get_enrichment_pathways"
    description = (
        "List the most significantly enriched biological pathways / GO terms for the "
        "selected comparison, optionally filtered by category or regulation. Use to "
        "answer questions about affected biological processes."
    )
    params_model = PathwaysParams
    figure_type = None

    async def execute(self, ctx: ToolContext, params: PathwaysParams) -> ToolResult:
        # Enrichment does not live on the DEG dataset the chat is scoped to. On a self-service
        # analysis it is a separate annoDB ENRICHMENT dataset, so reading `ctx.dataset_id`
        # returned nothing and the chat would answer "no pathways" while the enrichment panel on
        # the same page was full of them. Resolve it the way the panel does.
        pathways_dataset_id = ctx.dataset_id
        if ctx.dataset is not None and ctx.comparison_name:
            pathways_dataset_id = await resolve_pathway_dataset_id(
                ctx.db, ctx.dataset, ctx.comparison_name
            )

        base = (
            select(EnrichmentPathway)
            .where(EnrichmentPathway.dataset_id == pathways_dataset_id)
            .where(EnrichmentPathway.comparison_name == ctx.comparison_name)
        )

        # Resolve the requested category against what this dataset actually stores, so a model
        # asking for "GO:BP" against annoDB rows gets biological_process rather than silence.
        available = [
            c for c in (await ctx.db.execute(
                base.with_only_columns(EnrichmentPathway.category).distinct()
            )).scalars().all() if c
        ]
        unknown_category = None
        stmt = base
        if params.category:
            matched = _match_category(params.category, available)
            if matched is None:
                unknown_category = params.category
            else:
                stmt = stmt.where(EnrichmentPathway.category == matched)

        if params.regulation:
            stmt = stmt.where(EnrichmentPathway.regulation == params.regulation.upper())
        stmt = stmt.order_by(EnrichmentPathway.padj.asc()).limit(params.top_n)
        rows = [] if unknown_category else (await ctx.db.execute(stmt)).scalars().all()

        summary = {
            "returned": len(rows),
            "available_categories": sorted(available),
            "pathways": [
                {
                    "pathway_name": p.pathway_name,
                    "category": p.category,
                    "padj": p.padj,
                    "gene_count": p.gene_count,
                    "regulation": p.regulation,
                }
                for p in rows
            ],
        }
        if unknown_category:
            # Tell the model the vocabulary instead of letting it read an empty list as "none".
            summary["note"] = (
                f"No category {unknown_category!r} in this dataset. "
                f"Available: {', '.join(sorted(available)) or '(none)'}."
            )
        return ToolResult(summary_for_model=summary, params=params.model_dump(exclude_none=True))


def build_default_tools() -> List[BaseTool]:
    """Instantiate the v1 tool set."""
    return [
        GetDatasetSummaryTool(),
        ListDegGenesTool(),
        GenerateChartTool(),
        GetEnrichmentPathwaysTool(),
    ]
