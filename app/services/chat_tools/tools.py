"""
Concrete chat-mode tools.

Every tool is keyed on the selected context (DEG dataset + comparison) injected via
ToolContext. Tools reuse the existing analysis endpoint coroutines directly (in-process,
no HTTP round-trip) so their output is byte-identical to the REST API — the frontend can
render the same plot components from a `figure` SSE event that it would from a fetch.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.models.models import DegGene, EnrichmentPathway
from app.services.chat_tools.base import BaseTool, ToolContext, ToolResult


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


class VolcanoParams(BaseModel):
    padj_threshold: float = Field(
        0.05, ge=0.0, le=1.0, description="Adjusted p-value significance threshold."
    )
    logfc_threshold: float = Field(
        0.58, ge=0.0, le=10.0, description="Absolute log2 fold-change significance threshold."
    )
    max_points: int = Field(
        5000, ge=100, le=20000, description="Maximum number of points to plot."
    )


class PathwaysParams(BaseModel):
    category: Optional[str] = Field(
        None, description="Filter by category, e.g. 'GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'Reactome'."
    )
    regulation: Optional[str] = Field(
        None, description="Filter by regulation: 'ALL', 'UP' or 'DOWN'."
    )
    top_n: int = Field(15, ge=1, le=50, description="How many pathways to return (max 50).")


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

        summary = {
            "dataset_name": getattr(ctx.dataset, "name", None),
            "comparison_name": ctx.comparison_name,
            "deg_up": int(up),
            "deg_down": int(down),
            "deg_total": int(up) + int(down),
            "available_comparisons": list(comparisons),
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


class GenerateVolcanoPlotTool(BaseTool):
    name = "generate_volcano_plot"
    description = (
        "Generate a volcano plot (log2 fold-change vs -log10 adjusted p-value) for the "
        "selected comparison. Call this whenever the user asks to see, draw, plot or "
        "visualise a volcano plot."
    )
    params_model = VolcanoParams
    figure_type = "volcano"

    async def execute(self, ctx: ToolContext, params: VolcanoParams) -> ToolResult:
        from app.api.endpoints.datasets import get_volcano_plot_data

        payload = await get_volcano_plot_data(
            dataset_id=ctx.dataset_id,
            comparison_name=ctx.comparison_name,
            db=ctx.db,
            current_user=ctx.current_user,
            max_points=params.max_points,
            force_recalculate=False,
            padj_threshold=params.padj_threshold,
            logfc_threshold=params.logfc_threshold,
        )
        summary = {
            "total_genes": payload.get("total_genes"),
            "significant_genes": payload.get("significant_genes"),
            "thresholds": payload.get("thresholds"),
        }
        return ToolResult(
            summary_for_model=summary,
            figure_type=self.figure_type,
            figure_payload=payload,
            params=params.model_dump(),
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
        stmt = (
            select(EnrichmentPathway)
            .where(EnrichmentPathway.dataset_id == ctx.dataset_id)
            .where(EnrichmentPathway.comparison_name == ctx.comparison_name)
        )
        if params.category:
            stmt = stmt.where(EnrichmentPathway.category == params.category)
        if params.regulation:
            stmt = stmt.where(EnrichmentPathway.regulation == params.regulation.upper())
        stmt = stmt.order_by(EnrichmentPathway.padj.asc()).limit(params.top_n)
        rows = (await ctx.db.execute(stmt)).scalars().all()

        summary = {
            "returned": len(rows),
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
        return ToolResult(summary_for_model=summary, params=params.model_dump(exclude_none=True))


def build_default_tools() -> List[BaseTool]:
    """Instantiate the v1 tool set."""
    return [
        GetDatasetSummaryTool(),
        ListDegGenesTool(),
        GenerateVolcanoPlotTool(),
        GetEnrichmentPathwaysTool(),
    ]
