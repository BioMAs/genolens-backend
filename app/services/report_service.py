"""
ReportService: collects project analysis data and generates a PDF report
in the SciLicium format (cover + QC + PCA + per-comparison results + appendix).
"""
import asyncio
import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Dataset, DatasetType, DatasetStatus,
    Project, DegGene, EnrichmentPathway,
)

logger = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "report"


class ReportService:

    def __init__(self):
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=True,
        )

    # ── Data collection ────────────────────────────────────────────────────────

    async def collect_project_data(
        self, db: AsyncSession, project_id: UUID
    ) -> dict[str, Any]:
        """Gather all data needed for a full project report."""
        project = await db.get(Project, project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        stmt = select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.status == DatasetStatus.READY,
        )
        datasets = (await db.execute(stmt)).scalars().all()

        deg_datasets = [d for d in datasets if d.type == DatasetType.DEG]
        enrichment_datasets = [d for d in datasets if d.type == DatasetType.ENRICHMENT]
        matrix_datasets = [d for d in datasets if d.type == DatasetType.MATRIX]

        comparisons = []
        for deg_ds in deg_datasets:
            comp = await self._build_comparison_section(db, deg_ds, enrichment_datasets)
            comparisons.append(comp)

        qc_data = self._extract_qc_data(matrix_datasets)
        pca_b64 = await asyncio.to_thread(self._generate_pca_plot, matrix_datasets)

        return {
            "project": project,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d"),
            "reference_number": f"GL-{str(project_id)[:8].upper()}",
            "qc_data": qc_data,
            "pca_plot_b64": pca_b64,
            "comparisons": comparisons,
        }

    async def _build_comparison_section(
        self,
        db: AsyncSession,
        deg_ds: Dataset,
        enrichment_datasets: list,
    ) -> dict[str, Any]:
        stmt = (
            select(DegGene)
            .where(DegGene.dataset_id == deg_ds.id)
            .order_by(DegGene.padj.asc())
            .limit(20)
        )
        top_genes = (await db.execute(stmt)).scalars().all()
        top_up = [g for g in top_genes if g.regulation == "UP"][:10]
        top_down = [g for g in top_genes if g.regulation == "DOWN"][:10]

        stmt_all = select(DegGene).where(DegGene.dataset_id == deg_ds.id)
        all_genes = (await db.execute(stmt_all)).scalars().all()

        volcano_b64 = await asyncio.to_thread(
            self._generate_volcano_plot, all_genes, deg_ds.name
        )

        matching_enrichment = next(
            (e for e in enrichment_datasets
             if deg_ds.name in e.name or e.name in deg_ds.name),
            None,
        )
        enrichment_results = []
        dotplot_b64 = None
        if matching_enrichment:
            stmt_e = (
                select(EnrichmentPathway)
                .where(EnrichmentPathway.dataset_id == matching_enrichment.id)
                .order_by(EnrichmentPathway.padj.asc())
                .limit(50)
            )
            enrichment_results = (await db.execute(stmt_e)).scalars().all()
            dotplot_b64 = await asyncio.to_thread(
                self._generate_enrichment_dotplot, enrichment_results, deg_ds.name
            )

        meta = deg_ds.dataset_metadata or {}
        return {
            "name": deg_ds.name,
            "deg_up": meta.get("deg_up_count", sum(1 for g in all_genes if g.regulation == "UP")),
            "deg_down": meta.get("deg_down_count", sum(1 for g in all_genes if g.regulation == "DOWN")),
            "deg_total": len(all_genes),
            "top_up_genes": top_up,
            "top_down_genes": top_down,
            "volcano_b64": volcano_b64,
            "enrichment_results": enrichment_results,
            "dotplot_b64": dotplot_b64,
        }

    def _extract_qc_data(self, matrix_datasets: list) -> list[dict]:
        rows = []
        for ds in matrix_datasets:
            meta = ds.dataset_metadata or {}
            qc_report = meta.get("qc_report", {})
            if isinstance(qc_report, dict):
                for sample, stats in qc_report.items():
                    rows.append({
                        "sample": sample,
                        "total_reads": stats.get("total_reads", "N/A"),
                        "mapped_reads": stats.get("mapped_reads", "N/A"),
                        "mapping_rate": stats.get("mapping_rate", "N/A"),
                        "detected_genes": stats.get("detected_genes", "N/A"),
                    })
        return rows

    # ── Plot generation (sync — called via asyncio.to_thread) ─────────────────

    def _fig_to_b64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def _generate_pca_plot(self, matrix_datasets: list) -> Optional[str]:
        pca_data = None
        for ds in matrix_datasets:
            meta = ds.dataset_metadata or {}
            if "pca" in meta:
                pca_data = meta["pca"]
                break
        if not pca_data:
            return None

        pc1 = pca_data.get("PC1", [])
        pc2 = pca_data.get("PC2", [])
        groups = pca_data.get("groups", [""] * len(pc1))
        if not pc1:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        unique_groups = list(dict.fromkeys(groups))
        colors = plt.cm.tab10.colors
        for i, grp in enumerate(unique_groups):
            idx = [j for j, g in enumerate(groups) if g == grp]
            ax.scatter(
                [pc1[j] for j in idx],
                [pc2[j] for j in idx],
                label=grp,
                color=colors[i % len(colors)],
                s=80,
                alpha=0.8,
            )
        var_exp = pca_data.get("variance_explained", [])
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)" if var_exp else "PC1")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)" if len(var_exp) > 1 else "PC2")
        ax.set_title("PCA — Sample Distribution")
        ax.legend(loc="best", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return self._fig_to_b64(fig)

    def _generate_volcano_plot(self, genes, comparison_name: str) -> Optional[str]:
        if not genes:
            return None
        lfc = np.array([g.log_fc or 0.0 for g in genes])
        pval = np.array([g.padj or 1.0 for g in genes])
        neg_log_p = -np.log10(np.clip(pval, 1e-300, 1.0))
        sig = (pval < 0.05) & (np.abs(lfc) > 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(lfc[~sig], neg_log_p[~sig], s=5, color="#AAAAAA", alpha=0.5, label="NS")
        ax.scatter(lfc[sig & (lfc > 0)], neg_log_p[sig & (lfc > 0)], s=8, color="#D73027", alpha=0.7, label="Up")
        ax.scatter(lfc[sig & (lfc < 0)], neg_log_p[sig & (lfc < 0)], s=8, color="#4575B4", alpha=0.7, label="Down")
        ax.axhline(-np.log10(0.05), color="gray", lw=0.8, ls="--")
        ax.axvline(1, color="gray", lw=0.8, ls="--")
        ax.axvline(-1, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("log₂(Fold Change)")
        ax.set_ylabel("-log₁₀(adjusted p-value)")
        ax.set_title(f"Volcano — {comparison_name}")
        ax.legend(fontsize=8, markerscale=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return self._fig_to_b64(fig)

    def _generate_enrichment_dotplot(self, pathways, comparison_name: str) -> Optional[str]:
        if not pathways:
            return None
        top = sorted(pathways, key=lambda p: p.padj or 1.0)[:20]
        names = [p.pathway_name[:55] for p in top]
        pvals = [-np.log10(p.padj or 1e-10) for p in top]
        sizes = [max(20, (p.gene_count or 1) * 5) for p in top]

        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
        scatter = ax.scatter(pvals, range(len(top)), s=sizes, c=pvals, cmap="RdBu_r", alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("-log₁₀(adjusted p-value)")
        ax.set_title(f"Enrichment — {comparison_name}")
        plt.colorbar(scatter, ax=ax, label="-log₁₀(padj)")
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── PDF rendering ──────────────────────────────────────────────────────────

    async def render_pdf(self, context: dict[str, Any]) -> bytes:
        """Render the full report as PDF bytes via WeasyPrint."""
        html_str = await asyncio.to_thread(self._render_html, context)
        return await asyncio.to_thread(self._html_to_pdf, html_str)

    def _render_html(self, context: dict[str, Any]) -> str:
        return self.jinja_env.get_template("report.html.j2").render(**context)

    def _html_to_pdf(self, html_str: str) -> bytes:
        from weasyprint import HTML
        return HTML(string=html_str, base_url=str(TEMPLATE_DIR)).write_pdf()


report_service = ReportService()
