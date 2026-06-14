"""
ReportLatexService — generates the branded SciLicium PDF report from a
self-service analysis, using the vendored LaTeX template
(app/templates/report_latex/) compiled with pdflatex.

Ported/adapted from pipe_scilicium/workflow/scripts/python/fill_latex_template.py:
the Jinja2 LaTeX environment, tex_escape, the table/figure generators and the
explanatory texts are reused; data is sourced from the GenoLens DB instead of
pipeline TSVs.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jinja2
from jinja2 import pass_context
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Dataset, DatasetType, DegGene, EnrichmentPathway, AIInterpretation,
    Project, SelfServiceAnalysis,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "report_latex"
MAIN_TEMPLATE = "report_template.tex"
MM_TEMPLATE = "material_and_methods.tex"

GO_CATEGORIES = ["GO:BP", "GO:MF", "GO:CC"]


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX-escaping + formatting helpers (ported from fill_latex_template.py)
# ─────────────────────────────────────────────────────────────────────────────

def tex_escape(text) -> str:
    if text is None:
        return ""
    text = str(text)
    conv = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
        '\\': r'\textbackslash{}', '<': r'\textless{}', '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(k) for k in sorted(conv, key=lambda i: -len(i))))
    return regex.sub(lambda m: conv[m.group()], text)


def tex_escape_address(text) -> str:
    if text is None:
        return ""
    return r'\\'.join(tex_escape(line) for line in str(text).split('\\\\'))


def sanitize_condition_name(condition) -> str:
    if condition is None:
        return ""
    s = str(condition).replace('_', ' ')
    for ch in ['/', '\\', '|', ':', ';', ',']:
        s = s.replace(ch, ' ')
    return ' '.join(s.split())


def format_condition_display(condition, label) -> str:
    return f"{tex_escape(sanitize_condition_name(condition))} ({label})"


def format_number(number, decimals=2) -> str:
    if number is None:
        return "N/A"
    try:
        return f"{float(number):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_millions(number) -> str:
    if number is None:
        return "N/A"
    try:
        num = float(number)
        if num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        if num >= 1_000:
            return f"{num/1_000:.1f}K"
        return f"{num:.0f}"
    except (ValueError, TypeError):
        return "N/A"


@pass_context
def subrender_filter(context, value):
    """Render a template string within the current context (templates in templates)."""
    return context.eval_ctx.environment.from_string(value or "").render(**context)


def _fmt_p(value) -> str:
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if v != v:  # NaN
        return "N/A"
    if v == 0:
        return "<1e-300"
    return f"{v:.2e}" if v < 0.001 else f"{v:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Explanatory texts (subset relevant to GenoLens, from fill_latex_template.py)
# ─────────────────────────────────────────────────────────────────────────────

EXPLANATORY = {
    "pca": r"The principal component analysis summarises the expression profile (after normalisation) of all samples on a 2D plot. Each point is a sample; colours indicate the experimental condition. Samples with similar expression cluster together, which is an essential quality-control step to confirm that biological differences drive the results.",
    "qc": r"The following table summarises per-sample sequencing-depth metrics derived from the count matrix: the library size (total assigned counts) and the number of detected genes (genes with at least one count).",
    "volcano": r"The volcano plot shows which genes differ between the two conditions. Each point is a gene; the x-axis is the magnitude of change (log\textsubscript{2} fold-change, test/control) and the y-axis its statistical significance (--log\textsubscript{10} adjusted p-value). Red points are up-regulated in the test condition, blue points down-regulated, grey points non-significant.",
    "deg_multimethod_short": r"Differentially expressed genes (DEGs) were identified using a multi-method approach combining DESeq2, edgeR and limma-voom. P-values were meta-analysed using Stouffer's Z-score method and corrected for multiple testing (Benjamini-Hochberg FDR).",
    "enrichment_intro": r"Differentially expressed genes were further studied by over-representation analysis (ORA) against multiple databases (GO, KEGG, Reactome, MSigDB) using a hypergeometric test (Benjamini-Hochberg adjusted p-value $\leq$ 0.05). Dot size reflects the gene ratio and colour the significance.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation (matplotlib → vector PDF placed in the work dir)
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _figure_latex(rel_path: str, caption: str, label: str, width: str = "13cm") -> str:
    return (
        "\\begin{figure}[H]\n\\centering\n"
        f"\\includegraphics[width={width}]{{{rel_path}}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n"
    )


def _plot_volcano(genes: list[dict], out_path: str) -> bool:
    if not genes:
        return False
    lfc = np.array([g.get("log_fc") or 0.0 for g in genes], dtype=float)
    padj = np.array([g.get("padj") if g.get("padj") is not None else 1.0 for g in genes], dtype=float)
    neg_log_p = -np.log10(np.clip(padj, 1e-300, 1.0))
    sig = (padj < 0.05) & (np.abs(lfc) > 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lfc[~sig], neg_log_p[~sig], s=5, color="#AAAAAA", alpha=0.5, label="NS")
    ax.scatter(lfc[sig & (lfc > 0)], neg_log_p[sig & (lfc > 0)], s=8, color="#D73027", alpha=0.7, label="Up")
    ax.scatter(lfc[sig & (lfc < 0)], neg_log_p[sig & (lfc < 0)], s=8, color="#4575B4", alpha=0.7, label="Down")
    ax.axhline(-np.log10(0.05), color="gray", lw=0.8, ls="--")
    ax.axvline(1, color="gray", lw=0.8, ls="--")
    ax.axvline(-1, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, markerscale=2)
    _save_fig(fig, out_path)
    return True


def _plot_enrichment_dotplot(pathways: list[dict], out_path: str) -> bool:
    if not pathways:
        return False
    top = sorted(pathways, key=lambda p: p.get("padj") or 1.0)[:20]
    names = [str(p.get("pathway_name") or p.get("pathway_id") or "")[:55] for p in top]
    pvals = [-np.log10(max(p.get("padj") or 1e-300, 1e-300)) for p in top]
    sizes = [max(20, (p.get("gene_count") or 1) * 5) for p in top]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    sc = ax.scatter(pvals, range(len(top)), s=sizes, c=pvals, cmap="RdBu_r", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("-log10(adjusted p-value)")
    plt.colorbar(sc, ax=ax, label="-log10(padj)")
    fig.tight_layout()
    _save_fig(fig, out_path)
    return True


def _plot_pca(pca: dict, sample_conditions: dict[str, str], out_path: str) -> bool:
    points = pca.get("data") or pca.get("points") or pca.get("samples") or []
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    groups: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        cond = sample_conditions.get(str(p.get("sample", "")), "Unknown")
        groups.setdefault(cond, []).append((p.get("x", 0.0), p.get("y", 0.0)))
    colors = plt.cm.tab10.colors
    for i, (grp, pts) in enumerate(groups.items()):
        xs = [a for a, _ in pts]
        ys = [b for _, b in pts]
        ax.scatter(xs, ys, label=grp, color=colors[i % len(colors)], s=80, alpha=0.8)
    var = pca.get("explained_variance") or []
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)" if var else "PC1")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)" if len(var) > 1 else "PC2")
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_path)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generators (adapted to GenoLens DB dicts)
# ─────────────────────────────────────────────────────────────────────────────

def generate_qc_table(qc_rows: list[dict], title: str) -> str:
    if not qc_rows:
        return r"\textit{QC data not available}"
    lines = [
        r"\begin{longtable}{lS[table-format=4.2]S[table-format=6]l}",
        f"\\caption{{Quality Control Statistics --- {tex_escape(title)}}} \\\\",
        r"\toprule",
        r"\rowcolor{white}\multicolumn{1}{L{4cm}}{Sample} &",
        r"  \multicolumn{1}{L{3cm}}{Library size} &",
        r"  \multicolumn{1}{L{3cm}}{Genes detected} &",
        r"  \multicolumn{1}{L{4cm}}{Condition} \\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"\rowcolor{white}\multicolumn{1}{L{4cm}}{Sample} &",
        r"  \multicolumn{1}{L{3cm}}{Library size} &",
        r"  \multicolumn{1}{L{3cm}}{Genes detected} &",
        r"  \multicolumn{1}{L{4cm}}{Condition} \\",
        r"\midrule\endhead",
        r"\hline\endfoot\hline\endlastfoot",
    ]
    for r in qc_rows:
        lines.append(
            f"{tex_escape(str(r['sample'])[:20])} & {format_millions(r['library_size'])} "
            f"& {r['detected_genes']:,} & {tex_escape(r.get('condition', 'Unknown'))} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def generate_deg_table(degs: list[dict], max_rows: int = 15) -> str:
    if not degs:
        return r"\textit{No differentially expressed genes found}"
    rows = sorted(degs, key=lambda d: (d.get("padj") if d.get("padj") is not None else 1.0))[:max_rows]
    lines = [
        r"\begin{longtable}{lL{3cm}L{2.2cm}L{2.2cm}l}",
        r"\caption{Top Differentially Expressed Genes} \\",
        r"\toprule",
        r"\rowcolor{white}\multicolumn{1}{L{3.5cm}}{Gene} & {log\textsubscript{2}FC} & {\textit{P}-value} & {Adj. \textit{P}} & {Reg.} \\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"\rowcolor{white}\multicolumn{1}{L{3.5cm}}{Gene} & {log\textsubscript{2}FC} & {\textit{P}-value} & {Adj. \textit{P}} & {Reg.} \\",
        r"\midrule\endhead",
        r"\hline\endfoot\hline\endlastfoot",
    ]
    for d in rows:
        gene = tex_escape(d.get("gene_name") or d.get("gene_id") or "")
        lines.append(
            f"{gene} & {format_number(d.get('log_fc'))} & {_fmt_p(d.get('pvalue'))} "
            f"& {_fmt_p(d.get('padj'))} & {tex_escape(d.get('regulation') or '')} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def generate_enrichment_table(pathways: list[dict], label: str, max_terms: int = 10) -> str:
    if not pathways:
        return r"\textit{No enriched terms found}"
    terms = sorted(pathways, key=lambda p: (p.get("padj") if p.get("padj") is not None else 1.0))[:max_terms]
    lines = [
        r"\begin{longtable}{p{9cm}L{2.5cm}L{2.5cm}}",
        f"\\caption{{Functional enrichment --- {tex_escape(label)}}} \\\\",
        r"\toprule",
        r"\rowcolor{white}\multicolumn{1}{L{9cm}}{Term} & {\textit{P}-value} & {Adj. \textit{P}} \\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"\rowcolor{white}\multicolumn{1}{L{9cm}}{Term} & {\textit{P}-value} & {Adj. \textit{P}} \\",
        r"\midrule\endhead",
        r"\hline\endfoot\hline\endlastfoot",
    ]
    for t in terms:
        name = str(t.get("pathway_name") or t.get("pathway_id") or "").replace("_", " ")
        if len(name) > 80:
            name = name[:77] + "..."
        cat = t.get("category") or ""
        display = f"{name} [{cat}]" if cat else name
        lines.append(
            f"{{\\small {tex_escape(display)}}} & {_fmt_p(t.get('pvalue'))} & {_fmt_p(t.get('padj'))} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Per-comparison LaTeX section
# ─────────────────────────────────────────────────────────────────────────────

def _ai_box(text: str) -> str:
    return (
        r"\subsubsection{Biological Summary (AI)}" "\n"
        r"\begin{tcolorbox}[enhanced,breakable,colback=sciliciumgreen!5,"
        r"colframe=sciliciumgreen,boxrule=0.8pt,arc=1mm,"
        r"title={\textbf{AI-generated interpretation}},fonttitle=\bfseries,coltitle=white]" "\n"
        f"{tex_escape(text)}" "\n"
        r"\end{tcolorbox}" "\n"
    )


def build_comparison_section(comp: dict) -> str:
    """comp: dict produced by collect_analysis_data (see below)."""
    test = comp["test"]
    control = comp["control"]
    test_d = format_condition_display(test, "test")
    control_d = format_condition_display(control, "control")
    n_up = comp["n_up"]
    n_down = comp["n_down"]
    n_total = n_up + n_down

    s = [f"\\subsection{{{test_d} vs {control_d}}}", f"\\label{{sec:{comp['safe_id']}}}"]
    s.append(r"\subsubsection{Overview}")
    s.append(f"Gene expression was compared between {test_d} and {control_d}. ")
    if n_total == 0:
        s.append("No differentially expressed genes were detected.")
    else:
        s.append(
            f"A total of {n_total} differentially expressed genes were detected, "
            f"of which {n_up} are up-regulated and {n_down} down-regulated in the test condition."
        )
    s.append("")
    s.append(EXPLANATORY["deg_multimethod_short"])
    s.append("")

    s.append(r"\subsubsection{Detailed Comparison Results}")
    s.append(EXPLANATORY["volcano"])
    s.append("")
    if comp.get("volcano_fig"):
        s.append(_figure_latex(comp["volcano_fig"],
                               f"Volcano plot --- {test_d} vs {control_d}.",
                               f"fig:volcano_{comp['safe_id']}", "12.5cm"))
    s.append(generate_deg_table(comp["degs"]))
    s.append("")

    s.append(r"\subsubsection{Functional Enrichment}")
    s.append(EXPLANATORY["enrichment_intro"])
    s.append("")
    if comp.get("dotplot_up_fig"):
        s.append(r"\paragraph{Up-regulated genes}\mbox{}\\")
        s.append(_figure_latex(comp["dotplot_up_fig"],
                               f"Enrichment (up-regulated) --- {test_d} vs {control_d}.",
                               f"fig:enr_up_{comp['safe_id']}", "12cm"))
    if comp["enr_up"]:
        s.append(generate_enrichment_table(comp["enr_up"], "Up-regulated genes"))
    if comp.get("dotplot_down_fig"):
        s.append(r"\paragraph{Down-regulated genes}\mbox{}\\")
        s.append(_figure_latex(comp["dotplot_down_fig"],
                               f"Enrichment (down-regulated) --- {test_d} vs {control_d}.",
                               f"fig:enr_down_{comp['safe_id']}", "12cm"))
    if comp["enr_down"]:
        s.append(generate_enrichment_table(comp["enr_down"], "Down-regulated genes"))
    s.append("")

    if comp.get("ai_interpretation"):
        s.append(_ai_box(comp["ai_interpretation"]))

    return "\n".join(s)


class ReportLatexService:
    def __init__(self):
        self.env = jinja2.Environment(
            block_start_string="\\JINJABLOCK{", block_end_string="}",
            variable_start_string="\\JINJAVAR{", variable_end_string="}",
            comment_start_string="\\#{", comment_end_string="}",
            trim_blocks=True, autoescape=False,
            loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        )
        for name, fn in {
            "tex_escape": tex_escape, "tex_escape_address": tex_escape_address,
            "subrender": subrender_filter, "format_number": format_number,
        }.items():
            self.env.filters[name] = fn

    # ── Data collection (analysis-scoped) ────────────────────────────────────
    async def collect_analysis_data(self, db: AsyncSession, analysis_id: UUID,
                                    work_dir: str, generate_ai: bool = True) -> dict[str, Any]:
        analysis = await db.get(SelfServiceAnalysis, analysis_id)
        if not analysis:
            raise ValueError(f"Analysis {analysis_id} not found")
        project = await db.get(Project, analysis.project_id)
        params = analysis.params or {}

        result_ids = [UUID(x) for x in (analysis.result_dataset_ids or [])]
        datasets = []
        if result_ids:
            datasets = (await db.execute(select(Dataset).where(Dataset.id.in_(result_ids)))).scalars().all()
        deg_datasets = [d for d in datasets if d.type == DatasetType.DEG]
        enr_datasets = [d for d in datasets if d.type == DatasetType.ENRICHMENT]

        # condition map per comparison from the comparisons input dataset
        comp_conditions = await self._load_comparison_conditions(db, analysis.comparisons_dataset_id)

        comparisons = []
        for deg_ds in deg_datasets:
            comp = await self._build_comparison(db, deg_ds, enr_datasets, comp_conditions,
                                                work_dir, generate_ai)
            comparisons.append(comp)
        comparisons.sort(key=lambda c: c["comp_id"])

        # PCA + QC figures
        pca_fig = await self._make_pca(db, analysis, work_dir)
        qc_rows = await self._make_qc(db, analysis)

        return {
            "project": project,
            "analysis": analysis,
            "params": params,
            "comparisons": comparisons,
            "pca_fig": pca_fig,
            "qc_rows": qc_rows,
        }

    async def _load_comparison_conditions(self, db, comparisons_dataset_id) -> dict[str, dict]:
        if not comparisons_dataset_id:
            return {}
        ds = await db.get(Dataset, comparisons_dataset_id)
        if not ds or not ds.parquet_file_path:
            return {}
        try:
            from app.services.storage import storage_service
            data = await storage_service.download_file(ds.parquet_file_path)
            df = await asyncio.to_thread(lambda: pd.read_parquet(io.BytesIO(data)))
            df.columns = [c.lower() for c in df.columns]
            id_col = next((c for c in ("comparison_id", "comparison", "name", "contrast") if c in df.columns), None)
            c1 = next((c for c in ("condition1", "test", "group1") if c in df.columns), None)
            c2 = next((c for c in ("condition2", "ref", "control", "group2") if c in df.columns), None)
            out = {}
            if id_col and c1 and c2:
                for _, row in df.iterrows():
                    out[str(row[id_col])] = {"test": str(row[c1]), "control": str(row[c2])}
            return out
        except Exception as exc:
            logger.warning("Could not load comparison conditions: %s", exc)
            return {}

    async def _build_comparison(self, db, deg_ds, enr_datasets, comp_conditions,
                                work_dir, generate_ai) -> dict:
        meta = deg_ds.dataset_metadata if isinstance(deg_ds.dataset_metadata, dict) else {}
        comp_name = meta.get("comparison_name") or deg_ds.name
        safe_id = re.sub(r"[^a-zA-Z0-9]+", "-", comp_name).strip("-")

        all_genes = (await db.execute(
            select(DegGene).where(DegGene.dataset_id == deg_ds.id)
        )).scalars().all()
        deg_dicts = [{"gene_id": g.gene_id, "gene_name": g.gene_name, "log_fc": g.log_fc,
                      "padj": g.padj, "pvalue": g.pvalue, "regulation": g.regulation} for g in all_genes]
        n_up = sum(1 for g in all_genes if (g.regulation or "").upper() == "UP")
        n_down = sum(1 for g in all_genes if (g.regulation or "").upper() == "DOWN")

        # matched enrichment dataset (by comparison_name metadata)
        enr_ds = next(
            (e for e in enr_datasets
             if (e.dataset_metadata or {}).get("comparison_name") == comp_name),
            None,
        )
        enr_all, enr_up, enr_down = [], [], []
        if enr_ds:
            rows = (await db.execute(
                select(EnrichmentPathway).where(EnrichmentPathway.dataset_id == enr_ds.id)
            )).scalars().all()
            for ep in rows:
                d = {"pathway_id": ep.pathway_id, "pathway_name": ep.pathway_name,
                     "category": ep.category, "pvalue": ep.pvalue, "padj": ep.padj,
                     "gene_count": ep.gene_count, "regulation": ep.regulation}
                reg = (ep.regulation or "ALL").upper()
                (enr_up if reg == "UP" else enr_down if reg == "DOWN" else enr_all).append(d)

        # figures
        comp = {
            "comp_id": comp_name, "safe_id": safe_id,
            "test": comp_conditions.get(comp_name, {}).get("test", comp_name),
            "control": comp_conditions.get(comp_name, {}).get("control", ""),
            "n_up": n_up, "n_down": n_down,
            "degs": deg_dicts, "enr_up": enr_up, "enr_down": enr_down,
        }
        vfig = f"volcano_{safe_id}.pdf"
        if await asyncio.to_thread(_plot_volcano, deg_dicts, os.path.join(work_dir, vfig)):
            comp["volcano_fig"] = vfig
        if enr_up:
            f = f"enr_up_{safe_id}.pdf"
            if await asyncio.to_thread(_plot_enrichment_dotplot, enr_up, os.path.join(work_dir, f)):
                comp["dotplot_up_fig"] = f
        if enr_down:
            f = f"enr_down_{safe_id}.pdf"
            if await asyncio.to_thread(_plot_enrichment_dotplot, enr_down, os.path.join(work_dir, f)):
                comp["dotplot_down_fig"] = f

        # AI interpretation (existing, or generate)
        ai = await db.scalar(
            select(AIInterpretation).where(
                AIInterpretation.dataset_id == deg_ds.id,
                AIInterpretation.comparison_name == comp_name,
            )
        )
        if ai is None and generate_ai:
            # Time-boxed, best-effort: Ollama generation is slow (minutes) and the
            # r-worker is single-concurrency, so never let it block the report/worker.
            try:
                from app.services.ai_interpreter import generate_and_store  # T6
                ai = await asyncio.wait_for(
                    generate_and_store(db, deg_ds.id, enr_ds.id if enr_ds else None, comp_name),
                    timeout=120,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.info("AI interpretation skipped for %s: %s", comp_name, exc)
        if ai is not None:
            comp["ai_interpretation"] = ai.interpretation
        return comp

    async def _make_pca(self, db, analysis, work_dir) -> Optional[str]:
        vst_id = (analysis.intermediate_dataset_ids or {}).get("vst")
        if not vst_id:
            return None
        vst = await db.get(Dataset, UUID(vst_id) if isinstance(vst_id, str) else vst_id)
        if not vst or not vst.parquet_file_path:
            return None
        try:
            from app.services.storage import storage_service
            from app.services.data_processor import data_processor
            data = await storage_service.download_file(vst.parquet_file_path)
            pca = await data_processor.calculate_pca(data, n_components=2)
            sample_conditions = await self._sample_conditions(db, analysis.samples_dataset_id)
            out = "pca.pdf"
            if await asyncio.to_thread(_plot_pca, pca, sample_conditions, os.path.join(work_dir, out)):
                return out
        except Exception as exc:
            logger.warning("PCA figure failed: %s", exc)
        return None

    async def _sample_conditions(self, db, samples_dataset_id) -> dict[str, str]:
        if not samples_dataset_id:
            return {}
        ds = await db.get(Dataset, samples_dataset_id)
        if not ds or not ds.parquet_file_path:
            return {}
        try:
            from app.services.storage import storage_service
            data = await storage_service.download_file(ds.parquet_file_path)
            df = await asyncio.to_thread(lambda: pd.read_parquet(io.BytesIO(data)))
            lc = {c.lower(): c for c in df.columns}
            sid = next((lc[c] for c in ("sample_id", "sample", "sampleid", "id") if c in lc), None)
            cond = next((lc[c] for c in ("condition", "group", "treatment", "genotype") if c in lc), None)
            if sid and cond:
                return {str(r[sid]): str(r[cond]) for _, r in df.iterrows()}
        except Exception as exc:
            logger.warning("sample conditions failed: %s", exc)
        return {}

    async def _make_qc(self, db, analysis) -> list[dict]:
        if not analysis.matrix_dataset_id:
            return []
        ds = await db.get(Dataset, analysis.matrix_dataset_id)
        if not ds or not ds.parquet_file_path:
            return []
        try:
            from app.services.storage import storage_service
            data = await storage_service.download_file(ds.parquet_file_path)
            conditions = await self._sample_conditions(db, analysis.samples_dataset_id)

            def _compute():
                df = pd.read_parquet(io.BytesIO(data))
                non_sample = {"gene_id", "gene_name", "symbol"}
                cols = [c for c in df.columns if c.lower() not in non_sample]
                num = df[cols].apply(pd.to_numeric, errors="coerce")
                rows = []
                for c in cols:
                    col = num[c]
                    rows.append({
                        "sample": c,
                        "library_size": int(col.sum(skipna=True)),
                        "detected_genes": int((col > 0).sum()),
                        "condition": conditions.get(str(c), "Unknown"),
                    })
                return rows
            return await asyncio.to_thread(_compute)
        except Exception as exc:
            logger.warning("QC table failed: %s", exc)
            return []

    # ── Assemble template vars + render + compile ────────────────────────────
    def _read_mm(self) -> str:
        p = TEMPLATE_DIR / MM_TEMPLATE
        try:
            return p.read_text()
        except Exception:
            return r"\input{material_and_methods.tex}"

    def assemble_tex(self, data: dict) -> str:
        project = data["project"]
        analysis = data["analysis"]
        params = data["params"]
        species = str(params.get("species", "human"))
        proj_name = getattr(project, "name", "") or getattr(analysis, "name", "")

        # Results section: title + QC + PCA + per-comparison
        results = [f"\\section{{Results: {tex_escape(proj_name)}}}"]
        results.append(r"\subsection{Quality Control}")
        results.append(EXPLANATORY["qc"])
        results.append(generate_qc_table(data["qc_rows"], proj_name))
        results.append(r"\subsection{Principal Component Analysis}")
        results.append(EXPLANATORY["pca"])
        if data.get("pca_fig"):
            results.append(_figure_latex(data["pca_fig"],
                                         "PCA of VST-normalised expression across all samples.",
                                         "fig:pca", "12cm"))
        for comp in data["comparisons"]:
            results.append(build_comparison_section(comp))
        projects_results = "\n\n".join(results)

        template_vars = {
            "jj_file_title": tex_escape(f"GenoLens Report - {proj_name}"),
            "jj_report_title": tex_escape("Transcriptomics Analysis Report"),
            "jj_report_subtitle": tex_escape(f"Analysis: {getattr(analysis, 'name', '')}"),
            "jj_report_version": "1.0",
            "jj_report_number": tex_escape(f"GL-{str(analysis.id)[:8].upper()}"),
            "jj_report_author": tex_escape("SciLicium"),
            "jj_report_clientref": "",
            "jj_sponsor_name": "", "jj_sponsor_contact": "", "jj_sponsor_email": "", "jj_sponsor_address": "",
            "jj_test_facility_name": tex_escape("SciLicium"), "jj_test_facility_contact": "",
            "jj_test_facility_email": "", "jj_test_facility_address": "",
            "jj_test_site_name": tex_escape("SciLicium"), "jj_test_site_contact": "",
            "jj_test_site_email": "", "jj_test_site_address": "",
            "jj_report_project": tex_escape(proj_name),
            "jj_report_prepared_by": "", "jj_report_checked_by": "", "jj_report_approved_by": "",
            "jj_analysis_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "jj_pipeline_version": "2.0",
            "jj_genome_version": tex_escape(species),
            "jj_materials_methods": self._read_mm(),
            "jj_executive_summary": data.get("executive_summary", ""),
            "jj_conclusion": data.get("conclusion", ""),
            "jj_projects_results": projects_results,
            "jj_appendix": "",
            "ref_appendix_soft_versions": "A",
        }
        template = self.env.get_template(MAIN_TEMPLATE)
        return template.render(**template_vars)

    def compile_pdf(self, tex_str: str, work_dir: str) -> bytes:
        # Stage template assets + the rendered .tex into work_dir, compile.
        for item in TEMPLATE_DIR.iterdir():
            if item.name.endswith(".tex"):
                continue
            dst = Path(work_dir) / item.name
            if item.is_dir():
                if not dst.exists():
                    shutil.copytree(item, dst)
            else:
                shutil.copy(item, dst)
        # M&M file must be present for any \input fallback
        shutil.copy(TEMPLATE_DIR / MM_TEMPLATE, Path(work_dir) / MM_TEMPLATE)

        tex_path = Path(work_dir) / "report.tex"
        tex_path.write_text(tex_str)
        env = {**os.environ, "TEXINPUTS": f"{work_dir}//:"}
        for _ in range(2):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                cwd=work_dir, env=env, capture_output=True, text=True, timeout=600,
            )
        pdf_path = Path(work_dir) / "report.pdf"
        if not pdf_path.exists():
            log = (Path(work_dir) / "report.log")
            tail = log.read_text()[-2000:] if log.exists() else proc.stdout[-2000:]
            raise RuntimeError(f"pdflatex failed:\n{tail}")
        # Normalise font embedding with Ghostscript (best-effort)
        out = Path(work_dir) / "report_embedded.pdf"
        gs = shutil.which("gs")
        if gs:
            r = subprocess.run([gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
                                "-dEmbedAllFonts=true", "-dSubsetFonts=false",
                                f"-sOutputFile={out}", str(pdf_path)],
                               capture_output=True, text=True, timeout=300)
            if out.exists():
                return out.read_bytes()
        return pdf_path.read_bytes()

    async def render_pdf(self, db: AsyncSession, analysis_id: UUID, generate_ai: bool = False) -> bytes:
        with tempfile.TemporaryDirectory(prefix="report_") as work_dir:
            data = await self.collect_analysis_data(db, analysis_id, work_dir, generate_ai)
            tex_str = self.assemble_tex(data)
            return await asyncio.to_thread(self.compile_pdf, tex_str, work_dir)


report_latex_service = ReportLatexService()
