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

from app.core.config import settings
from app.models.models import (
    Dataset, DatasetType, DegGene, EnrichmentPathway, AIInterpretation,
    Project, SelfServiceAnalysis,
)
from app.services import report_latex_methods as rm
from app.services.enrichment_source import match_enrichment_dataset
from app.services.report_common import (
    ai_label,
    ai_text_record,
    comparison_text_is_current,
    merge_cover_info,
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


# Common Unicode punctuation that pdflatex can choke on, mapped to LaTeX-safe ASCII.
_UNICODE_PUNCT = {
    "’": "'", "‘": "'", "“": "``", "”": "''",
    "–": "--", "—": "---", "…": r"\ldots{}",
    " ": " ", " ": " ", "°": r"\textdegree{}",
    "×": r"\texttimes{}", "−": "-",
}


def plaintext_to_latex(text) -> str:
    """Convert user-entered prose into LaTeX-safe markup.

    User-provided conclusion / Material & Methods are plain text, not LaTeX, so
    every special character must be escaped (e.g. a literal '#' or '&' otherwise
    aborts pdflatex). Blank lines become paragraph breaks; single newlines are
    treated as soft wraps (joined with spaces).
    """
    if not text:
        return ""
    paragraphs = re.split(r"\n\s*\n", str(text).strip())
    out = []
    for para in paragraphs:
        line = " ".join(seg.strip() for seg in para.splitlines() if seg.strip())
        if not line:
            continue
        # Escape LaTeX specials first, then map Unicode punctuation to LaTeX
        # commands (tex_escape leaves these Unicode chars untouched).
        escaped = tex_escape(line)
        for uni, repl in _UNICODE_PUNCT.items():
            escaped = escaped.replace(uni, repl)
        out.append(escaped)
    return "\n\n".join(out)


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


def fmt_g(value) -> str:
    """A recorded number as it is written in the text: 5 not 5.0, 0.05, 1e-05, and an integer in
    full (1000000, never 1e+06: see `report_latex_methods.format_number`)."""
    try:
        return rm.format_number(value)
    except (ValueError, TypeError, OverflowError):
        return ""


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
}
# What depends on the run (the volcano plot's thresholds, how the DEGs were called, how the
# enrichment was made) is written from its record: see `report_latex_methods`.


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation (matplotlib → vector PDF placed in the work dir)
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> None:
    fmt = "svg" if str(path).lower().endswith(".svg") else "pdf"
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def _figure_latex(rel_path: str, caption: str, label: str, width: str = "13cm") -> str:
    return (
        "\\begin{figure}[H]\n\\centering\n"
        f"\\includegraphics[width={width}]{{{rel_path}}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n"
    )


def _plot_volcano(genes: list[dict], out_path: str, fdr: Optional[float] = None,
                  min_log2fc: Optional[float] = None) -> bool:
    """Volcano plot. The colours are the calls the dataset records (`regulation`), so the figure
    agrees with the counts and the table of the report; the dashed lines are the thresholds of
    the run, drawn only when they are recorded."""
    if not genes:
        return False
    lfc = np.array([g.get("log_fc") or 0.0 for g in genes], dtype=float)
    padj = np.array([g.get("padj") if g.get("padj") is not None else 1.0 for g in genes], dtype=float)
    neg_log_p = -np.log10(np.clip(padj, 1e-300, 1.0))
    regulation = np.array([str(g.get("regulation") or "").upper() for g in genes])
    up, down = regulation == "UP", regulation == "DOWN"
    ns = ~(up | down)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lfc[ns], neg_log_p[ns], s=5, color="#AAAAAA", alpha=0.5, label="NS")
    ax.scatter(lfc[up], neg_log_p[up], s=8, color="#D73027", alpha=0.7, label="Up")
    ax.scatter(lfc[down], neg_log_p[down], s=8, color="#4575B4", alpha=0.7, label="Down")
    if fdr is not None and 0 < fdr <= 1:
        ax.axhline(-np.log10(fdr), color="gray", lw=0.8, ls="--")
    if min_log2fc is not None:
        ax.axvline(min_log2fc, color="gray", lw=0.8, ls="--")
        ax.axvline(-min_log2fc, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, markerscale=2)
    _save_fig(fig, out_path)
    return True


def _plot_enrichment_dotplot(pathways: list[dict], out_path: str) -> bool:
    """Dotplot of the terms with the smallest adjusted p-values. Dot size is the number of genes
    (`gene_count`), colour is -log10 of the adjusted p-value: the caption says so."""
    if not pathways:
        return False

    def _padj(p: dict) -> float:
        # `or 1.0` would rank an adjusted p-value of exactly 0 (an underflow) last.
        value = p.get("padj")
        return 1.0 if value is None else value

    top = sorted(pathways, key=_padj)[:rm.DOTPLOT_TERMS]
    names = [str(p.get("pathway_name") or p.get("pathway_id") or "")[:55] for p in top]
    pvals = [-np.log10(max(_padj(p), 1e-300)) for p in top]
    sizes = [max(20, (p.get("gene_count") or 1) * 5) for p in top]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    sc = ax.scatter(pvals, range(len(top)), s=sizes, c=pvals, cmap="RdBu_r", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("-log10(adjusted p-value)")
    plt.colorbar(sc, ax=ax, label="-log10(adjusted p-value)")
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
        r"\begin{longtable}{L{4cm}S[table-format=4.2]S[table-format=6]L{4cm}}",
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
            f"{tex_escape(str(r['sample']))} & {format_millions(r['library_size'])} "
            f"& {r['detected_genes']} & {tex_escape(r.get('condition', 'Unknown'))} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def generate_deg_table(degs: list[dict], max_rows: int = 15) -> str:
    """The differentially expressed genes with the smallest adjusted p-values. `degs` are the
    genes of the comparison as tested: only those the dataset calls UP or DOWN are listed."""
    called = [d for d in degs if str(d.get("regulation") or "").upper() in ("UP", "DOWN")]
    if not called:
        return r"\textit{No differentially expressed genes found}"
    rows = sorted(called, key=lambda d: (d.get("padj") if d.get("padj") is not None else 1.0))[:max_rows]
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

def _ai_box(text: str, model: Optional[str] = None, generated_at: Optional[datetime] = None) -> str:
    """The AI comparison text in its box, titled with the label every AI text carries: its model
    and the date it was generated."""
    label = tex_escape(ai_label(model or "", generated_at))
    return (
        r"\subsubsection{Biological Summary (AI)}" "\n"
        r"\begin{tcolorbox}[enhanced,breakable,colback=sciliciumgreen!5,"
        r"colframe=sciliciumgreen,boxrule=0.8pt,arc=1mm,"
        rf"title={{\textbf{{{label}}}}},fonttitle=\bfseries,coltitle=white]" "\n"
        f"{tex_escape(text)}" "\n"
        r"\end{tcolorbox}" "\n"
    )


def printed_ai_texts(comparisons: list[dict]) -> list[dict]:
    """One record, `{kind, model, generated_at}`, per comparison whose AI text the report prints."""
    return [
        ai_text_record("comparison", comp.get("ai_model"), comp.get("ai_generated_at"))
        for comp in comparisons
        if comp.get("ai_interpretation")
    ]


def _dotplot_caption(direction: str, test_d: str, control_d: str) -> str:
    """The caption says what the plot encodes: size is the number of genes, colour the adjusted
    p-value (never the gene ratio)."""
    return (
        f"Enrichment ({direction}) --- {test_d} vs {control_d}. Dot size: number of genes; "
        r"colour: $-\log_{10}$ adjusted \textit{p}-value; "
        f"the terms with the smallest adjusted \\textit{{p}}-value are shown (at most {rm.DOTPLOT_TERMS})."
    )


def build_comparison_section(comp: dict) -> str:
    """comp: dict produced by collect_analysis_data (see below).

    `comp["methods"]` is the record of the run (`report_latex_methods.methods_context`); a
    comparison without one is written as an imported result, whose methods are not described.
    """
    methods = comp.get("methods") or rm.methods_context(None, {}, {}, [])
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
        s.append(rm.no_gene_tex(methods))
    else:
        s.append(
            f"A total of {n_total} differentially expressed genes were detected, "
            f"of which {n_up} are up-regulated and {n_down} down-regulated in the test condition."
        )
    s.append("")
    s.append(rm.de_overview_tex(methods))
    s.append("")

    s.append(r"\subsubsection{Detailed Comparison Results}")
    s.append(rm.volcano_tex(methods))
    s.append("")
    if comp.get("volcano_fig"):
        s.append(_figure_latex(comp["volcano_fig"],
                               f"Volcano plot --- {test_d} vs {control_d}.",
                               f"fig:volcano_{comp['safe_id']}", "12.5cm"))
    s.append(generate_deg_table(comp["degs"]))
    s.append("")

    s.append(r"\subsubsection{Functional Enrichment}")
    if comp["enr_up"] or comp["enr_down"]:
        s.append(rm.enrichment_intro_tex(methods))
        s.append("")
    else:
        s.append(rm.no_term_tex(methods, no_gene=n_total == 0))
        s.append("")
    if comp.get("dotplot_up_fig"):
        s.append(r"\paragraph{Up-regulated genes}\mbox{}\\")
        s.append(_figure_latex(comp["dotplot_up_fig"],
                               _dotplot_caption("up-regulated", test_d, control_d),
                               f"fig:enr_up_{comp['safe_id']}", "12cm"))
    if comp["enr_up"]:
        s.append(generate_enrichment_table(comp["enr_up"], "Up-regulated genes"))
    if comp.get("dotplot_down_fig"):
        s.append(r"\paragraph{Down-regulated genes}\mbox{}\\")
        s.append(_figure_latex(comp["dotplot_down_fig"],
                               _dotplot_caption("down-regulated", test_d, control_d),
                               f"fig:enr_down_{comp['safe_id']}", "12cm"))
    if comp["enr_down"]:
        s.append(generate_enrichment_table(comp["enr_down"], "Down-regulated genes"))
    s.append("")

    if comp.get("ai_interpretation"):
        s.append(
            _ai_box(comp["ai_interpretation"], comp.get("ai_model"), comp.get("ai_generated_at"))
        )

    return "\n".join(s)


# Columns of the sample sheet the R pipeline reads (run_multimethod_pipeline.R: sid_col, cond_col).
_SAMPLE_ID_ALIASES = ("sample_id", "sample", "sampleid", "id", "name")
_CONDITION_ALIASES = ("condition", "group", "groupe", "treatment", "genotype")


def pick_sample_conditions(df: pd.DataFrame, condition_column: Optional[str] = None) -> dict[str, str]:
    """sample -> condition, from the column the analysis was run on.

    The R pipeline uses the wizard's `condition_column` (case-insensitive) when there is one and
    never falls back to another column; without one it takes the first of its aliases. A column
    that is asked for and absent, or that is the sample-id column, gives no conditions.
    """
    columns = {str(c).strip().lower(): c for c in df.columns}
    sid = next((columns[a] for a in _SAMPLE_ID_ALIASES if a in columns), None)
    if sid is None:
        return {}
    if condition_column and str(condition_column).strip():
        cond = columns.get(str(condition_column).strip().lower())
        if cond is None or cond == sid:
            return {}
    else:
        cond = next((columns[a] for a in _CONDITION_ALIASES if a in columns and columns[a] != sid), None)
        if cond is None:
            return {}
    return {str(r[sid]): str(r[cond]) for _, r in df.iterrows()}


def _condition_column(analysis) -> Optional[str]:
    """The condition column the analysis was run on (None: the pipeline's alias detection)."""
    params = getattr(analysis, "params", None)
    return params.get("condition_column") if isinstance(params, dict) else None


def undefined_references(log: str) -> list[str]:
    """The labels and citations a LaTeX log reports as undefined (["?"] when it only says that
    there are some)."""
    names = re.findall(r"(?:Reference|Citation) `([^']*)' on page \S+ undefined", log)
    if not names and "There were undefined references" in log:
        return ["?"]
    return names


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
            "fmt_g": fmt_g, "method_name": rm.method_name,
        }.items():
            self.env.filters[name] = fn

    # ── Data collection (analysis-scoped) ────────────────────────────────────
    async def collect_analysis_data(self, db: AsyncSession, analysis_id: UUID,
                                    work_dir: str, generate_ai: bool = True,
                                    include_ai: bool = True) -> dict[str, Any]:
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

        vst_meta = await self._vst_meta(db, analysis)
        comparisons = []
        for deg_ds in deg_datasets:
            comp = await self._build_comparison(db, deg_ds, enr_datasets, comp_conditions,
                                                work_dir, generate_ai, include_ai=include_ai,
                                                params=params, vst_meta=vst_meta)
            comparisons.append(comp)
        comparisons.sort(key=lambda c: c["comp_id"])

        # PCA + QC figures
        pca_fig = await self._make_pca(db, analysis, work_dir)
        qc_rows = await self._make_qc(db, analysis)

        # The Material and Methods describe the whole analysis: what its comparisons record.
        methods = rm.methods_context(
            params,
            rm.merge_deg_metas([c.get("_deg_meta") or {} for c in comparisons]),
            vst_meta,
            [m for c in comparisons for m in c.get("_enr_metas") or []],
        )
        return {
            "project": project,
            "analysis": analysis,
            "params": params,
            "comparisons": comparisons,
            "pca_fig": pca_fig,
            "qc_rows": qc_rows,
            "methods": methods,
            "demo": any(c.get("demo") for c in comparisons),
        }

    # ── Data collection (comparison-scoped) ──────────────────────────────────
    async def _find_parent_analysis(self, db: AsyncSession, dataset_id: UUID,
                                    project_id) -> Optional[SelfServiceAnalysis]:
        """Resolve the analysis whose result_dataset_ids contains this DEG dataset."""
        rows = (await db.execute(
            select(SelfServiceAnalysis).where(SelfServiceAnalysis.project_id == project_id)
        )).scalars().all()
        sid = str(dataset_id)
        for a in rows:
            if sid in (a.result_dataset_ids or []):
                return a
        return None

    async def collect_comparison_data(self, db: AsyncSession, dataset_id: UUID,
                                      comparison_name: str, work_dir: str,
                                      generate_ai: bool = False,
                                      make_figures: bool = True,
                                      include_ai: bool = True) -> dict[str, Any]:
        """Data of one comparison report.

        make_figures=False (editorial report) skips the LaTeX PDF figures and returns the raw
        expression data instead, under "expression" (see `_expression_data`).
        include_ai=False reads no stored AI interpretation and generates none.
        """
        deg_ds = await db.get(Dataset, dataset_id)
        if not deg_ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        project = await db.get(Project, deg_ds.project_id)
        analysis = await self._find_parent_analysis(db, dataset_id, deg_ds.project_id)
        params = (analysis.params or {}) if analysis else {}

        # Enrichment datasets + comparison conditions come from the parent analysis.
        enr_datasets = []
        comp_conditions: dict[str, dict] = {}
        pca_fig = None
        expression: Optional[dict] = None
        qc_rows: list[dict] = []
        vst_meta: dict = {}
        if analysis:
            vst_meta = await self._vst_meta(db, analysis)
            result_ids = [UUID(x) for x in (analysis.result_dataset_ids or [])]
            if result_ids:
                datasets = (await db.execute(
                    select(Dataset).where(Dataset.id.in_(result_ids))
                )).scalars().all()
                enr_datasets = [d for d in datasets if d.type == DatasetType.ENRICHMENT]
            comp_conditions = await self._load_comparison_conditions(db, analysis.comparisons_dataset_id)
            if make_figures:
                pca_fig = await self._make_pca(db, analysis, work_dir)
            else:
                expression = await self._expression_data(db, analysis)
            qc_rows = await self._make_qc(db, analysis)

        # A DEG dataset with no parent analysis holds imported results: no params, no methods.
        comp = await self._build_comparison(db, deg_ds, enr_datasets, comp_conditions,
                                            work_dir, generate_ai, comparison_name=comparison_name,
                                            make_figures=make_figures, include_ai=include_ai,
                                            params=params if analysis else None,
                                            vst_meta=vst_meta)

        return {
            "project": project,
            "analysis": analysis,
            "params": params,
            "comparison_name": comparison_name,
            "dataset_id": dataset_id,
            "comparisons": [comp],
            "pca_fig": pca_fig,
            "qc_rows": qc_rows,
            "expression": expression,
            "methods": comp["methods"],
            "demo": bool(comp.get("demo")),
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
                                work_dir, generate_ai, comparison_name: Optional[str] = None,
                                make_figures: bool = True, include_ai: bool = True,
                                params: Optional[dict] = None,
                                vst_meta: Optional[dict] = None) -> dict:
        """`params` are the parent analysis's (None: the DEG dataset is an import, whose methods
        are not described); `vst_meta` is its VST dataset's metadata (the QC report)."""
        meta = deg_ds.dataset_metadata if isinstance(deg_ds.dataset_metadata, dict) else {}
        comp_name = comparison_name or meta.get("comparison_name") or deg_ds.name
        safe_id = re.sub(r"[^a-zA-Z0-9]+", "-", comp_name).strip("-")

        gene_stmt = select(DegGene).where(DegGene.dataset_id == deg_ds.id)
        if comparison_name:
            gene_stmt = gene_stmt.where(DegGene.comparison_name == comparison_name)
        all_genes = (await db.execute(gene_stmt)).scalars().all()
        deg_dicts = [{"gene_id": g.gene_id, "gene_name": g.gene_name, "log_fc": g.log_fc,
                      "padj": g.padj, "pvalue": g.pvalue, "regulation": g.regulation} for g in all_genes]
        n_up = sum(1 for g in all_genes if (g.regulation or "").upper() == "UP")
        n_down = sum(1 for g in all_genes if (g.regulation or "").upper() == "DOWN")

        # Matched enrichment dataset, via the shared rules (`enrichment_source`) rather than a
        # local `comparison_name` check — which missed an enrichment file that names the
        # comparison in `enrichment_comparisons` or in its own dataset name, and could pick a
        # FAILED duplicate over a READY one. Candidates stay scoped to this analysis's result
        # datasets, so the pure matcher is the right entry point.
        enr_ds = match_enrichment_dataset(enr_datasets, comp_name)
        enr_all, enr_up, enr_down = [], [], []
        if enr_ds:
            enr_stmt = select(EnrichmentPathway).where(EnrichmentPathway.dataset_id == enr_ds.id)
            if comparison_name:
                enr_stmt = enr_stmt.where(EnrichmentPathway.comparison_name == comparison_name)
            rows = (await db.execute(enr_stmt)).scalars().all()
            for ep in rows:
                d = {"pathway_id": ep.pathway_id, "pathway_name": ep.pathway_name,
                     "category": ep.category, "pvalue": ep.pvalue, "padj": ep.padj,
                     "gene_count": ep.gene_count, "regulation": ep.regulation}
                reg = (ep.regulation or "ALL").upper()
                (enr_up if reg == "UP" else enr_down if reg == "DOWN" else enr_all).append(d)

        # What the run recorded, for the text of the report. The mark of a demo dataset follows
        # the data (the DEG dataset's or its enrichment's), whoever asks for the report.
        enr_meta = (
            enr_ds.dataset_metadata
            if enr_ds is not None and isinstance(enr_ds.dataset_metadata, dict) else {}
        )
        enr_metas = [enr_meta] if enr_ds is not None else []
        methods = rm.methods_context(params, meta, vst_meta or {}, enr_metas)

        # figures
        comp = {
            "comp_id": comp_name, "safe_id": safe_id,
            "test": comp_conditions.get(comp_name, {}).get("test", comp_name),
            "control": comp_conditions.get(comp_name, {}).get("control", ""),
            "n_up": n_up, "n_down": n_down,
            "degs": deg_dicts, "enr_up": enr_up, "enr_down": enr_down,
            "methods": methods,
            "demo": bool(meta.get("demo") or enr_meta.get("demo")),
            "_deg_meta": meta, "_enr_metas": enr_metas,
        }
        if make_figures:
            vfig = f"volcano_{safe_id}.pdf"
            if await asyncio.to_thread(
                _plot_volcano, deg_dicts, os.path.join(work_dir, vfig),
                fdr=methods["fdr"], min_log2fc=methods["min_log2fc"],
            ):
                comp["volcano_fig"] = vfig
            if enr_up:
                f = f"enr_up_{safe_id}.pdf"
                if await asyncio.to_thread(_plot_enrichment_dotplot, enr_up, os.path.join(work_dir, f)):
                    comp["dotplot_up_fig"] = f
            if enr_down:
                f = f"enr_down_{safe_id}.pdf"
                if await asyncio.to_thread(_plot_enrichment_dotplot, enr_down, os.path.join(work_dir, f)):
                    comp["dotplot_down_fig"] = f

        # AI interpretation (existing, or generate); none at all without include_ai
        if not include_ai:
            return comp
        ai = await db.scalar(
            select(AIInterpretation).where(
                AIInterpretation.dataset_id == deg_ds.id,
                AIInterpretation.comparison_name == comp_name,
            )
        )
        stored = ai is not None  # a stored text is checked against today's data below
        if ai is None and generate_ai:
            # Time-boxed, best-effort: LLM generation can be slow (cold start) and the
            # r-worker is single-concurrency, so never let it block the report/worker.
            # ⚠️ Ce chemin ne passe pas par `reserve_ai_request` : il contournerait le
            # plafond de requêtes IA des comptes démo. Aucun appelant ne l'active
            # aujourd'hui (generate_ai=False partout) ; le brancher sur le plafond
            # avant de l'activer.
            try:
                from app.services.ai_interpreter import generate_and_store  # T6
                ai = await asyncio.wait_for(
                    generate_and_store(db, deg_ds.id, enr_ds.id if enr_ds else None, comp_name),
                    timeout=120,
                )
                stored = False  # written just now, from today's data
            except (asyncio.TimeoutError, Exception) as exc:
                logger.info("AI interpretation skipped for %s: %s", comp_name, exc)
        if ai is not None:
            if stored and not await self._comparison_text_is_current(db, ai):
                # Written from other counts of genes or pathways: never printed as current.
                comp["ai_stale"] = True
                return comp
            comp["ai_interpretation"] = ai.interpretation
            comp["ai_model"] = ai.model
            comp["ai_generated_at"] = ai.created_at
        return comp

    async def _comparison_text_is_current(self, db, ai) -> bool:
        """Whether a stored comparison interpretation was written from today's data: the counts
        it recorded (up, down, pathways) against those its generator would read now."""
        from app.services.ai_interpreter import build_comparison_ai_context

        context = await build_comparison_ai_context(db, ai.dataset_id, None, ai.comparison_name)
        if context is None:  # no differentially expressed gene any more
            return False
        summary = context["deg_summary"]
        return comparison_text_is_current(
            ai, summary["up_count"], summary["down_count"], len(context["top_pathways"])
        )

    async def _vst_dataset(self, db, analysis) -> Optional[Dataset]:
        """The analysis's VST-normalised matrix, when it has a stored one."""
        vst_id = (analysis.intermediate_dataset_ids or {}).get("vst")
        if not vst_id:
            return None
        vst = await db.get(Dataset, UUID(vst_id) if isinstance(vst_id, str) else vst_id)
        if not vst or not vst.parquet_file_path:
            return None
        return vst

    async def _vst_meta(self, db, analysis) -> dict:
        """The metadata of the analysis's VST dataset (its QC report), {} when it has none."""
        vst_id = (analysis.intermediate_dataset_ids or {}).get("vst")
        if not vst_id:
            return {}
        vst = await db.get(Dataset, UUID(vst_id) if isinstance(vst_id, str) else vst_id)
        meta = getattr(vst, "dataset_metadata", None)
        return meta if isinstance(meta, dict) else {}

    async def _expression_data(self, db, analysis) -> dict:
        """Raw expression data for reports that draw their own figures.

        {"vst": parquet bytes | None, "pca": calculate_pca output | None,
         "sample_conditions": {sample: condition}}
        """
        out: dict[str, Any] = {"vst": None, "pca": None, "sample_conditions": {}}
        vst = await self._vst_dataset(db, analysis)
        out["sample_conditions"] = await self._sample_conditions(
            db, analysis.samples_dataset_id, _condition_column(analysis)
        )
        if vst is None:
            return out
        try:
            from app.services.storage import storage_service
            from app.services.data_processor import data_processor
            out["vst"] = await storage_service.download_file(vst.parquet_file_path)
            out["pca"] = await data_processor.calculate_pca(out["vst"], n_components=2)
        except Exception as exc:
            logger.warning("Expression data for the report failed: %s", exc)
        return out

    async def _make_pca(self, db, analysis, work_dir) -> Optional[str]:
        vst = await self._vst_dataset(db, analysis)
        if vst is None:
            return None
        try:
            from app.services.storage import storage_service
            from app.services.data_processor import data_processor
            data = await storage_service.download_file(vst.parquet_file_path)
            pca = await data_processor.calculate_pca(data, n_components=2)
            sample_conditions = await self._sample_conditions(
                db, analysis.samples_dataset_id, _condition_column(analysis)
            )
            out = "pca.pdf"
            if await asyncio.to_thread(_plot_pca, pca, sample_conditions, os.path.join(work_dir, out)):
                return out
        except Exception as exc:
            logger.warning("PCA figure failed: %s", exc)
        return None

    async def _sample_conditions(self, db, samples_dataset_id,
                                 condition_column: Optional[str] = None) -> dict[str, str]:
        """sample -> condition, read as the R pipeline reads it (`pick_sample_conditions`):
        `condition_column` is the wizard's choice (params.condition_column)."""
        if not samples_dataset_id:
            return {}
        ds = await db.get(Dataset, samples_dataset_id)
        if not ds or not ds.parquet_file_path:
            return {}
        try:
            from app.services.storage import storage_service
            data = await storage_service.download_file(ds.parquet_file_path)
            df = await asyncio.to_thread(lambda: pd.read_parquet(io.BytesIO(data)))
            conditions = pick_sample_conditions(df, condition_column)
            if not conditions and condition_column:
                logger.warning("condition column %r not found in the samples table", condition_column)
            return conditions
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
            conditions = await self._sample_conditions(
                db, analysis.samples_dataset_id, _condition_column(analysis)
            )

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

    def render_methods(self, methods: dict) -> str:
        """The Material and Methods text of a run (`report_latex_methods.methods_context`)."""
        return self.env.from_string(self._read_mm()).render(jj_methods=methods)

    def assemble_tex(self, data: dict) -> str:
        project = data["project"]
        analysis = data.get("analysis")
        params = data["params"]
        species = str(params.get("species", "human"))
        proj_name = getattr(project, "name", "") or getattr(analysis, "name", "")

        # Subtitle + report number: comparison-scoped when a comparison is set,
        # otherwise the legacy analysis-scoped values.
        comparison_name = data.get("comparison_name")
        if comparison_name:
            report_subtitle = f"Comparison: {comparison_name}"
            number_src = data.get("dataset_id") or (analysis.id if analysis else "")
        else:
            report_subtitle = f"Analysis: {getattr(analysis, 'name', '')}"
            number_src = analysis.id if analysis else ""
        report_number = f"GL-{str(number_src)[:8].upper()}" if number_src else "GL-REPORT"

        # Branding (report customization module). Empty/None → SciLicium defaults.
        branding = data.get("branding") or {}
        default_address = r"35 bd du portugal\\35\,200 Rennes, France"
        institute_name = (
            tex_escape(branding["institute_name"]) if branding.get("institute_name")
            else tex_escape("SciLicium")
        )
        institute_address = (
            tex_escape(branding["institute_address"]) if branding.get("institute_address")
            else default_address
        )
        # User-provided M&M / conclusion are plain prose → LaTeX-escape them.
        # (Empty → fall back to the default LaTeX M&M template / no conclusion.)
        mm_raw = data.get("materials_methods")
        mm = plaintext_to_latex(mm_raw) if mm_raw else None
        conclusion = plaintext_to_latex(data.get("conclusion"))

        # Results section: title + QC + PCA + per-comparison
        results = [f"\\section{{Results: {tex_escape(proj_name)}}}"]
        results.append(r"\subsection{Quality Control}")
        results.append(EXPLANATORY["qc"])
        results.append(generate_qc_table(data["qc_rows"], proj_name))
        results.append(r"\subsection{Principal Component Analysis}")
        results.append(EXPLANATORY["pca"])
        if data.get("pca_fig"):
            results.append(_figure_latex(data["pca_fig"],
                                         "PCA of VST-normalised expression (samples retained after quality control).",
                                         "fig:pca", "12cm"))
        for comp in data["comparisons"]:
            results.append(build_comparison_section(comp))
        projects_results = "\n\n".join(results)

        # Cover / project information (escaped). Page-layout selection.
        cover = data.get("cover_info") or {}

        def _cv(key: str, default: str = "") -> str:
            val = cover.get(key)
            return tex_escape(val) if val else default

        first_page_type = data.get("first_page_type") or "detailed"
        last_page_type = data.get("last_page_type") or "color"

        template_vars = {
            "jj_file_title": tex_escape(f"GenoLens Report - {proj_name}"),
            "jj_report_title": tex_escape("Transcriptomics Analysis Report"),
            "jj_report_subtitle": tex_escape(report_subtitle),
            "jj_report_version": "1.0",
            "jj_report_number": tex_escape(report_number),
            "jj_report_author": institute_name,
            "jj_institute_name": institute_name,
            "jj_institute_address": institute_address,
            "jj_primary_color": branding.get("primary_color") or "",
            "jj_secondary_color": branding.get("secondary_color") or "",
            "jj_logo_path": branding.get("logo_file") or "",
            "jj_first_page_type": first_page_type,
            "jj_last_page_type": last_page_type,
            "jj_report_clientref": _cv("client_ref"),
            "jj_sponsor_name": _cv("sponsor_name"), "jj_sponsor_contact": _cv("sponsor_contact"),
            "jj_sponsor_email": _cv("sponsor_email"), "jj_sponsor_address": _cv("sponsor_address"),
            "jj_test_facility_name": _cv("test_facility_name", institute_name),
            "jj_test_facility_contact": _cv("test_facility_contact"),
            "jj_test_facility_email": _cv("test_facility_email"),
            "jj_test_facility_address": _cv("test_facility_address"),
            "jj_test_site_name": _cv("test_site_name", institute_name),
            "jj_test_site_contact": _cv("test_site_contact"),
            "jj_test_site_email": _cv("test_site_email"),
            "jj_test_site_address": _cv("test_site_address"),
            "jj_report_project": _cv("project_name", tex_escape(proj_name)),
            "jj_report_prepared_by": _cv("prepared_by"),
            "jj_report_checked_by": _cv("checked_by"),
            "jj_report_approved_by": _cv("approved_by"),
            "jj_contact_email": _cv("contact_email"),
            "jj_contact_phone": _cv("contact_phone"),
            "jj_analysis_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "jj_pipeline_version": tex_escape(settings.APP_VERSION),
            "jj_genome_version": tex_escape(species),
            "jj_materials_methods": mm if mm else self._read_mm(),
            # What the run recorded: the Material and Methods template reads it.
            "jj_methods": data.get("methods") or rm.methods_context(None, {}, {}, []),
            "jj_executive_summary": data.get("executive_summary", ""),
            "jj_conclusion": conclusion,
            "jj_projects_results": projects_results,
            "jj_appendix": "",
            "ref_appendix_soft_versions": "A",
            # Filigrane DEMO sur chaque page : rapport demandé par un compte de démonstration
            # (users.is_demo), ou jeu de données de démonstration, quel que soit le demandeur.
            "jj_demo_watermark": bool(data.get("demo_watermark") or data.get("demo")),
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
        log_path = Path(work_dir) / "report.log"

        def _tail(proc) -> str:
            return log_path.read_text(errors="replace")[-2000:] if log_path.exists() else proc.stdout[-2000:]

        for pass_no in (1, 2):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                cwd=work_dir, env=env, capture_output=True, text=True, timeout=600,
            )
            # A failed pass is a failed report, even when an earlier pass left a PDF behind.
            if proc.returncode != 0:
                raise RuntimeError(f"pdflatex failed (pass {pass_no}, exit {proc.returncode}):\n{_tail(proc)}")
        pdf_path = Path(work_dir) / "report.pdf"
        if not pdf_path.exists():
            raise RuntimeError(f"pdflatex failed:\n{_tail(proc)}")
        # After the second pass every reference resolves; one that does not prints as "??".
        undefined = undefined_references(log_path.read_text(errors="replace") if log_path.exists() else "")
        if undefined:
            raise RuntimeError(
                "pdflatex left undefined references: " + ", ".join(dict.fromkeys(undefined))
            )
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

    async def render_pdf(self, db: AsyncSession, analysis_id: UUID, generate_ai: bool = False,
                         demo_watermark: bool = False, use_ai: bool = True,
                         ai_texts: Optional[list[dict]] = None) -> bytes:
        """The analysis report. `use_ai=False`: no AI text (none read, none generated). When a
        list is given as `ai_texts` it receives one record per AI text printed."""
        with tempfile.TemporaryDirectory(prefix="report_") as work_dir:
            data = await self.collect_analysis_data(
                db, analysis_id, work_dir, generate_ai and use_ai, include_ai=use_ai
            )
            data["demo_watermark"] = demo_watermark or bool(data.get("demo"))
            if ai_texts is not None:
                ai_texts.extend(printed_ai_texts(data["comparisons"]))
            tex_str = self.assemble_tex(data)
            return await asyncio.to_thread(self.compile_pdf, tex_str, work_dir)

    async def is_demo_requester(self, db: AsyncSession, user_id) -> bool:
        """True quand le rapport est demandé par un compte de démonstration.

        Lu à la génération, sur le demandeur du job : le filigrane est figé dans
        le PDF stocké, que le téléchargement sert ensuite tel quel.
        """
        if not user_id:
            return False
        from app.models.models import User
        user = await db.get(User, user_id)
        return bool(user and user.is_demo)

    async def _load_branding(self, db: AsyncSession, user_id, work_dir: str) -> Optional[dict]:
        """Load persistent report branding for a user, gated by the customization module.

        Returns None (→ SciLicium defaults) when the user lacks the module or has
        no settings. Stages the custom logo into the work dir when present.
        """
        if not user_id:
            return None
        from app.models.models import User, UserReportSettings
        user = await db.get(User, user_id)
        if not user or not user.has_report_customization:
            return None
        settings = await db.get(UserReportSettings, user_id)
        if not settings:
            return None
        out = {
            "institute_name": settings.institute_name,
            "institute_address": settings.institute_address,
            "primary_color": (settings.primary_color or "").lstrip("#") or None,
            "secondary_color": (settings.secondary_color or "").lstrip("#") or None,
            "default_conclusion": settings.default_conclusion,
            "default_materials_methods": settings.default_materials_methods,
            "first_page_type": settings.first_page_type or "detailed",
            "last_page_type": settings.last_page_type or "color",
            "cover_info": settings.cover_info or {},
            "logo_file": None,
        }
        if settings.logo_path:
            try:
                from app.services.storage import storage_service
                data = await storage_service.download_file(settings.logo_path)
                ext = os.path.splitext(settings.logo_path)[1] or ".png"
                fname = f"custom_logo{ext}"
                Path(work_dir, fname).write_bytes(data)
                out["logo_file"] = fname
            except Exception as exc:
                logger.warning("Custom logo staging failed: %s", exc)
        return out

    async def render_comparison_pdf(self, db: AsyncSession, dataset_id: UUID,
                                    comparison_name: str, requested_by=None,
                                    conclusion: Optional[str] = None,
                                    materials_methods: Optional[str] = None,
                                    first_page_type: Optional[str] = None,
                                    last_page_type: Optional[str] = None,
                                    cover_info: Optional[dict] = None,
                                    generate_ai: bool = False,
                                    demo_watermark: bool = False,
                                    use_ai: bool = True,
                                    ai_texts: Optional[list[dict]] = None) -> bytes:
        """The comparison report. When a list is given as `ai_texts` it receives one record per
        AI text printed, `{kind, model, generated_at}` (none when `use_ai` is False)."""
        with tempfile.TemporaryDirectory(prefix="report_") as work_dir:
            data = await self.collect_comparison_data(
                db, dataset_id, comparison_name, work_dir, generate_ai and use_ai,
                include_ai=use_ai,
            )
            data["demo_watermark"] = demo_watermark or bool(data.get("demo"))
            if ai_texts is not None:
                ai_texts.extend(printed_ai_texts(data["comparisons"]))
            branding = await self._load_branding(db, requested_by, work_dir)
            data["branding"] = branding
            defaults = branding or {}

            # Per-report content (job override → user default)
            if conclusion:
                data["conclusion"] = conclusion
            elif defaults.get("default_conclusion"):
                data["conclusion"] = defaults["default_conclusion"]
            if materials_methods:
                data["materials_methods"] = materials_methods
            elif defaults.get("default_materials_methods"):
                data["materials_methods"] = defaults["default_materials_methods"]

            # Page layout (job override → user default → template default)
            data["first_page_type"] = first_page_type or defaults.get("first_page_type") or "detailed"
            data["last_page_type"] = last_page_type or defaults.get("last_page_type") or "color"

            # Cover info: merge user defaults with per-report overrides (override wins)
            data["cover_info"] = merge_cover_info(branding, cover_info)

            tex_str = self.assemble_tex(data)
            return await asyncio.to_thread(self.compile_pdf, tex_str, work_dir)


report_latex_service = ReportLatexService()
