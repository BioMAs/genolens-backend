"""
What the LaTeX report says about the run it reports.

The report used to carry one fixed Material and Methods text, written for a pipeline that no
longer exists as described: filterByExpr with a default of 10, "robust" three-method agreement,
a fixed list of enrichment databases, a Stouffer p-value "provided for information", a dot size
that was the gene ratio. `methods_context` replaces it with what the run recorded:

  * the analysis parameters (`SelfServiceAnalysis.params`);
  * the DEG dataset's metadata: the method that decided the calls, the design each comparison was
    fitted with, the versions of R and of the packages, whether the sample thresholds were relaxed
    (all copied from the R manifest by `deg_dataset_metadata`);
  * the VST dataset's `qc_report`: the thresholds applied, the samples excluded, the gene filter;
  * the ENRICHMENT datasets' metadata: the cut-off the script was given (`padj_cutoff`), the
    anno.db file, the categories and the background (`functional_enrichment.R --meta-out`).

A value that is not recorded comes back `None` (or an empty dict / list where that is its natural
absence) and the text says "not recorded": never a default, never the analysis parameters alone
where the script ran at something else (the enrichment cut-off, see `cosmomics_nova._enrichment_fdr`).
Results without a parent analysis are imported: their methods are not described. When their DEG
dataset's metadata still names the analysis that produced them, that analysis is gone (deleted):
the report says so instead of saying the results were not produced in GenoLens.

Everything here is pure: no database, no TeX. The Material and Methods template
(`templates/report_latex/material_and_methods.tex`) is rendered from `methods_context`, and the
sentences of the report body that depend on the run (`de_overview_tex`, `enrichment_intro_tex`,
`volcano_tex`, `no_gene_tex`, `no_term_tex`) are built from the same record.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from app.services.cosmomics_nova import _enrichment_fdr

# Terms the enrichment dotplot shows (the smallest adjusted p-values).
DOTPLOT_TERMS = 20

# The order in which `--method all` lets a method decide: DESeq2, and when it gave no result for a
# comparison, edgeR, then limma (run_multimethod_pipeline.R, primary_order).
_DECIDING_ORDER = ("DESeq2", "edgeR", "limma")
_CANONICAL = {"deseq2": "DESeq2", "edger": "edgeR", "limma": "limma"}
_RUN_BY_METHOD = {
    "all": _DECIDING_ORDER,
    "deseq2": ("DESeq2",),
    "edger": ("edgeR",),
    "limma": ("limma",),
}
_VERSION_KEYS = ("R", "DESeq2", "edgeR", "limma")


def method_name(method: Any) -> str:
    """How a method is named in the text: limma is run through voom."""
    return "limma-voom" if method == "limma" else str(method)


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _canonical_method(value: Any) -> Optional[str]:
    """DESeq2 / edgeR / limma from the way the R manifest or the wizard names a method."""
    return _CANONICAL.get(str(value).strip().lower()) if value is not None else None


def _de_method(*values: Any) -> Optional[str]:
    for value in values:
        key = str(value).strip().lower() if value is not None else ""
        if key in _RUN_BY_METHOD:
            return key
    return None


def _strings(value: Any) -> Optional[list[str]]:
    """A list of non-empty strings, None when the value is not one (or is absent)."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    return [str(v) for v in value if v is not None and str(v).strip()]


def _min_reps(qc_report: dict, params: dict) -> tuple[Optional[float], Optional[str]]:
    """(samples a gene must reach the minimum count in, how it was set).

    The pipeline records the number it resolved (`min_reps_threshold`) and its mode. A run that
    predates the mode ("fixed" was the only one) has a number and a mode only through the params.
    """
    asked = params.get("min_reps")
    mode = qc_report.get("min_reps_mode")
    if mode not in ("auto", "fixed"):
        mode = "auto" if asked == "auto" else ("fixed" if _num(asked) is not None else None)
    recorded = _num(qc_report.get("min_reps_threshold"))
    return _first(recorded, _num(asked)), mode


def _versions(raw: Any, de_method: Optional[str]) -> Optional[dict[str, str]]:
    """R and the packages of the methods that ran, as the manifest recorded them. Never the
    version of a package that was not part of the run."""
    if not isinstance(raw, dict):
        return None
    wanted = {"R", *_RUN_BY_METHOD.get(de_method or "", ())}
    versions = {
        key: str(raw[key])
        for key in _VERSION_KEYS
        if key in wanted and isinstance(raw.get(key), str) and raw[key].strip()
    }
    return versions or None


def _design_used(deg_meta: dict) -> dict[str, str]:
    """Comparison name -> the design it was fitted with ("" names a single, unnamed comparison)."""
    used = deg_meta.get("design_used")
    if isinstance(used, dict):
        return {str(k): str(v) for k, v in used.items() if isinstance(v, str) and v.strip()}
    if isinstance(used, str) and used.strip():
        return {str(deg_meta.get("comparison_name") or ""): used}
    return {}


def _distinct(values: Iterable[Any]) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value is not None and value not in seen:
            seen.append(value)
    return seen


def _enrichment(metas: list[dict], params: Optional[dict]) -> dict[str, Any]:
    """What the enrichment behind the report was, from its datasets' metadata.

    kind "annodb": GenoLens ran the over-representation analysis; "stored": rows deposited with
    the dataset (their test is not described); "mixed": both, for different comparisons; "none":
    no enrichment dataset for the comparisons of the report.
    """
    annodb = [m for m in metas if m.get("source") == "annodb"]
    kind = "none" if not metas else "annodb" if len(annodb) == len(metas) else (
        "stored" if not annodb else "mixed"
    )
    asked = (params or {}).get("enrichment_databases")

    # The cut-off the script ran at is the one it was given and the worker recorded; the params
    # only stand in when nothing was recorded, and only where they cannot mislead.
    recorded = [_num(m.get("padj_cutoff")) for m in annodb]
    if annodb and all(v is not None for v in recorded) and len(set(recorded)) == 1:
        fdr = recorded[0]
    elif annodb and any(v is not None for v in recorded):
        fdr = None  # records that disagree
    elif kind in ("none", "stored"):
        fdr = None  # no run stands behind a cut-off: no result, or tables deposited as they are
    else:
        fdr = _enrichment_fdr(None, params)

    categories = _distinct(c for m in annodb for c in (_strings(m.get("categories")) or []))
    if annodb and not categories:
        # An older result did not record them: the categories the analysis asked for.
        picked = _strings(asked) if isinstance(asked, list) else None
        categories = picked or []
    files = _distinct(m.get("annodb_file") for m in annodb if isinstance(m.get("annodb_file"), str))
    backgrounds = _distinct(
        m.get("background") for m in annodb if isinstance(m.get("background"), str)
    )
    return {
        "kind": kind,
        "fdr": fdr,
        "categories": categories or None,
        "annodb_file": ", ".join(files) or None,
        "background": backgrounds[0] if len(backgrounds) == 1 else None,
        # An explicit empty pick ("0 databases selected") runs no enrichment at all.
        "requested": asked is None or bool(asked),
    }


def methods_context(
    params: Optional[dict],
    deg_meta: dict,
    vst_meta: dict,
    enrichment_metas: list[dict],
) -> dict[str, Any]:
    """The record of the run a report describes.

    `params` is the parent analysis's parameters: None when the DEG dataset has no parent analysis
    (imported results, whose methods GenoLens does not describe). `deg_meta`, `vst_meta` and
    `enrichment_metas` are the `dataset_metadata` of the DEG dataset (merged over comparisons when
    the report has several: `merge_deg_metas`), of the analysis's VST dataset and of the
    ENRICHMENT datasets of the comparisons.
    """
    deg_meta = deg_meta if isinstance(deg_meta, dict) else {}
    vst_meta = vst_meta if isinstance(vst_meta, dict) else {}
    metas = [m for m in (enrichment_metas or []) if isinstance(m, dict)]
    imported = params is None
    # The metadata of results produced by an analysis keeps naming it; if the analysis cannot be
    # found any more, the results are not imported ones, only orphaned.
    analysis_gone = imported and bool(deg_meta.get("analysis_id"))
    p = params if isinstance(params, dict) else {}
    qc_report = vst_meta.get("qc_report") if isinstance(vst_meta.get("qc_report"), dict) else {}
    if imported:
        # Not a GenoLens analysis: nothing of how the calls were made is known or described.
        deg_meta, qc_report = {}, {}

    de_method = _de_method(p.get("de_method"), deg_meta.get("de_method"))
    primary = _canonical_method(deg_meta.get("primary_method"))
    per_comparison = deg_meta.get("primary_methods")
    primary_by_comparison = (
        [
            (str(name), method_name(method))
            for name, method in per_comparison.items()
            if _canonical_method(method) and isinstance(name, str)
        ]
        if primary is None and isinstance(per_comparison, dict)
        else []
    )
    if primary is None and de_method in ("deseq2", "edger", "limma"):
        primary = _CANONICAL[de_method]  # a single method run is the one that decided
    other_methods = (
        [method_name(m) for m in _DECIDING_ORDER if m != primary]
        if primary and de_method == "all"
        else []
    )
    ahead = (
        [method_name(m) for m in _DECIDING_ORDER[: _DECIDING_ORDER.index(primary)]]
        if primary and de_method == "all" and primary in _DECIDING_ORDER
        else []
    )
    min_reps, min_reps_mode = _min_reps(qc_report, p)
    relaxed = deg_meta.get("qc_relaxed")
    return {
        "imported": imported,
        "analysis_gone": analysis_gone,
        "de_method": de_method,
        "primary_method": primary,
        # [(comparison, method)]: the deciding method of each comparison that records one, when
        # the comparisons of the report do not all share it; `primary_differs` when they name more
        # than one.
        "primary_by_comparison": primary_by_comparison,
        "primary_differs": len({method for _name, method in primary_by_comparison}) > 1,
        "other_methods": other_methods,
        "ahead_methods": ahead,
        "fdr": _first(_num(p.get("fdr")), _num(deg_meta.get("fdr"))),
        "min_log2fc": _first(_num(p.get("min_log2fc")), _num(deg_meta.get("min_log2fc"))),
        "min_count": _first(_num(qc_report.get("min_count_threshold")), _num(p.get("min_count"))),
        "min_reps": min_reps,
        "min_reps_mode": min_reps_mode,
        "qc": {
            "min_reads": _first(_num(qc_report.get("min_reads_threshold")), _num(p.get("min_reads"))),
            "min_genes": _first(_num(qc_report.get("min_genes_threshold")), _num(p.get("min_genes"))),
            "removed_sample_ids": _strings(qc_report.get("removed_sample_ids")),
            "relaxed": relaxed if isinstance(relaxed, bool) else None,
        },
        "design_used": _design_used(deg_meta),
        "enrichment": _enrichment(metas, None if imported else p),
        "versions": _versions(deg_meta.get("software_versions"), de_method),
        "dotplot_terms": DOTPLOT_TERMS,
    }


def merge_deg_metas(metas: list[dict]) -> dict:
    """One `deg_meta` out of those of several comparisons, for a report that covers them all.

    A setting the comparisons agree on is kept; one they disagree on, or that one of them does not
    record, is dropped (the text then says "not recorded"); the designs are listed per comparison,
    and so is the method that decided the calls when it is not the same for all of them
    (`primary_methods`); the sample thresholds were relaxed if they were for any of them.
    """
    metas = [m for m in metas if isinstance(m, dict)]
    if not metas:
        return {}
    merged: dict[str, Any] = {}
    for key in ("de_method", "primary_method", "min_log2fc", "fdr", "software_versions"):
        values = [m.get(key) for m in metas]
        if values[0] is not None and all(v == values[0] for v in values):
            merged[key] = values[0]
    if "primary_method" not in merged:
        # The comparisons disagree on the method that decided their calls (or some record none):
        # name it comparison by comparison, for those that record one.
        primary = {
            str(m.get("comparison_name") or ""): _canonical_method(m.get("primary_method"))
            for m in metas
        }
        merged["primary_methods"] = {name: method for name, method in primary.items() if method}
    relaxed = [m.get("qc_relaxed") for m in metas if isinstance(m.get("qc_relaxed"), bool)]
    if relaxed:
        merged["qc_relaxed"] = any(relaxed)
    designs: dict[str, str] = {}
    for m in metas:
        designs.update(_design_used(m))
    if designs:
        merged["design_used"] = designs
    return merged


# ── sentences of the report body, from the same record ─────────────────────────────


def format_number(value: Any) -> str:
    """A recorded number as it is written in the text: 5 not 5.0, 0.05, 1e-05, and an integer in
    full (1000000, never 1e+06; a recorded integer is never rounded to six digits). Raises on a
    value that is not a number."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    short = f"{number:g}"
    return short if float(short) == number else repr(number)


_g = format_number


def thresholds_tex(ctx: dict) -> str:
    """"adjusted p-value below 0.05 and absolute log2 fold-change at or above 1": the thresholds
    that are recorded, joined; "" when none is."""
    parts = []
    if ctx.get("fdr") is not None:
        parts.append(f"adjusted \\textit{{p}}-value below {_g(ctx['fdr'])}")
    if ctx.get("min_log2fc") is not None:
        parts.append(f"absolute log$_2$ fold-change at or above {_g(ctx['min_log2fc'])}")
    return " and ".join(parts)


def not_described_tex(ctx: dict) -> str:
    """Why the methods of results with no parent analysis are not described."""
    if ctx.get("analysis_gone"):
        return (
            "The analysis that produced these results is no longer available; "
            "its methods are not described here."
        )
    return "The differential analysis was not run in GenoLens; its methods are not described here."


def de_overview_tex(ctx: dict) -> str:
    """The sentence of a comparison's overview on how its DEGs were called."""
    if ctx["imported"]:
        return not_described_tex(ctx)
    thresholds = thresholds_tex(ctx)
    primary = ctx.get("primary_method")
    if not primary:
        text = (
            "The method that decided the differentially expressed gene (DEG) calls is not "
            "recorded for this comparison"
        )
        return text + (f" ({thresholds})." if thresholds else ".")
    text = f"Differentially expressed genes (DEGs) were called with {method_name(primary)}"
    text += f" (Benjamini-Hochberg {thresholds})." if thresholds else "."
    if ctx.get("ahead_methods"):
        text += (
            f" {' and '.join(ctx['ahead_methods'])} gave no result for this comparison, "
            f"so {method_name(primary)} decided the calls."
        )
    if ctx.get("other_methods"):
        text += (
            f" {' and '.join(ctx['other_methods'])} were also run on the same counts; "
            "they do not decide the calls and are not reported here."
        )
    return text


def volcano_tex(ctx: dict) -> str:
    """What the volcano plot shows, with the thresholds drawn on it when they are recorded."""
    text = (
        "The volcano plot shows which genes differ between the two conditions. Each point is a gene; "
        "the x-axis is the magnitude of change (log\\textsubscript{2} fold-change, test/control) and "
        "the y-axis its statistical significance (--log\\textsubscript{10} adjusted p-value). "
        "Red points are the genes called up-regulated in the test condition, blue points the genes "
        "called down-regulated, grey points the genes not called."
    )
    if ctx.get("fdr") is not None and ctx.get("min_log2fc") is not None:
        text += (
            f" Dashed lines mark the thresholds of the analysis (adjusted p-value {_g(ctx['fdr'])}, "
            f"log\\textsubscript{{2}} fold-change $\\pm${_g(ctx['min_log2fc'])})."
        )
    return text


def no_gene_tex(ctx: dict) -> str:
    """The overview of a comparison in which no gene was called."""
    thresholds = thresholds_tex(ctx)
    if thresholds:
        return f"0 genes passed the thresholds ({thresholds})."
    if ctx["imported"] and not ctx.get("analysis_gone"):
        return "0 genes passed the thresholds of the imported results."
    return "0 genes passed the thresholds (the thresholds are not recorded)."


def no_enrichment_import_tex(ctx: dict) -> str:
    """The enrichment line of results that have no parent analysis: what the report omits, and why.

    The LaTeX report does not read the rows an import carries (the legacy Python enrichment stores
    them under the DEG dataset, a delivered ENRICHMENT dataset may exist), so it neither shows nor
    denies them. For an orphaned analysis the results are not imported ones: the sentence names the
    analysis that is gone, as the differential paragraph does.
    """
    if ctx.get("analysis_gone"):
        return (
            "Enrichment results are not included in this report: the analysis that produced "
            "these results is no longer available."
        )
    return "Enrichment results of imported data are not included in this report."


def no_term_tex(ctx: dict, no_gene: bool = False) -> str:
    """The enrichment section of a comparison that has no enriched term."""
    enr = ctx["enrichment"]
    if ctx["imported"]:
        return no_enrichment_import_tex(ctx)
    if not enr["requested"]:
        return "No annotation database was selected for this analysis: no enrichment was run."
    if enr["kind"] == "none":
        # No ENRICHMENT dataset is registered: the record cannot tell "ran, nothing passed" from
        # "did not run" (the worker registers none when the script fails), so no cut-off and no
        # "no term passed" is stated, as in the Material and Methods and in the editorial.
        text = "No enrichment result is stored for this comparison."
    elif enr["fdr"] is not None:
        text = f"No term passed (adjusted p $<$ {_g(enr['fdr'])})."
    else:
        text = "No term passed (the cut-off of the run is not recorded)."
    if no_gene:
        text += " The enrichment tests the differentially expressed genes, and none was called."
    return text


def enrichment_intro_tex(ctx: dict) -> str:
    """The sentence above the enrichment figures and tables of a comparison."""
    enr = ctx["enrichment"]
    if enr["kind"] in ("stored", "none"):
        text = (
            "The enrichment results are those stored with the dataset; the test and the "
            "correction that produced them are not described here."
        )
    else:
        if enr["fdr"] is not None:
            kept = f"terms with an adjusted \\textit{{p}}-value below {_g(enr['fdr'])} are shown"
        else:
            kept = "terms are shown at the cut-off of the run, which is not recorded"
        text = (
            "Differentially expressed genes were further studied by over-representation analysis "
            "(ORA) of the up- and of the down-regulated genes, with a hypergeometric test and "
            f"Benjamini-Hochberg adjustment; {kept}."
        )
    return text + (
        " Dot size is the number of genes of the term found among the DEGs; colour is the "
        "significance ($-\\log_{10}$ adjusted \\textit{p}-value)."
    )
