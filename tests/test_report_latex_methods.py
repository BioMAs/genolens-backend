"""
The LaTeX report describes the run it reports.

`methods_context` gathers what the run recorded (analysis parameters, DEG / VST / ENRICHMENT dataset
metadata); `material_and_methods.tex` is rendered from it, with Jinja alone (no TeX), and prints a
value only when it is recorded: otherwise it says "not recorded", never a default.
"""
import io
import re
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app.models.models import Dataset, DatasetStatus, DatasetType
from app.services import report_latex_methods as rm
from app.services import report_latex_service as rl

svc = rl.report_latex_service

BACKGROUND = "genes annotated to at least one term of the category tested"

PARAMS = {
    "design": "auto",
    "fdr": 0.05,
    "min_log2fc": 1.0,
    "min_reads": 10,
    "min_genes": 200,
    "min_count": 5,
    "min_reps": "auto",
    "enrichment_databases": None,
    "enrichment_fdr": 0.05,
    "species": "human",
    "de_method": "all",
    "condition_column": None,
}
DEG_META = {
    "analysis_id": "an-1",
    "comparison_name": "A_vs_B",
    "min_log2fc": 1.0,
    "de_method": "all",
    "primary_method": "DESeq2",
    "design_used": "~ batch + condition",
    "qc_relaxed": False,
    "software_versions": {"R": "4.4.1", "DESeq2": "1.44.0", "edgeR": "4.2.1", "limma": "3.60.4"},
}
VST_META = {
    "source": "vst",
    "analysis_id": "an-1",
    "qc_report": {
        "total_input_samples": 8,
        "samples_passed": 7,
        "samples_removed": 1,
        "removed_sample_ids": ["S_3"],
        "min_reads_threshold": 10,
        "min_genes_threshold": 200,
        "min_count_threshold": 5,
        "min_reps_threshold": 3,
        "min_reps_mode": "fixed",
    },
}
ENR_META = {
    "analysis_id": "an-1",
    "comparison_name": "A_vs_B",
    "source": "annodb",
    "padj_cutoff": 0.05,
    "annodb_file": "anno.db.human.gene_symbol.20260324.rds",
    "categories": ["GO_BP", "kegg_pathway"],
    "background": BACKGROUND,
}


def _ctx(params=PARAMS, deg_meta=DEG_META, vst_meta=VST_META, enr=(ENR_META,), de_method=None, **change):
    """`methods_context` of a recorded run; `change` overrides keys of the inputs it names:
    the `params_`, `deg_`, `vst_` (its QC report) and `enr_` prefixes set one key of the matching
    metadata; `de_method` sets the method both in the params and in the DEG dataset's metadata."""
    params, deg_meta, vst_meta = dict(params or {}), dict(deg_meta), {**vst_meta}
    if de_method:
        params["de_method"] = deg_meta["de_method"] = de_method
    vst_meta["qc_report"] = dict(vst_meta.get("qc_report") or {})
    enr = [dict(e) for e in enr]
    for key, value in change.items():
        prefix, name = key.split("_", 1) if key.split("_", 1)[0] in {"params", "deg", "vst", "enr"} else (None, key)
        if prefix == "params":
            params[name] = value
        elif prefix == "deg":
            deg_meta[name] = value
        elif prefix == "vst":
            vst_meta["qc_report"][name] = value
        elif prefix == "enr":
            for meta in enr:
                meta[name] = value
    return rm.methods_context(params, deg_meta, vst_meta, enr)


def _text(ctx: dict) -> str:
    return svc.render_methods(ctx)


def _drop(meta: dict, *keys) -> dict:
    return {k: v for k, v in meta.items() if k not in keys}


# ── the context ──────────────────────────────────────────────────────────────────


def test_context_keys_and_values_of_a_recorded_run():
    ctx = _ctx()
    for key in (
        "imported", "de_method", "primary_method", "fdr", "min_log2fc", "min_count", "min_reps",
        "min_reps_mode", "qc", "design_used", "enrichment", "versions",
    ):
        assert key in ctx
    assert ctx["imported"] is False
    assert (ctx["de_method"], ctx["primary_method"]) == ("all", "DESeq2")
    assert (ctx["fdr"], ctx["min_log2fc"]) == (0.05, 1.0)
    assert (ctx["min_count"], ctx["min_reps"], ctx["min_reps_mode"]) == (5, 3, "fixed")
    assert ctx["qc"] == {
        "min_reads": 10, "min_genes": 200, "removed_sample_ids": ["S_3"], "relaxed": False,
    }
    assert ctx["design_used"] == {"A_vs_B": "~ batch + condition"}
    assert ctx["versions"]["DESeq2"] == "1.44.0"
    enr = ctx["enrichment"]
    assert (enr["fdr"], enr["categories"], enr["background"]) == (0.05, ["GO_BP", "kegg_pathway"], BACKGROUND)
    assert enr["annodb_file"] == "anno.db.human.gene_symbol.20260324.rds"


def test_a_missing_record_is_none_never_a_default():
    ctx = rm.methods_context({}, {"analysis_id": "an-1"}, {}, [])
    assert ctx["imported"] is False
    for key in ("de_method", "primary_method", "fdr", "min_log2fc", "min_count", "min_reps",
                "min_reps_mode", "versions"):
        assert ctx[key] is None, key
    assert ctx["design_used"] == {}
    assert ctx["qc"] == {"min_reads": None, "min_genes": None, "removed_sample_ids": None, "relaxed": None}


def test_a_run_with_one_method_names_it_as_the_deciding_one():
    ctx = _ctx(de_method="edger", deg_primary_method=None)
    assert (ctx["de_method"], ctx["primary_method"]) == ("edger", "edgeR")


def test_the_enrichment_cutoff_is_the_recorded_one_not_the_params():
    # Before the script was given `--padj-cutoff` it always ran at 0.05, whatever the params said.
    assert _ctx(params_enrichment_fdr=0.01, enr_padj_cutoff=0.05)["enrichment"]["fdr"] == 0.05
    assert _ctx(params_enrichment_fdr=0.01, enr_padj_cutoff=0.01)["enrichment"]["fdr"] == 0.01
    # No record: the params only count when they ask for 0.05 (what every run did before).
    old = rm.methods_context(
        {**PARAMS, "enrichment_fdr": 0.01}, DEG_META, VST_META, [_drop(ENR_META, "padj_cutoff")]
    )
    assert old["enrichment"]["fdr"] is None
    old = rm.methods_context(PARAMS, DEG_META, VST_META, [_drop(ENR_META, "padj_cutoff")])
    assert old["enrichment"]["fdr"] == 0.05


# ── Material and Methods: only the methods that were run ─────────────────────────


def test_methods_name_only_the_methods_run():
    only = _text(_ctx(de_method="deseq2"))
    assert "DESeq2" in only
    assert "edgeR" not in only and "limma" not in only
    only_edger = _text(_ctx(de_method="edger", deg_primary_method="edgeR"))
    assert "edgeR" in only_edger and "DESeq2" not in only_edger and "limma" not in only_edger
    everything = _text(_ctx())
    assert all(name in everything for name in ("DESeq2", "edgeR", "limma"))


def test_imported_results_say_methods_not_described():
    ctx = rm.methods_context(None, {}, {}, [])
    assert ctx["imported"] is True
    text = _text(ctx)
    assert NOT_RUN in text
    for word in ("DESeq2", "edgeR", "limma", "Benjamini", "Wald", "filterByExpr"):
        assert word not in text


def test_a_run_whose_dataset_has_no_parent_analysis_is_imported():
    # Even when its metadata carries a method or a threshold: the run is not GenoLens's.
    ctx = rm.methods_context(None, {"de_method": "all", "min_log2fc": 0.5}, {}, [])
    assert ctx["imported"] is True and ctx["de_method"] is None and ctx["min_log2fc"] is None


IMPORTED_ENRICHMENT = "Enrichment results of imported data are not included in this report."
GONE_ENRICHMENT = (
    "Enrichment results are not included in this report: the analysis that produced these "
    "results is no longer available."
)
NOT_RUN = "The differential analysis was not run in GenoLens; its methods are not described here"
GONE = "The analysis that produced these results is no longer available; its methods are not described here."


def _imported(analysis_id=None) -> dict:
    """The record of results with no parent analysis; `analysis_id` is what their DEG dataset's
    metadata still names (an analysis that no longer exists)."""
    return rm.methods_context(None, {"analysis_id": analysis_id} if analysis_id else {}, {}, [])


def test_imported_results_never_say_that_no_enrichment_is_stored():
    """The legacy Python enrichment writes its rows under the DEG dataset's own id, and a delivered
    ENRICHMENT dataset may exist: the report does not read them, and does not deny them either."""
    for ctx, line in ((_imported(), IMPORTED_ENRICHMENT), (_imported("an-9"), GONE_ENRICHMENT)):
        for text in (_text(ctx), rl.build_comparison_section(_comp(methods=ctx))):
            assert "No enrichment result is stored" not in text
            assert line in text
    # A GenoLens analysis with no ENRICHMENT dataset is another matter: nothing is stored there.
    assert "No enrichment result is stored" in _text(_ctx(enr=()))


def test_results_whose_analysis_is_gone_say_so():
    gone = _imported("an-9")
    assert gone["imported"] is True and gone["analysis_gone"] is True
    for text in (_text(gone), rl.build_comparison_section(_comp(methods=gone))):
        assert GONE in text and NOT_RUN not in text
    # No analysis id in the metadata: the results were never produced by a GenoLens analysis.
    never = _imported()
    assert never["imported"] is True and never["analysis_gone"] is False
    for text in (_text(never), rl.build_comparison_section(_comp(methods=never))):
        assert NOT_RUN in text and GONE not in text
    # The enrichment line of an orphaned analysis does not call its data imported (R-F3), and says
    # what the DE paragraph says; the line of results that were never GenoLens's is unchanged.
    for text in (_text(gone), rl.build_comparison_section(_comp(methods=gone))):
        assert "imported data" not in text and GONE_ENRICHMENT in text
    for text in (_text(never), rl.build_comparison_section(_comp(methods=never))):
        assert IMPORTED_ENRICHMENT in text and GONE_ENRICHMENT not in text
    # A GenoLens analysis is neither.
    assert _ctx()["analysis_gone"] is False
    # Neither names a method.
    for ctx in (gone, never):
        assert all(w not in _text(ctx) for w in ("DESeq2", "edgeR", "limma", "Benjamini"))


def test_thresholds_come_from_params():
    text = _text(_ctx(params_fdr=0.01, params_min_log2fc=1.0, deg_min_log2fc=1.0))
    assert re.search(r"below 0\.01\b", text) and re.search(r"at or above 1\b", text)
    assert "log$_2$(1.5)" not in text and "1.5" not in text
    low = _text(_ctx(params_min_log2fc=0.58, deg_min_log2fc=0.58))
    assert re.search(r"at or above 0\.58\b", low)
    unknown = _text(rm.methods_context({}, {"analysis_id": "an-1", "primary_method": "DESeq2"}, {}, []))
    assert "not recorded" in unknown and "at or above" not in unknown


def test_filter_rule_is_the_real_one():
    text = _text(_ctx(vst_min_count_threshold=5, vst_min_reps_threshold=3, vst_min_reps_mode="fixed"))
    assert "raw counts" in text
    assert re.search(r"reached 5 in at least 3 samples", text)
    assert "filterByExpr" not in text and "10 by default" not in text
    other = _text(_ctx(vst_min_count_threshold=7, vst_min_reps_threshold=2))
    assert "reached 7 in at least 2 samples" in other
    # `auto` is the smallest compared group, said as such.
    auto = _text(_ctx(vst_min_reps_mode="auto", vst_min_reps_threshold=4))
    assert "at least 4 samples" in auto and "smallest compared group" in auto
    assert "filterByExpr" not in auto
    # No record, no number.
    none = _text(rm.methods_context({}, {"analysis_id": "an-1"}, {}, []))
    assert "reached" not in none and "not recorded" in none


def test_no_robust_and_no_fixed_database_list():
    text = _text(_ctx(enr_categories=None, params_enrichment_databases=None))
    assert "robust" not in text.lower()
    for name in ("KEGG", "Reactome", "WikiPathways", "MSigDB", "Hallmark", "Gene Ontology"):
        assert name not in text
    assert "all categories of the annotation database" in text
    picked = _text(_ctx())
    assert r"GO\_BP" in picked and r"kegg\_pathway" in picked
    for name in ("Reactome", "WikiPathways", "MSigDB"):
        assert name not in picked


def test_unrecorded_categories_are_said_unrecorded():
    """Imported results carry no analysis params: that R ran every category is not known."""
    ctx = rm.methods_context(None, {}, {}, [_drop(ENR_META, "categories")])
    assert ctx["enrichment"]["categories_known"] is False
    text = _text(ctx)
    assert "were not recorded" in text
    assert "all categories" not in text


def test_no_pick_means_all_categories():
    """The worker passes `--databases` only for a non-empty pick: with the params of the analysis
    and no pick, R ran all the categories, even when the result did not record them."""
    ctx = rm.methods_context({}, {}, {}, [_drop(ENR_META, "categories")])
    assert ctx["enrichment"]["categories_known"] is True
    text = _text(ctx)
    assert "all categories" in text
    assert "were not recorded" not in text


def test_recorded_categories_are_listed_whatever_the_params():
    for params in (None, {}, PARAMS):
        ctx = rm.methods_context(params, {}, {}, [ENR_META])
        assert ctx["enrichment"]["categories_known"] is True
        text = _text(ctx)
        assert r"The categories tested were GO\_BP, kegg\_pathway." in text
        assert "all categories" not in text and "were not recorded" not in text


def test_enrichment_is_described_as_it_was_run():
    text = _text(_ctx())
    assert "hypergeometric" in text and "Benjamini" in text
    assert r"anno.db.human.gene\_symbol.20260324.rds" in text
    assert "adjusted \\textit{p}-value below 0.05" in text or "below 0.05" in text
    assert BACKGROUND in text
    # The older result records neither the file nor the background: it says so.
    old = _text(_ctx(enr_annodb_file=None, enr_background=None))
    assert "not recorded" in old and BACKGROUND not in old


def test_a_stored_enrichment_is_not_said_to_be_hypergeometric():
    upload = _text(_ctx(enr_source="upload"))
    assert "hypergeometric" not in upload and "Benjamini" not in upload.split("Functional Enrichment")[1]
    assert "stored with the dataset" in upload and "not described here" in upload
    none = _text(_ctx(enr=()))
    assert "hypergeometric" not in none and "No enrichment result is stored" in none


def test_excluded_samples_are_named():
    text = _text(_ctx(vst_removed_sample_ids=["S_3", "S4"]))
    assert r"S\_3" in text and "S4" in text and "excluded" in text
    assert "No sample was excluded" in _text(_ctx(vst_removed_sample_ids=[]))
    unknown = _text(rm.methods_context(PARAMS, DEG_META, {}, []))
    assert "samples excluded are not recorded" in unknown


def test_relaxed_qc_is_stated():
    relaxed = _text(_ctx(deg_qc_relaxed=True))
    assert "relaxed" in relaxed and "two samples" in relaxed
    assert "relaxed" not in _text(_ctx(deg_qc_relaxed=False))
    assert "relaxed" not in _text(_ctx(deg_qc_relaxed=None))


def test_versions_printed_when_recorded_else_not_recorded():
    text = _text(_ctx())
    assert "R 4.4.1" in text and "DESeq2 1.44.0" in text and "edgeR 4.2.1" in text and "limma 3.60.4" in text
    only = _text(_ctx(de_method="deseq2"))
    assert "DESeq2 1.44.0" in only and "edgeR 4.2.1" not in only and "limma 3.60.4" not in only
    old = _text(_ctx(deg_software_versions=None))
    assert "versions of R and of the packages used are not recorded" in old


def test_stouffer_is_not_said_to_be_reported():
    text = _text(_ctx())
    assert "Stouffer" not in text and "combined p-value" not in text
    assert "number of methods in which" not in text
    section = rl.build_comparison_section(_comp())
    assert "Stouffer" not in section and "how many methods support" not in section


def test_the_design_used_is_printed_per_comparison():
    text = _text(_ctx())
    assert r"\textasciitilde{} batch + condition" in text
    assert "design used is not recorded" in _text(_ctx(deg_design_used=None))
    two = rm.methods_context(
        PARAMS,
        {**DEG_META, "design_used": {"A_vs_B": "~ condition", "A_vs_C": "~ batch + condition"}},
        VST_META,
        [],
    )
    text = _text(two)
    assert "A\\_vs\\_B" in text and "A\\_vs\\_C" in text


def test_merging_the_metadata_of_several_comparisons():
    metas = [
        {**DEG_META, "comparison_name": "A_vs_B", "design_used": "~ condition"},
        {**DEG_META, "comparison_name": "A_vs_C", "design_used": "~ batch + condition", "qc_relaxed": True},
    ]
    merged = rm.merge_deg_metas(metas)
    assert merged["design_used"] == {"A_vs_B": "~ condition", "A_vs_C": "~ batch + condition"}
    assert merged["qc_relaxed"] is True
    assert merged["primary_method"] == "DESeq2" and merged["min_log2fc"] == 1.0
    assert merged["software_versions"] == DEG_META["software_versions"]
    differing = rm.merge_deg_metas([DEG_META, {**DEG_META, "primary_method": "edgeR"}])
    assert "primary_method" not in differing and differing["de_method"] == "all"


# ── the report body follows the same record ──────────────────────────────────────


def _comp(**extra) -> dict:
    return {
        "comp_id": "A_vs_B",
        "safe_id": "A-vs-B",
        "test": "A",
        "control": "B",
        "n_up": 1,
        "n_down": 1,
        "degs": [],
        "enr_up": [],
        "enr_down": [],
        "methods": _ctx(),
        **extra,
    }


def test_zero_deg_and_no_enrichment_states():
    section = rl.build_comparison_section(_comp(n_up=0, n_down=0))
    assert "0 genes passed" in section
    assert "No term passed (adjusted p $<$ 0.05)" in section
    assert "No differentially expressed genes were detected" not in section
    # The cut-off printed is the recorded one.
    low = rl.build_comparison_section(
        _comp(n_up=0, n_down=0, methods=_ctx(enr_padj_cutoff=0.01, params_enrichment_fdr=0.01))
    )
    assert "No term passed (adjusted p $<$ 0.01)" in low
    # Some DEGs and no term: the enrichment says no term passed, the overview does not say 0 genes.
    some = rl.build_comparison_section(_comp())
    assert "0 genes passed" not in some and "No term passed (adjusted p $<$ 0.05)" in some
    # An imported result has no cut-off to print, and no term it can say did not pass.
    imported = rl.build_comparison_section(_comp(methods=rm.methods_context(None, {}, {}, [])))
    assert IMPORTED_ENRICHMENT in imported and "No term passed" not in imported
    assert "adjusted p $<$" not in imported
    assert "The differential analysis was not run in GenoLens" in imported


def test_the_overview_follows_the_run():
    section = rl.build_comparison_section(_comp())
    assert "DESeq2" in section and "edgeR and limma-voom were also run" in section
    assert "do not decide the calls" in section
    only = rl.build_comparison_section(_comp(methods=_ctx(de_method="deseq2")))
    assert "edgeR" not in only and "limma" not in only
    unknown = rl.build_comparison_section(
        _comp(methods=rm.methods_context({}, {"analysis_id": "an-1"}, {}, []))
    )
    assert "not recorded" in unknown and "DESeq2" not in unknown


def test_a_legacy_enrichment_intro_names_no_test():
    comp = _comp(
        methods=_ctx(enr_source="upload"),
        enr_up=[{"pathway_name": "t", "category": "GO", "pvalue": None, "padj": 1e-3, "gene_count": 4}],
        dotplot_up_fig="enr_up.pdf",
    )
    section = rl.build_comparison_section(comp)
    assert "hypergeometric" not in section and "Benjamini" not in section.split("Functional Enrichment")[1]


def test_dotplot_caption_matches_the_plot(monkeypatch):
    comp = _comp(
        enr_up=[{"pathway_name": "t", "category": "GO", "pvalue": 1e-4, "padj": 1e-3, "gene_count": 4}],
        dotplot_up_fig="enr_up.pdf",
    )
    section = rl.build_comparison_section(comp)
    assert "number of genes" in section and "gene ratio" not in section.lower()
    assert "adjusted" in section.split("Dot size")[1]
    box = {}
    monkeypatch.setattr(rl, "_save_fig", lambda fig, path: box.setdefault("fig", fig))
    rows = [
        {"pathway_name": "small", "padj": 1e-2, "gene_count": 4},
        {"pathway_name": "medium", "padj": 1e-3, "gene_count": 8},
        {"pathway_name": "large", "padj": 1e-4, "gene_count": 16},
    ]
    assert rl._plot_enrichment_dotplot(rows, "x.pdf")
    fig = box["fig"]
    scatter = fig.axes[0].collections[0]
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    sizes = dict(zip(labels, scatter.get_sizes()))
    assert sizes["small"] < sizes["medium"] < sizes["large"]  # size = number of genes
    colours = dict(zip(labels, scatter.get_array()))
    assert colours["small"] < colours["large"]  # colour = -log10(adjusted p)
    assert fig.axes[1].get_ylabel() == "-log10(adjusted p-value)"
    plt.close(fig)


def test_dotplot_ranks_padj_zero_first(monkeypatch):
    box = {}
    monkeypatch.setattr(rl, "_save_fig", lambda fig, path: box.setdefault("fig", fig))
    rows = [
        {"pathway_name": "weak", "padj": 0.5, "gene_count": 5},
        {"pathway_name": "underflow", "padj": 0.0, "gene_count": 5},
        {"pathway_name": "strong", "padj": 1e-6, "gene_count": 5},
        {"pathway_name": "unknown", "padj": None, "gene_count": 5},
    ]
    assert rl._plot_enrichment_dotplot(rows, "x.pdf")
    labels = [t.get_text() for t in box["fig"].axes[0].get_yticklabels()]
    assert labels[:3] == ["underflow", "strong", "weak"] and labels[-1] == "unknown"
    plt.close(box["fig"])


def test_top_genes_table_lists_degs_only():
    tested = [
        {"gene_name": "NOT_A_DEG", "log_fc": 0.1, "pvalue": 1e-9, "padj": 1e-9, "regulation": None},
        {"gene_name": "BELOW_LFC", "log_fc": 0.2, "pvalue": 1e-8, "padj": 1e-8, "regulation": ""},
        {"gene_name": "UPGENE", "log_fc": 2.5, "pvalue": 1e-6, "padj": 1e-4, "regulation": "UP"},
        {"gene_name": "DOWNGENE", "log_fc": -2.0, "pvalue": 1e-5, "padj": 1e-3, "regulation": "down"},
    ]
    table = rl.generate_deg_table(tested)
    assert "UPGENE" in table and "DOWNGENE" in table
    assert "NOT\\_A\\_DEG" not in table and "BELOW\\_LFC" not in table
    assert table.index("UPGENE") < table.index("DOWNGENE")  # by adjusted p
    assert "No differentially expressed genes" in rl.generate_deg_table(tested[:2])


def test_qc_table_keeps_the_whole_sample_name():
    name = "a_very_long_sample_name_that_is_not_unique_A"
    other = "a_very_long_sample_name_that_is_not_unique_B"
    table = rl.generate_qc_table(
        [
            {"sample": name, "library_size": 10, "detected_genes": 5, "condition": "x"},
            {"sample": other, "library_size": 10, "detected_genes": 5, "condition": "x"},
        ],
        "P",
    )
    assert tex_name(name) in table and tex_name(other) in table


def tex_name(name: str) -> str:
    return rl.tex_escape(name)


# ── the volcano plot follows the thresholds of the run ───────────────────────────


def _volcano_genes():
    return [
        {"log_fc": 3.0, "padj": 1e-9, "regulation": "UP"},
        {"log_fc": -3.0, "padj": 1e-9, "regulation": "DOWN"},
        {"log_fc": 0.7, "padj": 1e-9, "regulation": "UP"},  # a DEG at |log2FC| >= 0.58
        {"log_fc": 0.1, "padj": 0.9, "regulation": None},
    ]


def test_volcano_thresholds_come_from_the_run(monkeypatch):
    box = {}
    monkeypatch.setattr(rl, "_save_fig", lambda fig, path: box.setdefault("fig", fig))
    assert rl._plot_volcano(_volcano_genes(), "x.pdf", fdr=0.01, min_log2fc=0.58)
    ax = box["fig"].axes[0]
    vertical = sorted(round(float(line.get_xdata()[0]), 2) for line in ax.lines if len(set(line.get_xdata())) == 1)
    horizontal = [float(line.get_ydata()[0]) for line in ax.lines if len(set(line.get_ydata())) == 1]
    assert vertical == [-0.58, 0.58]
    assert horizontal == pytest.approx([2.0])  # -log10(0.01)
    # The colours are the calls the dataset records: 1 NS, 2 up, 1 down.
    assert [len(c.get_offsets()) for c in ax.collections] == [1, 2, 1]
    plt.close(box["fig"])


def test_volcano_draws_no_threshold_it_does_not_know(monkeypatch):
    box = {}
    monkeypatch.setattr(rl, "_save_fig", lambda fig, path: box.setdefault("fig", fig))
    assert rl._plot_volcano(_volcano_genes(), "x.pdf")
    assert len(box["fig"].axes[0].lines) == 0
    plt.close(box["fig"])


async def _built(params=None, deg_meta=None, vst_meta=None, enrichment=(), tmp="/tmp", **kwargs):
    deg_ds = SimpleNamespace(id=uuid4(), dataset_metadata=deg_meta or {}, name="A_vs_B")
    gene_rows = MagicMock()
    gene_rows.scalars.return_value.all.return_value = []
    db = MagicMock()
    db.execute = AsyncMock(return_value=gene_rows)
    db.scalar = AsyncMock(return_value=None)
    enr = [
        Dataset(
            id=uuid4(),
            project_id=uuid4(),
            name="A_vs_B - enrichment",
            type=DatasetType.ENRICHMENT,
            status=DatasetStatus.READY,
            dataset_metadata=meta,
        )
        for meta in enrichment
    ]
    return await svc._build_comparison(
        db, deg_ds, enr, {}, tmp, False, comparison_name="A_vs_B", include_ai=False,
        params=params, vst_meta=vst_meta, **kwargs,
    )


async def test_the_volcano_is_drawn_with_the_params_of_the_analysis(tmp_path):
    seen = {}

    def fake(genes, out_path, fdr=None, min_log2fc=None):
        seen.update(fdr=fdr, min_log2fc=min_log2fc)
        return True

    with patch.object(rl, "_plot_volcano", fake):
        comp = await _built(
            params={**PARAMS, "fdr": 0.01, "min_log2fc": 0.58},
            deg_meta={**DEG_META, "min_log2fc": 0.58},
            vst_meta=VST_META,
            tmp=str(tmp_path),
        )
    assert seen == {"fdr": 0.01, "min_log2fc": 0.58}
    assert comp["volcano_fig"].startswith("volcano_")
    # An imported result is drawn with no threshold.
    seen.clear()
    with patch.object(rl, "_plot_volcano", fake):
        await _built(params=None, deg_meta={"min_log2fc": 2.0}, tmp=str(tmp_path))
    assert seen == {"fdr": None, "min_log2fc": None}


async def test_a_comparison_carries_its_methods_and_the_demo_mark():
    comp = await _built(
        params=PARAMS, deg_meta={**DEG_META, "demo": True}, vst_meta=VST_META, enrichment=[ENR_META],
        make_figures=False,
    )
    assert comp["methods"]["imported"] is False
    assert comp["methods"]["enrichment"]["fdr"] == 0.05
    assert comp["methods"]["design_used"] == {"A_vs_B": "~ batch + condition"}
    assert comp["demo"] is True
    imported = await _built(params=None, deg_meta={}, make_figures=False)
    assert imported["methods"]["imported"] is True and imported["demo"] is False
    assert imported["methods"]["analysis_gone"] is False
    # The metadata names an analysis that is no longer a parent: its results are not imported ones.
    gone = await _built(params=None, deg_meta={"analysis_id": "an-9"}, make_figures=False)
    assert gone["methods"]["imported"] is True and gone["methods"]["analysis_gone"] is True
    # The enrichment dataset's own mark counts too.
    marked = await _built(
        params=PARAMS, deg_meta=DEG_META, vst_meta=VST_META,
        enrichment=[{**ENR_META, "demo": True}], make_figures=False,
    )
    assert marked["demo"] is True


# ── the assembled report ─────────────────────────────────────────────────────────


def _report_data(**extra) -> dict:
    return {
        "project": SimpleNamespace(name="Proj"),
        "analysis": SimpleNamespace(id=uuid4(), name="An"),
        "params": PARAMS,
        "comparison_name": "A_vs_B",
        "dataset_id": uuid4(),
        "comparisons": [_comp(degs=[{"gene_name": "G1", "log_fc": 2.0, "pvalue": 1e-5, "padj": 1e-4,
                                      "regulation": "UP"}])],
        "qc_rows": [{"sample": "s1", "library_size": 1234567, "detected_genes": 12000, "condition": "A"}],
        "pca_fig": None,
        "methods": _ctx(),
        **extra,
    }


def test_the_assembled_report_prints_the_methods_of_the_run():
    tex = svc.assemble_tex(_report_data())
    assert "DESeq2 1.44.0" in tex and r"anno.db.human.gene\_symbol.20260324.rds" in tex
    assert "JINJA" not in tex.split(r"\begin{document}")[1]
    # A report without a record of its own is described as imported.
    bare = _report_data()
    del bare["methods"]
    assert "The differential analysis was not run in GenoLens" in svc.assemble_tex(bare)


def test_a_materials_methods_text_written_by_the_user_still_replaces_the_template():
    tex = svc.assemble_tex(_report_data(materials_methods="Our own methods."))
    assert "Our own methods." in tex and "hypergeometric" not in tex


def test_demo_dataset_latex_report_is_watermarked():
    # The mark follows the dataset, not only the account that asks for the report.
    marked = svc.assemble_tex(_report_data(demo=True, demo_watermark=False))
    assert "AddToShipoutPictureFG" in marked
    assert "AddToShipoutPictureFG" not in svc.assemble_tex(_report_data(demo=False, demo_watermark=False))
    # A demo account's request stays marked whatever the dataset says.
    assert "AddToShipoutPictureFG" in svc.assemble_tex(_report_data(demo=False, demo_watermark=True))


async def test_the_comparison_report_is_marked_when_its_dataset_is_demo():
    comp = _comp(demo=True)
    data = {**_report_data(), "comparisons": [comp], "demo": True}
    seen = {}

    async def collect(*a, **k):
        return data

    def assemble(d):
        seen["mark"] = d.get("demo_watermark")
        return "tex"

    with (
        patch.object(svc, "collect_comparison_data", collect),
        patch.object(svc, "_load_branding", AsyncMock(return_value=None)),
        patch.object(svc, "assemble_tex", assemble),
        patch.object(svc, "compile_pdf", MagicMock(return_value=b"%PDF")),
    ):
        await svc.render_comparison_pdf(MagicMock(), uuid4(), "A_vs_B", demo_watermark=False)
    assert seen["mark"] is True


# ── the conditions of the samples follow the wizard's choice ─────────────────────


def _samples(**cols) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], **cols})


def test_sample_conditions_follow_the_condition_column():
    df = _samples(condition=["x", "x", "y", "y"], Genotype=["KO", "WT", "KO", "WT"])
    assert rl.pick_sample_conditions(df, "genotype") == {"s1": "KO", "s2": "WT", "s3": "KO", "s4": "WT"}
    # No column asked: the aliases the R pipeline detects, in its order.
    assert rl.pick_sample_conditions(df) == {"s1": "x", "s2": "x", "s3": "y", "s4": "y"}
    french = pd.DataFrame({"name": ["s1", "s2"], "groupe": ["a", "b"]})
    assert rl.pick_sample_conditions(french) == {"s1": "a", "s2": "b"}
    # A column that is asked for and absent: nothing, never another column.
    assert rl.pick_sample_conditions(df, "nope") == {}
    # The sample-id column is not a condition column.
    assert rl.pick_sample_conditions(df, "sample_id") == {}


async def test_the_sample_conditions_of_the_report_use_the_analysis_column():
    df = _samples(condition=["x", "x", "y", "y"], genotype=["KO", "WT", "KO", "WT"])
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    db = MagicMock()
    db.get = AsyncMock(return_value=SimpleNamespace(parquet_file_path="samples.parquet"))
    with patch("app.services.storage.storage_service.download_file", AsyncMock(return_value=buffer.getvalue())):
        asked = await svc._sample_conditions(db, uuid4(), "genotype")
        default = await svc._sample_conditions(db, uuid4())
    assert asked["s1"] == "KO" and default["s1"] == "x"


async def test_the_qc_table_reads_the_conditions_with_the_analysis_column():
    matrix = pd.DataFrame({"gene_id": ["g1", "g2"], "s1": [5, 0], "s2": [1, 1]})
    buffer = io.BytesIO()
    matrix.to_parquet(buffer)
    analysis = SimpleNamespace(
        matrix_dataset_id=uuid4(), samples_dataset_id=uuid4(), params={"condition_column": "genotype"}
    )
    db = MagicMock()
    db.get = AsyncMock(return_value=SimpleNamespace(parquet_file_path="m.parquet"))
    conditions = AsyncMock(return_value={"s1": "KO", "s2": "WT"})
    with (
        patch("app.services.storage.storage_service.download_file", AsyncMock(return_value=buffer.getvalue())),
        patch.object(svc, "_sample_conditions", conditions),
    ):
        rows = await svc._make_qc(db, analysis)
    conditions.assert_awaited_once()
    assert conditions.await_args.args[-1] == "genotype" or conditions.await_args.kwargs.get("condition_column") == "genotype"
    assert [r["condition"] for r in rows] == ["KO", "WT"]


# ── compilation: a failed run is a failed report ─────────────────────────────────

MINIMAL = "\\documentclass{article}\n\\begin{document}\n%s\n\\end{document}\n"


def test_undefined_references_are_read_from_the_log():
    log = (
        "LaTeX Warning: Reference `fig:missing' on page 2 undefined on input line 9.\n"
        "LaTeX Warning: Citation `smith' on page 3 undefined on input line 12.\n"
        "LaTeX Warning: There were undefined references.\n"
    )
    assert rl.undefined_references(log) == ["fig:missing", "smith"]
    assert rl.undefined_references("LaTeX Warning: There were undefined references.\n") == ["?"]
    assert rl.undefined_references("all fine\n") == []


def test_compile_fails_on_a_nonzero_return_code_even_when_a_pdf_is_there(tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"%PDF-stale")
    failed = MagicMock(returncode=1, stdout="! Undefined control sequence.", stderr="")
    with patch.object(rl.subprocess, "run", return_value=failed):
        with pytest.raises(rl.LatexCompileError, match="could not be compiled"):
            svc.compile_pdf(MINIMAL % "text", str(tmp_path))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_compile_fails_on_undefined_references(tmp_path):
    with pytest.raises(RuntimeError, match="undefined reference"):
        svc.compile_pdf(MINIMAL % r"See \ref{nowhere}.", str(tmp_path))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_compile_accepts_references_the_second_pass_resolves(tmp_path):
    pdf = svc.compile_pdf(MINIMAL % r"\section{A}\label{a} See \ref{a}.", str(tmp_path))
    assert pdf.startswith(b"%PDF")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
@pytest.mark.parametrize(
    "variant",
    ["recorded", "imported", "gone", "upload", "unrecorded"],
)
def test_the_report_of_each_kind_of_run_compiles(tmp_path, variant):
    methods = {
        "recorded": _ctx(deg_qc_relaxed=True, vst_removed_sample_ids=["S_3", "S4"]),
        "imported": rm.methods_context(None, {}, {}, []),
        "gone": rm.methods_context(None, {"analysis_id": "an-9"}, {}, []),
        "upload": _ctx(enr_source="upload"),
        "unrecorded": rm.methods_context({}, {"analysis_id": "an-1"}, {}, []),
    }[variant]
    data = _report_data(methods=methods)
    data["comparisons"][0]["methods"] = methods
    pdf = svc.compile_pdf(svc.assemble_tex(data), str(tmp_path))
    assert pdf.startswith(b"%PDF")


# ── R-F2: no ENRICHMENT dataset is "no result stored", not "no term passed" ──────


def test_no_enrichment_dataset_says_no_result_is_stored_not_that_no_term_passed():
    """With no ENRICHMENT dataset registered the record cannot tell "ran, nothing passed" from
    "did not run" (the worker registers none when the script fails): the body says what the
    Material and Methods and the editorial say."""
    none = _comp(methods=_ctx(enr=()))
    for no_gene in (False, True):
        section = rl.build_comparison_section({**none, "n_up": 0 if no_gene else 1, "n_down": 0})
        assert "No enrichment result is stored for this comparison." in section
        assert "No term passed" not in section
        assert "adjusted p $<$" not in section and "cut-off of the run" not in section
    # The cut-off is no business of a comparison that has no result.
    zero = rl.build_comparison_section(_comp(n_up=0, n_down=0, methods=_ctx(enr=())))
    assert "No enrichment result is stored for this comparison." in zero
    assert "none was called" in zero  # the enrichment tests the DEGs: what explains it
    # An enrichment result that exists with no kept term keeps the sentence with its cut-off.
    kept = rl.build_comparison_section(_comp())
    assert "No term passed (adjusted p $<$ 0.05)" in kept
    assert "No enrichment result is stored" not in kept
    # No result, no run: no cut-off is a fact of this comparison either.
    assert _ctx(enr=())["enrichment"]["fdr"] is None
    # No database picked: nothing ran, and it says so.
    picked_none = _comp(methods=_ctx(enr=(), params_enrichment_databases=[]))
    assert "No annotation database was selected" in rl.build_comparison_section(picked_none)


# ── D1: a recorded threshold prints exactly ──────────────────────────────────────


@pytest.mark.parametrize(
    "value, printed",
    [
        (1000000, "1000000"),
        (1234567, "1234567"),
        (1000000.0, "1000000"),
        (5, "5"),
        (5.0, "5"),
        (0.05, "0.05"),
        (0.585, "0.585"),
        (1e-05, "1e-05"),
        (1234567.5, "1234567.5"),
    ],
)
def test_a_recorded_number_prints_exactly(value, printed):
    assert rl.fmt_g(value) == printed
    assert rm.format_number(value) == printed


def test_a_recorded_minimum_count_of_a_million_is_not_written_in_scientific_notation():
    text = _text(_ctx(vst_min_count_threshold=1000000, vst_min_reps_threshold=2))
    assert "reached 1000000 in at least 2 samples" in text
    assert "1e+06" not in text
    reads = _text(_ctx(vst_min_reads_threshold=1234567))
    assert "at least 1234567 total counts" in reads and "1.23457e+06" not in reads
    thresholds = rm.thresholds_tex({"fdr": 0.05, "min_log2fc": 1000000})
    assert "1000000" in thresholds and "e+06" not in thresholds


# ── D2: comparisons that disagree on the deciding method ─────────────────────────


def _de_section(text: str) -> str:
    return text.split("Differential Gene Expression Analysis")[1].split("Functional Enrichment")[0]


def _merged_ctx(*metas):
    merged = rm.merge_deg_metas(list(metas))
    return rm.methods_context(PARAMS, merged, VST_META, [ENR_META])


def test_comparisons_that_disagree_on_the_method_are_named_one_by_one():
    ctx = _merged_ctx(
        {**DEG_META, "comparison_name": "A_vs_B", "primary_method": "DESeq2"},
        {**DEG_META, "comparison_name": "A_vs_C", "primary_method": "edgeR"},
    )
    assert ctx["primary_method"] is None
    assert ctx["primary_by_comparison"] == [("A_vs_B", "DESeq2"), ("A_vs_C", "edgeR")]
    section = _de_section(_text(ctx))
    assert "differs between the comparisons" in section
    assert "A\\_vs\\_B, DESeq2" in section and "A\\_vs\\_C, edgeR" in section
    assert "is not recorded for this analysis" not in section
    # The BH sentence and the recorded thresholds stay: every method of the pipeline adjusts so.
    assert "Benjamini-Hochberg (BH) False Discovery Rate" in section
    assert re.search(r"adjusted \\textit\{p\}-value was below 0\.05", section)
    assert re.search(r"fold-change was at or above 1\b", section)
    assert "are not recorded for this analysis" not in section


def test_a_method_that_only_some_comparisons_record_is_listed_for_those_alone():
    ctx = _merged_ctx(
        {**DEG_META, "comparison_name": "A_vs_B", "primary_method": "DESeq2"},
        _drop({**DEG_META, "comparison_name": "A_vs_C"}, "primary_method"),
    )
    assert ctx["primary_method"] is None and ctx["primary_by_comparison"] == [("A_vs_B", "DESeq2")]
    section = _de_section(_text(ctx))
    assert "not recorded for every comparison" in section
    sentence = section.split("not recorded for every comparison")[1].split("Benjamini")[0]
    assert "A\\_vs\\_B, DESeq2" in sentence
    assert "A\\_vs\\_C" not in sentence and "edgeR" not in sentence
    assert "Benjamini-Hochberg (BH) False Discovery Rate" in section


def test_comparisons_that_agree_on_the_method_keep_the_single_sentence():
    ctx = _merged_ctx(
        {**DEG_META, "comparison_name": "A_vs_B"}, {**DEG_META, "comparison_name": "A_vs_C"}
    )
    assert ctx["primary_method"] == "DESeq2" and not ctx["primary_by_comparison"]
    assert "differs between the comparisons" not in _text(ctx)
    # No comparison records one: the paragraph says so and keeps nothing it cannot back.
    none = _merged_ctx(*(_drop({**DEG_META, "comparison_name": n}, "primary_method") for n in "AB"))
    section = _de_section(_text(none))
    assert none["primary_by_comparison"] == []
    assert "calls is not recorded for this analysis" in section
    assert "Benjamini-Hochberg (BH)" not in section
