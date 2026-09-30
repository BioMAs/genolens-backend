"""
The wizard's "Enrichment FDR threshold" reaches the R enrichment step.

It used to stop at the browser: the field was never sent, `AnalysisParams` had no slot for it, and
the Rscript call never passed `--padj-cutoff`, so every analysis kept its terms at the script's
own 0.05 whatever the user chose.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.endpoints.analyses import AnalysisParams
from app.worker.tasks import build_functional_enrichment_cmd, enrichment_requested


def _cmd(params: dict) -> list[str]:
    return build_functional_enrichment_cmd(
        Path("/app/r_scripts/functional_enrichment.R"),
        Path("/tmp/deg.csv"),
        "/app/anno_db",
        "A_vs_B",
        Path("/tmp/out.csv"),
        params,
    )


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


pytestmark = pytest.mark.unit


def test_enrichment_fdr_is_passed_as_the_term_cutoff():
    cmd = _cmd({"fdr": 0.05, "min_log2fc": 0.585, "enrichment_fdr": 0.01})
    assert _flag(cmd, "--padj-cutoff") == "0.01"


def test_deg_thresholds_stay_on_their_own_flags():
    # `fdr` selects the DEGs; it must not leak into the term cut-off, nor the reverse.
    cmd = _cmd({"fdr": 0.1, "min_log2fc": 0.585, "enrichment_fdr": 0.01})
    assert _flag(cmd, "--fdr") == "0.1"
    assert _flag(cmd, "--min-log2fc") == "0.585"


def test_analyses_created_before_the_field_keep_running_at_005():
    cmd = _cmd({"fdr": 0.05, "min_log2fc": 1.0})
    assert _flag(cmd, "--padj-cutoff") == "0.05"


def test_the_stored_params_carry_the_field():
    dumped = AnalysisParams(enrichment_fdr=0.02).model_dump()
    assert dumped["enrichment_fdr"] == 0.02
    assert _flag(_cmd(dumped), "--padj-cutoff") == "0.02"


def test_default_is_the_cutoff_the_script_always_used():
    assert AnalysisParams().enrichment_fdr == 0.05


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_out_of_range_cutoff_is_rejected(bad):
    with pytest.raises(ValidationError):
        AnalysisParams(enrichment_fdr=bad)


# ── the database pick ─────────────────────────────────────────────────────────────
# The wizard's picker lists raw anno.db category names. A partial pick used to be passed as
# `--enrichment-databases` to run_multimethod_pipeline.R, whose optparse rejects unknown flags,
# so the whole analysis failed; functional_enrichment.R, which does the enrichment, never saw it.


def test_database_pick_is_passed_to_the_enrichment_script():
    cmd = _cmd({"enrichment_databases": ["GSEA_hallmark", "kegg_pathway"]})
    assert _flag(cmd, "--databases") == "GSEA_hallmark,kegg_pathway"


@pytest.mark.parametrize("pick", [None, [], ["", "  "]])
def test_no_pick_sends_no_flag(pick):
    assert "--databases" not in _cmd({"enrichment_databases": pick})


def test_all_databases_and_a_pick_run_the_enrichment():
    assert enrichment_requested({}) is True
    assert enrichment_requested({"enrichment_databases": None}) is True
    assert enrichment_requested({"enrichment_databases": ["kegg_pathway"]}) is True


# "Clear" in the wizard reads "0 databases selected"; the empty list used to run them all.
def test_an_explicit_empty_pick_skips_the_enrichment():
    assert enrichment_requested({"enrichment_databases": []}) is False


# ── every flag the worker passes is one the R script defines ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
TASKS = (ROOT / "app" / "worker" / "tasks.py").read_text()


def _r_options(script: str) -> set[str]:
    source = (ROOT / "r_scripts" / script).read_text()
    return set(re.findall(r'make_option\(\s*"(--[a-z0-9-]+)"', source))


def _flags_between(start: str, end: str) -> set[str]:
    begin = TASKS.index(start)
    finish = TASKS.index(end, begin)
    block = TASKS[begin:finish]
    return set(re.findall(r'"(--[a-z0-9-]+)"', block))


def test_pipeline_flags_are_all_defined():
    passed = _flags_between('Path("/app/r_scripts/run_multimethod_pipeline.R")', "subprocess.run(")
    assert passed, "the pipeline command was not found"
    assert passed <= _r_options("run_multimethod_pipeline.R")
    assert "--enrichment-databases" not in passed


def test_enrichment_flags_are_all_defined():
    passed = set(a for a in _cmd({"enrichment_databases": ["x"]}) if a.startswith("--"))
    assert passed <= _r_options("functional_enrichment.R")


# ── the R side of the pick ────────────────────────────────────────────────────────


def _select_categories(available: list[str], databases: str | None) -> list[str]:
    """Evaluate `select_categories` alone: the script itself needs an anno.db to run."""
    script = ROOT / "r_scripts" / "functional_enrichment.R"
    r_vec = "c(" + ",".join(f'"{a}"' for a in available) + ")"
    r_db = "NULL" if databases is None else f'"{databases}"'
    code = (
        f'exprs <- parse("{script}"); '
        'fn <- Filter(function(e) is.call(e) && identical(e[[1]], as.name("<-")) && '
        'identical(e[[2]], as.name("select_categories")), exprs)[[1]]; '
        "eval(fn); "
        f'cat(select_categories({r_vec}, {r_db}), sep = "\\n")'
    )
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
ANNO = ["biological_process", "kegg_pathway", "GSEA_hallmark", "NCBI_gene_info", "xrefs"]


@needs_r
def test_r_runs_every_category_without_a_pick():
    assert _select_categories(ANNO, None) == ["biological_process", "kegg_pathway", "GSEA_hallmark"]


@needs_r
def test_r_keeps_only_the_picked_categories():
    assert _select_categories(ANNO, "GSEA_hallmark, kegg_pathway") == [
        "kegg_pathway",
        "GSEA_hallmark",
    ]


@needs_r
def test_r_ignores_names_the_species_lacks():
    assert _select_categories(ANNO, "kegg_pathway,GSEA_c8.CellType") == ["kegg_pathway"]
