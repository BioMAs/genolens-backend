"""
The wizard's "Enrichment FDR threshold" reaches the R enrichment step.

It used to stop at the browser: the field was never sent, `AnalysisParams` had no slot for it, and
the Rscript call never passed `--padj-cutoff`, so every analysis kept its terms at the script's
own 0.05 whatever the user chose.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.endpoints.analyses import AnalysisParams
from app.worker.tasks import build_functional_enrichment_cmd


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
