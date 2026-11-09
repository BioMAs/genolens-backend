"""
How a GenoLens analysis decides which genes are DEGs, on the Python side.

- `--min-reps auto` (the wizard's default) reaches the R pipeline: the low-count filter then
  requires as many samples as the smallest compared group, not a fixed 2.
- The R pipeline records, per comparison, the method whose padj decided (`primary_method`,
  DESeq2 with `--method all`). The DEG dataset carries it so ingestion makes that column the
  active padj, the one the volcano, the counts and the DEG table read.
"""

import pytest
from pydantic import ValidationError

from app.api.endpoints.analyses import AnalysisParams
from app.worker.tasks import _build_pipeline_command, deg_dataset_metadata

pytestmark = pytest.mark.unit


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def _cmd(params: dict) -> list[str]:
    return _build_pipeline_command("run.R", "m.tsv", "s.tsv", "c.tsv", "/out", params)


class TestMinReps:
    def test_auto_is_accepted_and_reaches_the_pipeline(self):
        params = AnalysisParams(min_reps="auto").model_dump()
        assert params["min_reps"] == "auto"
        assert _flag(_cmd(params), "--min-reps") == "auto"

    def test_whole_number_still_works(self):
        params = AnalysisParams(min_reps=3).model_dump()
        assert _flag(_cmd(params), "--min-reps") == "3"

    def test_default_stays_two_for_api_callers(self):
        assert AnalysisParams().min_reps == 2

    @pytest.mark.parametrize("bad", [0, -1, "two", "", 2.5])
    def test_invalid_values_are_rejected(self, bad):
        with pytest.raises(ValidationError):
            AnalysisParams(min_reps=bad)


class TestDegDatasetMetadata:
    manifest = {
        "comparison_stats": {
            "KO_vs_WT": {"n_up": 3, "n_down": 1, "primary_method": "DESeq2"},
            "Old_vs_WT": {"n_up": 3, "n_down": 1},
        }
    }

    def test_carries_the_deciding_method(self):
        meta = deg_dataset_metadata("an-1", "KO_vs_WT", {"min_log2fc": 0.58, "de_method": "all"},
                                    self.manifest)
        assert meta == {
            "analysis_id": "an-1",
            "comparison_name": "KO_vs_WT",
            "min_log2fc": 0.58,
            "de_method": "all",
            "primary_method": "DESeq2",
        }

    def test_no_record_no_key(self):
        """A manifest from an older pipeline has no primary_method: ingestion keeps its priority."""
        meta = deg_dataset_metadata("an-1", "Old_vs_WT", {}, self.manifest)
        assert "primary_method" not in meta
        assert meta["min_log2fc"] == 1.0 and meta["de_method"] == "all"

    def test_missing_manifest(self):
        assert "primary_method" not in deg_dataset_metadata("an-1", "KO_vs_WT", {}, {})

    def test_copies_the_provenance_the_run_recorded(self):
        """The LaTeX report prints what the run observed: its software versions, the design each
        comparison was fitted with, and whether the sample thresholds had to be relaxed."""
        manifest = {
            "software_versions": {"R": "4.4.1", "DESeq2": "1.44.0", "edgeR": "4.2.1", "limma": "3.60.4"},
            "qc_relaxed": True,
            "comparison_stats": {
                "KO_vs_WT": {"primary_method": "DESeq2", "design_used": "~ batch + condition"}
            },
        }
        meta = deg_dataset_metadata("an-1", "KO_vs_WT", {}, manifest)
        assert meta["software_versions"] == manifest["software_versions"]
        assert meta["design_used"] == "~ batch + condition"
        assert meta["qc_relaxed"] is True
        # Each comparison keeps its own design.
        other = {**manifest, "comparison_stats": {"B_vs_C": {"design_used": "~ condition"}}}
        assert deg_dataset_metadata("an-1", "B_vs_C", {}, other)["design_used"] == "~ condition"

    def test_a_manifest_without_provenance_gives_no_key(self):
        """An older run recorded none of it: the report says "not recorded", the metadata no value."""
        meta = deg_dataset_metadata("an-1", "Old_vs_WT", {}, self.manifest)
        for key in ("software_versions", "design_used", "qc_relaxed"):
            assert key not in meta
        # A malformed record is not copied either.
        bad = {"software_versions": "4.4.1", "qc_relaxed": "yes", "comparison_stats": {}}
        meta = deg_dataset_metadata("an-1", "KO_vs_WT", {}, bad)
        assert "software_versions" not in meta and "qc_relaxed" not in meta
