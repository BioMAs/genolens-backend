"""Building a Drug Discovery signature from a user's comparison.

This is where the risk of the feature actually lives. The HTTP layer is a passthrough and the
scoring belongs to genolens-dd; what happens here is the translation of *our* data model into
*their* contract, and every step of it has a plausible wrong answer:

- a replicate count that includes samples the QC dropped, which walks straight past the
  underpowered-signature gate;
- a condition name guessed with a separator list that shreds `Treated_24h_vs_Ctrl_24h`;
- a direction sent with zero genes, which upstream rejects as SIG004;
- a truncation applied silently, so 1 000 genes out of 5 000 read as "there were 1 000".

Each of those is a wrong answer that looks like a working feature, so each has a test.
"""
from unittest.mock import MagicMock
from uuid import UUID

import pandas as pd
import pytest

from app.services.drug_discovery_signature import (
    build_signature_payload,
    parse_condition_names,
)

DATASET_ID = UUID("00000000-0000-0000-0000-0000000000d1")
PROJECT_ID = UUID("00000000-0000-0000-0000-0000000000d2")
ANALYSIS_ID = UUID("00000000-0000-0000-0000-0000000000d3")
SAMPLES_ID = UUID("00000000-0000-0000-0000-0000000000d4")
CONTRASTS_ID = UUID("00000000-0000-0000-0000-0000000000d5")
VST_ID = UUID("00000000-0000-0000-0000-0000000000d6")


def _deg(gene_name, log_fc, padj, regulation, gene_id=None):
    row = MagicMock()
    row.gene_name = gene_name
    row.gene_id = gene_id or f"ENSG_{gene_name}"
    row.log_fc = log_fc
    row.padj = padj
    row.regulation = regulation
    return row


def _dataset(metadata=None):
    dataset = MagicMock()
    dataset.id = DATASET_ID
    dataset.project_id = PROJECT_ID
    dataset.dataset_metadata = metadata if metadata is not None else {}
    return dataset


class FakeDb:
    """Minimal async stand-in: `get` by id, `execute` returning preloaded scalars."""

    def __init__(self, objects=None, scalars=()):
        self._objects = objects or {}
        self._scalars = list(scalars)

    async def get(self, _model, identifier):
        return self._objects.get(identifier)

    async def execute(self, _stmt):
        result = MagicMock()
        scalars = MagicMock()
        scalars.all.return_value = self._scalars
        result.scalars.return_value = scalars
        return result


@pytest.fixture
def deg_rows(monkeypatch):
    """Patch the shared DEG query so these tests exercise the translation, not SQLAlchemy."""
    calls: list[dict] = []
    table: dict[str, list] = {"UP": [], "DOWN": []}

    async def fake_select(db, dataset_id, comparison_name, **kwargs):
        calls.append({"comparison_name": comparison_name, **kwargs})
        return table[kwargs["regulation"]]

    monkeypatch.setattr(
        "app.api.endpoints.datasets.select_deg_genes", fake_select, raising=True
    )
    return table, calls


@pytest.fixture
def no_frames(monkeypatch):
    """No samplesheet, no contrast table — the uploaded-DEG case."""
    async def none(_db, _dataset_id):
        return None

    monkeypatch.setattr(
        "app.services.drug_discovery_signature._load_dataset_frame", none, raising=True
    )


# ── Condition names ──────────────────────────────────────────────────────────


class TestConditionNames:
    def test_only_an_explicit_vs_is_split(self):
        assert parse_condition_names("Treated_vs_Control") == ("Treated", "Control")
        assert parse_condition_names("A-vs-B") == ("A", "B")

    def test_a_condition_name_containing_underscores_survives(self):
        """THE CASE THAT FORBIDS REUSING `_parse_comparison_name`.

        That helper's separator list includes bare `_`, `-` and `.`, so it would return
        ("Treated", "24h") here. A wrong condition name is not cosmetic: it is the key of the
        replicate dictionary, so the power gate would be applied to an arm that does not exist.
        """
        assert parse_condition_names("Treated_24h_vs_Ctrl_24h") == (
            "Treated_24h",
            "Ctrl_24h",
        )

    def test_a_name_that_says_nothing_returns_none(self):
        """`None` is a useful answer: the caller will ask the user rather than invent labels."""
        assert parse_condition_names("comparison_1") is None
        assert parse_condition_names("_vs_B") is None

    @pytest.mark.asyncio
    async def test_the_contrast_table_wins_over_the_name(self, deg_rows, monkeypatch):
        """The R pipeline used those names; a guess from the filename is a second-hand source."""
        table, _ = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]

        analysis = MagicMock()
        analysis.comparisons_dataset_id = CONTRASTS_ID
        analysis.samples_dataset_id = None
        analysis.intermediate_dataset_ids = {}
        analysis.params = {}

        frame = pd.DataFrame([{
            "comparison": "Treated_vs_Control",
            "condition1": "TRT",
            "condition2": "CTRL",
        }])

        async def load(_db, dataset_id):
            return frame if dataset_id == CONTRASTS_ID else None

        monkeypatch.setattr(
            "app.services.drug_discovery_signature._load_dataset_frame", load, raising=True
        )

        payload = await build_signature_payload(
            FakeDb({ANALYSIS_ID: analysis}),
            _dataset({"analysis_id": str(ANALYSIS_ID)}),
            "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
        )
        assert [c.name for c in payload.conditions] == ["TRT"]


# ── Direction mapping and truncation ─────────────────────────────────────────


class TestDirections:
    @pytest.mark.asyncio
    async def test_up_and_down_become_two_named_conditions(self, deg_rows, no_frames):
        """One pseudo-condition holding every DEG would force an `n1 + n2` count, which sails
        past the SIG001/SIG002 gate that exists to catch underpowered arms."""
        table, _ = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]
        table["DOWN"] = [_deg("TP53", -2.0, 0.001, "DOWN")]

        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0,
            replicates_override={"Treated": 4, "Control": 4},
        )
        assert [(c.name, c.direction) for c in payload.conditions] == [
            ("Treated", "UP"), ("Control", "DOWN")
        ]
        assert payload.genes_by_condition() == {"Treated": ["EGFR"], "Control": ["TP53"]}

    @pytest.mark.asyncio
    async def test_an_empty_direction_is_omitted_not_sent_empty(self, deg_rows, no_frames):
        """SIG004 refuses an empty condition, and rightly: an enrichment over the empty set
        returns meaningless p-values instead of an error. A one-sided comparison sends one arm."""
        table, _ = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]
        table["DOWN"] = []

        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, replicates_override={"Treated": 4},
        )
        assert len(payload.conditions) == 1
        assert "Control" not in payload.genes_by_condition()

    @pytest.mark.asyncio
    async def test_the_direction_filter_skips_the_query_entirely(self, deg_rows, no_frames):
        table, calls = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]

        await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
            replicates_override={"Treated": 4},
        )
        assert [c["regulation"] for c in calls] == ["UP"]

    @pytest.mark.asyncio
    async def test_truncation_is_capped_counted_and_announced(self, deg_rows, no_frames):
        """Never silent.

        `n_available` stays the real number, so "1 000 sent" cannot be read as "there were
        only 1 000" — the same failure as the silent cap upstream.
        """
        table, _ = deg_rows
        table["UP"] = [_deg(f"G{i}", 2.0, 0.001, "UP") for i in range(10)]

        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
            max_genes_per_condition=3, replicates_override={"Treated": 4},
        )
        condition = payload.conditions[0]
        assert len(condition.genes) == 3
        assert condition.n_available == 10
        assert condition.truncated is True
        assert any("10 genes match" in w for w in payload.warnings)

    @pytest.mark.asyncio
    async def test_duplicate_symbols_are_collapsed(self, deg_rows, no_frames):
        """Several Ensembl ids can carry one symbol; a duplicate would inflate the signature
        size, and size is what makes the mean-percentile statistic degenerate."""
        table, _ = deg_rows
        table["UP"] = [
            _deg("EGFR", 2.0, 0.001, "UP", gene_id="ENSG1"),
            _deg("EGFR", 2.1, 0.002, "UP", gene_id="ENSG2"),
        ]
        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
            replicates_override={"Treated": 4},
        )
        assert payload.conditions[0].genes == ("EGFR",)

    @pytest.mark.asyncio
    async def test_the_thresholds_reach_the_query(self, deg_rows, no_frames):
        table, calls = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]
        await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.01, logfc_min=1.5, directions="up",
            replicates_override={"Treated": 4},
        )
        assert calls[0]["padj_max"] == 0.01
        assert calls[0]["logfc_min"] == 1.5


# ── Replicates ───────────────────────────────────────────────────────────────


class TestReplicates:
    @pytest.mark.asyncio
    async def test_qc_removed_samples_are_not_counted(self, deg_rows, monkeypatch):
        """THE TEST THAT EARNS ITS KEEP.

        A sample dropped by the pipeline's QC was never in the differential model. Counting it
        can push a genuine n=2 arm past the SIG002 gate — the precise failure that gate exists
        to prevent, and one that would produce a confident p-value on an underpowered design.
        """
        table, _ = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]

        analysis = MagicMock()
        analysis.comparisons_dataset_id = None
        analysis.samples_dataset_id = SAMPLES_ID
        analysis.intermediate_dataset_ids = {"vst": str(VST_ID)}
        analysis.params = {}

        vst = MagicMock()
        vst.dataset_metadata = {"qc_report": {"removed_sample_ids": ["S3"]}}

        samples = pd.DataFrame(
            [
                {"sample_id": "S1", "condition": "Treated"},
                {"sample_id": "S2", "condition": "Treated"},
                {"sample_id": "S3", "condition": "Treated"},
                {"sample_id": "S4", "condition": "Control"},
                {"sample_id": "S5", "condition": "Control"},
                {"sample_id": "S6", "condition": "Control"},
            ]
        )

        async def load(_db, dataset_id):
            return samples if dataset_id == SAMPLES_ID else None

        monkeypatch.setattr(
            "app.services.drug_discovery_signature._load_dataset_frame", load, raising=True
        )

        payload = await build_signature_payload(
            FakeDb({ANALYSIS_ID: analysis, VST_ID: vst}),
            _dataset({"analysis_id": str(ANALYSIS_ID)}),
            "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
        )
        condition = payload.conditions[0]
        assert condition.replicates == 2, "S3 was QC-dropped, so the arm is n=2 not n=3"
        assert condition.replicates_source == "analysis_samplesheet"

    @pytest.mark.asyncio
    async def test_an_unresolvable_count_is_reported_not_guessed(self, deg_rows, no_frames):
        """SIG005: "assuming it is sufficient would be the most dangerous default of the lot"."""
        table, _ = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]

        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
        )
        assert payload.conditions[0].replicates is None
        assert payload.conditions[0].replicates_source == "unknown"
        assert payload.needs_replicates is True

    @pytest.mark.asyncio
    async def test_a_user_override_wins_and_is_labelled(self, deg_rows, no_frames):
        table, _ = deg_rows
        table["UP"] = [_deg("EGFR", 2.0, 0.001, "UP")]

        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
            replicates_override={"Treated": 7},
        )
        assert payload.conditions[0].replicates == 7
        assert payload.conditions[0].replicates_source == "user"
        assert payload.needs_replicates is False


# ── Warnings that save an upstream call ──────────────────────────────────────


class TestWarnings:
    @pytest.mark.asyncio
    async def test_ensembl_ids_masquerading_as_symbols_are_flagged(self, deg_rows, no_frames):
        """The R script sets `gene_name = gene_id` when the matrix has no symbol column.

        Detecting it here avoids returning "no gene resolved", which reads like an outage
        rather than a missing column in the user's input.
        """
        table, _ = deg_rows
        table["UP"] = [
            _deg(f"ENSG0000000{i}", 2.0, 0.001, "UP") for i in range(5)
        ]
        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
            replicates_override={"Treated": 4},
        )
        assert any("Ensembl IDs" in w for w in payload.warnings)

    @pytest.mark.asyncio
    async def test_a_non_human_comparison_warns_without_blocking(self, deg_rows, no_frames):
        """Warned, not blocked.

        It is the user's call, but the upstream ranking is human TCGA, so most murine symbols
        will not resolve.
        """
        table, _ = deg_rows
        table["UP"] = [_deg("Egfr", 2.0, 0.001, "UP")]

        payload = await build_signature_payload(
            FakeDb(), _dataset({"species": "mouse"}), "Treated_vs_Control",
            padj_max=0.05, logfc_min=1.0, directions="up",
            replicates_override={"Treated": 4},
        )
        assert payload.conditions, "not blocked"
        assert any("human" in w for w in payload.warnings)

    @pytest.mark.asyncio
    async def test_no_gene_at_all_says_so(self, deg_rows, no_frames):
        table, _ = deg_rows
        payload = await build_signature_payload(
            FakeDb(), _dataset(), "Treated_vs_Control", padj_max=0.05, logfc_min=1.0
        )
        assert payload.conditions == ()
        assert any("No gene passes" in w for w in payload.warnings)
