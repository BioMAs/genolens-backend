"""
Unit tests for enrichment_source — which dataset holds a comparison's enrichment.

The bug this closes: `interpret_comparison` and the chat's pathways tool read
`EnrichmentPathway` under the id they were handed, which on a comparison page is the DEG dataset.
A self-service analysis keeps its enrichment in a separate annoDB ENRICHMENT dataset, and the
worker deliberately skips the legacy Python enrichment that would otherwise write rows under the
DEG id — so the AI and the chat saw **no pathways at all** while the enrichment panel next to them
was full of them.

The rules mirror the client (`useComparisonContext.ts`) on purpose, so the two cannot drift apart
again. These tests pin the mirroring, not just the happy path.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.models.models import DatasetStatus, DatasetType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PROJECT = uuid4()


def make_dataset(
    *,
    name: str = "ds",
    type_=DatasetType.ENRICHMENT,
    status=DatasetStatus.READY,
    project_id=None,
    metadata: dict | None = None,
):
    d = MagicMock()
    d.id = uuid4()
    d.name = name
    d.type = type_
    d.status = status
    d.project_id = project_id or PROJECT
    d.dataset_metadata = metadata
    return d


def make_db(candidates):
    """A db whose single ENRICHMENT query returns `candidates`."""
    db = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = candidates
    db.execute = AsyncMock(return_value=result)
    return db


def deg_dataset(metadata: dict | None = None):
    return make_dataset(
        name="KO_vs_WT", type_=DatasetType.DEG, metadata=metadata
    )


# ─────────────────────────────────────────────────────────────────────────────
# find_enrichment_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestFindEnrichmentDataset:

    @pytest.mark.asyncio
    async def test_matches_on_metadata_comparison_name(self):
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="whatever", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_matches_on_dataset_name(self):
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="KO_vs_WT", metadata={})
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_matches_a_multi_comparison_file(self):
        """An enrichment file that merely lists the comparison among its columns."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(
            name="All enrichment",
            metadata={"enrichment_comparisons": ["A_vs_B", "KO_vs_WT"]},
        )
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_direct_name_match_beats_a_multi_comparison_file(self):
        """`byName || byComparisons` precedence, as the client applies it."""
        from app.services.enrichment_source import find_enrichment_dataset

        listed = make_dataset(
            name="All enrichment",
            metadata={"enrichment_comparisons": ["KO_vs_WT"]},
        )
        named = make_dataset(name="x", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([listed, named])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is named

    @pytest.mark.asyncio
    async def test_prefers_a_ready_dataset(self):
        """A failed or half-written duplicate must never be picked over a READY one."""
        from app.services.enrichment_source import find_enrichment_dataset

        failed = make_dataset(
            name="x", status=DatasetStatus.FAILED, metadata={"comparison_name": "KO_vs_WT"}
        )
        ready = make_dataset(
            name="y", status=DatasetStatus.READY, metadata={"comparison_name": "KO_vs_WT"}
        )
        db = make_db([failed, ready])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is ready

    @pytest.mark.asyncio
    async def test_scopes_to_the_same_analysis(self):
        """
        An enrichment file from another analysis sharing the comparison name must not bleed in.

        The client hit this and fixed it the same way ("mélange entre analyses").
        """
        from app.services.enrichment_source import find_enrichment_dataset

        analysis = str(uuid4())
        other = make_dataset(
            name="other-analysis",
            metadata={"comparison_name": "KO_vs_WT", "analysis_id": str(uuid4())},
        )
        mine = make_dataset(
            name="my-analysis",
            metadata={"comparison_name": "KO_vs_WT", "analysis_id": analysis},
        )
        db = make_db([other, mine])

        found = await find_enrichment_dataset(
            db, deg_dataset({"analysis_id": analysis}), "KO_vs_WT"
        )
        assert found is mine

    @pytest.mark.asyncio
    async def test_falls_back_to_project_scope_when_analysis_has_none(self):
        """Analysis scoping narrows only when it would leave something behind."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="x", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([enr])

        found = await find_enrichment_dataset(
            db, deg_dataset({"analysis_id": str(uuid4())}), "KO_vs_WT"
        )
        assert found is enr

    @pytest.mark.asyncio
    async def test_returns_none_when_nothing_matches(self):
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="unrelated", metadata={"comparison_name": "A_vs_B"})
        db = make_db([enr])

        assert await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_project_has_no_enrichment(self):
        from app.services.enrichment_source import find_enrichment_dataset

        db = make_db([])
        assert await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT") is None

    @pytest.mark.asyncio
    async def test_tolerates_null_metadata(self):
        """`dataset_metadata` is nullable; a None must not raise."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="KO_vs_WT", metadata=None)
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(None), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_tolerates_non_list_enrichment_comparisons(self):
        """Malformed metadata must not raise — it just fails to match."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="x", metadata={"enrichment_comparisons": "KO_vs_WT"})
        db = make_db([enr])

        assert await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT") is None


# ─────────────────────────────────────────────────────────────────────────────
# resolve_pathway_dataset_id
# ─────────────────────────────────────────────────────────────────────────────

class TestResolvePathwayDatasetId:

    @pytest.mark.asyncio
    async def test_returns_the_enrichment_dataset_id(self):
        """The regression: the AI must read annoDB pathways, not the empty DEG dataset."""
        from app.services.enrichment_source import resolve_pathway_dataset_id

        enr = make_dataset(name="x", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([enr])
        deg = deg_dataset({"analysis_id": str(uuid4())})

        assert await resolve_pathway_dataset_id(db, deg, "KO_vs_WT") == enr.id

    @pytest.mark.asyncio
    async def test_falls_back_to_the_deg_dataset(self):
        """
        A plain DEG upload has no ENRICHMENT dataset: the legacy Python enrichment writes its rows
        under the DEG id, and that is the only enrichment it has. Same `?? dataset.id` fallback the
        enrichment panel applies, so the AI reads what the user sees.
        """
        from app.services.enrichment_source import resolve_pathway_dataset_id

        db = make_db([])
        deg = deg_dataset()

        assert await resolve_pathway_dataset_id(db, deg, "KO_vs_WT") == deg.id


# ─────────────────────────────────────────────────────────────────────────────
# deg_comparison_names
# ─────────────────────────────────────────────────────────────────────────────

class TestDegComparisonNames:

    def test_reads_a_single_comparison_name(self):
        from app.services.enrichment_source import deg_comparison_names
        deg = deg_dataset({"comparison_name": "KO_vs_WT"})
        assert deg_comparison_names(deg) == ["KO_vs_WT"]

    def test_reads_the_keys_of_a_comparisons_dict(self):
        """Multi-comparison DEG files key `comparisons` by name."""
        from app.services.enrichment_source import deg_comparison_names
        deg = deg_dataset({"comparisons": {"A_vs_B": {}, "C_vs_D": {}}})
        assert set(deg_comparison_names(deg)) == {"A_vs_B", "C_vs_D"}

    def test_reads_a_comparisons_list(self):
        from app.services.enrichment_source import deg_comparison_names
        deg = deg_dataset({"comparisons": ["A_vs_B", "C_vs_D"]})
        assert deg_comparison_names(deg) == ["A_vs_B", "C_vs_D"]

    def test_the_named_comparison_comes_first(self):
        """Most specific first, so a per-comparison name beats the file's whole list."""
        from app.services.enrichment_source import deg_comparison_names
        deg = deg_dataset({"comparison_name": "KO_vs_WT", "comparisons": ["A_vs_B"]})
        assert deg_comparison_names(deg)[0] == "KO_vs_WT"

    def test_deduplicates(self):
        from app.services.enrichment_source import deg_comparison_names
        deg = deg_dataset({"comparison_name": "KO_vs_WT", "comparisons": ["KO_vs_WT"]})
        assert deg_comparison_names(deg) == ["KO_vs_WT"]

    def test_empty_when_nothing_is_declared(self):
        from app.services.enrichment_source import deg_comparison_names
        assert deg_comparison_names(deg_dataset(None)) == []
        assert deg_comparison_names(deg_dataset({"comparisons": "not-a-list"})) == []


# ─────────────────────────────────────────────────────────────────────────────
# match_enrichment_for_deg — the project report, which has no comparison in hand
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchEnrichmentForDeg:

    def test_matches_on_a_declared_comparison(self):
        from app.services.enrichment_source import match_enrichment_for_deg

        enr = make_dataset(name="enr", metadata={"comparison_name": "KO_vs_WT"})
        deg = deg_dataset({"comparison_name": "KO_vs_WT"})

        assert match_enrichment_for_deg([enr], deg) is enr

    def test_matches_a_multi_comparison_enrichment_file(self):
        """
        The pair the old rule missed: names that do not contain one another, but the enrichment
        file lists the comparison. 'All DEG' / 'All enrichment' on the dev database.
        """
        from app.services.enrichment_source import match_enrichment_for_deg

        enr = make_dataset(
            name="All enrichment",
            metadata={"enrichment_comparisons": ["KO_vs_WT", "A_vs_B"]},
        )
        deg = deg_dataset({"comparisons": {"KO_vs_WT": {}}})

        assert match_enrichment_for_deg([enr], deg) is enr

    def test_comparison_rules_win_over_name_pairing(self):
        """
        Precedence matters: the name fallback is fragile and must never pre-empt a real match.
        """
        from app.services.enrichment_source import match_enrichment_for_deg

        # `paired` would win on substring alone — its name contains the DEG's.
        paired = make_dataset(name="KO_vs_WT enrichment", metadata={})
        correct = make_dataset(name="unrelated", metadata={"comparison_name": "KO_vs_WT"})
        deg = deg_dataset({"comparison_name": "KO_vs_WT"})
        deg.name = "KO_vs_WT"

        assert match_enrichment_for_deg([paired, correct], deg) is correct

    def test_falls_back_to_name_pairing(self):
        """The project report's original and only rule, kept as a last resort."""
        from app.services.enrichment_source import match_enrichment_for_deg

        enr = make_dataset(name="KO_vs_WT enrichment", metadata={})
        deg = deg_dataset(None)
        deg.name = "KO_vs_WT"

        assert match_enrichment_for_deg([enr], deg) is enr

    def test_returns_none_when_nothing_pairs(self):
        from app.services.enrichment_source import match_enrichment_for_deg

        enr = make_dataset(name="totally other", metadata={})
        deg = deg_dataset(None)
        deg.name = "KO_vs_WT"

        assert match_enrichment_for_deg([enr], deg) is None

    def test_prefers_ready_in_the_name_fallback_too(self):
        from app.services.enrichment_source import match_enrichment_for_deg

        failed = make_dataset(name="KO_vs_WT a", status=DatasetStatus.FAILED, metadata={})
        ready = make_dataset(name="KO_vs_WT b", status=DatasetStatus.READY, metadata={})
        deg = deg_dataset(None)
        deg.name = "KO_vs_WT"

        assert match_enrichment_for_deg([failed, ready], deg) is ready

    def test_tolerates_a_nameless_candidate(self):
        from app.services.enrichment_source import match_enrichment_for_deg

        nameless = make_dataset(name=None, metadata={})
        deg = deg_dataset(None)
        deg.name = "KO_vs_WT"

        assert match_enrichment_for_deg([nameless], deg) is None


# ─────────────────────────────────────────────────────────────────────────────
# recorded_comparison_names — what lets the nova tell an unknown comparison from an empty one
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordedComparisonNames:

    @pytest.mark.asyncio
    async def test_the_metadata_answers_without_reading_the_deg_rows(self):
        from app.services.enrichment_source import recorded_comparison_names

        db = make_db([])
        deg = deg_dataset({"comparisons": {"A_vs_B": {}, "C_vs_D": {}}})

        assert await recorded_comparison_names(db, deg) == {"A_vs_B", "C_vs_D"}
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_deg_rows_answer_when_the_metadata_does_not(self):
        from app.services.enrichment_source import recorded_comparison_names

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = ["A_vs_B", None, "C_vs_D"]
        db.execute = AsyncMock(return_value=result)

        assert await recorded_comparison_names(db, deg_dataset(None)) == {"A_vs_B", "C_vs_D"}

    @pytest.mark.asyncio
    async def test_a_dataset_that_records_none_answers_with_an_empty_set(self):
        from app.services.enrichment_source import recorded_comparison_names

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        assert await recorded_comparison_names(db, deg_dataset({})) == set()


# ─────────────────────────────────────────────────────────────────────────────
# Rows without an adjusted p are never stored as if they had one
# ─────────────────────────────────────────────────────────────────────────────

GO_ROW_WITH_FDR = {
    "go_id": "GO:0000001",
    "namespace": "biological_process",
    "term_name": "kept",
    "definition": None,
    "study_count": 2,
    "pvalue": 0.012,
    "fdr": 0.04,
    "background_count": 10,
    "study_genes": ["G1", "G2"],
}
GO_ROW_WITHOUT_FDR = {**GO_ROW_WITH_FDR, "go_id": "GO:0000002", "term_name": "left out", "pvalue": 0.001}
GO_ROW_WITHOUT_FDR.pop("fdr")
ORA_ROW_WITH_FDR = {
    "pathway_id": "P1",
    "pathway_name": "kept",
    "study_count": 2,
    "pvalue": 0.012,
    "fdr": 0.04,
    "study_genes": ["G1"],
}
ORA_ROW_WITHOUT_FDR = {**ORA_ROW_WITH_FDR, "pathway_id": "P2", "pathway_name": "left out", "pvalue": 0.001}
ORA_ROW_WITHOUT_FDR["fdr"] = None


def test_split_by_adjusted_p_keeps_only_rows_that_carry_one():
    from app.services.enrichment_source import split_by_adjusted_p

    nan = float("nan")
    rows = [{"fdr": 0.5}, {"fdr": 0.0}, {}, {"fdr": None}, {"fdr": nan}]
    kept, n_without = split_by_adjusted_p(rows)
    assert kept == [{"fdr": 0.5}, {"fdr": 0.0}]
    assert n_without == 3


async def test_rows_without_fdr_are_not_stored_as_adjusted_by_the_worker():
    """The worker's automatic enrichment (GO and ORA) stores `padj` only from a real `fdr`."""
    import sys

    import app.worker.tasks  # noqa: F401 — the package loads tasks.py as this module

    tasks = sys.modules["app.worker._tasks_legacy"]
    deg = [
        SimpleNamespace(comparison_name="A_vs_B", padj=0.001, log_fc=2.0, gene_name="G1", gene_id="g1"),
        SimpleNamespace(comparison_name="A_vs_B", padj=0.001, log_fc=-2.0, gene_name="G2", gene_id="g2"),
    ]
    stored: list[dict] = []

    async def execute(stmt, params=None):
        if params is not None:
            stored.extend(params)
        result = MagicMock()
        result.fetchall.return_value = [("A_vs_B",)]
        result.scalars.return_value.all.return_value = deg
        result.scalar_one_or_none.return_value = 1  # every ORA database has gene sets
        return result

    db = MagicMock()
    db.execute = AsyncMock(side_effect=execute)
    db.commit = AsyncMock()
    go = MagicMock()
    go.go_enrichment_analysis = AsyncMock(return_value=[GO_ROW_WITH_FDR, GO_ROW_WITHOUT_FDR])
    go.run_ora_enrichment = AsyncMock(return_value=[ORA_ROW_WITH_FDR, ORA_ROW_WITHOUT_FDR])

    with patch.object(tasks, "GOService", return_value=go):
        n_without_padj = await tasks._auto_run_enrichment(db, str(uuid4()))

    per_regulation = 1 + len(tasks._ORA_DATABASES)  # one GO row and one row per ORA database
    assert n_without_padj == 3 * per_regulation  # ALL, UP and DOWN
    assert len(stored) == 3 * per_regulation
    assert {r["padj"] for r in stored} == {0.04}
    assert {r["pathway_name"] for r in stored} == {"kept"}


async def test_rows_without_fdr_are_not_stored_as_adjusted_by_the_go_enrichment_route():
    from app.api.endpoints import datasets as route

    dataset = make_dataset(name="A_vs_B", type_=DatasetType.DEG, metadata={"rows": 10})
    degs = [SimpleNamespace(gene_name="G1", gene_id="g1"), SimpleNamespace(gene_name="G2", gene_id="g2")]
    result = MagicMock()
    result.scalar_one_or_none.return_value = dataset
    result.scalars.return_value.all.return_value = degs
    result.fetchall.return_value = [("G1",), ("G2",), ("G3",)]
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    added = []
    db.add = MagicMock(side_effect=added.append)

    with (
        patch.object(route, "assert_project_read_access", new=AsyncMock()),
        patch.object(
            route.go_service,
            "go_enrichment_analysis",
            new=AsyncMock(return_value=[GO_ROW_WITH_FDR, GO_ROW_WITHOUT_FDR]),
        ),
        patch.object(route.history_service, "log_activity", new=AsyncMock()),
    ):
        await route.run_go_enrichment_analysis(
            dataset.id,
            "A_vs_B",
            db,
            SimpleNamespace(user_id=uuid4()),
            namespace=None,
            regulation=None,
            padj_threshold=0.05,
            log_fc_threshold=0.5,
            min_term_size=5,
            max_term_size=500,
            pvalue_threshold=0.05,
            fdr_method="fdr_bh",
            propagate_annotations=True,
            organism="Homo sapiens",
        )

    assert [row.pathway_id for row in added] == ["GO:0000001"]
    assert [row.padj for row in added] == [0.04]
    assert dataset.dataset_metadata["rows_without_padj"] == 1
    assert dataset.dataset_metadata["rows"] == 10  # what was there stays


async def test_the_rerun_route_records_the_rows_the_worker_left_out():
    from tests.test_project_access_control import make_client, make_fixtures, make_user

    from app.main import app

    owner = make_user()
    dataset, project = make_fixtures(owner_id=owner.id)
    dataset.dataset_metadata = {"min_log2fc": 1.5, "rows_without_padj": 9}
    client = make_client(caller=owner, dataset=dataset, project=project, member=None)
    try:
        with patch("app.worker.tasks._auto_run_enrichment", new=AsyncMock(return_value=4)):
            async with client as c:
                resp = await c.post(f"/api/v1/datasets/{dataset.id}/rerun-enrichment")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    # The run replaced the stored rows: its count replaces the previous one.
    assert dataset.dataset_metadata == {"min_log2fc": 1.5, "rows_without_padj": 4}


# ─────────────────────────────────────────────────────────────────────────────
# Uploaded enrichment tables: the column read as padj, and what could not be read
# ─────────────────────────────────────────────────────────────────────────────

def _enrichment_parquet(padj_header: str, padj_values: list, extra: dict | None = None) -> bytes:
    import io

    import pandas as pd

    n = len(padj_values)
    frame = pd.DataFrame(
        {
            "ID": [f"GO:{i:07d}" for i in range(n)],
            "Description": [f"term {i}" for i in range(n)],
            "pvalue": [0.001] * n,
            padj_header: padj_values,
            **(extra or {}),
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


async def test_upload_records_the_padj_column():
    from app.services.data_processor import data_processor

    # "0.01" reads; "n/a" cannot be parsed; a missing cell cannot either. Both are stored at 1.0.
    parquet = _enrichment_parquet("qvalue", ["0.01", "n/a", None])
    report: dict = {}
    result = await data_processor.extract_enrichment_pathways_for_db(
        parquet, {"A_vs_B": {}}, report=report
    )

    assert report == {"padj_column": "qvalue", "padj_unparsable": 2}
    assert [p["padj"] for p in result["A_vs_B"]] == [0.01, 1.0, 1.0]


async def test_upload_counts_the_unparsable_rows_of_every_comparison():
    from app.services.data_processor import data_processor

    parquet = _enrichment_parquet(
        "FDR",
        ["0.2", "oops", "0.3", "oops"],
        {"gene.cluster": ["A_vs_B (up)", "A_vs_B (up)", "A_vs_B (down)", "A_vs_B (down)"]},
    )
    report: dict = {}
    await data_processor.extract_enrichment_pathways_for_db(parquet, {"A_vs_B": {}}, report=report)

    assert report == {"padj_column": "FDR", "padj_unparsable": 2}


async def test_upload_without_a_padj_column_records_nothing():
    from app.services.data_processor import data_processor

    parquet = _enrichment_parquet("score", ["0.01"])
    report: dict = {}
    assert await data_processor.extract_enrichment_pathways_for_db(
        parquet, {"A_vs_B": {}}, report=report
    ) == {}
    assert report == {}


async def test_the_padj_report_is_optional():
    from app.services.data_processor import data_processor

    parquet = _enrichment_parquet("padj", [0.01])
    result = await data_processor.extract_enrichment_pathways_for_db(parquet, {"A_vs_B": {}})
    assert result["A_vs_B"][0]["padj"] == 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Upload route: the optional `source_version` form field
# ─────────────────────────────────────────────────────────────────────────────

async def _upload(source_version):
    from tests.test_dataset_upload_quota_rollback import make_client, make_user

    from app.main import app

    user = make_user(used=0)
    client, db, project = make_client(as_user=user, update_returns=1)
    data = {"project_id": str(project.id), "name": "Enrichment tables", "dataset_type": "ENRICHMENT"}
    if source_version is not None:
        data["source_version"] = source_version
    files = {"file": ("enrichment.csv", b"ID,Description,pvalue,padj\n", "text/csv")}
    try:
        with (
            patch(
                "app.api.endpoints.datasets.storage_service.upload_file",
                new=AsyncMock(return_value="projects/x/raw/enrichment.csv"),
            ),
            patch("app.api.endpoints.datasets.process_dataset_upload.delay", MagicMock()),
            patch("app.api.endpoints.datasets.history_service.log_activity", new=AsyncMock()),
        ):
            async with client as c:
                resp = await c.post("/api/v1/datasets/upload", data=data, files=files)
    finally:
        app.dependency_overrides.clear()
    return resp, db


async def test_upload_records_the_source_version():
    resp, db = await _upload("pipe_scilicium v3.0.0")
    assert resp.status_code == 201, resp.text
    dataset = db.add.call_args[0][0]
    assert dataset.dataset_metadata["source_version"] == "pipe_scilicium v3.0.0"


@pytest.mark.parametrize("value", [None, "", "   "])
async def test_upload_without_a_source_version_records_none(value):
    resp, db = await _upload(value)
    assert resp.status_code == 201, resp.text
    assert "source_version" not in db.add.call_args[0][0].dataset_metadata


@pytest.mark.parametrize(
    "value",
    ["v3; rm -rf /", "<script>", "v3\nv4", "naïve", "x" * 101],
    ids=["shell", "markup", "newline", "non-ascii", "too-long"],
)
async def test_a_malformed_source_version_is_422(value):
    resp, db = await _upload(value)
    assert resp.status_code == 422, resp.text
    db.add.assert_not_called()


async def test_a_source_version_of_exactly_100_characters_is_accepted():
    resp, db = await _upload("v" * 100)
    assert resp.status_code == 201, resp.text
