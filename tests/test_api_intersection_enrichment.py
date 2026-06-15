"""
Tests for ad-hoc intersection enrichment: the R-output CSV parser and the
trigger/poll endpoints (Celery dispatch mocked; license overridden).
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_USER_ID, TEST_PROJECT_ID, make_fake_supabase_user, make_project

PATH_DATASET_ID = UUID("00000000-0000-0000-0000-0000000000c1")


def _make_dataset(dataset_id, project):
    ds = MagicMock()
    ds.id = dataset_id
    ds.project = project
    ds.project_id = project.id
    ds.dataset_metadata = {"species": "human"}
    return ds


def _scalar_one(value):
    res = MagicMock()
    res.scalar_one_or_none.return_value = value
    return res


# ── CSV parser ───────────────────────────────────────────────────────────────

class TestParseEnrichmentCsv:
    def test_parses_rows_and_sorts_by_padj(self, tmp_path):
        from app.worker.tasks.intersection_enrichment_task import _parse_enrichment_csv

        csv = tmp_path / "enr.csv"
        csv.write_text(
            "term,Description,category,pvalue,p.adjust,genes,Count,GeneRatio,BgRatio,gene.cluster\n"
            "GO:0001,Apoptosis,GO:BP,0.001,0.02,TP53/BAX,2,2/10,50/10000,intersection\n"
            "hsa04210,Apoptosis KEGG,KEGG,0.0001,0.005,TP53/CASP3/BAX,3,3/10,80/10000,intersection\n"
        )
        rows = _parse_enrichment_csv(csv)
        assert len(rows) == 2
        # Sorted by padj ascending → KEGG row first
        assert rows[0]["category"] == "KEGG"
        assert rows[0]["padj"] == 0.005
        assert rows[0]["gene_count"] == 3
        assert rows[0]["genes"] == ["TP53", "CASP3", "BAX"]
        assert rows[1]["pathway_id"] == "GO:0001"

    def test_empty_csv_returns_empty(self, tmp_path):
        from app.worker.tasks.intersection_enrichment_task import _parse_enrichment_csv

        csv = tmp_path / "empty.csv"
        csv.write_text(
            "term,Description,category,pvalue,p.adjust,genes,Count,GeneRatio,BgRatio,gene.cluster\n"
        )
        assert _parse_enrichment_csv(csv) == []


# ── Endpoints ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def enrich_client():
    from app.main import app
    from app.api.deps import get_current_user, get_db
    from app.api.deps.license import require_active_license

    fake_user = make_fake_supabase_user()
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    async def _override_user():
        return fake_user

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[require_active_license] = lambda: None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


class TestTriggerEndpoint:
    @pytest.mark.asyncio
    async def test_trigger_creates_job_and_dispatches(self, enrich_client):
        client, mock_db = enrich_client
        project = make_project(project_id=TEST_PROJECT_ID, owner_id=TEST_USER_ID)
        dataset = _make_dataset(PATH_DATASET_ID, project)
        mock_db.execute = AsyncMock(return_value=_scalar_one(dataset))

        # db.refresh should give the job an id
        async def _refresh(obj):
            obj.id = UUID("00000000-0000-0000-0000-0000000000d1")
        mock_db.refresh = AsyncMock(side_effect=_refresh)

        with patch(
            "app.worker.tasks.intersection_enrichment_task.run_intersection_enrichment.apply_async"
        ) as mock_apply:
            mock_apply.return_value = MagicMock(id="celery-task-1")
            resp = await client.post(
                f"/api/v1/datasets/{PATH_DATASET_ID}/intersection-enrichment",
                json={"genes": ["TP53", "BAX"], "label": "A ∩ B"},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "PENDING"
        assert "job_id" in body
        # Dispatched explicitly to the r_analysis queue
        mock_apply.assert_called_once()
        assert mock_apply.call_args.kwargs.get("queue") == "r_analysis"

    @pytest.mark.asyncio
    async def test_trigger_rejects_empty_genes(self, enrich_client):
        client, mock_db = enrich_client
        resp = await client.post(
            f"/api/v1/datasets/{PATH_DATASET_ID}/intersection-enrichment",
            json={"genes": [], "label": "x"},
        )
        assert resp.status_code == 400
