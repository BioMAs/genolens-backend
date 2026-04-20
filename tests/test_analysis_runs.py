"""
Unit tests for GET /datasets/{dataset_id}/analysis-runs endpoint.

Uses dependency_overrides to skip real DB and auth.

Covers:
1. Returns 200 with correct shape when AnalysisRun records exist
2. Returns empty runs list when no records for that dataset
3. analysis_type filter is applied (only matching records returned)
4. offset and limit are passed to the query
5. Returns 404 when dataset not found
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
from datetime import datetime
from httpx import AsyncClient, ASGITransport

from tests.conftest import (
    TEST_USER_ID,
    TEST_PROJECT_ID,
    TEST_DATASET_ID,
    make_project,
    make_fake_supabase_user,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(dataset_id: UUID = TEST_DATASET_ID, project_id: UUID = TEST_PROJECT_ID):
    """Create a minimal Dataset mock."""
    from app.models.models import Dataset
    ds = MagicMock(spec=Dataset)
    ds.id = dataset_id
    ds.project_id = project_id
    ds.status = "READY"
    ds.created_at = datetime.utcnow()
    ds.updated_at = datetime.utcnow()
    return ds


def _make_analysis_run(
    dataset_id: UUID = TEST_DATASET_ID,
    analysis_type: str = "VOLCANO",
    comparison_name: str = "KO_vs_WT",
):
    """Create a minimal AnalysisRun mock."""
    from app.models.models import AnalysisRun
    r = MagicMock(spec=AnalysisRun)
    r.id = uuid4()
    r.dataset_id = dataset_id
    r.user_id = str(TEST_USER_ID)
    r.analysis_type = analysis_type
    r.comparison_name = comparison_name
    r.parameters = {"fc_threshold": 1.5, "pval_threshold": 0.05}
    r.algorithm_versions = {"scipy": "1.12.0", "pandas": "2.2.0"}
    r.reference_db_versions = None
    r.result_summary = {"gene_count": 42}
    r.created_at = datetime.utcnow()
    return r


def _make_db_with_runs(dataset=None, project=None, runs=None, total=None):
    """
    Build a mock DB whose execute() side_effect serves:
      call 0 → dataset lookup (scalar_one_or_none)
      call 1 → project lookup (scalar_one)
      call 2 → analysis runs query (scalars().all())
      call 3 → count query (scalar_one)
    """
    dataset = dataset or _make_dataset()
    project = project or make_project()
    # Make owner match the test user so the access check passes
    project.owner_id = TEST_USER_ID
    runs = runs if runs is not None else [_make_analysis_run()]
    if total is None:
        total = len(runs)

    ds_result = MagicMock()
    ds_result.scalar_one_or_none.return_value = dataset

    proj_result = MagicMock()
    proj_result.scalar_one.return_value = project

    runs_result = MagicMock()
    runs_result.scalars.return_value.all.return_value = runs

    count_result = MagicMock()
    count_result.scalar_one.return_value = total

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    db.execute = AsyncMock(side_effect=[ds_result, proj_result, runs_result, count_result])
    return db


def _make_db_dataset_not_found():
    """DB whose first execute returns None (dataset not found)."""
    not_found = MagicMock()
    not_found.scalar_one_or_none.return_value = None

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock(return_value=not_found)
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def analysis_runs_client():
    """
    AsyncClient with auth and DB dependencies overridden.
    Returns (client, mock_db_factory) — tests that need custom DB behaviour
    call the factory inside a new dependency override.
    """
    from app.main import app
    from app.api.deps import get_current_user, get_db

    fake_user = make_fake_supabase_user()
    mock_db = _make_db_with_runs()

    async def _override_user():
        return fake_user

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAnalysisRuns:

    # ── 1. Returns 200 with correct shape ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_returns_200_with_correct_shape(self, analysis_runs_client):
        client, _ = analysis_runs_client
        response = await client.get(f"/api/v1/datasets/{TEST_DATASET_ID}/analysis-runs")

        assert response.status_code == 200
        data = response.json()

        # Top-level pagination envelope
        assert "dataset_id" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data
        assert "runs" in data
        assert isinstance(data["runs"], list)

        # Per-run shape
        run = data["runs"][0]
        assert "id" in run
        assert "analysis_type" in run
        assert "comparison_name" in run
        assert "parameters" in run
        assert "algorithm_versions" in run
        assert "reference_db_versions" in run
        assert "result_summary" in run
        assert "user_id" in run
        assert "created_at" in run

    # ── 2. Returns empty runs list when no records ────────────────────────────

    @pytest.mark.asyncio
    async def test_returns_empty_runs_when_none_exist(self):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        fake_user = make_fake_supabase_user()
        mock_db = _make_db_with_runs(runs=[], total=0)

        async def _override_user():
            return fake_user

        async def _override_db():
            yield mock_db

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(f"/api/v1/datasets/{TEST_DATASET_ID}/analysis-runs")

        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["total"] == 0

    # ── 3. analysis_type filter is applied ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_analysis_type_filter_returns_only_matching_runs(self):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        fake_user = make_fake_supabase_user()
        volcano_run = _make_analysis_run(analysis_type="VOLCANO")
        mock_db = _make_db_with_runs(runs=[volcano_run], total=1)

        async def _override_user():
            return fake_user

        async def _override_db():
            yield mock_db

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                f"/api/v1/datasets/{TEST_DATASET_ID}/analysis-runs",
                params={"analysis_type": "VOLCANO"},
            )

        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["analysis_type"] == "VOLCANO"

    # ── 4. offset and limit are forwarded to the query ───────────────────────

    @pytest.mark.asyncio
    async def test_offset_and_limit_are_reflected_in_response(self):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        fake_user = make_fake_supabase_user()
        mock_db = _make_db_with_runs(runs=[], total=100)

        async def _override_user():
            return fake_user

        async def _override_db():
            yield mock_db

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                f"/api/v1/datasets/{TEST_DATASET_ID}/analysis-runs",
                params={"offset": 20, "limit": 10},
            )

        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        # The response must echo back the requested pagination params
        assert data["offset"] == 20
        assert data["limit"] == 10
        # Total reflects the full count, not just the page
        assert data["total"] == 100

    # ── 5. Returns 404 when dataset not found ────────────────────────────────

    @pytest.mark.asyncio
    async def test_returns_404_when_dataset_not_found(self):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        fake_user = make_fake_supabase_user()
        mock_db = _make_db_dataset_not_found()

        async def _override_user():
            return fake_user

        async def _override_db():
            yield mock_db

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db

        nonexistent_id = uuid4()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(f"/api/v1/datasets/{nonexistent_id}/analysis-runs")

        app.dependency_overrides.clear()

        assert response.status_code == 404
        assert response.json()["detail"] == "Dataset not found"

    # ── 6. Returns 403 when user is not owner and not a member ───────────────

    @pytest.mark.asyncio
    async def test_returns_403_when_user_not_authorized(self):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        fake_user = make_fake_supabase_user()

        # Build a project owned by a *different* user
        project = make_project()
        project.owner_id = uuid4()  # not TEST_USER_ID

        dataset = _make_dataset()

        # DB: dataset found → project found → member query returns None (not a member)
        ds_result = MagicMock()
        ds_result.scalar_one_or_none.return_value = dataset

        proj_result = MagicMock()
        proj_result.scalar_one.return_value = project

        member_result = MagicMock()
        member_result.scalar_one_or_none.return_value = None  # not a member

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=[ds_result, proj_result, member_result])

        async def _override_user():
            return fake_user

        async def _override_db():
            yield db

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(f"/api/v1/datasets/{TEST_DATASET_ID}/analysis-runs")

        app.dependency_overrides.clear()

        assert response.status_code == 403
        assert "Not authorized" in response.json()["detail"]
