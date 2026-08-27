"""
Integration tests for report endpoint access control.

Report jobs expose analysis/comparison content (including the generated PDF), so
every route must enforce project access — owner or ProjectMember — not merely a
valid JWT.

Covers, for both the analysis-scoped and comparison-scoped routers:
- Stranger (neither project owner nor member) gets 403
- Project member passes the gate and reaches the job lookup
"""
import pytest
import pytest_asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, make_fake_current_user, make_project

OWNER_ID = TEST_USER_ID
MEMBER_ID = UUID("00000000-0000-0000-0000-0000000000aa")
STRANGER_ID = UUID("00000000-0000-0000-0000-0000000000bb")
ANALYSIS_ID = uuid4()
DATASET_ID = uuid4()


def make_client(*, as_user: UUID, member: bool):
    """Client where the analysis/dataset exists in a project owned by OWNER_ID."""
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db
    from app.models.models import Dataset, SelfServiceAnalysis

    analysis = SimpleNamespace(id=ANALYSIS_ID, project_id=TEST_PROJECT_ID, name="Demo")
    dataset = SimpleNamespace(id=DATASET_ID, project_id=TEST_PROJECT_ID, name="DEG")
    project = make_project(owner_id=OWNER_ID)
    membership = SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=as_user) if member else None

    async def _fake_db():
        mock = AsyncMock()

        async def _get(model, _id):
            if model is SelfServiceAnalysis:
                return analysis
            if model is Dataset:
                return dataset
            return None

        async def _scalar(stmt):
            # assert_project_access queries Project first, then ProjectMember
            entity = stmt.column_descriptions[0]["entity"].__name__
            return project if entity == "Project" else membership

        mock.get = AsyncMock(side_effect=_get)
        mock.scalar = AsyncMock(side_effect=_scalar)
        mock.execute = AsyncMock()
        mock.execute.return_value.scalar_one_or_none = lambda: None  # no report job
        yield mock

    app.dependency_overrides[get_current_user] = lambda: make_fake_current_user(user_id=as_user)
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    app.dependency_overrides.clear()


# ── Analysis-scoped routes ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_report_403_for_stranger():
    async with make_client(as_user=STRANGER_ID, member=False) as c:
        resp = await c.post(f"/api/v1/analyses/{ANALYSIS_ID}/report")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_report_status_403_for_stranger():
    async with make_client(as_user=STRANGER_ID, member=False) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}/report/status")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_download_report_403_for_stranger():
    async with make_client(as_user=STRANGER_ID, member=False) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}/report/download")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_report_status_member_passes_gate():
    """A member is let through — 404 here means "no job yet", not "denied"."""
    async with make_client(as_user=MEMBER_ID, member=True) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}/report/status")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No report job found for this analysis"


# ── Comparison-scoped routes ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_comparison_report_status_403_for_stranger():
    async with make_client(as_user=STRANGER_ID, member=False) as c:
        resp = await c.get(
            f"/api/v1/datasets/{DATASET_ID}/report/status",
            params={"comparison_name": "A_vs_B"},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_comparison_download_403_for_stranger():
    async with make_client(as_user=STRANGER_ID, member=False) as c:
        resp = await c.get(
            f"/api/v1/datasets/{DATASET_ID}/report/download",
            params={"comparison_name": "A_vs_B"},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_comparison_report_status_member_passes_gate():
    async with make_client(as_user=MEMBER_ID, member=True) as c:
        resp = await c.get(
            f"/api/v1/datasets/{DATASET_ID}/report/status",
            params={"comparison_name": "A_vs_B"},
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No report job found for this comparison"
