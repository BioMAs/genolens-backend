"""
Integration tests for analysis access control (GET/DELETE /analyses/{id}).

An analysis belongs to a project. Access must follow project access — owner or
ProjectMember — like every other project-scoped resource, not the identity of
whoever launched the job.

Covers:
- Creator gets 200
- Project member who did not create the analysis gets 200 (shared project)
- Stranger (neither owner nor member) gets 403
- DELETE stays restricted to the creator and the project owner
"""
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, make_project, make_fake_supabase_user

OWNER_ID = TEST_USER_ID
MEMBER_ID = UUID("00000000-0000-0000-0000-0000000000aa")
STRANGER_ID = UUID("00000000-0000-0000-0000-0000000000bb")
ANALYSIS_ID = uuid4()


def make_analysis(user_id: UUID = OWNER_ID, project_id: UUID = TEST_PROJECT_ID):
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=ANALYSIS_ID,
        project_id=project_id,
        name="Demo analysis",
        data_type="transcriptomics",
        status="DONE",
        matrix_dataset_id=uuid4(),
        samples_dataset_id=uuid4(),
        comparisons_dataset_id=uuid4(),
        params={},
        result_dataset_ids=[],
        intermediate_dataset_ids={},
        celery_task_id=None,
        current_step=None,
        progress_log=[],
        error_message=None,
        user_id=user_id,
        created_at=now,
        updated_at=now,
    )


def make_client(*, as_user: UUID, scalars: list):
    """Client whose db.scalar() returns `scalars` in order."""
    from app.main import app
    from app.api.deps.supabase_deps import get_current_user, get_db

    async def _fake_db():
        mock = AsyncMock()
        mock.scalar = AsyncMock(side_effect=list(scalars))
        yield mock

    app.dependency_overrides[get_current_user] = lambda: make_fake_supabase_user(user_id=as_user)
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_analysis_creator_200():
    analysis = make_analysis(user_id=OWNER_ID)
    project = make_project(owner_id=OWNER_ID)
    async with make_client(as_user=OWNER_ID, scalars=[analysis, project, None]) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(ANALYSIS_ID)


@pytest.mark.asyncio
async def test_get_analysis_project_member_200():
    """A shared project's member must be able to open an analysis they didn't launch."""
    analysis = make_analysis(user_id=OWNER_ID)
    project = make_project(owner_id=OWNER_ID)
    member = SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=MEMBER_ID)
    async with make_client(as_user=MEMBER_ID, scalars=[analysis, project, member]) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == str(ANALYSIS_ID)


@pytest.mark.asyncio
async def test_get_analysis_stranger_403():
    analysis = make_analysis(user_id=OWNER_ID)
    project = make_project(owner_id=OWNER_ID)
    async with make_client(as_user=STRANGER_ID, scalars=[analysis, project, None]) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_get_analysis_404_when_missing():
    async with make_client(as_user=OWNER_ID, scalars=[None]) as c:
        resp = await c.get(f"/api/v1/analyses/{ANALYSIS_ID}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_analysis_member_403():
    """Members may read but not delete someone else's analysis."""
    analysis = make_analysis(user_id=OWNER_ID)
    project = make_project(owner_id=OWNER_ID)
    async with make_client(as_user=MEMBER_ID, scalars=[analysis, project]) as c:
        resp = await c.delete(f"/api/v1/analyses/{ANALYSIS_ID}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_analysis_project_owner_200():
    """The project owner can delete an analysis launched by another member."""
    analysis = make_analysis(user_id=MEMBER_ID)
    project = make_project(owner_id=OWNER_ID)
    async with make_client(as_user=OWNER_ID, scalars=[analysis, project]) as c:
        resp = await c.delete(f"/api/v1/analyses/{ANALYSIS_ID}")
    assert resp.status_code == 204
