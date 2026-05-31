"""
Integration tests for report generation endpoints.

Covers:
- GET /projects/{id}/report/status  → 404 when no job exists
- GET /projects/{id}/report/download → 404 when no completed job exists
- POST /projects/{id}/report → 404 when project not found
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_USER_ID, make_fake_supabase_user

_AUTH_PATH = "app.api.deps.auth.get_current_user"
_DB_PATH = "app.api.deps.db.get_db"

FAKE_PROJECT_ID = str(uuid4())


@pytest_asyncio.fixture
async def reports_client():
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db

    fake_user = make_fake_supabase_user(user_id=TEST_USER_ID)

    async def _fake_db():
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)  # no project found
        mock.execute = AsyncMock()
        mock.execute.return_value.scalar_one_or_none = lambda: None
        yield mock

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = _fake_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_report_status_404_no_job(reports_client):
    resp = await reports_client.get(f"/api/v1/projects/{FAKE_PROJECT_ID}/report/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_download_404_no_done_job(reports_client):
    resp = await reports_client.get(f"/api/v1/projects/{FAKE_PROJECT_ID}/report/download")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trigger_report_404_project_not_found(reports_client):
    resp = await reports_client.post(f"/api/v1/projects/{FAKE_PROJECT_ID}/report")
    assert resp.status_code == 404
