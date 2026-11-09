"""
Integration tests for report generation endpoints (analysis-scoped).

Covers:
- GET /analyses/{id}/report/status   → 404 when no job exists
- GET /analyses/{id}/report/download → 404 when no completed job exists
- POST /analyses/{id}/report → 404 when analysis not found
- POST /analyses/{id}/report reads `use_ai`, stores it on the job and keys the reuse of a job on it
"""
import re

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from uuid import uuid4
from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_USER_ID, make_fake_current_user

FAKE_ANALYSIS_ID = str(uuid4())


@pytest_asyncio.fixture
async def reports_client():
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db

    fake_user = make_fake_current_user(user_id=TEST_USER_ID)

    async def _fake_db():
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)  # no analysis found
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
    resp = await reports_client.get(f"/api/v1/analyses/{FAKE_ANALYSIS_ID}/report/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_download_404_no_done_job(reports_client):
    resp = await reports_client.get(f"/api/v1/analyses/{FAKE_ANALYSIS_ID}/report/download")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trigger_report_404_analysis_not_found(reports_client):
    resp = await reports_client.post(f"/api/v1/analyses/{FAKE_ANALYSIS_ID}/report")
    assert resp.status_code == 404


# ── use_ai on the analysis-scoped report ────────────────────────────────────


def _analysis_client(*, in_progress=None, executed=None):
    """A client for POST /analyses/{id}/report on an analysis the caller may see."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.api.deps.subscription import get_or_create_user
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db
    from app.models.models import SelfServiceAnalysis
    from tests.conftest import make_fake_db_user, make_project

    user = make_fake_db_user(user_id=TEST_USER_ID)
    analysis = SimpleNamespace(id=uuid4(), project_id=uuid4(), name="Analysis")
    project = make_project(owner_id=TEST_USER_ID)
    added: list = []

    async def _fake_db():
        mock = AsyncMock()

        async def _get(model, _id):
            return analysis if model is SelfServiceAnalysis else None

        async def _scalar(stmt):
            return project

        async def _execute(stmt, *a, **k):
            if executed is not None:
                executed.append(stmt)
            result = MagicMock()
            result.scalar_one_or_none.return_value = in_progress
            return result

        mock.get = AsyncMock(side_effect=_get)
        mock.scalar = AsyncMock(side_effect=_scalar)
        mock.execute = AsyncMock(side_effect=_execute)
        mock.add = MagicMock(side_effect=added.append)
        mock.commit = AsyncMock()
        mock.refresh = AsyncMock()
        yield mock

    fake_current = make_fake_current_user(user_id=TEST_USER_ID)
    app.dependency_overrides[get_current_user] = lambda: fake_current
    app.dependency_overrides[get_or_create_user] = lambda: user
    app.dependency_overrides[get_db] = _fake_db
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
    return client, analysis, added


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,expected",
    [(None, True), ({}, True), ({"use_ai": True}, True), ({"use_ai": False}, False)],
)
async def test_analysis_report_honours_use_ai_false(body, expected):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from app.main import app

    client, analysis, added = _analysis_client()
    delay = MagicMock(return_value=SimpleNamespace(id="task-1"))
    try:
        with patch("app.worker.tasks.report_task.generate_analysis_report.delay", new=delay):
            async with client as c:
                resp = await c.post(f"/api/v1/analyses/{analysis.id}/report", json=body)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 202, resp.text
    assert len(added) == 1 and added[0].use_ai is expected
    delay.assert_called_once()


@pytest.mark.asyncio
async def test_analysis_report_reuses_only_a_job_made_with_the_same_use_ai():
    """An in-progress job of the other setting is never handed back."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from app.main import app

    executed: list = []
    client, analysis, _added = _analysis_client(executed=executed)
    try:
        with patch(
            "app.worker.tasks.report_task.generate_analysis_report.delay",
            new=MagicMock(return_value=SimpleNamespace(id="task-1")),
        ):
            async with client as c:
                await c.post(f"/api/v1/analyses/{analysis.id}/report", json={"use_ai": False})
    finally:
        app.dependency_overrides.clear()
    where = str(executed[0].compile(compile_kwargs={"literal_binds": True})).split("WHERE")[1]
    assert re.search(r"report_jobs\.use_ai = (false|0)", where)  # next to the analysis filter
