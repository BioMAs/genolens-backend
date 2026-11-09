"""
Comparison report format (`template`): latex | editorial.

Covers:
- the editorial format requires the cosmetics module (403 otherwise, nothing queued)
- each format is stored on the job and dispatched to its own Celery task
- status / download filter on the format, and the downloaded filename carries it
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tests.conftest import (
    TEST_PROJECT_ID,
    make_fake_current_user,
    make_fake_db_user,
    make_project,
)

DATASET_ID = uuid4()
URL = f"/api/v1/datasets/{DATASET_ID}/report"


def _client(user, *, job=None, added=None):
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db
    from app.api.deps.subscription import get_or_create_user
    from app.main import app
    from app.models.models import Dataset

    dataset = SimpleNamespace(id=DATASET_ID, project_id=TEST_PROJECT_ID, name="DEG")
    project = make_project(owner_id=user.id)

    async def _fake_db():
        mock = AsyncMock()

        async def _get(model, _id):
            return dataset if model is Dataset else None

        async def _scalar(stmt):
            entity = stmt.column_descriptions[0]["entity"].__name__
            return project if entity == "Project" else None

        result = MagicMock()
        result.scalars.return_value.first.return_value = None  # no job in progress
        result.scalar_one_or_none.return_value = job
        mock.get = AsyncMock(side_effect=_get)
        mock.scalar = AsyncMock(side_effect=_scalar)
        mock.execute = AsyncMock(return_value=result)
        mock.add = MagicMock(side_effect=(added.append if added is not None else None))
        mock.commit = AsyncMock()
        mock.refresh = AsyncMock()
        yield mock

    app.dependency_overrides[get_current_user] = lambda: make_fake_current_user(user_id=user.id)
    app.dependency_overrides[get_or_create_user] = lambda: user
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app

    app.dependency_overrides.clear()


def _user(cosmetics: bool):
    user = make_fake_db_user()
    user.cosmetics_module_enabled = cosmetics
    return user


def _patch_tasks():
    latex = MagicMock(return_value=SimpleNamespace(id="latex-task"))
    editorial = MagicMock(return_value=SimpleNamespace(id="editorial-task"))
    return (
        patch("app.worker.tasks.report_task.generate_comparison_report.delay", new=latex),
        patch(
            "app.worker.tasks.report_task.generate_comparison_report_editorial.delay", new=editorial
        ),
        latex,
        editorial,
    )


@pytest.mark.asyncio
async def test_editorial_requires_the_cosmetics_module():
    p_latex, p_edit, latex, editorial = _patch_tasks()
    with p_latex, p_edit:
        async with _client(_user(cosmetics=False)) as c:
            resp = await c.post(
                URL, params={"comparison_name": "KO_vs_WT", "template": "editorial"}
            )
    assert resp.status_code == 403
    latex.assert_not_called()
    editorial.assert_not_called()


@pytest.mark.asyncio
async def test_editorial_is_stored_and_dispatched_to_the_chromium_worker():
    added: list = []
    p_latex, p_edit, latex, editorial = _patch_tasks()
    with p_latex, p_edit:
        async with _client(_user(cosmetics=True), added=added) as c:
            resp = await c.post(
                URL, params={"comparison_name": "KO_vs_WT", "template": "editorial"}
            )
    assert resp.status_code == 202, resp.text
    editorial.assert_called_once()
    latex.assert_not_called()
    assert added and added[0].template == "editorial"


@pytest.mark.asyncio
async def test_latex_stays_the_default():
    added: list = []
    p_latex, p_edit, latex, editorial = _patch_tasks()
    with p_latex, p_edit:
        async with _client(_user(cosmetics=False), added=added) as c:
            resp = await c.post(URL, params={"comparison_name": "KO_vs_WT"})
    assert resp.status_code == 202, resp.text
    latex.assert_called_once()
    editorial.assert_not_called()
    assert added[0].template == "latex"


@pytest.mark.asyncio
async def test_unknown_template_is_rejected():
    async with _client(_user(cosmetics=True)) as c:
        resp = await c.post(URL, params={"comparison_name": "KO_vs_WT", "template": "docx"})
    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["editorial", "latex"])
async def test_download_filename_carries_the_format(template):
    job = SimpleNamespace(pdf_path="reports/x.pdf", template=template)
    with patch(
        "app.services.storage.storage_service.download_file",
        new=AsyncMock(return_value=b"%PDF-1.7"),
    ):
        async with _client(_user(cosmetics=True), job=job) as c:
            resp = await c.get(
                f"{URL}/download", params={"comparison_name": "KO_vs_WT", "template": template}
            )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == (
        f'attachment; filename="report_KO_vs_WT_{template}.pdf"'
    )


@pytest.mark.asyncio
async def test_status_reports_the_format():
    from app.models.report_job import ReportJob, ReportJobStatus

    now = datetime.now(timezone.utc)
    job = ReportJob(
        id=uuid4(),
        project_id=TEST_PROJECT_ID,
        dataset_id=DATASET_ID,
        comparison_name="KO_vs_WT",
        template="editorial",
        status=ReportJobStatus.DONE,
        requested_by=uuid4(),
        pdf_path="p.pdf",
    )
    job.created_at = job.updated_at = now
    async with _client(_user(cosmetics=True), job=job) as c:
        resp = await c.get(
            f"{URL}/status", params={"comparison_name": "KO_vs_WT", "template": "editorial"}
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["template"] == "editorial"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ai_texts",
    [
        None,
        [],
        [
            {"kind": "nova", "model": "m-1", "generated_at": "2026-11-05T09:30:00+00:00"},
            {"kind": "comparison", "model": "m-2", "generated_at": "2026-11-04T08:00:00+00:00"},
        ],
    ],
)
async def test_status_returns_ai_texts(ai_texts):
    """The AI texts the report printed come back with the job (null: none recorded, [] : none)."""
    from app.models.report_job import ReportJob, ReportJobStatus

    now = datetime.now(timezone.utc)
    job = ReportJob(
        id=uuid4(),
        project_id=TEST_PROJECT_ID,
        dataset_id=DATASET_ID,
        comparison_name="KO_vs_WT",
        template="editorial",
        status=ReportJobStatus.DONE,
        requested_by=uuid4(),
        pdf_path="p.pdf",
        ai_texts=ai_texts,
    )
    job.created_at = job.updated_at = now
    async with _client(_user(cosmetics=True), job=job) as c:
        resp = await c.get(
            f"{URL}/status", params={"comparison_name": "KO_vs_WT", "template": "editorial"}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()["ai_texts"]
    if ai_texts is None:
        assert body is None
        return
    assert [(t["kind"], t["model"]) for t in body] == [(t["kind"], t["model"]) for t in ai_texts]
    assert [datetime.fromisoformat(t["generated_at"].replace("Z", "+00:00")) for t in body] == [
        datetime.fromisoformat(t["generated_at"]) for t in ai_texts
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["status", "download"])
async def test_a_job_id_must_be_a_uuid(route):
    async with _client(_user(cosmetics=True)) as c:
        resp = await c.get(
            f"{URL}/{route}", params={"comparison_name": "KO_vs_WT", "job_id": "not-a-uuid"}
        )
    assert resp.status_code == 422


def test_report_filename_is_sanitised():
    from app.api.endpoints.reports import report_filename

    assert report_filename("A/B vs C", "editorial") == "report_A_B_vs_C_editorial.pdf"
    assert report_filename("KO_vs_WT", None) == "report_KO_vs_WT_latex.pdf"
