"""
The AI texts a report really printed are recorded on its job (`report_jobs.ai_texts`).

One record per text printed, `{kind, model, generated_at}`; `[]` when the job asked for no AI. The
renderers fill a list they are handed, the Celery task stores it on the job with the PDF.
No LLM, no pdflatex, no Chromium: the heavy steps are replaced.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.models.report_job import ReportJob, ReportJobStatus
from app.services.report_latex_service import report_latex_service
from app.worker.tasks import report_task

DAY = datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc)
RECORD = {"kind": "comparison", "model": "org/model_v1", "generated_at": DAY.isoformat()}


def make_job(template: str = "latex", use_ai: bool = True, analysis: bool = False) -> ReportJob:
    return ReportJob(
        id=uuid4(),
        project_id=uuid4(),
        analysis_id=uuid4() if analysis else None,
        dataset_id=None if analysis else uuid4(),
        comparison_name=None if analysis else "AL-7_vs_Vehicle",
        template=template,
        use_ai=use_ai,
        status=ReportJobStatus.PENDING,
        requested_by=uuid4(),
    )


def latex_data(include_ai: bool) -> dict:
    comp = {
        "comp_id": "A_vs_B",
        "safe_id": "A-vs-B",
        "test": "A",
        "control": "B",
        "n_up": 1,
        "n_down": 0,
        "degs": [],
        "enr_up": [],
        "enr_down": [],
    }
    if include_ai:  # what the collector hands over when the job asks for AI and a text is current
        comp.update(ai_interpretation="Reading A.", ai_model="org/model_v1", ai_generated_at=DAY)
    return {"comparisons": [comp], "project": None, "analysis": None, "params": {}, "qc_rows": []}


def latex_steps(seen: list):
    """The LaTeX renderers with the database and pdflatex taken out."""

    async def collect(*args, include_ai=True, **kwargs):
        seen.append(include_ai)
        return latex_data(include_ai)

    return (
        patch.object(report_latex_service, "collect_comparison_data", new=collect),
        patch.object(report_latex_service, "collect_analysis_data", new=collect),
        patch.object(report_latex_service, "_load_branding", new=AsyncMock(return_value=None)),
        patch.object(report_latex_service, "is_demo_requester", new=AsyncMock(return_value=False)),
        patch.object(report_latex_service, "assemble_tex", new=MagicMock(return_value="tex")),
        patch.object(report_latex_service, "compile_pdf", new=MagicMock(return_value=b"%PDF-1.7")),
    )


async def run_latex(render, job):
    seen: list = []
    steps = latex_steps(seen)
    for step in steps:
        step.start()
    try:
        return await render(MagicMock(), job), seen
    finally:
        for step in steps:
            step.stop()


async def test_report_job_records_ai_texts_for_the_latex_comparison_report():
    job = make_job("latex", use_ai=True)
    pdf, seen = await run_latex(report_task._render_latex, job)
    assert pdf == b"%PDF-1.7"
    assert job.ai_texts == [RECORD]
    assert seen == [True]


async def test_report_job_records_no_ai_text_when_use_ai_is_false():
    job = make_job("latex", use_ai=False)
    _pdf, seen = await run_latex(report_task._render_latex, job)
    assert job.ai_texts == []
    assert seen == [False]


def editorial_inputs(**overrides):
    from tests.test_report_editorial import GENERATED_AT, _dated

    return _dated(**overrides), GENERATED_AT


async def run_editorial(job, inputs):
    from app.services import report_editorial_service as mod

    with (
        patch.object(
            mod.report_editorial_service, "collect", new=AsyncMock(return_value=inputs)
        ) as collect,
        patch.object(mod, "html_to_pdf", new=AsyncMock(return_value=(b"%PDF-1.7", []))),
        patch.object(report_latex_service, "is_demo_requester", new=AsyncMock(return_value=False)),
    ):
        pdf = await report_task._render_editorial(MagicMock(), job)
    return pdf, collect


async def test_report_job_records_ai_texts_for_the_editorial_report():
    from tests.test_report_editorial import FRENCH_AI

    job = make_job("editorial", use_ai=True)
    inputs, day = editorial_inputs()
    pdf, collect = await run_editorial(job, inputs)
    assert pdf == b"%PDF-1.7"
    assert collect.await_args.kwargs["use_ai"] is True
    model = "google/gemma-4-E4B-it"
    assert job.ai_texts == [
        {"kind": "nova", "model": model, "generated_at": day.isoformat()},
        {"kind": "comparison", "model": model, "generated_at": day.isoformat()},
    ]

    # A text the report does not print (here a French comparison text) is not recorded.
    job = make_job("editorial", use_ai=True)
    inputs, _ = editorial_inputs(comparison_ai=FRENCH_AI)
    await run_editorial(job, inputs)
    assert [t["kind"] for t in job.ai_texts] == ["nova"]


async def test_report_job_records_no_ai_text_for_an_editorial_report_without_ai():
    job = make_job("editorial", use_ai=False)
    # Even a collector that handed texts over would see none of them printed.
    inputs, _ = editorial_inputs(use_ai=False)
    _pdf, collect = await run_editorial(job, inputs)
    assert collect.await_args.kwargs["use_ai"] is False
    assert job.ai_texts == []


# ── The task lifecycle ──────────────────────────────────────────────────────


def session_of(job):
    @asynccontextmanager
    async def fake():
        db = MagicMock()
        db.get = AsyncMock(return_value=job)
        db.commit = AsyncMock()
        yield db

    return fake


async def test_the_task_stores_ai_texts_with_the_pdf():
    job = make_job("latex", use_ai=True)

    async def render(db, j):
        j.ai_texts = [RECORD]
        return b"%PDF-1.7"

    with (
        patch.object(report_task, "_fresh_session", new=session_of(job)),
        patch("app.services.storage.storage_service.upload_file", new=AsyncMock()),
    ):
        out = await report_task._run_comparison_job(str(job.id), render)
    assert out["status"] == "done"
    assert job.status == ReportJobStatus.DONE and job.ai_texts == [RECORD]


async def test_a_failed_job_records_no_ai_text():
    job = make_job("latex", use_ai=True)

    async def render(db, j):
        j.ai_texts = [RECORD]
        return b"%PDF-1.7"

    with (
        patch.object(report_task, "_fresh_session", new=session_of(job)),
        patch(
            "app.services.storage.storage_service.upload_file",
            new=AsyncMock(side_effect=RuntimeError("storage down")),
        ),
    ):
        out = await report_task._run_comparison_job(str(job.id), render)
    assert out["status"] == "failed"
    assert job.status == ReportJobStatus.FAILED and job.ai_texts is None


@pytest.mark.parametrize("use_ai", [True, False])
async def test_the_analysis_report_follows_use_ai_and_records_its_texts(use_ai):
    job = make_job(use_ai=use_ai, analysis=True)
    seen: list = []
    steps = latex_steps(seen)
    for step in steps:
        step.start()
    try:
        with (
            patch.object(report_task, "_fresh_session", new=session_of(job)),
            patch("app.services.storage.storage_service.upload_file", new=AsyncMock()),
        ):
            out = await report_task._async_generate(None, str(job.id), str(job.analysis_id))
    finally:
        for step in steps:
            step.stop()
    assert out["status"] == "done"
    assert seen == [use_ai]
    assert job.ai_texts == ([RECORD] if use_ai else [])
