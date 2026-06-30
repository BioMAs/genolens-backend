"""
Report generation endpoints (analysis-scoped SciLicium LaTeX PDF).

POST   /analyses/{analysis_id}/report          — trigger background generation
GET    /analyses/{analysis_id}/report/status   — poll job status
GET    /analyses/{analysis_id}/report/download  — download generated PDF
"""
import logging
from typing import Annotated, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.auth import get_current_user
from app.api.deps.db import get_db
from app.api.deps.subscription import get_or_create_user
from app.core.security import CurrentUser
from app.models.models import Dataset, SelfServiceAnalysis, User
from app.models.report_job import ReportJob, ReportJobStatus
from app.schemas.report import (
    ComparisonReportTriggerRequest,
    ReportJobResponse,
    ReportTriggerResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyses", tags=["reports"])

# Comparison-scoped report routes (the new primary scope). A comparison is the
# logical pair (DEG dataset, comparison_name); comparison_name is a query param
# because it may contain characters unsafe in a path segment.
comparison_router = APIRouter(prefix="/datasets", tags=["reports"])


async def _get_analysis_or_404(analysis_id: UUID, db: AsyncSession) -> SelfServiceAnalysis:
    analysis = await db.get(SelfServiceAnalysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.post(
    "/{analysis_id}/report",
    response_model=ReportTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_report(
    analysis_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Trigger background SciLicium PDF report generation for an analysis."""
    analysis = await _get_analysis_or_404(analysis_id, db)

    # Return existing in-progress job rather than spawning a duplicate
    stmt = select(ReportJob).where(
        ReportJob.analysis_id == analysis_id,
        ReportJob.status.in_([ReportJobStatus.PENDING, ReportJobStatus.RUNNING]),
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        return ReportTriggerResponse(
            job_id=existing.id, status=existing.status,
            message="Report generation already in progress.",
        )

    job = ReportJob(
        id=uuid4(),
        project_id=analysis.project_id,
        analysis_id=analysis_id,
        requested_by=current_user.id,
        status=ReportJobStatus.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from app.worker.tasks.report_task import generate_analysis_report
        task = generate_analysis_report.delay(str(job.id), str(analysis_id))
        job.celery_task_id = task.id
        await db.commit()
    except Exception as exc:
        logger.exception("Failed to dispatch report task for job %s", job.id)
        job.status = ReportJobStatus.FAILED
        job.error_message = f"Failed to queue report generation: {exc}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report generation service is unavailable. Please try again later.",
        )

    return ReportTriggerResponse(
        job_id=job.id, status=job.status, message="Report generation started.",
    )


@router.get("/{analysis_id}/report/status", response_model=ReportJobResponse)
async def report_status(
    analysis_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Get the status of the latest report job for an analysis."""
    stmt = (
        select(ReportJob)
        .where(ReportJob.analysis_id == analysis_id)
        .order_by(desc(ReportJob.created_at))
        .limit(1)
    )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="No report job found for this analysis")
    return job


@router.get("/{analysis_id}/report/download")
async def download_report(
    analysis_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Download the generated PDF report for an analysis."""
    stmt = (
        select(ReportJob)
        .where(ReportJob.analysis_id == analysis_id, ReportJob.status == ReportJobStatus.DONE)
        .order_by(desc(ReportJob.created_at))
        .limit(1)
    )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job or not job.pdf_path:
        raise HTTPException(status_code=404, detail="No completed report available")

    from app.services.storage import storage_service
    pdf_bytes = await storage_service.download_file(job.pdf_path)

    analysis = await db.get(SelfServiceAnalysis, analysis_id)
    name = (analysis.name if analysis else str(analysis_id)) or str(analysis_id)
    filename = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"report_{name}.pdf")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


# ── Comparison-scoped report ────────────────────────────────────────────────────

async def _get_dataset_or_404(dataset_id: UUID, db: AsyncSession) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@comparison_router.post(
    "/{dataset_id}/report",
    response_model=ReportTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_comparison_report(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_or_create_user)],
    comparison_name: Annotated[str, Query(...)],
    payload: Annotated[Optional[ComparisonReportTriggerRequest], Body()] = None,
):
    """Trigger background PDF report generation for a single comparison.

    Per-report customization content (conclusion / materials & methods) is only
    applied when the requesting user has the report customization module.
    """
    dataset = await _get_dataset_or_404(dataset_id, db)

    # Return existing in-progress job rather than spawning a duplicate
    stmt = select(ReportJob).where(
        ReportJob.dataset_id == dataset_id,
        ReportJob.comparison_name == comparison_name,
        ReportJob.status.in_([ReportJobStatus.PENDING, ReportJobStatus.RUNNING]),
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        return ReportTriggerResponse(
            job_id=existing.id, status=existing.status,
            message="Report generation already in progress.",
        )

    conclusion = None
    materials_methods = None
    first_page_type = None
    last_page_type = None
    cover_info = None
    if payload and user.has_report_customization:
        conclusion = payload.conclusion
        materials_methods = payload.materials_methods
        first_page_type = payload.first_page_type
        last_page_type = payload.last_page_type
        cover_info = payload.cover_info.model_dump(exclude_none=True) if payload.cover_info else None

    job = ReportJob(
        id=uuid4(),
        project_id=dataset.project_id,
        dataset_id=dataset_id,
        comparison_name=comparison_name,
        conclusion=conclusion,
        materials_methods=materials_methods,
        first_page_type=first_page_type,
        last_page_type=last_page_type,
        cover_info=cover_info,
        requested_by=user.id,
        status=ReportJobStatus.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from app.worker.tasks.report_task import generate_comparison_report
        task = generate_comparison_report.delay(str(job.id))
        job.celery_task_id = task.id
        await db.commit()
    except Exception as exc:
        logger.exception("Failed to dispatch comparison report task for job %s", job.id)
        job.status = ReportJobStatus.FAILED
        job.error_message = f"Failed to queue report generation: {exc}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report generation service is unavailable. Please try again later.",
        )

    return ReportTriggerResponse(
        job_id=job.id, status=job.status, message="Report generation started.",
    )


@comparison_router.get("/{dataset_id}/report/status", response_model=ReportJobResponse)
async def comparison_report_status(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    comparison_name: Annotated[str, Query(...)],
):
    """Get the status of the latest report job for a comparison."""
    stmt = (
        select(ReportJob)
        .where(
            ReportJob.dataset_id == dataset_id,
            ReportJob.comparison_name == comparison_name,
        )
        .order_by(desc(ReportJob.created_at))
        .limit(1)
    )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="No report job found for this comparison")
    return job


@comparison_router.get("/{dataset_id}/report/download")
async def download_comparison_report(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    comparison_name: Annotated[str, Query(...)],
):
    """Download the generated PDF report for a comparison."""
    stmt = (
        select(ReportJob)
        .where(
            ReportJob.dataset_id == dataset_id,
            ReportJob.comparison_name == comparison_name,
            ReportJob.status == ReportJobStatus.DONE,
        )
        .order_by(desc(ReportJob.created_at))
        .limit(1)
    )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job or not job.pdf_path:
        raise HTTPException(status_code=404, detail="No completed report available")

    from app.services.storage import storage_service
    pdf_bytes = await storage_service.download_file(job.pdf_path)

    filename = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in f"report_{comparison_name}.pdf"
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
