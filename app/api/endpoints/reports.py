"""
Report generation endpoints.

POST   /projects/{project_id}/report          — trigger background generation
GET    /projects/{project_id}/report/status   — poll job status
GET    /projects/{project_id}/report/download — download generated PDF
"""
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.auth import get_current_user
from app.api.deps.db import get_db
from app.core.supabase_auth import SupabaseUser
from app.models.models import Project
from app.models.report_job import ReportJob, ReportJobStatus
from app.schemas.report import ReportJobResponse, ReportTriggerResponse

router = APIRouter(prefix="/projects", tags=["reports"])


async def _get_project_or_404(project_id: UUID, db: AsyncSession) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post(
    "/{project_id}/report",
    response_model=ReportTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_report(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
):
    """Trigger background PDF report generation for a project."""
    await _get_project_or_404(project_id, db)

    # Return existing in-progress job rather than spawning a duplicate
    stmt = select(ReportJob).where(
        ReportJob.project_id == project_id,
        ReportJob.status.in_([ReportJobStatus.PENDING, ReportJobStatus.RUNNING]),
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        return ReportTriggerResponse(
            job_id=existing.id,
            status=existing.status,
            message="Report generation already in progress.",
        )

    job = ReportJob(
        id=uuid4(),
        project_id=project_id,
        requested_by=UUID(str(current_user.user_id)),
        status=ReportJobStatus.PENDING,
    )
    db.add(job)
    await db.flush()

    from app.worker.tasks.report_task import generate_project_report
    task = generate_project_report.delay(str(job.id), str(project_id))
    job.celery_task_id = task.id
    await db.commit()

    return ReportTriggerResponse(
        job_id=job.id,
        status=job.status,
        message="Report generation started.",
    )


@router.get("/{project_id}/report/status", response_model=ReportJobResponse)
async def report_status(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
):
    """Get the status of the latest report generation job for a project."""
    stmt = (
        select(ReportJob)
        .where(ReportJob.project_id == project_id)
        .order_by(desc(ReportJob.created_at))
        .limit(1)
    )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="No report job found for this project")
    return job


@router.get("/{project_id}/report/download")
async def download_report(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
):
    """Download the generated PDF report for a project."""
    stmt = (
        select(ReportJob)
        .where(
            ReportJob.project_id == project_id,
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

    project = await db.get(Project, project_id)
    filename = f"report_{project.name if project else project_id}.pdf"
    filename = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
