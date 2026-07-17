"""
Asynchronous pre-ranked GSEA for a comparison.

POST /datasets/{dataset_id}/gsea-async  — trigger a job
GET  /gsea-jobs/{job_id}                — poll status / results

GSEA over a full gene-set database with permutation FDR takes minutes, exceeding
HTTP timeouts, so it runs as a Celery task (pure Python, default queue), mirroring
the intersection-enrichment job pattern.
"""
import logging
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.api.deps import get_current_user, get_db
from app.api.deps.license import require_active_license
from app.core.supabase_auth import SupabaseUser
from app.models.models import Dataset, Project, ProjectMember
from app.models.gsea_job import GSEAJob, GSEAJobStatus

logger = logging.getLogger(__name__)
router = APIRouter(tags=["gsea"])


class GSEATriggerResponse(BaseModel):
    job_id: UUID
    status: GSEAJobStatus


class GSEAJobResponse(BaseModel):
    job_id: UUID
    status: GSEAJobStatus
    comparison_name: Optional[str] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None


async def _assert_project_access(db: AsyncSession, project: Project, user: SupabaseUser) -> None:
    if project.owner_id == user.user_id:
        return
    member = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == user.user_id,
        )
    )
    if member.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Access denied")


@router.post(
    "/datasets/{dataset_id}/gsea-async",
    response_model=GSEATriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_active_license)],
)
async def trigger_gsea(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    comparison_name: str = Body(...),
    gene_set_database: str = Body("GO_BP"),
    ranking_metric: str = Body("signed_pvalue"),
    min_size: int = Body(15),
    max_size: int = Body(500),
    n_permutations: int = Body(1000),
    fdr_threshold: float = Body(0.25),
) -> GSEATriggerResponse:
    result = await db.execute(
        select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await _assert_project_access(db, dataset.project, current_user)

    job = GSEAJob(
        dataset_id=dataset.id,
        project_id=dataset.project_id,
        status=GSEAJobStatus.PENDING,
        params={
            "comparison_name": comparison_name,
            "gene_set_database": gene_set_database,
            "ranking_metric": ranking_metric,
            "min_size": min_size,
            "max_size": max_size,
            "n_permutations": n_permutations,
            "fdr_threshold": fdr_threshold,
        },
        requested_by=current_user.user_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from app.worker.tasks.gsea_task import run_gsea_job

        task = run_gsea_job.apply_async(args=[str(job.id)], queue="default")
        job.celery_task_id = task.id
        await db.commit()
    except Exception as exc:  # pragma: no cover - broker failure path
        job.status = GSEAJobStatus.FAILED
        job.error_message = f"Failed to queue GSEA: {exc}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to queue GSEA",
        )

    return GSEATriggerResponse(job_id=job.id, status=job.status)


@router.get("/gsea-jobs/{job_id}", response_model=GSEAJobResponse)
async def get_gsea_job(
    job_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> GSEAJobResponse:
    job = await db.get(GSEAJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="GSEA job not found")

    project = await db.get(Project, job.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await _assert_project_access(db, project, current_user)

    return GSEAJobResponse(
        job_id=job.id,
        status=job.status,
        comparison_name=(job.params or {}).get("comparison_name"),
        result=job.result,
        error_message=job.error_message,
    )
