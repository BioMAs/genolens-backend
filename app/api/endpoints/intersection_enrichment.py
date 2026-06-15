"""
Ad-hoc functional enrichment of a Venn/UpSet intersection (annoDB, async).

POST /datasets/{path_dataset_id}/intersection-enrichment  — trigger a job
GET  /intersection-enrichment/{job_id}                    — poll status / results

The selected intersection's gene symbols are enriched on the r-worker via
functional_enrichment.R --gene-list, mirroring the report-job pattern.
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
from app.models.models import Dataset, Project, ProjectMember, SelfServiceAnalysis
from app.models.intersection_enrichment_job import (
    IntersectionEnrichmentJob,
    IntersectionEnrichmentStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["intersection-enrichment"])

MAX_GENES = 20000


class EnrichmentTriggerResponse(BaseModel):
    job_id: UUID
    status: IntersectionEnrichmentStatus


class EnrichmentJobResponse(BaseModel):
    job_id: UUID
    status: IntersectionEnrichmentStatus
    label: Optional[str] = None
    gene_count: Optional[int] = None
    species: Optional[str] = None
    result: Optional[list] = None
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


async def _resolve_species(db: AsyncSession, dataset: Dataset) -> str:
    """Best-effort species for the dataset's project (defaults to human)."""
    meta = dataset.dataset_metadata or {}
    if meta.get("species"):
        return str(meta["species"])
    analysis_id = meta.get("analysis_id")
    if analysis_id:
        try:
            analysis = await db.get(SelfServiceAnalysis, UUID(str(analysis_id)))
            if analysis and (analysis.params or {}).get("species"):
                return str(analysis.params["species"])
        except (ValueError, TypeError):
            pass
    return "human"


@router.post(
    "/datasets/{path_dataset_id}/intersection-enrichment",
    response_model=EnrichmentTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_active_license)],
)
async def trigger_intersection_enrichment(
    path_dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    genes: list[str] = Body(..., description="Gene symbols in the selected intersection"),
    label: str = Body("intersection", description="Display label of the intersection"),
    species: Optional[str] = Body(None, description="Override species (else resolved)"),
) -> EnrichmentTriggerResponse:
    genes = [g.strip() for g in genes if isinstance(g, str) and g.strip()]
    if len(genes) < 1:
        raise HTTPException(status_code=400, detail="No genes provided")
    if len(genes) > MAX_GENES:
        raise HTTPException(status_code=400, detail=f"Too many genes (max {MAX_GENES})")

    result = await db.execute(
        select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == path_dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await _assert_project_access(db, dataset.project, current_user)

    resolved_species = species or await _resolve_species(db, dataset)

    job = IntersectionEnrichmentJob(
        project_id=dataset.project_id,
        status=IntersectionEnrichmentStatus.PENDING,
        params={
            "label": label,
            "gene_count": len(genes),
            "species": resolved_species,
            "genes": genes,
        },
        requested_by=current_user.user_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from app.worker.tasks.intersection_enrichment_task import run_intersection_enrichment

        task = run_intersection_enrichment.apply_async(args=[str(job.id)], queue="r_analysis")
        job.celery_task_id = task.id
        await db.commit()
    except Exception as exc:  # pragma: no cover - broker failure path
        job.status = IntersectionEnrichmentStatus.FAILED
        job.error_message = f"Failed to queue enrichment: {exc}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to queue enrichment",
        )

    return EnrichmentTriggerResponse(job_id=job.id, status=job.status)


@router.get(
    "/intersection-enrichment/{job_id}",
    response_model=EnrichmentJobResponse,
)
async def get_intersection_enrichment(
    job_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> EnrichmentJobResponse:
    job = await db.get(IntersectionEnrichmentJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Enrichment job not found")

    project = await db.get(Project, job.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await _assert_project_access(db, project, current_user)

    params = job.params or {}
    return EnrichmentJobResponse(
        job_id=job.id,
        status=job.status,
        label=params.get("label"),
        gene_count=params.get("gene_count"),
        species=params.get("species"),
        result=job.result,
        error_message=job.error_message,
    )
