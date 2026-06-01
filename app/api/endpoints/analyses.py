"""
Self-service analysis endpoints.
Allows users to launch DESeq2 / multi-method differential expression pipelines
from existing datasets (matrix + samples + comparisons).
"""
import json
import logging
from pathlib import Path
from typing import Annotated, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.license import require_active_license
from app.api.deps.subscription import get_or_create_user
from app.core.supabase_auth import SupabaseUser
from app.models.models import SelfServiceAnalysis, SelfServiceAnalysisStatus, User, Project, ProjectMember

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyses", tags=["analyses"])

# Path to the pre-generated anno_db categories JSON
_ANNO_DB_JSON = Path(__file__).parent.parent.parent.parent / "app" / "data" / "anno_db_categories.json"


# ---------------------------------------------------------------------------
# Pydantic schemas (inline — avoids circular import with schemas package)
# ---------------------------------------------------------------------------

class AnalysisParams(BaseModel):
    design: str = "auto"
    fdr: float = 0.05
    min_log2fc: float = 1.0
    min_reads: int = 10
    min_genes: int = 200
    min_count: int = 5
    min_reps: int = 2
    threads: int = 4
    enrichment_databases: Optional[List[str]] = None
    species: Optional[str] = "human"


class SelfServiceAnalysisCreate(BaseModel):
    project_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    matrix_dataset_id: UUID
    samples_dataset_id: UUID
    comparisons_dataset_id: UUID
    params: AnalysisParams = Field(default_factory=AnalysisParams)


class SelfServiceAnalysisResponse(BaseModel):
    id: UUID
    project_id: UUID
    name: str
    status: SelfServiceAnalysisStatus
    matrix_dataset_id: Optional[UUID]
    samples_dataset_id: Optional[UUID]
    comparisons_dataset_id: Optional[UUID]
    params: dict
    result_dataset_ids: list
    intermediate_dataset_ids: dict
    celery_task_id: Optional[str]
    current_step: Optional[str]
    progress_log: list
    error_message: Optional[str]
    user_id: UUID
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj: SelfServiceAnalysis) -> "SelfServiceAnalysisResponse":
        return cls(
            id=obj.id,
            project_id=obj.project_id,
            name=obj.name,
            status=obj.status,
            matrix_dataset_id=obj.matrix_dataset_id,
            samples_dataset_id=obj.samples_dataset_id,
            comparisons_dataset_id=obj.comparisons_dataset_id,
            params=obj.params or {},
            result_dataset_ids=obj.result_dataset_ids or [],
            intermediate_dataset_ids=obj.intermediate_dataset_ids or {},
            celery_task_id=obj.celery_task_id,
            current_step=obj.current_step,
            progress_log=obj.progress_log or [],
            error_message=obj.error_message,
            user_id=obj.user_id,
            created_at=obj.created_at.isoformat(),
            updated_at=obj.updated_at.isoformat(),
        )


class SelfServiceAnalysisListResponse(BaseModel):
    items: List[SelfServiceAnalysisResponse]
    total: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/anno-db-categories")
async def get_anno_db_categories(
    species: str = Query(..., description="Species name (e.g. 'human', 'mouse')"),
    _current_user: Annotated[SupabaseUser, Depends(get_current_user)] = None,
) -> dict:
    """Return available annotation DB categories for a given species."""
    if not _ANNO_DB_JSON.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Annotation DB catalogue not available on this server.",
        )
    try:
        data = json.loads(_ANNO_DB_JSON.read_text())
    except Exception as exc:
        logger.error("Failed to read anno_db_categories.json: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read annotation DB catalogue.")

    categories = data.get(species)
    if categories is None:
        available = [k for k in data.keys() if not k.startswith("_")]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Species '{species}' not found. Available: {available}",
        )
    return {"species": species, "categories": categories}


@router.get("", response_model=SelfServiceAnalysisListResponse)
async def list_analyses(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    project_id: UUID = Query(...),
) -> SelfServiceAnalysisListResponse:
    """List all analyses for a project (owner or member)."""
    # Verify project access: owner or project member
    proj = await db.scalar(
        select(Project).where(Project.id == project_id)
    )
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if proj.owner_id != current_user.user_id:
        member = await db.scalar(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == current_user.user_id,
            )
        )
        if not member:
            raise HTTPException(status_code=404, detail="Project not found")

    result = await db.execute(
        select(SelfServiceAnalysis)
        .where(SelfServiceAnalysis.project_id == project_id)
        .order_by(SelfServiceAnalysis.created_at.desc())
    )
    items = result.scalars().all()
    return SelfServiceAnalysisListResponse(
        items=[SelfServiceAnalysisResponse.from_orm(a) for a in items],
        total=len(items),
    )


@router.get("/{analysis_id}", response_model=SelfServiceAnalysisResponse)
async def get_analysis(
    analysis_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> SelfServiceAnalysisResponse:
    """Get a single analysis by ID."""
    analysis = await db.scalar(
        select(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id)
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return SelfServiceAnalysisResponse.from_orm(analysis)


@router.post("", response_model=SelfServiceAnalysisResponse, status_code=201, dependencies=[Depends(require_active_license)])
async def create_analysis(
    payload: SelfServiceAnalysisCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db_user: Annotated[User, Depends(get_or_create_user)],
) -> SelfServiceAnalysisResponse:
    """Create and launch a self-service analysis job."""
    # Verify project ownership
    proj = await db.scalar(
        select(Project).where(
            Project.id == payload.project_id,
            Project.owner_id == current_user.user_id,
        )
    )
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    analysis = SelfServiceAnalysis(
        id=uuid4(),
        project_id=payload.project_id,
        name=payload.name,
        status=SelfServiceAnalysisStatus.PENDING,
        matrix_dataset_id=payload.matrix_dataset_id,
        samples_dataset_id=payload.samples_dataset_id,
        comparisons_dataset_id=payload.comparisons_dataset_id,
        params=payload.params.model_dump(),
        result_dataset_ids=[],
        intermediate_dataset_ids={},
        progress_log=[],
        user_id=current_user.user_id,
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    # Dispatch Celery task
    try:
        from app.worker.tasks import run_self_service_analysis
        task = run_self_service_analysis.delay(str(analysis.id))
        await db.execute(
            update(SelfServiceAnalysis)
            .where(SelfServiceAnalysis.id == analysis.id)
            .values(celery_task_id=task.id)
        )
        await db.commit()
        await db.refresh(analysis)
        logger.info("[ANALYSES] Dispatched task %s for analysis %s", task.id, analysis.id)
    except Exception as exc:
        # Task dispatch failed — mark as failed but don't lose the record
        logger.error("[ANALYSES] Failed to dispatch Celery task: %s", exc)
        await db.execute(
            update(SelfServiceAnalysis)
            .where(SelfServiceAnalysis.id == analysis.id)
            .values(
                status=SelfServiceAnalysisStatus.FAILED,
                error_message=f"Failed to dispatch analysis task: {exc}",
            )
        )
        await db.commit()
        await db.refresh(analysis)

    return SelfServiceAnalysisResponse.from_orm(analysis)


@router.delete("/{analysis_id}", status_code=204)
async def delete_analysis(
    analysis_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> None:
    """Delete an analysis record (cancels if still running)."""
    analysis = await db.scalar(
        select(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id)
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Attempt to revoke the Celery task if still pending/running
    if analysis.celery_task_id and analysis.status in (
        SelfServiceAnalysisStatus.PENDING,
        SelfServiceAnalysisStatus.RUNNING,
    ):
        try:
            from app.worker.celery_app import celery_app
            celery_app.control.revoke(analysis.celery_task_id, terminate=True)
        except Exception as exc:
            logger.warning("[ANALYSES] Could not revoke task %s: %s", analysis.celery_task_id, exc)

    await db.execute(delete(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id))
    await db.commit()
