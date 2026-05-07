"""
Analyses API endpoints — self-service multi-method DE analysis.
"""
import json
import logging
from typing import Optional
from uuid import UUID, uuid4

from celery import states as celery_states
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.supabase_auth import SupabaseUser
from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    Project,
    ProjectMember,
    SelfServiceAnalysis,
    SelfServiceAnalysisStatus,
)
from app.schemas.analysis import (
    SelfServiceAnalysisCreate,
    SelfServiceAnalysisListResponse,
    SelfServiceAnalysisResponse,
    SelfServiceAnalysisUploadCreate,
)
from app.services.anno_db_service import SUPPORTED_SPECIES, get_categories, resolve_species
from app.services.storage import storage_service
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyses", tags=["analyses"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_project_or_403(
    project_id: UUID,
    user_id: str,
    db: AsyncSession,
) -> Project:
    """Return the project if the current user is owner or member, else 403."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project.owner_id) == str(user_id):
        return project

    member = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == str(user_id),
        )
    )
    if not member.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Access denied to this project")
    return project


async def _get_analysis_or_404(
    analysis_id: UUID,
    user_id: str,
    db: AsyncSession,
) -> SelfServiceAnalysis:
    """Return the analysis if owned by user, else 404."""
    result = await db.execute(
        select(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Verify user has access to the parent project
    await _get_project_or_403(analysis.project_id, user_id, db)
    return analysis


# ---------------------------------------------------------------------------
# POST /analyses — Create and launch a new analysis
# ---------------------------------------------------------------------------


@router.post("", response_model=SelfServiceAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    payload: SelfServiceAnalysisCreate,
    current_user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SelfServiceAnalysisResponse:
    """
    Create a self-service DESeq2 analysis and dispatch it to the r-worker.

    The analysis starts with status=PENDING. Poll GET /analyses/{id} for progress.
    """
    await _get_project_or_403(payload.project_id, current_user.user_id, db)

    analysis = SelfServiceAnalysis(
        project_id=payload.project_id,
        name=payload.name,
        user_id=current_user.user_id,
        status=SelfServiceAnalysisStatus.PENDING,
        matrix_dataset_id=payload.matrix_dataset_id,
        samples_dataset_id=payload.samples_dataset_id,
        comparisons_dataset_id=payload.comparisons_dataset_id,
        params=payload.params.model_dump(),
        result_dataset_ids=[],
        progress_log=[],
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    # Dispatch to r_analysis queue
    from app.worker.tasks import run_self_service_analysis  # noqa: PLC0415

    task = run_self_service_analysis.apply_async(
        args=[str(analysis.id)],
        queue="r_analysis",
    )

    analysis.celery_task_id = task.id
    await db.commit()
    await db.refresh(analysis)

    logger.info(f"Dispatched analysis {analysis.id} → Celery task {task.id}")
    return SelfServiceAnalysisResponse.model_validate(analysis)


# ---------------------------------------------------------------------------
# GET /analyses — List analyses for a project
# ---------------------------------------------------------------------------


@router.get("", response_model=SelfServiceAnalysisListResponse)
async def list_analyses(
    project_id: UUID = Query(..., description="Filter by project"),
    current_user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SelfServiceAnalysisListResponse:
    """Return all analyses for a project ordered by creation date (newest first)."""
    await _get_project_or_403(project_id, current_user.user_id, db)

    result = await db.execute(
        select(SelfServiceAnalysis)
        .where(SelfServiceAnalysis.project_id == project_id)
        .order_by(SelfServiceAnalysis.created_at.desc())
    )
    items = result.scalars().all()
    return SelfServiceAnalysisListResponse(
        items=[SelfServiceAnalysisResponse.model_validate(a) for a in items],
        total=len(items),
    )


# ---------------------------------------------------------------------------
# GET /analyses/anno-db-categories — List available annotation databases
# IMPORTANT: must be declared BEFORE /{analysis_id} routes
# ---------------------------------------------------------------------------


@router.get("/anno-db-categories")
async def get_anno_db_categories(
    species: str = Query(..., description="Species key: human, mouse, rat, zebrafish, pig"),
    current_user: SupabaseUser = Depends(get_current_user),
) -> dict:
    """
    Return the annotation database categories available in the anno.db for a given species.

    Used by the frontend to populate the enrichment database selector (advanced mode).
    """
    canonical = resolve_species(species)
    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported species '{species}'. Supported species: {', '.join(SUPPORTED_SPECIES.keys())}",
        )
    categories = get_categories(species)
    return {"species": canonical, "categories": categories}


# ---------------------------------------------------------------------------
# GET /analyses/{analysis_id} — Get one analysis
# ---------------------------------------------------------------------------


@router.get("/{analysis_id}", response_model=SelfServiceAnalysisResponse)
async def get_analysis(
    analysis_id: UUID,
    current_user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SelfServiceAnalysisResponse:
    """Return a single analysis with its current status and progress."""
    analysis = await _get_analysis_or_404(analysis_id, current_user.user_id, db)
    return SelfServiceAnalysisResponse.model_validate(analysis)


# ---------------------------------------------------------------------------
# DELETE /analyses/{analysis_id} — Cancel / delete
# ---------------------------------------------------------------------------


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: UUID,
    current_user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Cancel a running analysis or delete a finished one.

    - If PENDING or RUNNING: revoke the Celery task and set status=CANCELLED.
    - If DONE or FAILED: remove the record (result datasets are preserved).
    """
    analysis = await _get_analysis_or_404(analysis_id, current_user.user_id, db)

    if analysis.status in (SelfServiceAnalysisStatus.PENDING, SelfServiceAnalysisStatus.RUNNING):
        if analysis.celery_task_id:
            try:
                celery_app.control.revoke(analysis.celery_task_id, terminate=True)
                logger.info(f"Revoked Celery task {analysis.celery_task_id}")
            except Exception as exc:
                logger.warning(f"Could not revoke task {analysis.celery_task_id}: {exc}")

        analysis.status = SelfServiceAnalysisStatus.CANCELLED
        analysis.current_step = "cancelled"
        await db.commit()
    else:
        await db.delete(analysis)
        await db.commit()


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/analyses/upload — Create analysis from file uploads
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}


@router.post(
    "/upload",
    response_model=SelfServiceAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_analysis_from_upload(
    project_id: UUID = Form(...),
    name: str = Form(...),
    params_json: str = Form("{}"),
    counts_file: UploadFile = File(...),
    samples_file: UploadFile = File(...),
    comparisons_file: UploadFile = File(...),
    current_user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SelfServiceAnalysisResponse:
    """
    Create a multi-method DE analysis by uploading count matrix, sample metadata
    and comparisons files directly (no pre-existing datasets required).

    Files are immediately converted to Parquet and stored, then the analysis is
    dispatched to the r-worker queue.
    """
    from pathlib import Path

    from app.schemas.analysis import AnalysisParams
    from app.services.data_processor import data_processor

    await _get_project_or_403(project_id, current_user.user_id, db)

    # Parse analysis params from JSON string
    try:
        params_dict = json.loads(params_json) if params_json.strip() else {}
        params = AnalysisParams(**params_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid params_json: {exc}") from exc

    # Validate name
    import re
    if not re.match(r"^[a-zA-Z0-9\s_\-()\.,]+$", name):
        raise HTTPException(status_code=422, detail="Name contains invalid characters")
    name = name.strip()

    # Helper: upload one file as a ready Dataset
    async def _ingest_file(
        upload: UploadFile, dtype: DatasetType, label: str
    ) -> UUID:
        ext = Path(upload.filename or "").suffix.lower() or ".tsv"
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File '{upload.filename}' has unsupported extension '{ext}'. "
                       f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            )
        raw_bytes = await upload.read()
        parquet_bytes = await data_processor.convert_to_parquet(raw_bytes, ext)

        dataset_id = uuid4()
        raw_path = (
            f"projects/{project_id}/datasets/{dataset_id}/raw/{upload.filename}"
        )
        parquet_path = (
            f"projects/{project_id}/datasets/{dataset_id}/processed/data.parquet"
        )
        await storage_service.upload_file(raw_path, raw_bytes, content_type="text/plain")
        await storage_service.upload_file(
            parquet_path, parquet_bytes, content_type="application/octet-stream"
        )

        import pandas as pd
        import io as _io
        df = pd.read_parquet(_io.BytesIO(parquet_bytes))

        dataset = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=f"{name} — {label}",
            description=f"Auto-uploaded for analysis '{name}'",
            type=dtype,
            status=DatasetStatus.READY,
            raw_file_path=raw_path,
            parquet_file_path=parquet_path,
            dataset_metadata={
                "original_filename": upload.filename,
                "generated_by": "analysis_upload",
            },
            total_genes=len(df),
        )
        db.add(dataset)
        await db.flush()
        return dataset_id

    matrix_id = await _ingest_file(counts_file,      DatasetType.MATRIX,           "counts")
    samples_id = await _ingest_file(samples_file,    DatasetType.METADATA_SAMPLE,  "samples")
    comps_id   = await _ingest_file(comparisons_file, DatasetType.METADATA_CONTRAST, "comparisons")
    await db.commit()

    # Create the analysis record
    analysis = SelfServiceAnalysis(
        project_id=project_id,
        name=name,
        user_id=current_user.user_id,
        status=SelfServiceAnalysisStatus.PENDING,
        matrix_dataset_id=matrix_id,
        samples_dataset_id=samples_id,
        comparisons_dataset_id=comps_id,
        params=params.model_dump(),
        result_dataset_ids=[],
        progress_log=[],
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    # Dispatch to r_analysis queue
    from app.worker.tasks import run_self_service_analysis  # noqa: PLC0415

    task = run_self_service_analysis.apply_async(
        args=[str(analysis.id)],
        queue="r_analysis",
    )
    analysis.celery_task_id = task.id
    await db.commit()
    await db.refresh(analysis)

    logger.info(f"Dispatched upload-analysis {analysis.id} → Celery task {task.id}")
    return SelfServiceAnalysisResponse.model_validate(analysis)


# ---------------------------------------------------------------------------
# GET /analyses/{analysis_id}/comparisons/{comparison_id}/legacy-enrichment
# ---------------------------------------------------------------------------


@router.get("/{analysis_id}/comparisons/{comparison_id}/legacy-enrichment")
async def get_legacy_enrichment(
    analysis_id: UUID,
    comparison_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    category: Optional[str] = Query(None, description="Filter by annotation category"),
    current_user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Return paginated legacy functional enrichment results for a comparison.

    Results are read from Supabase Storage (TSV file whose path is stored in the
    DEG dataset metadata under ``enrichment_legacy_path``).
    """
    import io

    import pandas as pd

    analysis = await _get_analysis_or_404(analysis_id, current_user.user_id, db)

    # Find the DEG dataset for this comparison
    deg_dataset: Optional[Dataset] = None
    for ds_id in analysis.result_dataset_ids or []:
        result = await db.execute(select(Dataset).where(Dataset.id == ds_id))
        ds = result.scalar_one_or_none()
        if ds and ds.dataset_metadata:
            comparisons = ds.dataset_metadata.get("comparisons", [])
            if comparison_id in comparisons:
                deg_dataset = ds
                break

    if deg_dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"DEG dataset not found for comparison '{comparison_id}'",
        )

    enrich_path: Optional[str] = (deg_dataset.dataset_metadata or {}).get(
        "enrichment_legacy_path"
    )
    if not enrich_path:
        raise HTTPException(
            status_code=404,
            detail="No legacy enrichment results available for this comparison",
        )

    try:
        tsv_bytes = await storage_service.download_file(enrich_path)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not retrieve enrichment file: {exc}",
        ) from exc

    df = pd.read_csv(io.BytesIO(tsv_bytes), sep="\t")

    if category:
        df = df[df["category"] == category] if "category" in df.columns else df

    total = len(df)
    start = (page - 1) * page_size
    df_page = df.iloc[start : start + page_size]

    return {
        "comparison_id": comparison_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "categories": sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else [],
        "items": df_page.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# GET /analyses/anno-db-categories — moved above /{analysis_id} — see above
# ---------------------------------------------------------------------------
