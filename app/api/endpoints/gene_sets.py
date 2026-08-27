"""
Project-scoped custom gene sets.

GET    /projects/{project_id}/gene-sets              — list a project's custom sets
POST   /projects/{project_id}/gene-sets              — create from pasted genes
POST   /projects/{project_id}/gene-sets/upload-gmt   — create from an uploaded GMT
DELETE /projects/{project_id}/gene-sets/{gene_set_id} — delete a custom set

Custom sets are stored as GeneSet rows with database=CUSTOM, scoped by project_id.
They feed GSEA (gene_set_database="CUSTOM") and ORA (via intersection-enrichment).
"""
import logging
import tempfile
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.license import require_active_license
from app.api.deps.subscription import require_scientific_access
from app.core.supabase_auth import SupabaseUser
from app.models.models import Project, ProjectMember, GeneSet, GeneSetDatabase
from app.services.gene_set_loader import GMTParser

logger = logging.getLogger(__name__)
router = APIRouter(tags=["gene-sets"])

MAX_GENES_PER_SET = 20000
MAX_SETS_PER_GMT = 2000


class GeneSetResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    size: int
    genes: list[str]
    database: str


class GeneSetCreateResponse(BaseModel):
    id: UUID
    name: str
    size: int


class GMTUploadResponse(BaseModel):
    created: int
    updated: int


async def _assert_project_access(db: AsyncSession, project_id: UUID, user: SupabaseUser) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id == user.user_id:
        return project
    member = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user.user_id,
        )
    )
    if member.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Access denied")
    return project


@router.get(
    "/projects/{project_id}/gene-sets",
    response_model=list[GeneSetResponse],
    dependencies=[Depends(require_scientific_access)],
)
async def list_custom_gene_sets(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> list[GeneSetResponse]:
    await _assert_project_access(db, project_id, current_user)
    result = await db.execute(
        select(GeneSet)
        .where(GeneSet.database == GeneSetDatabase.CUSTOM, GeneSet.project_id == project_id)
        .order_by(GeneSet.name)
    )
    sets = result.scalars().all()
    return [
        GeneSetResponse(
            id=gs.id, name=gs.name, description=gs.description,
            size=gs.size, genes=list(gs.genes or []), database=gs.database.value,
        )
        for gs in sets
    ]


@router.post(
    "/projects/{project_id}/gene-sets",
    response_model=GeneSetCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_active_license), Depends(require_scientific_access)],
)
async def create_custom_gene_set(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    name: str = Body(...),
    genes: list[str] = Body(...),
    description: Optional[str] = Body(None),
) -> GeneSetCreateResponse:
    await _assert_project_access(db, project_id, current_user)

    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    clean_genes = sorted({g.strip() for g in genes if g and str(g).strip()})
    if not clean_genes:
        raise HTTPException(status_code=400, detail="At least one gene is required")
    if len(clean_genes) > MAX_GENES_PER_SET:
        raise HTTPException(status_code=400, detail=f"Too many genes (max {MAX_GENES_PER_SET})")

    # Enforce per-project name uniqueness (also guarded by the partial unique index)
    existing = await db.execute(
        select(GeneSet).where(
            GeneSet.database == GeneSetDatabase.CUSTOM,
            GeneSet.project_id == project_id,
            GeneSet.name == name,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail=f"A custom gene set named '{name}' already exists")

    gene_set = GeneSet(
        name=name,
        database=GeneSetDatabase.CUSTOM,
        description=description,
        genes=clean_genes,
        size=len(clean_genes),
        organism="Homo sapiens",
        project_id=project_id,
        user_id=current_user.user_id,
        gene_set_metadata={"source": "paste"},
    )
    db.add(gene_set)
    await db.commit()
    await db.refresh(gene_set)
    return GeneSetCreateResponse(id=gene_set.id, name=gene_set.name, size=gene_set.size)


@router.post(
    "/projects/{project_id}/gene-sets/upload-gmt",
    response_model=GMTUploadResponse,
    dependencies=[Depends(require_active_license), Depends(require_scientific_access)],
)
async def upload_gmt(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    file: UploadFile = File(...),
) -> GMTUploadResponse:
    await _assert_project_access(db, project_id, current_user)

    content = await file.read()
    with tempfile.NamedTemporaryFile("wb", suffix=".gmt", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            parsed = GMTParser.parse_file(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse GMT: {e}")

    if not parsed:
        raise HTTPException(status_code=400, detail="No gene sets found in file")
    if len(parsed) > MAX_SETS_PER_GMT:
        raise HTTPException(status_code=400, detail=f"Too many gene sets (max {MAX_SETS_PER_GMT})")

    # Existing custom sets for this project (upsert by name)
    existing_result = await db.execute(
        select(GeneSet).where(
            GeneSet.database == GeneSetDatabase.CUSTOM,
            GeneSet.project_id == project_id,
        )
    )
    by_name = {gs.name: gs for gs in existing_result.scalars().all()}

    created = 0
    updated = 0
    for gs_data in parsed:
        genes = [g for g in gs_data["genes"] if g][:MAX_GENES_PER_SET]
        if not genes:
            continue
        existing = by_name.get(gs_data["name"])
        if existing:
            existing.description = gs_data.get("description")
            existing.genes = genes
            existing.size = len(genes)
            updated += 1
        else:
            db.add(GeneSet(
                name=gs_data["name"],
                database=GeneSetDatabase.CUSTOM,
                description=gs_data.get("description"),
                genes=genes,
                size=len(genes),
                organism="Homo sapiens",
                project_id=project_id,
                user_id=current_user.user_id,
                gene_set_metadata={"source": "gmt", "filename": file.filename},
            ))
            created += 1

    await db.commit()
    return GMTUploadResponse(created=created, updated=updated)


@router.delete(
    "/projects/{project_id}/gene-sets/{gene_set_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_active_license), Depends(require_scientific_access)],
)
async def delete_custom_gene_set(
    project_id: UUID,
    gene_set_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> None:
    await _assert_project_access(db, project_id, current_user)
    gs = await db.get(GeneSet, gene_set_id)
    if not gs or gs.project_id != project_id or gs.database != GeneSetDatabase.CUSTOM:
        raise HTTPException(status_code=404, detail="Custom gene set not found in this project")
    await db.delete(gs)
    await db.commit()
