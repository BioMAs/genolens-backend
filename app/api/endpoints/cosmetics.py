"""
Cosmetics module endpoints.

  - GET  /cosmetics/{dataset_id}/{comparison_name}            -> claim scores
  - POST /cosmetics/{dataset_id}/{comparison_name}/interpret  -> AI narrative
  - admin /admin/claim-mappings ...                           -> referential CRUD

The Cosmetics tab stays visible to every user in the frontend; users without the
module see a locked teaser rendered from static demo data and never call these
endpoints. Access here is therefore gated by require_cosmetics_access (403 for
users without the add-on).
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.exc import IntegrityError

from app.api.deps import get_db
from app.api.deps.subscription import (
    increment_ai_usage,
    require_admin,
    require_ai_access,
    require_cosmetics_access,
)
from app.models.models import (
    ClaimDirection,
    ClaimPathwayMapping,
    CosmeticInterpretation,
    Dataset,
    EvidenceLevel,
    Project,
    ProjectMember,
    User,
    UserRole,
)
from app.services.ai_interpreter import LocalAIInterpreter
from app.services.claim_import_service import (
    import_claim_workbooks,
    import_workbook_bytes,
)
from app.services.cosmetics_service import score_claims

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cosmetics", tags=["Cosmetics"])
admin_router = APIRouter(prefix="/admin", tags=["Admin - Cosmetics"])


async def _verify_dataset_access(db: AsyncSession, dataset_id: UUID, user: User) -> Dataset:
    """Ensure the user can read this dataset (owner / member / admin)."""
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    ).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        return dataset

    project = (
        await db.execute(select(Project).where(Project.id == dataset.project_id))
    ).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.owner_id != user.id:
        member = (
            await db.execute(
                select(ProjectMember).where(
                    and_(
                        ProjectMember.project_id == project.id,
                        ProjectMember.user_id == user.id,
                    )
                )
            )
        ).scalar_one_or_none()
        if not member:
            raise HTTPException(
                status_code=403, detail="Not authorized to access this dataset"
            )
    return dataset


@router.get("/{dataset_id}/{comparison_name}")
async def get_cosmetics_scores(
    dataset_id: UUID,
    comparison_name: str,
    max_padj: float = Query(0.05, description="Adjusted p-value cutoff"),
    user: User = Depends(require_cosmetics_access),
    db: AsyncSession = Depends(get_db),
):
    """Compute cosmetic claim scores and skin-zone activity for a comparison."""
    await _verify_dataset_access(db, dataset_id, user)
    return await score_claims(db, dataset_id, comparison_name, max_padj=max_padj)


@router.post("/{dataset_id}/{comparison_name}/interpret")
async def interpret_cosmetics(
    dataset_id: UUID,
    comparison_name: str,
    force_regenerate: bool = Query(False),
    user: User = Depends(require_cosmetics_access),
    _ai: User = Depends(require_ai_access),
    db: AsyncSession = Depends(get_db),
):
    """Generate (and cache) a cosmetic-focused AI interpretation for a comparison."""
    await _verify_dataset_access(db, dataset_id, user)

    existing = (
        await db.execute(
            select(CosmeticInterpretation).where(
                and_(
                    CosmeticInterpretation.dataset_id == dataset_id,
                    CosmeticInterpretation.comparison_name == comparison_name,
                )
            )
        )
    ).scalar_one_or_none()

    if existing and not force_regenerate:
        return {
            "interpretation": existing.interpretation,
            "cached": True,
            "model": existing.model,
            "claims_count": existing.claims_count,
            "generated_at": existing.updated_at.isoformat() if existing.updated_at else None,
        }

    scores = await score_claims(db, dataset_id, comparison_name)
    claims_count = sum(1 for c in scores.get("claims", []) if c.get("n_supporting", 0) > 0)

    interpreter = LocalAIInterpreter()
    try:
        text = await interpreter.interpret_cosmetics(scores, comparison_name)
    except Exception as e:  # noqa: BLE001
        logger.error("Cosmetics AI interpretation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service unavailable: {e}",
        )

    try:
        if existing:
            existing.interpretation = text
            existing.model = interpreter.model
            existing.claims_count = claims_count
        else:
            db.add(
                CosmeticInterpretation(
                    dataset_id=dataset_id,
                    comparison_name=comparison_name,
                    interpretation=text,
                    model=interpreter.model,
                    claims_count=claims_count,
                )
            )
        await db.commit()
    except IntegrityError:
        # Concurrent request created it first — return that one.
        await db.rollback()
        existing = (
            await db.execute(
                select(CosmeticInterpretation).where(
                    and_(
                        CosmeticInterpretation.dataset_id == dataset_id,
                        CosmeticInterpretation.comparison_name == comparison_name,
                    )
                )
            )
        ).scalar_one_or_none()
        if existing:
            return {
                "interpretation": existing.interpretation,
                "cached": True,
                "model": existing.model,
                "claims_count": existing.claims_count,
                "generated_at": existing.updated_at.isoformat() if existing.updated_at else None,
            }

    try:
        await increment_ai_usage(
            user,
            db,
            action_type="cosmetics_interpretation",
            dataset_id=dataset_id,
            comparison_name=comparison_name,
            model_used=interpreter.model,
            tokens_used=interpreter.last_usage["total_tokens"],
        )
    except Exception:  # noqa: BLE001 — usage logging must never block the response
        logger.warning("Failed to log cosmetics AI usage", exc_info=True)

    return {
        "interpretation": text,
        "cached": False,
        "model": interpreter.model,
        "claims_count": claims_count,
        "generated_at": None,
    }


# ---------------------------------------------------------------------------
# Admin — claim referential management
# ---------------------------------------------------------------------------


class ClaimMappingResponse(BaseModel):
    id: UUID
    term_id: str
    term_id_normalized: str
    description: Optional[str] = None
    original_claims: Optional[str] = None
    updated_claim_framing: Optional[str] = None
    updated_direction: str
    category: Optional[str] = None
    evidence_level: str
    rationale: Optional[str] = None
    caveats: Optional[str] = None
    ref_cat: Optional[str] = None
    canonical_claims: list = []
    is_active: bool

    class Config:
        from_attributes = True


class ClaimMappingUpdate(BaseModel):
    updated_claim_framing: Optional[str] = None
    updated_direction: Optional[ClaimDirection] = None
    category: Optional[str] = None
    evidence_level: Optional[EvidenceLevel] = None
    rationale: Optional[str] = None
    caveats: Optional[str] = None
    canonical_claims: Optional[list] = None
    is_active: Optional[bool] = None


@admin_router.get("/claim-mappings", response_model=List[ClaimMappingResponse])
async def list_claim_mappings(
    search: Optional[str] = Query(None, description="Filter by term id / description"),
    category: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List curated claim mappings (admin)."""
    query = select(ClaimPathwayMapping)
    if search:
        like = f"%{search}%"
        query = query.where(
            ClaimPathwayMapping.term_id.ilike(like)
            | ClaimPathwayMapping.description.ilike(like)
        )
    if category:
        query = query.where(ClaimPathwayMapping.category == category)
    query = query.order_by(ClaimPathwayMapping.term_id).limit(limit).offset(offset)
    rows = (await db.execute(query)).scalars().all()
    return rows


@admin_router.get("/claim-mappings/stats")
async def claim_mappings_stats(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Quick counts for the admin referential screen."""
    total = (
        await db.execute(select(func.count()).select_from(ClaimPathwayMapping))
    ).scalar() or 0
    active = (
        await db.execute(
            select(func.count())
            .select_from(ClaimPathwayMapping)
            .where(ClaimPathwayMapping.is_active.is_(True))
        )
    ).scalar() or 0
    return {"total": total, "active": active}


@admin_router.post("/claim-mappings/import")
async def import_claim_mappings(
    file: Optional[UploadFile] = File(
        None, description="Optional .xlsx workbook. If omitted, re-imports bundled files."
    ),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Import the claim referential. Upload a workbook, or re-seed bundled files."""
    try:
        if file is not None:
            data = await file.read()
            result = await import_workbook_bytes(db, data)
        else:
            result = await import_claim_workbooks(db)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error("Claim referential import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")


@admin_router.patch("/claim-mappings/{mapping_id}", response_model=ClaimMappingResponse)
async def update_claim_mapping(
    mapping_id: UUID,
    update: ClaimMappingUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Edit a single claim mapping (admin)."""
    obj = (
        await db.execute(
            select(ClaimPathwayMapping).where(ClaimPathwayMapping.id == mapping_id)
        )
    ).scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Claim mapping not found")

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)
    await db.commit()
    await db.refresh(obj)
    return obj


@admin_router.delete("/claim-mappings/{mapping_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_claim_mapping(
    mapping_id: UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a claim mapping (admin)."""
    obj = (
        await db.execute(
            select(ClaimPathwayMapping).where(ClaimPathwayMapping.id == mapping_id)
        )
    ).scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Claim mapping not found")
    await db.delete(obj)
    await db.commit()
