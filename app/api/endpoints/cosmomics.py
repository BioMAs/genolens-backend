"""
CosmOmics Pathway Nova endpoints — the Skin claims add-on.

  GET  /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova
  GET  /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova.svg
  POST /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova/interpret

Both are gated by the existing per-user add-on flag (`require_cosmetics_access`, 403 without
it), then by project read access (owner or member, 404 otherwise). The claim referential itself
never leaves the server: the payload only carries the terms enriched in this comparison, and
says what it was computed from. A comparison the dataset does not have is a 404; one it has but
with no enrichment rows is a payload with `status` "no_enrichment".
"""

import logging
from typing import Any, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.deps.license import require_active_license
from app.api.deps.project_access import assert_project_read_access
from app.api.deps.subscription import (
    increment_ai_usage,
    require_ai_access,
    require_cosmetics_access,
    reserve_ai_request,
)
from app.models.models import CosmeticInterpretation, Dataset, User
from app.services.ai_interpreter import LocalAIInterpreter
from app.services.cosmomics_nova import DEFAULT_ALPHA, NOVA_MODEL_TAG, compute_nova
from app.services.cosmomics_nova_svg import render_nova_svg
from app.services.enrichment_source import recorded_comparison_names

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/datasets",
    tags=["CosmOmics"],
    dependencies=[Depends(require_active_license)],
)


async def _readable_dataset(db: AsyncSession, dataset_id: UUID, user: User) -> Dataset:
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    ).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    await assert_project_read_access(db, dataset.project_id, user.id)
    return dataset


async def _nova(db: AsyncSession, dataset: Dataset, comparison_name: str, alpha: float) -> dict:
    # A dataset that records its comparisons refuses one it does not have. One that records none
    # (no metadata, no DEG rows) cannot tell: it answers "no_enrichment" when no row matches.
    recorded = await recorded_comparison_names(db, dataset)
    if recorded and comparison_name not in recorded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No comparison '{comparison_name}' in this dataset",
        )
    nova, _hits = await compute_nova(db, dataset, comparison_name, alpha)
    return nova


def _interpretation_payload(row: CosmeticInterpretation, cached: bool) -> dict[str, Any]:
    return {
        "interpretation": row.interpretation,
        "cached": cached,
        "model": (row.model or "").removeprefix(NOVA_MODEL_TAG),
        "claims_count": row.claims_count,
        "generated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@router.get("/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova")
async def get_cosmomics_nova(
    dataset_id: UUID,
    comparison_name: str,
    alpha: float = Query(
        DEFAULT_ALPHA, gt=0, le=0.25, description="Significance threshold on adjusted p-values"
    ),
    user: User = Depends(require_cosmetics_access),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Pathway sub-groups for and against each cosmetic claim, for one comparison."""
    dataset = await _readable_dataset(db, dataset_id, user)
    return await _nova(db, dataset, comparison_name, alpha)


@router.get("/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova.svg")
async def get_cosmomics_nova_svg(
    dataset_id: UUID,
    comparison_name: str,
    theme: Literal["light", "dark"] = Query("light", description="Colour theme of the wheel"),
    alpha: float = Query(
        DEFAULT_ALPHA, gt=0, le=0.25, description="Significance threshold on adjusted p-values"
    ),
    user: User = Depends(require_cosmetics_access),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """The nova wheel as a standalone SVG (same payload as `cosmomics-nova`)."""
    dataset = await _readable_dataset(db, dataset_id, user)
    nova = await _nova(db, dataset, comparison_name, alpha)
    svg = render_nova_svg(nova, theme=theme, wheel_only=True)
    return Response(
        content=svg, media_type="image/svg+xml", headers={"Cache-Control": "no-store"}
    )


@router.post("/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova/interpret")
async def interpret_cosmomics_nova(
    dataset_id: UUID,
    comparison_name: str,
    force_regenerate: bool = Query(False),
    user: User = Depends(require_cosmetics_access),
    _ai: User = Depends(require_ai_access),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Generate (and cache) the cosmetic AI reading of the nova."""
    dataset = await _readable_dataset(db, dataset_id, user)

    async def _existing() -> Optional[CosmeticInterpretation]:
        return (
            await db.execute(
                select(CosmeticInterpretation).where(
                    and_(
                        CosmeticInterpretation.dataset_id == dataset.id,
                        CosmeticInterpretation.comparison_name == comparison_name,
                    )
                )
            )
        ).scalar_one_or_none()

    existing = await _existing()
    if existing and (existing.model or "").startswith(NOVA_MODEL_TAG) and not force_regenerate:
        return _interpretation_payload(existing, cached=True)

    nova = await _nova(db, dataset, comparison_name, DEFAULT_ALPHA)
    claims_count = len(nova["claims"])

    # The cache above was free; one AI request is reserved for the LLM call (429 at the
    # account cap) and refunded if the call fails.
    reservation = await reserve_ai_request(user, db)
    interpreter = LocalAIInterpreter()
    try:
        text = await interpreter.interpret_cosmomics(nova, comparison_name)
    except Exception as e:  # noqa: BLE001
        logger.error("CosmOmics AI interpretation failed: %s", e)
        await reservation.refund()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service unavailable: {e}",
        )

    model_tag = f"{NOVA_MODEL_TAG}{interpreter.model}"
    try:
        if existing:
            existing.interpretation = text
            existing.model = model_tag
            existing.claims_count = claims_count
        else:
            db.add(
                CosmeticInterpretation(
                    dataset_id=dataset.id,
                    comparison_name=comparison_name,
                    interpretation=text,
                    model=model_tag,
                    claims_count=claims_count,
                )
            )
        await db.commit()
    except IntegrityError:
        # A concurrent request stored its interpretation first — return that one.
        await db.rollback()
        await db.refresh(user)
        existing = await _existing()
        if existing:
            return _interpretation_payload(existing, cached=True)

    try:
        await increment_ai_usage(
            user,
            db,
            action_type="cosmetics_interpretation",
            dataset_id=dataset.id,
            comparison_name=comparison_name,
            model_used=interpreter.model,
            tokens_used=interpreter.last_usage["total_tokens"],
        )
    except Exception:  # noqa: BLE001 — usage logging must never block the response
        logger.warning("Failed to log CosmOmics AI usage", exc_info=True)

    return {
        "interpretation": text,
        "cached": False,
        "model": interpreter.model,
        "claims_count": claims_count,
        "generated_at": None,
    }
