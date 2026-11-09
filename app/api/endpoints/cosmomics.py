"""
CosmOmics Pathway Nova endpoints — the Skin claims add-on.

  GET  /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova
  GET  /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova.svg
  POST /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova/interpret
  GET  /datasets/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova/interpretation

All are gated by the existing per-user add-on flag (`require_cosmetics_access`, 403 without
it), then by project read access (owner or member, 404 otherwise). The claim referential itself
never leaves the server: the payload only carries the terms enriched in this comparison, and
says what it was computed from. A comparison the dataset does not have is a 404; one it has but
with no enrichment rows is a payload with `status` "no_enrichment".

The AI reading is cached once per (dataset, comparison), tied to the nova it was written from
(`inputs_sha256`, see `report_common.nova_inputs_sha256`): a cached text whose fingerprint is not
the current nova's — another alpha, another referential, another enrichment — is `stale`. POST
never serves a stale text (it writes a new one); GET shows it, flagged, and never calls the LLM.
The two AI routes also need the AI entitlement.
"""

import logging
from datetime import datetime, timezone
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
from app.services.ai_interpreter import LocalAIInterpreter, prompt_claims_sent
from app.services.cosmomics_nova import DEFAULT_ALPHA, NOVA_MODEL_TAG, compute_nova
from app.services.cosmomics_nova_svg import render_nova_svg
from app.services.enrichment_source import recorded_comparison_names
from app.services.report_common import nova_inputs_sha256

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


def _model_of(row: CosmeticInterpretation) -> str:
    return (row.model or "").removeprefix(NOVA_MODEL_TAG)


def _is_nova_reading(row: Optional[CosmeticInterpretation]) -> bool:
    return row is not None and (row.model or "").startswith(NOVA_MODEL_TAG)


def _is_stale(row: CosmeticInterpretation, nova: dict) -> bool:
    """True unless the text was written from this very nova (a text with no fingerprint cannot
    be tied to any data: stale)."""
    return row.inputs_sha256 != nova_inputs_sha256(nova, _model_of(row))


def _interpretation_payload(
    row: CosmeticInterpretation, cached: bool, nova: dict
) -> dict[str, Any]:
    """The text and what it is tied to. `claims_total` / `claims_sent` are those of the text (the
    claims of the nova it was written from, and how many of them its prompt carried), not of the
    nova of today: a stale text keeps its own."""
    moment = row.updated_at or row.created_at
    return {
        "interpretation": row.interpretation,
        "cached": cached,
        "stale": _is_stale(row, nova),
        "model": _model_of(row),
        "generated_at": moment.isoformat() if moment else None,
        "inputs_sha256": row.inputs_sha256,
        "claims_sent": prompt_claims_sent(row.claims_count or 0),
        "claims_total": row.claims_count or 0,
        "claims_count": row.claims_count,  # the former name of claims_total, for older clients
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


async def _cached_reading(
    db: AsyncSession, dataset: Dataset, comparison_name: str
) -> Optional[CosmeticInterpretation]:
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


@router.get("/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova/interpretation")
async def get_cosmomics_nova_interpretation(
    dataset_id: UUID,
    comparison_name: str,
    alpha: float = Query(
        DEFAULT_ALPHA, gt=0, le=0.25, description="Significance threshold on adjusted p-values"
    ),
    user: User = Depends(require_cosmetics_access),
    _ai: User = Depends(require_ai_access),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The cached AI reading of the nova and whether it is `stale` for this nova (this `alpha`).

    A read: it never calls the LLM and spends no AI request. 404 when no reading is cached.
    """
    dataset = await _readable_dataset(db, dataset_id, user)
    row = await _cached_reading(db, dataset, comparison_name)
    if not _is_nova_reading(row):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No AI reading has been generated for this comparison",
        )
    nova = await _nova(db, dataset, comparison_name, alpha)
    return _interpretation_payload(row, cached=True, nova=nova)


@router.post("/{dataset_id}/comparisons/{comparison_name}/cosmomics-nova/interpret")
async def interpret_cosmomics_nova(
    dataset_id: UUID,
    comparison_name: str,
    force_regenerate: bool = Query(False),
    alpha: float = Query(
        DEFAULT_ALPHA, gt=0, le=0.25, description="Significance threshold on adjusted p-values"
    ),
    user: User = Depends(require_cosmetics_access),
    _ai: User = Depends(require_ai_access),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Generate (and cache) the AI reading of the nova — or serve the cached one while it is
    written from this very nova. A cached reading of another nova is not served: a new one is
    written, which costs one AI request."""
    dataset = await _readable_dataset(db, dataset_id, user)

    existing = await _cached_reading(db, dataset, comparison_name)
    nova = await _nova(db, dataset, comparison_name, alpha)
    if _is_nova_reading(existing) and not force_regenerate and not _is_stale(existing, nova):
        return _interpretation_payload(existing, cached=True, nova=nova)

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
    digest = nova_inputs_sha256(nova, interpreter.model)
    # The moment of generation is set here, never left to the database default: the response, the
    # label and the report all print it.
    generated_at = datetime.now(timezone.utc)
    try:
        if existing:
            existing.interpretation = text
            existing.model = model_tag
            existing.claims_count = claims_count
            existing.inputs_sha256 = digest
            existing.updated_at = generated_at
        else:
            db.add(
                CosmeticInterpretation(
                    dataset_id=dataset.id,
                    comparison_name=comparison_name,
                    interpretation=text,
                    model=model_tag,
                    claims_count=claims_count,
                    inputs_sha256=digest,
                    created_at=generated_at,
                    updated_at=generated_at,
                )
            )
        await db.commit()
    except IntegrityError:
        # A concurrent request stored its interpretation first — return that one.
        await db.rollback()
        await db.refresh(user)
        existing = await _cached_reading(db, dataset, comparison_name)
        if existing:
            return _interpretation_payload(existing, cached=True, nova=nova)

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
        "stale": False,
        "model": interpreter.model,
        "generated_at": generated_at.isoformat(),
        "inputs_sha256": digest,
        "claims_sent": prompt_claims_sent(claims_count),
        "claims_total": claims_count,
        "claims_count": claims_count,  # the former name of claims_total, for older clients
    }
