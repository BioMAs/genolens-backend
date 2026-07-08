"""
Admin endpoints for AI cost governance.

Per-user AI cost estimation for a calendar month:
    rate       = modal_spend_eur(month) / total_tokens(month)
    cost_user  = tokens_user(month) * rate

Token counts come from AIUsageLog.tokens_used (real vLLM usage.total_tokens).
The monthly Modal spend is entered by an admin (ModalMonthlyCost).

All endpoints require ADMIN or SCILICIUM_ADMIN role.
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_admin
from app.core.supabase_auth import SupabaseUser
from app.models.models import AIUsageLog, ModalMonthlyCost, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin - AI Costs"])


class SpendUpsert(BaseModel):
    year: int = Field(..., ge=2020, le=2100)
    month: int = Field(..., ge=1, le=12)
    spend_eur: float = Field(..., ge=0)


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start, end


@router.get("/ai-costs")
async def get_ai_costs(
    db: AsyncSession = Depends(get_db),
    _admin: SupabaseUser = Depends(require_admin),
    year: int = Query(..., ge=2020, le=2100),
    month: int = Query(..., ge=1, le=12),
) -> dict:
    """
    Per-user AI token usage and estimated cost for the given month.

    Estimated cost is derived from the admin-entered Modal spend for that month
    (rate = spend / total tokens). If no spend is recorded, costs are 0 and only
    token counts are meaningful.
    """
    start, end = _month_bounds(year, month)

    rows = (
        await db.execute(
            select(
                User.id,
                User.email,
                func.coalesce(func.sum(AIUsageLog.tokens_used), 0).label("tokens"),
            )
            .join(AIUsageLog, AIUsageLog.user_id == User.id)
            .where(AIUsageLog.created_at >= start, AIUsageLog.created_at < end)
            .group_by(User.id, User.email)
        )
    ).all()

    total_tokens = sum(int(r.tokens) for r in rows)

    spend = await db.scalar(
        select(ModalMonthlyCost.spend_eur)
        .where(ModalMonthlyCost.year == year, ModalMonthlyCost.month == month)
    )
    spend_eur = float(spend) if spend is not None else 0.0
    rate = (spend_eur / total_tokens) if total_tokens > 0 else 0.0

    users = sorted(
        (
            {
                "user_id": str(r.id),
                "email": r.email,
                "tokens": int(r.tokens),
                "cost_eur": round(int(r.tokens) * rate, 4),
            }
            for r in rows
        ),
        key=lambda u: u["tokens"],
        reverse=True,
    )

    return {
        "year": year,
        "month": month,
        "spend_eur": spend_eur,
        "total_tokens": total_tokens,
        # €/1M tokens, for transparency in the UI
        "rate_per_million_tokens": round(rate * 1_000_000, 4),
        "total_cost_eur": round(total_tokens * rate, 2),
        "users": users,
    }


@router.put("/ai-costs/spend")
async def upsert_ai_spend(
    payload: SpendUpsert,
    db: AsyncSession = Depends(get_db),
    _admin: SupabaseUser = Depends(require_admin),
) -> dict:
    """Set (upsert) the Modal spend for a given month."""
    existing = await db.scalar(
        select(ModalMonthlyCost)
        .where(ModalMonthlyCost.year == payload.year, ModalMonthlyCost.month == payload.month)
    )
    if existing:
        existing.spend_eur = payload.spend_eur
    else:
        db.add(ModalMonthlyCost(year=payload.year, month=payload.month, spend_eur=payload.spend_eur))
    await db.commit()
    return {"year": payload.year, "month": payload.month, "spend_eur": payload.spend_eur}
