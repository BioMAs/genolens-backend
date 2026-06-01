"""
Subscription and quota dependencies for API endpoints.
"""
from typing import Annotated
from uuid import UUID
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.supabase_deps import get_current_user
from app.core.supabase_auth import SupabaseUser
from app.db.session import get_db
from app.models.models import User, UserRole, SubscriptionPlan


# ── User resolution ────────────────────────────────────────────────────────────

async def get_or_create_user(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """Get or create User profile from Supabase auth. Defaults to STARTER plan."""
    query = select(User).where(User.id == current_user.user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            id=current_user.user_id,
            email=current_user.email,
            full_name=current_user.user_metadata.get("full_name"),
            role=current_user.role,
            subscription_plan=SubscriptionPlan.STARTER,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    else:
        if user.role != current_user.role:
            user.role = current_user.role
            db.add(user)
            await db.commit()
            await db.refresh(user)

    return user


# ── Admin guard ────────────────────────────────────────────────────────────────

async def require_admin(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """Require ADMIN or SCILICIUM_ADMIN role."""
    if user.role not in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


# ── AI access ──────────────────────────────────────────────────────────────────

async def require_ai_access(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """Require TEAM or ON_PREMISE plan for AI interpretation."""
    if not user.can_use_ai:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"AI interpretation requires TEAM or ON_PREMISE plan. "
                f"Current plan: {user.subscription_plan.value}"
            )
        )
    return user


async def check_ai_quota(
    user: Annotated[User, Depends(get_or_create_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    AI quota check.
    - ON_PREMISE / ADMIN / SCILICIUM_ADMIN: unlimited.
    - TEAM: 1 AI interpretation per comparison (enforced upstream via DB unique
      index on AIInterpretation(dataset_id, comparison_name) — always passes here).
    - STARTER: blocked by require_ai_access before reaching here.
    """
    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        return user
    if user.subscription_plan in (SubscriptionPlan.ON_PREMISE, SubscriptionPlan.TEAM):
        return user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="AI features not available on STARTER plan"
    )


async def increment_ai_usage(
    user: User,
    db: AsyncSession,
    action_type: str = "interpretation",
    dataset_id: UUID = None,
    comparison_name: str = None,
    model_used: str = "biomistral"
) -> None:
    """Log AI usage. Quota enforcement is via DB unique index, not counters."""
    from app.models.models import AIUsageLog
    log = AIUsageLog(
        user_id=user.id,
        dataset_id=dataset_id,
        action_type=action_type,
        comparison_name=comparison_name,
        model_used=model_used,
        tokens_used=1,
        was_free=True,
    )
    db.add(log)
    await db.commit()


# ── TEAM plan guard ────────────────────────────────────────────────────────────

async def require_team_plan(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """Require TEAM or ON_PREMISE plan (multi-comparison, export PDF, API access)."""
    if not user.can_use_multi_comparison:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"This feature requires a TEAM or ON_PREMISE plan. "
                f"Current plan: {user.subscription_plan.value}"
            )
        )
    return user


# ── Comparison quota ───────────────────────────────────────────────────────────

def _reset_quota_if_new_month(user: User) -> bool:
    """
    Reset comparisons_used_this_month if we're in a new calendar month.
    Returns True if a reset was performed.
    """
    now = datetime.now(timezone.utc)
    if user.quota_reset_at is None:
        user.comparisons_used_this_month = 0
        user.quota_reset_at = now
        return True
    if (
        now.year != user.quota_reset_at.year
        or now.month != user.quota_reset_at.month
    ):
        user.comparisons_used_this_month = 0
        user.quota_reset_at = now
        return True
    return False


async def check_comparison_quota(
    user: Annotated[User, Depends(get_or_create_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    Check that the user has remaining comparison quota for this month.
    Resets the counter automatically if a new month has started.
    Raises HTTP 429 if quota is exhausted.
    """
    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        return user

    reset_happened = _reset_quota_if_new_month(user)

    remaining = user.comparisons_remaining
    if remaining is not None and remaining <= 0:
        # Persist any reset even when blocking, so next request doesn't re-reset
        if reset_happened:
            db.add(user)
            await db.commit()
        quota = user.comparisons_quota
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Monthly comparison quota exhausted ({quota}/{quota}). "
                f"Quota resets on the 1st of next month. "
                f"Upgrade to a higher plan for more comparisons."
            )
        )

    if reset_happened:
        db.add(user)
        await db.commit()
    return user


async def increment_comparison_usage(user: User, db: AsyncSession) -> None:
    """
    Increment the comparison counter after a successful DEG dataset upload.
    Also sends a warning email at 80% quota.
    Call this AFTER the dataset has been committed.
    """
    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or user.comparisons_quota is None:
        return  # Unlimited — skip

    await db.execute(
        update(User)
        .where(User.id == user.id)
        .values(comparisons_used_this_month=User.comparisons_used_this_month + 1)
    )
    await db.commit()
    await db.refresh(user)

    # 80% warning — fire and forget, never block the upload
    quota = user.comparisons_quota
    used = user.comparisons_used_this_month
    if quota and used == int(quota * 0.8):
        try:
            from app.services.email_service import send_quota_warning_email
            await send_quota_warning_email(
                to=user.email,
                used=used,
                quota=quota,
                plan=user.subscription_plan.value,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to send 80%% quota warning email to %s", user.email
            )


# ── Legacy alias (backward compat with existing callers) ─────────────────────

async def require_analysis_access(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """Backward-compat alias — previously required ADVANCED plan.
    Now all paid plans can launch analyses (subject to quota).
    Use check_comparison_quota for new code."""
    return user
