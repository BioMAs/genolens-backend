"""
Subscription and quota dependencies for API endpoints.
"""
from typing import Annotated
from uuid import UUID
from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.supabase_deps import get_current_user
from app.core.supabase_auth import SupabaseUser
from app.db.session import get_db
from app.models.models import User, UserRole, SubscriptionPlan


async def get_or_create_user(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    Get or create User profile from Supabase auth user.
    Creates a BASIC user by default if not exists.
    """
    query = select(User).where(User.id == current_user.user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user, respecting role from Supabase (if available)
        user = User(
            id=current_user.user_id,
            email=current_user.email,
            full_name=current_user.user_metadata.get("full_name"),
            role=current_user.role,
            subscription_plan=SubscriptionPlan.BASIC
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    else:
        # Update role if changed in Supabase
        if user.role != current_user.role:
            user.role = current_user.role
            db.add(user)
            await db.commit()
            await db.refresh(user)
    
    return user


async def require_admin(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """
    Require user to be ADMIN.
    Used for admin-only endpoints (user management, analytics, etc.)
    """
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


async def require_ai_access(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """
    Require user to have AI access (PREMIUM or ADVANCED plan).
    Used for /interpret and /ask endpoints.
    """
    if not user.can_use_ai:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"AI features require PREMIUM or ADVANCED subscription. Current plan: {user.subscription_plan}"
        )
    return user


async def check_ai_quota(
    user: Annotated[User, Depends(get_or_create_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    Check if user has remaining AI interpretation quota.
    For ADMIN users: unlimited (no quota check)
    For PREMIUM users: 15 free + purchased tokens
    For ADVANCED users: unlimited (quota check skipped)
    Raises HTTPException if quota exceeded.
    """
    # Admins have unlimited access
    if user.role == UserRole.ADMIN:
        return user
    
    if user.subscription_plan == SubscriptionPlan.ADVANCED:
        # Unlimited for ADVANCED
        return user
    
    if user.subscription_plan == SubscriptionPlan.PREMIUM:
        remaining = user.ai_interpretations_remaining
        if remaining <= 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"AI interpretation quota exceeded. "
                    f"You have used {user.ai_interpretations_used} free interpretations. "
                    f"Purchase additional tokens or upgrade to ADVANCED plan."
                )
            )
        return user
    
    # BASIC users shouldn't reach here (blocked by require_ai_access)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="AI features not available on BASIC plan"
    )


async def increment_ai_usage(
    user: User,
    db: AsyncSession,
    action_type: str = "interpretation",
    dataset_id: UUID = None,
    comparison_name: str = None,
    model_used: str = "llama3.2:3b"
) -> None:
    """
    Increment AI usage counters and log usage.
    ADMIN users are logged but not counted against quotas.
    
    Args:
        user: User object
        db: Database session
        action_type: "interpretation" or "question"
        dataset_id: Dataset ID (optional)
        comparison_name: Comparison name (optional)
        model_used: Model name
    """
    from app.models.models import AIUsageLog
    
    # Determine if this uses free quota or purchased token
    was_free = True
    
    # Admins don't consume quota, but we still log their usage
    if user.role != UserRole.ADMIN:
        if user.subscription_plan == SubscriptionPlan.PREMIUM:
            if user.ai_interpretations_used < 15:
                # Within free quota
                user.ai_interpretations_used += 1
            else:
                # Use purchased token
                user.ai_tokens_used += 1
                was_free = False
    
    # Log usage
    log = AIUsageLog(
        user_id=user.id,
        dataset_id=dataset_id,
        action_type=action_type,
        comparison_name=comparison_name,
        model_used=model_used,
        tokens_used=1,
        was_free=was_free
    )
    db.add(log)
    db.add(user)  # Update user counters
    await db.commit()


async def require_analysis_access(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """
    Require user to have analysis launch access (ADVANCED plan).
    Used for endpoints that launch new analyses.
    """
    if not user.can_launch_analyses:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Analysis launch requires ADVANCED subscription. "
                f"Current plan: {user.subscription_plan}. "
                f"Contact support to upgrade or request admin to upload analyses for you."
            )
        )
    return user
