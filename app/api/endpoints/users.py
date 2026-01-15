from typing import Any, Annotated
from fastapi import APIRouter, Depends, Body
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps.subscription import get_or_create_user
from app.api.deps import get_db
from app.models.models import User, SubscriptionPlan
from app.schemas import user as user_schemas

router = APIRouter()

@router.get("/me", response_model=user_schemas.UserSelf)
async def read_user_me(
    current_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """
    Get current user profile and subscription details.
    """
    return current_user

@router.patch("/me/subscription", response_model=user_schemas.UserSelf)
async def update_my_subscription(
    plan: Annotated[SubscriptionPlan, Body(embed=True)],
    current_user: Annotated[User, Depends(get_or_create_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Any:
    """
    Update own subscription plan (Demo/Testing purpose).
    In production, this would be handled via Stripe webhooks.
    """
    current_user.subscription_plan = plan
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user
