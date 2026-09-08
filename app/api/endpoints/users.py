from typing import Any, Annotated, Optional, Literal
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.api.deps.subscription import get_or_create_user
from app.models.models import Project, User
from app.schemas import user as user_schemas
from app.services import email_service

router = APIRouter()


class AccessRequest(BaseModel):
    """A user's request to change plan or unlock a module (emailed to sales)."""
    type: Literal["plan", "module"]
    item: str = Field(..., max_length=120)
    details: Optional[str] = Field(None, max_length=500)

@router.get("/me", response_model=user_schemas.UserSelf)
async def read_user_me(
    current_user: Annotated[User, Depends(get_or_create_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Any:
    """
    Get current user profile and subscription details.
    """
    # Same predicate as the limit enforced in create_project
    # (app/api/endpoints/projects.py) — owned projects only, not shared ones.
    # GET /projects counts owned-or-shared, which is the wrong number for a
    # client trying to show "used/max" against the plan's project cap.
    current_user.project_count = await db.scalar(
        select(func.count()).select_from(Project).where(Project.owner_id == current_user.id)
    )
    return current_user

# NOTE: PATCH /me/subscription was removed deliberately. It let any authenticated
# caller set their own subscription_plan, which nullified every plan-keyed limit
# (comparison quota, project/dataset caps, AI access, advanced export). A plan
# change now has exactly two legitimate paths, both of them privileged:
#   - the Stripe webhook           -> app/api/endpoints/billing.py
#   - PATCH /admin/users/{id}/subscription (Depends(require_admin))
# Do not reintroduce a self-service variant without require_admin.


@router.post("/requests")
async def submit_access_request(
    payload: AccessRequest,
    current_user: Annotated[User, Depends(get_or_create_user)],
) -> dict:
    """
    Submit a plan-change or module-access request. Emails the sales inbox so the
    team can follow up. Always returns 200 so the UI can confirm receipt even if
    the mail transport is momentarily unavailable (the attempt is logged).
    """
    sent = await email_service.send_access_request(
        requester_email=current_user.email or "unknown user",
        kind=payload.type,
        item=payload.item,
        details=payload.details,
    )
    return {"status": "received", "emailed": sent}
