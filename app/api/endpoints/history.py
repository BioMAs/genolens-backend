"""
API endpoints for project activity history.
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.db import get_db
from app.api.deps.auth import get_current_user
from app.api.deps.project_access import assert_project_read_access
from app.core.security import CurrentUser
from app.models.models import ActivityEventType
from app.schemas.history import ActivityLogListResponse
from app.services import history_service

logger = logging.getLogger(__name__)

router = APIRouter()




@router.get(
    "/projects/{project_id}/history",
    response_model=ActivityLogListResponse,
    summary="Get project activity history",
    description=(
        "Returns a paginated, newest-first list of activity events for a project. "
        "Requires the authenticated user to be the project owner or a project member."
    ),
)
async def get_project_history(
    project_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Number of entries to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    event_type: Optional[ActivityEventType] = Query(None, description="Filter by event type"),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get the activity history for a project.

    Returns activity log entries sorted by newest first.
    Members and owners can query the history.
    """
    try:
        user_id = current_user.id
        await assert_project_read_access(db, project_id, user_id)

        result = await history_service.get_activity_log(
            db,
            project_id=project_id,
            limit=limit,
            offset=offset,
            event_type_filter=event_type,
        )
        return ActivityLogListResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching project history for %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve project history")
