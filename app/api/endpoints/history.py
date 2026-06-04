"""
API endpoints for project activity history.
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps.db import get_db
from app.api.deps.auth import get_current_user
from app.core.security import CurrentUser
from app.models.models import Project, ProjectMember, ActivityEventType
from app.schemas.history import ActivityLogListResponse
from app.services import history_service

logger = logging.getLogger(__name__)

router = APIRouter()


async def _require_project_access(project_id: UUID, user_id: UUID, db: AsyncSession) -> Project:
    """
    Verify that the project exists and the user has access (owner or member).
    Raises 404 if project not found, 403 if no access.
    """
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Owner always has access
    if project.user_id == user_id:
        return project

    # Check member access
    member_result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id,
        )
    )
    member = member_result.scalar_one_or_none()

    if not member:
        raise HTTPException(status_code=403, detail="Access denied to this project")

    return project


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
        await _require_project_access(project_id, user_id, db)

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
