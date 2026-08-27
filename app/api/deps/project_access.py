"""Shared project-scoped access control.

A project's resources — analyses, datasets, generated reports — are visible to
the project owner and to every `ProjectMember`. Keeping that rule in one place
avoids the class of bug where an endpoint gates on the resource's *creator*
instead of on project access, which locks legitimate members out of shared
projects.
"""
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Project, ProjectMember


async def assert_project_access(db: AsyncSession, project_id: UUID, user_id: UUID) -> Project:
    """Return the project if `user_id` is its owner or a member, else raise.

    Raises:
        HTTPException 404: the project does not exist.
        HTTPException 403: the user is neither owner nor member.
    """
    project = await db.scalar(select(Project).where(Project.id == project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != user_id:
        member = await db.scalar(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
            )
        )
        if member is None:
            raise HTTPException(status_code=403, detail="Access denied")
    return project
