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

from app.models.models import Project, ProjectMember, UserRole


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


async def assert_project_read_access(db: AsyncSession, project_id: UUID, user_id: UUID) -> Project:
    """Variante de `assert_project_access` qui refuse en 404, jamais en 403.

    Même règle — propriétaire ou membre — mais un tiers reçoit « Project not
    found » plutôt que « Access denied » : la route ne lui confirme pas que le
    projet existe. C'est le comportement historique de la trentaine de routes de
    `datasets.py`, et le frontend distingue les deux codes ; les fusionner
    changerait des réponses d'API et sort du périmètre d'une extraction.

    Les deux fonctions coexistent donc volontairement. Choisir : cette
    variante-ci pour une ressource dont l'existence même est confidentielle,
    `assert_project_access` quand un 403 explicite aide l'appelant légitime.

    Raises:
        HTTPException 404: projet inexistant, ou appelant ni propriétaire ni membre.
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
            raise HTTPException(status_code=404, detail="Project not found")
    return project


async def is_project_admin(db: AsyncSession, project: Project, user_id: UUID) -> bool:
    """Vrai si `user_id` est le propriétaire du projet ou un membre ADMIN.

    Rend un booléen au lieu de lever : les appelants n'ont pas le même refus à
    formuler — « Only project admins can update datasets » en 403 ici, un 404
    opaque là — et la politique leur appartient.

    Le projet est reçu déjà chargé, parce que tous les appelants viennent de le
    lire pour une autre raison ; le refaire ici doublerait la requête.

    Vivait en double, mot pour mot, dans `endpoints/datasets.py` et
    `endpoints/projects.py`. Trois modules d'endpoints importaient l'une de ces
    copies privées — ce qui explique probablement que les routes plus tardives
    aient réécrit le contrôle à la main plutôt que d'aller le chercher.
    """
    if project.owner_id == user_id:
        return True
    member = await db.scalar(
        select(ProjectMember).where(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == user_id,
            ProjectMember.access_level == UserRole.ADMIN,
        )
    )
    return member is not None
