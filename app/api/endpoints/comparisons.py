"""
Cross-project comparison listing.

Everything else in the API is scoped to one project; this endpoint answers
"what comparisons do I have, anywhere?" for the workspace-level Comparisons page.
"""
from typing import Annotated, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.supabase_auth import SupabaseUser
from app.models.models import Dataset, Project, ProjectMember
from app.schemas.project import PaginatedUserComparisonsResponse, UserComparisonItem
from app.services.comparison_catalog import build_comparisons_from_datasets

router = APIRouter(prefix="/comparisons", tags=["comparisons"])

SortField = Literal["name", "project_name", "deg_total", "updated_at"]


def _sort_key(item: UserComparisonItem, sort_by: SortField):
    """Sort key that never compares None against a real value."""
    if sort_by == "name":
        return item.name.lower()
    if sort_by == "project_name":
        return item.project_name.lower()
    if sort_by == "deg_total":
        return item.deg_total
    # updated_at is nullable; missing dates sort as the epoch.
    return item.updated_at.timestamp() if item.updated_at else 0.0


@router.get("", response_model=PaginatedUserComparisonsResponse)
@router.get("/", response_model=PaginatedUserComparisonsResponse)
async def list_user_comparisons(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, max_length=200),
    project_id: Optional[UUID] = Query(None, description="Restrict to a single project"),
    sort_by: SortField = Query("updated_at"),
    sort_order: Literal["asc", "desc"] = Query("desc"),
) -> dict:
    """
    List every comparison across the projects the current user owns or is a member of.
    """
    member_subquery = select(ProjectMember.project_id).where(
        ProjectMember.user_id == current_user.user_id
    )
    access_filter = or_(
        Project.owner_id == current_user.user_id,
        Project.id.in_(member_subquery),
    )

    # One query for every dataset the user can reach, with its project's name
    # alongside — the alternative is a lookup per project.
    query = (
        select(Dataset, Project.name.label("project_name"))
        .join(Project, Project.id == Dataset.project_id)
        .where(access_filter)
        .order_by(Dataset.created_at.desc())
    )
    if project_id is not None:
        query = query.where(Dataset.project_id == project_id)

    result = await db.execute(query)
    rows = result.all()

    # Group by project: the catalog attributes enrichment across the datasets it
    # is given, and enrichment never crosses a project boundary.
    datasets_by_project: dict[UUID, list[Dataset]] = {}
    project_names: dict[UUID, str] = {}
    dataset_meta: dict[UUID, tuple[str, object]] = {}
    for dataset, project_name in rows:
        datasets_by_project.setdefault(dataset.project_id, []).append(dataset)
        project_names[dataset.project_id] = project_name
        dataset_meta[dataset.id] = (dataset.name, dataset.updated_at)

    items: list[UserComparisonItem] = []
    for proj_id, datasets in datasets_by_project.items():
        for summary in build_comparisons_from_datasets(datasets):
            name, updated_at = dataset_meta[summary.dataset_id]
            items.append(
                UserComparisonItem(
                    **summary.model_dump(),
                    project_id=proj_id,
                    project_name=project_names[proj_id],
                    dataset_name=name,
                    updated_at=updated_at,
                )
            )

    if search:
        needle = search.strip().lower()
        items = [
            i for i in items
            if needle in i.name.lower() or needle in i.project_name.lower()
        ]

    items.sort(key=lambda i: _sort_key(i, sort_by), reverse=(sort_order == "desc"))

    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size

    return {
        "comparisons": items[offset:offset + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }
