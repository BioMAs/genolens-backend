"""
Service for tracking and retrieving project activity history.
"""
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.models import ProjectActivityLog, ActivityEventType

logger = logging.getLogger(__name__)


async def log_activity(
    db: AsyncSession,
    project_id: UUID,
    user_id: UUID,
    event_type: ActivityEventType,
    *,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    entity_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an activity event. Fire-and-forget — never raises.
    Call this after every successful state-changing operation.

    Usage:
        await log_activity(db, project_id, user_id, ActivityEventType.BOOKMARK_CREATED,
                           entity_type="bookmark", entity_name=gene_symbol)
    """
    try:
        entry = ProjectActivityLog(
            project_id=project_id,
            user_id=user_id,
            event_type=event_type,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id else None,
            entity_name=entity_name,
            extra_metadata=extra_metadata or {},
        )
        db.add(entry)
        await db.commit()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to log activity event %s for project %s: %s", event_type, project_id, exc)


async def get_activity_log(
    db: AsyncSession,
    project_id: UUID,
    limit: int = 50,
    offset: int = 0,
    event_type_filter: Optional[ActivityEventType] = None,
) -> Dict[str, Any]:
    """
    Return a paginated list of activity log entries for a project.
    Entries are sorted newest-first.
    """
    conditions = [ProjectActivityLog.project_id == project_id]
    if event_type_filter:
        conditions.append(ProjectActivityLog.event_type == event_type_filter)

    where_clause = and_(*conditions)

    # Total count
    count_stmt = select(func.count()).select_from(ProjectActivityLog).where(where_clause)
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Paginated items
    items_stmt = (
        select(ProjectActivityLog)
        .where(where_clause)
        .order_by(ProjectActivityLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items_result = await db.execute(items_stmt)
    items: List[ProjectActivityLog] = list(items_result.scalars().all())

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
