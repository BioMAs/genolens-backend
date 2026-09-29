"""
API endpoints for project comments and annotations.

Every route is scoped to project access (owner or `ProjectMember`). A caller
without access gets a 404, never a 403, so the API does not confirm that a
project or comment exists. That matches `datasets.py`.
"""
import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.db import get_db
from app.api.deps.auth import get_current_user
from app.api.deps.project_access import assert_project_read_access
from app.core.security import CurrentUser
from app.models.models import CommentType, ProjectComment
from app.schemas.comment import (
    ProjectCommentCreate,
    ProjectCommentUpdate,
    ProjectCommentResponse,
    CommentCountResponse,
    CommentThreadResponse
)
from app.services.comments_service import comments_service
from app.services import history_service, email_service
from app.core.supabase_auth import lookup_user_by_id
from app.models.models import ActivityEventType

logger = logging.getLogger(__name__)

router = APIRouter()


async def _load_accessible_comment(
    db: AsyncSession, comment_id: UUID, user_id: UUID
) -> ProjectComment:
    """Return the comment if the caller can access its project, else 404.

    A missing comment and a comment in someone else's project look the same to
    the caller, so a comment id cannot be used to probe for existence.
    """
    comment = await comments_service.get_comment_by_id(db, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    try:
        await assert_project_read_access(db, comment.project_id, user_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Comment not found")
    return comment


# ============================================================================
# Comment CRUD
# ============================================================================

@router.get("/projects/{project_id}/comments", response_model=List[ProjectCommentResponse])
async def get_comments(
    project_id: UUID,
    comment_type: Optional[str] = Query(None, description="Filter by comment type"),
    target_id: Optional[str] = Query(None, description="Filter by target entity ID"),
    include_resolved: bool = Query(True, description="Include resolved comments"),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get comments for a project with optional filters.

    Args:
        project_id: Project UUID
        comment_type: Optional type filter (GENERAL, GENE, COMPARISON, PATHWAY)
        target_id: Optional target entity ID (gene_symbol, comparison_name, etc.)
        include_resolved: Whether to include resolved comments

    Returns:
        List of comments (top-level only, use /thread endpoint for replies)
    """
    await assert_project_read_access(db, project_id, current_user.id)

    try:
        type_filter = CommentType(comment_type) if comment_type else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid comment_type")

    return await comments_service.get_comments(
        db,
        project_id=project_id,
        comment_type=type_filter,
        target_id=target_id,
        include_resolved=include_resolved
    )


@router.get("/comments/{comment_id}", response_model=ProjectCommentResponse)
async def get_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get a single comment by ID."""
    return await _load_accessible_comment(db, comment_id, current_user.id)


@router.get("/comments/{comment_id}/thread", response_model=CommentThreadResponse)
async def get_comment_thread(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get a comment thread (comment + all replies)."""
    await _load_accessible_comment(db, comment_id, current_user.id)

    # Replies always share the root's project: create_comment rejects a
    # parent_id from another project.
    thread = await comments_service.get_comment_thread(db, comment_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Comment not found")

    return {
        "comment": thread[0],
        "reply_count": len(thread) - 1
    }


@router.post("/projects/{project_id}/comments", response_model=ProjectCommentResponse, status_code=201)
async def create_comment(
    project_id: UUID,
    comment: ProjectCommentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Create a new comment or reply."""
    user_id = current_user.id

    # Access is checked before anything else, notification emails included:
    # a rejected request must not send mail on GenoLens' behalf.
    project = await assert_project_read_access(db, project_id, user_id)

    parent = None
    if comment.parent_id:
        parent = await comments_service.get_comment_by_id(db, comment.parent_id)
        if not parent or parent.project_id != project_id:
            raise HTTPException(status_code=400, detail="Invalid parent_id")

    try:
        new_comment = await comments_service.create_comment(
            db,
            project_id=project_id,
            user_id=user_id,
            content=comment.content,
            comment_type=CommentType(comment.comment_type.value),
            target_id=comment.target_id,
            parent_id=comment.parent_id,
            extra_metadata=comment.extra_metadata
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid comment")

    await history_service.log_activity(
        db, project_id, user_id, ActivityEventType.COMMENT_ADDED,
        entity_type="comment",
        entity_id=str(new_comment.id),
    )

    # ── Email notifications (fire-and-forget, non-blocking) ──────────
    # Wrapped in its own try/except: notification failures must never
    # break the main comment creation flow.
    try:
        author_email: str = current_user.email or ""
        project_name = project.name

        # 1. @email mentions
        mentioned_emails = email_service.extract_mentions(comment.content)
        for mentioned_email in mentioned_emails:
            if mentioned_email == author_email:
                continue  # Don't notify yourself
            await email_service.send_mention_notification(
                mentioned_email=mentioned_email,
                author_email=author_email,
                project_id=str(project_id),
                project_name=project_name,
                comment_id=str(new_comment.id),
                comment_excerpt=comment.content[:300],
            )

        # 2. Reply notification to parent comment author
        if parent and parent.user_id != user_id:
            parent_user = await lookup_user_by_id(parent.user_id)
            if parent_user:
                parent_author_email = parent_user.get("email", "")
                if parent_author_email and parent_author_email != author_email:
                    await email_service.send_reply_notification(
                        parent_author_email=parent_author_email,
                        replier_email=author_email,
                        project_id=str(project_id),
                        project_name=project_name,
                        comment_id=str(new_comment.id),
                        original_excerpt=parent.content[:200],
                        reply_excerpt=comment.content[:200],
                    )
    except Exception as notif_err:
        logger.warning("Email notification dispatch failed (non-critical): %s", notif_err)

    return new_comment


@router.patch("/comments/{comment_id}", response_model=ProjectCommentResponse)
async def update_comment(
    comment_id: UUID,
    update: ProjectCommentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Update a comment (content, resolved status, metadata).

    Any project member may resolve or reopen a comment; content and metadata
    stay with the author.
    """
    user_id = current_user.id
    await _load_accessible_comment(db, comment_id, user_id)

    try:
        updated_comment = await comments_service.update_comment(
            db,
            comment_id=comment_id,
            user_id=user_id,
            content=update.content,
            is_resolved=update.is_resolved,
            extra_metadata=update.extra_metadata
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Only the comment author can edit it")

    if not updated_comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    return updated_comment


@router.delete("/comments/{comment_id}", status_code=204)
async def delete_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Delete a comment (and all its replies)."""
    user_id = current_user.id
    await _load_accessible_comment(db, comment_id, user_id)

    try:
        success = await comments_service.delete_comment(
            db,
            comment_id=comment_id,
            user_id=user_id
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Only the comment author can delete it")

    if not success:
        raise HTTPException(status_code=404, detail="Comment not found")

    return None


# ============================================================================
# Comment Statistics
# ============================================================================

@router.get("/projects/{project_id}/comments/count", response_model=CommentCountResponse)
async def get_comment_count(
    project_id: UUID,
    target_id: Optional[str] = Query(None, description="Filter by target entity"),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comment count for a project or specific target."""
    await assert_project_read_access(db, project_id, current_user.id)

    total_count = await comments_service.get_comment_count(
        db,
        project_id=project_id,
        target_id=target_id
    )

    # Get counts by type
    by_type = {}
    for comment_type in CommentType:
        count = await comments_service.get_comment_count(
            db,
            project_id=project_id,
            target_id=target_id,
            comment_type=comment_type
        )
        if count > 0:
            by_type[comment_type.value] = count

    return {
        "count": total_count,
        "by_type": by_type
    }


@router.get("/users/me/comments", response_model=List[ProjectCommentResponse])
async def get_my_comments(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of comments"),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comments created by the current user."""
    return await comments_service.get_user_comments(
        db,
        user_id=current_user.id,
        project_id=project_id,
        limit=limit
    )
