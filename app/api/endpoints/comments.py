"""
API endpoints for project comments and annotations.
"""
import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.db import get_db
from app.api.deps.auth import get_current_user
from app.models.models import CommentType, Project
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
    current_user: dict = Depends(get_current_user)
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
    try:
        # Convert string to enum if provided
        type_filter = CommentType(comment_type) if comment_type else None
        
        comments = await comments_service.get_comments(
            db,
            project_id=project_id,
            comment_type=type_filter,
            target_id=target_id,
            include_resolved=include_resolved
        )
        return comments
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid comment_type: {e}")
    except Exception as e:
        logger.error(f"Error fetching comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comments/{comment_id}", response_model=ProjectCommentResponse)
async def get_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a single comment by ID."""
    try:
        comment = await comments_service.get_comment_by_id(db, comment_id)
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        return comment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comments/{comment_id}/thread", response_model=CommentThreadResponse)
async def get_comment_thread(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a comment thread (comment + all replies)."""
    try:
        thread = await comments_service.get_comment_thread(db, comment_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        return {
            "comment": thread[0],
            "reply_count": len(thread) - 1
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching comment thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/comments", response_model=ProjectCommentResponse, status_code=201)
async def create_comment(
    project_id: UUID,
    comment: ProjectCommentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new comment or reply."""
    try:
        user_id = UUID(current_user["sub"])

        # Convert enum to model enum
        comment_type = CommentType(comment.comment_type.value)

        new_comment = await comments_service.create_comment(
            db,
            project_id=project_id,
            user_id=user_id,
            content=comment.content,
            comment_type=comment_type,
            target_id=comment.target_id,
            parent_id=comment.parent_id,
            extra_metadata=comment.extra_metadata
        )
        await history_service.log_activity(
            db, project_id, user_id, ActivityEventType.COMMENT_ADDED,
            entity_type="comment",
            entity_id=str(new_comment.id),
        )

        # ── Email notifications (fire-and-forget, non-blocking) ──────────
        # Wrapped in its own try/except: notification failures must never
        # break the main comment creation flow.
        try:
            author_email: str = current_user.get("email", "")

            # Fetch project name once for all notifications
            project_result = await db.execute(select(Project).where(Project.id == project_id))
            project_obj = project_result.scalar_one_or_none()
            project_name = project_obj.name if project_obj else str(project_id)

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
            if comment.parent_id:
                parent = await comments_service.get_comment_by_id(db, comment.parent_id)
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/comments/{comment_id}", response_model=ProjectCommentResponse)
async def update_comment(
    comment_id: UUID,
    update: ProjectCommentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a comment (content, resolved status, metadata)."""
    try:
        user_id = UUID(current_user["sub"])
        
        updated_comment = await comments_service.update_comment(
            db,
            comment_id=comment_id,
            user_id=user_id,
            content=update.content,
            is_resolved=update.is_resolved,
            extra_metadata=update.extra_metadata
        )
        
        if not updated_comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        return updated_comment
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/comments/{comment_id}", status_code=204)
async def delete_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a comment (and all its replies)."""
    try:
        user_id = UUID(current_user["sub"])
        
        success = await comments_service.delete_comment(
            db,
            comment_id=comment_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        return None
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Comment Statistics
# ============================================================================

@router.get("/projects/{project_id}/comments/count", response_model=CommentCountResponse)
async def get_comment_count(
    project_id: UUID,
    target_id: Optional[str] = Query(None, description="Filter by target entity"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get comment count for a project or specific target."""
    try:
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
    except Exception as e:
        logger.error(f"Error getting comment count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/me/comments", response_model=List[ProjectCommentResponse])
async def get_my_comments(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of comments"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get comments created by the current user."""
    try:
        user_id = UUID(current_user["sub"])
        
        comments = await comments_service.get_user_comments(
            db,
            user_id=user_id,
            project_id=project_id,
            limit=limit
        )
        return comments
    except Exception as e:
        logger.error(f"Error fetching user comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))
