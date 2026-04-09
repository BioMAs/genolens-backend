"""
Service for managing project comments and annotations.
"""
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, delete
from sqlalchemy.orm import joinedload

from app.models.models import ProjectComment, CommentType, Project
from app.core.monitoring import timing_decorator

logger = logging.getLogger(__name__)


class CommentsService:
    """Service for project comments and annotations."""

    @timing_decorator(name="get_comments")
    async def get_comments(
        self,
        db: AsyncSession,
        project_id: UUID,
        comment_type: Optional[CommentType] = None,
        target_id: Optional[str] = None,
        parent_id: Optional[UUID] = None,
        include_resolved: bool = True
    ) -> List[ProjectComment]:
        """
        Get comments for a project with optional filters.
        
        Args:
            db: Database session
            project_id: Project ID
            comment_type: Optional filter by comment type
            target_id: Optional filter by target entity
            parent_id: Optional filter by parent comment (None = top-level only)
            include_resolved: Whether to include resolved comments
            
        Returns:
            List of comments
        """
        query = select(ProjectComment).where(
            ProjectComment.project_id == project_id
        ).options(
            joinedload(ProjectComment.replies)
        ).order_by(ProjectComment.created_at.desc())
        
        if comment_type:
            query = query.where(ProjectComment.comment_type == comment_type)
        
        if target_id is not None:
            query = query.where(ProjectComment.target_id == target_id)
        
        if parent_id is not None:
            query = query.where(ProjectComment.parent_id == parent_id)
        elif parent_id is None and target_id is None:
            # If not filtering by parent_id or target_id, only show top-level
            query = query.where(ProjectComment.parent_id.is_(None))
        
        if not include_resolved:
            query = query.where(ProjectComment.is_resolved == False)
        
        result = await db.execute(query)
        return result.scalars().all()

    @timing_decorator(name="get_comment_by_id")
    async def get_comment_by_id(
        self,
        db: AsyncSession,
        comment_id: UUID
    ) -> Optional[ProjectComment]:
        """
        Get a comment by ID.
        
        Args:
            db: Database session
            comment_id: Comment ID
            
        Returns:
            Comment or None
        """
        query = select(ProjectComment).where(
            ProjectComment.id == comment_id
        ).options(
            joinedload(ProjectComment.replies),
            joinedload(ProjectComment.parent)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @timing_decorator(name="get_comment_thread")
    async def get_comment_thread(
        self,
        db: AsyncSession,
        comment_id: UUID
    ) -> List[ProjectComment]:
        """
        Get a comment and all its replies (thread).
        
        Args:
            db: Database session
            comment_id: Comment ID (root of thread)
            
        Returns:
            List of comments in thread
        """
        # Get the root comment
        root = await self.get_comment_by_id(db, comment_id)
        if not root:
            return []
        
        # Get all replies recursively
        thread = [root]
        
        async def get_replies_recursive(parent: ProjectComment):
            if parent.replies:
                for reply in parent.replies:
                    thread.append(reply)
                    await get_replies_recursive(reply)
        
        await get_replies_recursive(root)
        return thread

    @timing_decorator(name="create_comment")
    async def create_comment(
        self,
        db: AsyncSession,
        project_id: UUID,
        user_id: UUID,
        content: str,
        comment_type: CommentType = CommentType.GENERAL,
        target_id: Optional[str] = None,
        parent_id: Optional[UUID] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> ProjectComment:
        """
        Create a new comment.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID (from Supabase Auth)
            content: Comment content (markdown supported)
            comment_type: Type of comment
            target_id: Optional target entity ID
            parent_id: Optional parent comment ID for replies
            extra_metadata: Optional metadata (mentions, tags, etc.)
            
        Returns:
            Created comment
        """
        # Verify project exists
        project_query = select(Project).where(Project.id == project_id)
        project_result = await db.execute(project_query)
        project = project_result.scalar_one_or_none()
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # If parent_id provided, verify parent exists and belongs to same project
        if parent_id:
            parent_query = select(ProjectComment).where(ProjectComment.id == parent_id)
            parent_result = await db.execute(parent_query)
            parent = parent_result.scalar_one_or_none()
            if not parent:
                raise ValueError(f"Parent comment {parent_id} not found")
            if parent.project_id != project_id:
                raise ValueError("Parent comment belongs to different project")
        
        comment = ProjectComment(
            project_id=project_id,
            user_id=user_id,
            content=content,
            comment_type=comment_type,
            target_id=target_id,
            parent_id=parent_id,
            extra_metadata=extra_metadata or {}
        )
        
        db.add(comment)
        await db.commit()
        await db.refresh(comment)
        
        logger.info(f"Created comment {comment.id} for project {project_id}")
        return comment

    @timing_decorator(name="update_comment")
    async def update_comment(
        self,
        db: AsyncSession,
        comment_id: UUID,
        user_id: UUID,
        content: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProjectComment]:
        """
        Update a comment (only by owner).
        
        Args:
            db: Database session
            comment_id: Comment ID
            user_id: User ID (must match comment owner)
            content: Optional new content
            is_resolved: Optional resolved status
            extra_metadata: Optional new metadata
            
        Returns:
            Updated comment or None if not found/unauthorized
        """
        query = select(ProjectComment).where(ProjectComment.id == comment_id)
        result = await db.execute(query)
        comment = result.scalar_one_or_none()
        
        if not comment:
            return None
        
        # Only owner can update (except for is_resolved which can be set by project members)
        if comment.user_id != user_id and content is not None:
            raise PermissionError("Only comment owner can edit content")
        
        if content is not None:
            comment.content = content
        
        if is_resolved is not None:
            comment.is_resolved = is_resolved
        
        if extra_metadata is not None:
            comment.extra_metadata = extra_metadata
        
        await db.commit()
        await db.refresh(comment)
        
        logger.info(f"Updated comment {comment_id}")
        return comment

    @timing_decorator(name="delete_comment")
    async def delete_comment(
        self,
        db: AsyncSession,
        comment_id: UUID,
        user_id: UUID
    ) -> bool:
        """
        Delete a comment (only by owner). Cascades to replies.
        
        Args:
            db: Database session
            comment_id: Comment ID
            user_id: User ID (must match comment owner)
            
        Returns:
            True if deleted, False if not found/unauthorized
        """
        query = select(ProjectComment).where(ProjectComment.id == comment_id)
        result = await db.execute(query)
        comment = result.scalar_one_or_none()
        
        if not comment:
            return False
        
        # Only owner can delete
        if comment.user_id != user_id:
            raise PermissionError("Only comment owner can delete")
        
        await db.delete(comment)
        await db.commit()
        
        logger.info(f"Deleted comment {comment_id}")
        return True

    @timing_decorator(name="get_comment_count")
    async def get_comment_count(
        self,
        db: AsyncSession,
        project_id: UUID,
        target_id: Optional[str] = None,
        comment_type: Optional[CommentType] = None
    ) -> int:
        """
        Get count of comments for project/target.
        
        Args:
            db: Database session
            project_id: Project ID
            target_id: Optional target entity ID
            comment_type: Optional comment type filter
            
        Returns:
            Count of comments
        """
        query = select(func.count(ProjectComment.id)).where(
            ProjectComment.project_id == project_id
        )
        
        if target_id is not None:
            query = query.where(ProjectComment.target_id == target_id)
        
        if comment_type is not None:
            query = query.where(ProjectComment.comment_type == comment_type)
        
        result = await db.execute(query)
        return result.scalar()

    @timing_decorator(name="get_user_comments")
    async def get_user_comments(
        self,
        db: AsyncSession,
        user_id: UUID,
        project_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[ProjectComment]:
        """
        Get comments by a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            project_id: Optional project filter
            limit: Maximum number of comments
            
        Returns:
            List of comments
        """
        query = select(ProjectComment).where(
            ProjectComment.user_id == user_id
        ).order_by(ProjectComment.created_at.desc()).limit(limit)
        
        if project_id:
            query = query.where(ProjectComment.project_id == project_id)
        
        result = await db.execute(query)
        return result.scalars().all()


# Singleton instance
comments_service = CommentsService()
