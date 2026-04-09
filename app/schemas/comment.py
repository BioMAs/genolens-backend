"""
Pydantic schemas for project comments.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class CommentTypeEnum(str, Enum):
    """Comment types."""
    GENERAL = "GENERAL"
    GENE = "GENE"
    COMPARISON = "COMPARISON"
    PATHWAY = "PATHWAY"


# ============================================================================
# Comment Schemas
# ============================================================================

class ProjectCommentBase(BaseModel):
    """Base schema for project comments."""
    content: str = Field(..., description="Comment content (markdown supported)")
    comment_type: CommentTypeEnum = Field(
        CommentTypeEnum.GENERAL,
        description="Type of comment"
    )
    target_id: Optional[str] = Field(
        None,
        description="Target entity ID (gene_symbol, comparison_name, etc.)"
    )
    parent_id: Optional[UUID] = Field(
        None,
        description="Parent comment ID for replies"
    )
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (mentions, tags, etc.)"
    )


class ProjectCommentCreate(ProjectCommentBase):
    """Schema for creating a comment."""
    pass


class ProjectCommentUpdate(BaseModel):
    """Schema for updating a comment."""
    content: Optional[str] = Field(None, description="New comment content")
    is_resolved: Optional[bool] = Field(None, description="Resolved status")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class ProjectCommentResponse(ProjectCommentBase):
    """Schema for comment response."""
    id: UUID
    project_id: UUID
    user_id: UUID
    is_resolved: bool
    created_at: datetime
    updated_at: datetime
    replies: List["ProjectCommentResponse"] = Field(
        default_factory=list,
        description="Nested replies"
    )

    class Config:
        from_attributes = True


# Enable forward references
ProjectCommentResponse.model_rebuild()


class CommentThreadResponse(BaseModel):
    """Schema for comment thread response."""
    comment: ProjectCommentResponse
    reply_count: int = Field(..., description="Total number of replies in thread")


class CommentCountResponse(BaseModel):
    """Schema for comment count response."""
    count: int = Field(..., description="Number of comments")
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by comment type"
    )


class CommentStatsResponse(BaseModel):
    """Schema for comment statistics."""
    total_comments: int
    by_type: Dict[str, int]
    by_target: Dict[str, int]
    resolved_count: int
    unresolved_count: int
