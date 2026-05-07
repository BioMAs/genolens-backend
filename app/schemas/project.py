"""
Pydantic schemas for Project endpoints.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class UserRole(str, Enum):
    """User role enum matching backend models."""
    ADMIN = "ADMIN"
    USER = "USER"


class ProjectBase(BaseModel):
    """Base project schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    species: Optional[str] = Field(
        None,
        description="Organism species for functional enrichment (human, mouse, rat, zebrafish, pig)"
    )


class ProjectCreate(ProjectBase):
    """Schema for creating a project."""
    pass


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    species: Optional[str] = None


class ProjectResponse(ProjectBase):
    """Schema for project response."""
    id: UUID
    owner_id: UUID
    species: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectListResponse(BaseModel):
    """Schema for paginated project list."""
    items: list[ProjectResponse]
    total: int
    page: int
    page_size: int


class ComparisonSummary(BaseModel):
    """Summary of a single comparison."""
    name: str
    deg_up: int = 0
    deg_down: int = 0
    deg_total: int = 0
    has_enrichment: bool = False
    dataset_id: UUID
    dataset_type: str  # 'SINGLE' or 'GLOBAL'


class PaginatedComparisonsResponse(BaseModel):
    """Paginated list of comparisons."""
    comparisons: list[ComparisonSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class ProjectStats(BaseModel):
    """Project statistics."""
    total_datasets: int
    total_comparisons: int
    processing_count: int
    ready_count: int
    failed_count: int
    original_files_count: int


class ProjectSummaryResponse(BaseModel):
    """Optimized project summary with pre-computed stats."""
    project: ProjectResponse
    stats: ProjectStats
    comparisons: list[ComparisonSummary]
    original_files: list[str]  # Just file names


class DatasetBreakdown(BaseModel):
    """Dataset counts broken down by type."""
    matrix: int = 0
    deg: int = 0
    enrichment: int = 0
    metadata: int = 0
    other: int = 0


class ActivityBreakdown(BaseModel):
    """Last 7 days activity counts."""
    datasets_uploaded: int = 0
    bookmarks_created: int = 0
    comments_added: int = 0
    analyses_run: int = 0


class ProjectDashboardStats(BaseModel):
    """
    Rich project statistics for the dashboard overview card.
    Aggregates counts across all project entities.
    """
    # Core
    project_id: UUID
    project_name: str

    # Datasets
    total_datasets: int = 0
    datasets_ready: int = 0
    datasets_processing: int = 0
    datasets_failed: int = 0
    dataset_breakdown: DatasetBreakdown = DatasetBreakdown()

    # Analysis output
    total_comparisons: int = 0
    total_deg_genes: int = 0
    total_enrichment_pathways: int = 0

    # Collaboration
    total_bookmarks: int = 0
    total_gene_lists: int = 0
    total_comments: int = 0
    total_members: int = 0

    # Activity
    total_activity_events: int = 0
    activity_last_7_days: ActivityBreakdown = ActivityBreakdown()
    last_activity_at: Optional[datetime] = None


# ============================================
# Project Member Schemas (Collaboration)
# ============================================

class ProjectMemberBase(BaseModel):
    """Base schema for project member."""
    access_level: UserRole = UserRole.USER


class ProjectMemberCreate(ProjectMemberBase):
    """Schema for inviting a user to a project."""
    user_email: EmailStr = Field(..., description="Email of user to invite")


class ProjectMemberUpdate(BaseModel):
    """Schema for updating a project member's access level."""
    access_level: UserRole


class ProjectMemberResponse(ProjectMemberBase):
    """Schema for project member response."""
    id: UUID
    project_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    # Optional user info (populated from external service)
    user_email: Optional[str] = None
    user_name: Optional[str] = None

    model_config = {"from_attributes": True}


class ProjectMemberListResponse(BaseModel):
    """Schema for listing project members."""
    members: list[ProjectMemberResponse]
    total: int
