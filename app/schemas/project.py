"""
Pydantic schemas for Project endpoints.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ProjectBase(BaseModel):
    """Base project schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    """Schema for creating a project."""
    pass


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectResponse(ProjectBase):
    """Schema for project response."""
    id: UUID
    owner_id: UUID
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
