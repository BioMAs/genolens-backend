"""
Pydantic schemas for Dataset endpoints.
"""
from datetime import datetime
from typing import Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field

from app.models.models import DatasetType, DatasetStatus


class DatasetBase(BaseModel):
    """Base dataset schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    type: DatasetType
    column_mapping: dict[str, str] = Field(default_factory=dict)


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset."""
    project_id: UUID


class DatasetUpdate(BaseModel):
    """Schema for updating a dataset."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    type: Optional[DatasetType] = None
    column_mapping: Optional[dict[str, str]] = None
    dataset_metadata: Optional[dict[str, Any]] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset response."""
    id: UUID
    project_id: UUID
    status: DatasetStatus
    raw_file_path: Optional[str] = None
    parquet_file_path: Optional[str] = None
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    """Schema for dataset upload response."""
    dataset_id: UUID
    message: str
    status: DatasetStatus


class DatasetQueryParams(BaseModel):
    """Schema for dataset query parameters."""
    gene_ids: Optional[list[str]] = Field(None, description="Filter by gene IDs")
    sample_ids: Optional[list[str]] = Field(None, description="Filter by sample IDs")
    limit: int = Field(default=100, ge=1, le=100000, description="Maximum rows to return")
    offset: int = Field(default=0, ge=0, description="Number of rows to skip")


class DatasetQueryResponse(BaseModel):
    """Schema for dataset query response."""
    columns: list[str]
    data: list[dict[str, Any]]
    total_rows: int
    returned_rows: int
