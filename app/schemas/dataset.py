"""
Pydantic schemas for Dataset endpoints with enhanced validation.
"""
from datetime import datetime
from typing import Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator
import re

from app.models.models import DatasetType, DatasetStatus


class DatasetBase(BaseModel):
    """Base dataset schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    type: DatasetType
    column_mapping: dict[str, str] = Field(default_factory=dict)


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset."""
    project_id: UUID

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate dataset name: alphanumeric, spaces, hyphens, underscores only."""
        if not re.match(r'^[a-zA-Z0-9\s_\-\'éèêëàâîïôùûüçœæ().,+&]+$', v, re.UNICODE):
            raise ValueError(
                'Dataset name contains invalid characters'
            )
        return v.strip()

    @field_validator('column_mapping')
    @classmethod
    def validate_column_mapping(cls, v: dict) -> dict:
        """Validate column mapping contains only safe characters."""
        for key, value in v.items():
            if not re.match(r'^[a-zA-Z0-9_.:/ -]+$', key):
                raise ValueError(f'Invalid column name: {key}')
            if not re.match(r'^[a-zA-Z0-9_.:/ -]+$', value):
                raise ValueError(f'Invalid mapped column name: {value}')
        return v


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
    """Schema for dataset query parameters with validation."""
    gene_ids: Optional[list[str]] = Field(None, description="Filter by gene IDs", max_length=10000)
    sample_ids: Optional[list[str]] = Field(None, description="Filter by sample IDs", max_length=1000)
    limit: int = Field(default=100, ge=1, le=100000, description="Maximum rows to return")
    offset: int = Field(default=0, ge=0, description="Number of rows to skip")
    # Performance optimization: backend filtering for DEG data
    padj_max: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter rows where padj <= this value")
    padj_min: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter rows where padj >= this value")
    logfc_min: Optional[float] = Field(None, ge=-50.0, le=50.0, description="Filter rows where |logFC| >= this value (absolute)")
    logfc_max: Optional[float] = Field(None, ge=-50.0, le=50.0, description="Filter rows where |logFC| <= this value (absolute)")
    columns: Optional[list[str]] = Field(None, description="Select specific columns to return", max_length=500)
    sort_by: Optional[str] = Field(None, max_length=100, description="Column name to sort by")
    sort_desc: bool = Field(default=True, description="Sort in descending order")
    
    @field_validator('gene_ids', 'sample_ids', 'columns')
    @classmethod
    def validate_id_lists(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate ID lists contain safe characters only."""
        if v is None:
            return v
        for item in v:
            if not re.match(r'^[a-zA-Z0-9_.:/-]+$', item):
                raise ValueError(f'Invalid identifier: {item}. Must contain only alphanumeric, underscore, dot, hyphen, colon, or slash.')
        return v
    
    @field_validator('sort_by')
    @classmethod
    def validate_sort_by(cls, v: Optional[str]) -> Optional[str]:
        """Validate sort_by column name."""
        if v is None:
            return v
        if not re.match(r'^[a-zA-Z0-9_.:/-]+$', v):
            raise ValueError(f'Invalid column name for sorting: {v}')
        return v
    
    @model_validator(mode='after')
    def validate_padj_range(self):
        """Validate padj_min <= padj_max."""
        if self.padj_min is not None and self.padj_max is not None:
            if self.padj_min > self.padj_max:
                raise ValueError('padj_min must be less than or equal to padj_max')
        return self
    
    @model_validator(mode='after')
    def validate_logfc_range(self):
        """Validate logfc ranges are logical."""
        if self.logfc_min is not None and self.logfc_max is not None:
            if abs(self.logfc_min) > abs(self.logfc_max):
                raise ValueError('logfc_min (absolute) must be less than or equal to logfc_max (absolute)')
        return self


class DatasetQueryResponse(BaseModel):
    """Schema for dataset query response."""
    columns: list[str]
    data: list[dict[str, Any]]
    total_rows: int
    returned_rows: int


class DatasetColumnsResponse(BaseModel):
    """Schema for dataset columns response."""
    columns: list[str]
    total_columns: int


class DatasetStatsResponse(BaseModel):
    """Schema for dataset statistics response."""
    total_genes: int
    up_genes: Optional[int] = None
    down_genes: Optional[int] = None
    significant_genes: Optional[int] = None
    comparisons: Optional[list[str]] = None
    cached: bool = Field(default=False, description="Whether stats were from cache")


class GeneListResponse(BaseModel):
    """Schema for gene list response."""
    genes: list[str]
    total_genes: int
