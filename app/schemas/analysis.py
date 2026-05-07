"""
Pydantic schemas for SelfServiceAnalysis endpoints.
"""
import re
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class AnalysisParams(BaseModel):
    """Parameters for a self-service DESeq2 analysis."""

    design: str = Field(
        "auto",
        description=(
            "DESeq2 design formula: "
            "'auto' (uses batch+condition if batch column is present), "
            "'condition' (~ condition), "
            "'batch_condition' (~ batch + condition)"
        ),
    )
    fdr: float = Field(0.05, ge=0.001, le=0.5, description="FDR threshold (adjusted p-value)")
    min_log2fc: float = Field(1.0, ge=0.0, le=10.0, description="Minimum absolute log2 fold-change")
    min_reads: int = Field(100000, ge=0, description="Minimum total reads per sample for QC")
    min_genes: int = Field(500, ge=0, description="Minimum genes detected per sample for QC")
    min_count: int = Field(10, ge=1, description="Minimum count for gene-level filtering")
    min_reps: int = Field(2, ge=1, description="Min samples with min_count for gene to pass")
    threads: int = Field(2, ge=1, le=8, description="Number of parallel threads for DESeq2")
    enrichment_databases: Optional[list[str]] = Field(
        None,
        description="Annotation databases to use for functional enrichment. null = all available databases."
    )
    species: str = Field(
        "human",
        description="Organism species for functional enrichment (human, mouse, rat, zebrafish, pig)"
    )


class SelfServiceAnalysisCreate(BaseModel):
    """Payload to create and launch a new self-service DE analysis."""

    project_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    matrix_dataset_id: UUID
    samples_dataset_id: UUID
    comparisons_dataset_id: UUID
    params: AnalysisParams = Field(default_factory=AnalysisParams)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9\s_\-()\.,]+$", v):
            raise ValueError("Name contains invalid characters")
        return v.strip()


class SelfServiceAnalysisResponse(BaseModel):
    """Full representation of a self-service analysis (returned by GET and POST)."""

    id: UUID
    project_id: UUID
    name: str
    status: str
    matrix_dataset_id: Optional[UUID]
    samples_dataset_id: Optional[UUID]
    comparisons_dataset_id: Optional[UUID]
    params: dict
    result_dataset_ids: list
    intermediate_dataset_ids: dict = {}
    celery_task_id: Optional[str]
    current_step: Optional[str]
    progress_log: list
    error_message: Optional[str]
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SelfServiceAnalysisUploadCreate(BaseModel):
    """Form-parsed payload for the multipart upload endpoint."""

    project_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    params: AnalysisParams = Field(default_factory=AnalysisParams)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9\s_\-()\.,]+$", v):
            raise ValueError("Name contains invalid characters")
        return v.strip()


class SelfServiceAnalysisListResponse(BaseModel):
    """Paginated list of self-service analyses."""

    items: list[SelfServiceAnalysisResponse]
    total: int
