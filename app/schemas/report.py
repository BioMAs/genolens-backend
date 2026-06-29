from uuid import UUID
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from app.models.report_job import ReportJobStatus


class ReportJobResponse(BaseModel):
    id: UUID
    project_id: UUID
    analysis_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    comparison_name: Optional[str] = None
    celery_task_id: Optional[str]
    status: ReportJobStatus
    pdf_path: Optional[str]
    error_message: Optional[str]
    requested_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReportTriggerResponse(BaseModel):
    job_id: UUID
    status: ReportJobStatus
    message: str


class ComparisonReportTriggerRequest(BaseModel):
    """Optional per-report customization content (requires the report
    customization module to be applied; ignored otherwise)."""
    conclusion: Optional[str] = None
    materials_methods: Optional[str] = None


class ReportDownloadResponse(BaseModel):
    signed_url: str


class ReportSettingsResponse(BaseModel):
    logo_path: Optional[str] = None
    institute_name: Optional[str] = None
    institute_address: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    default_materials_methods: Optional[str] = None
    default_conclusion: Optional[str] = None

    model_config = {"from_attributes": True}


class ReportSettingsUpdate(BaseModel):
    institute_name: Optional[str] = None
    institute_address: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    default_materials_methods: Optional[str] = None
    default_conclusion: Optional[str] = None
