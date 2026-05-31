from uuid import UUID
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from app.models.report_job import ReportJobStatus


class ReportJobResponse(BaseModel):
    id: UUID
    project_id: UUID
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


class ReportDownloadResponse(BaseModel):
    signed_url: str
