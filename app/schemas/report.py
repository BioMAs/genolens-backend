from uuid import UUID
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, field_validator
from app.models.report_job import ReportJobStatus

# Comparison report formats: the branded LaTeX report, or the OmicSkin editorial PDF.
ReportTemplate = Literal["latex", "editorial"]


class AITextRecord(BaseModel):
    """One AI text a report printed: which one ("nova": the reading of the nova; "comparison": the
    interpretation of the comparison), the model that wrote it, and when."""

    kind: Literal["nova", "comparison"]
    model: str
    generated_at: Optional[datetime] = None


class ReportJobResponse(BaseModel):
    id: UUID
    project_id: UUID
    analysis_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    comparison_name: Optional[str] = None
    template: str = "latex"
    use_ai: bool = True
    # The AI texts the report printed (null: not recorded, e.g. a job not finished or made by an
    # older server; []: none printed).
    ai_texts: Optional[list[AITextRecord]] = None
    celery_task_id: Optional[str]
    status: ReportJobStatus
    pdf_path: Optional[str]
    error_message: Optional[str]
    requested_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("use_ai", mode="before")
    @classmethod
    def _unset_use_ai_is_the_default(cls, value):
        # A job not yet flushed has no column default applied: it is the default, True.
        return True if value is None else value


class ReportTriggerResponse(BaseModel):
    job_id: UUID
    status: ReportJobStatus
    message: str


class CoverInfo(BaseModel):
    """Project/cover information shown on the report's first (and contact) page."""
    project_name: Optional[str] = None
    client_ref: Optional[str] = None
    sponsor_name: Optional[str] = None
    sponsor_contact: Optional[str] = None
    sponsor_email: Optional[str] = None
    sponsor_address: Optional[str] = None
    test_facility_name: Optional[str] = None
    test_facility_contact: Optional[str] = None
    test_facility_email: Optional[str] = None
    test_facility_address: Optional[str] = None
    test_site_name: Optional[str] = None
    test_site_contact: Optional[str] = None
    test_site_email: Optional[str] = None
    test_site_address: Optional[str] = None
    prepared_by: Optional[str] = None
    checked_by: Optional[str] = None
    approved_by: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


# Allowed page-layout values
FIRST_PAGE_TYPES = ("detailed", "simple", "cover")
LAST_PAGE_TYPES = ("color", "contact")


class AnalysisReportTriggerRequest(BaseModel):
    """Optional body of the analysis report request. `use_ai` False gives a report without any AI
    text (no LLM call, no stored interpretation read, no AI label)."""

    use_ai: bool = True


class ComparisonReportTriggerRequest(BaseModel):
    """Optional per-report customization (requires the report customization
    module to be applied; ignored otherwise).

    `use_ai` is read for every caller: False gives a report without any AI text (no LLM
    call, no stored interpretation read, no AI label)."""
    conclusion: Optional[str] = None
    materials_methods: Optional[str] = None
    first_page_type: Optional[str] = None
    last_page_type: Optional[str] = None
    cover_info: Optional[CoverInfo] = None
    use_ai: bool = True


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
    first_page_type: str = "detailed"
    last_page_type: str = "color"
    cover_info: Optional[CoverInfo] = None

    model_config = {"from_attributes": True}


class ReportSettingsUpdate(BaseModel):
    institute_name: Optional[str] = None
    institute_address: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    default_materials_methods: Optional[str] = None
    default_conclusion: Optional[str] = None
    first_page_type: Optional[str] = None
    last_page_type: Optional[str] = None
    cover_info: Optional[CoverInfo] = None
