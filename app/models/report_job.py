import enum
from uuid import UUID, uuid4
from typing import Optional
from sqlalchemy import String, ForeignKey, Text, Enum as SQLEnum, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ReportJobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE    = "DONE"
    FAILED  = "FAILED"


class ReportJob(Base, TimestampMixin):
    __tablename__ = "report_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Analysis-scoped report (the SciLicium LaTeX report covers one analysis).
    analysis_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("self_service_analyses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[ReportJobStatus] = mapped_column(
        SQLEnum(ReportJobStatus, name="report_job_status"),
        nullable=False,
        default=ReportJobStatus.PENDING,
    )
    pdf_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requested_by: Mapped[UUID] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_report_jobs_project_status", "project_id", "status"),
    )
