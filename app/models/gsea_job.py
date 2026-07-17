import enum
from uuid import UUID, uuid4
from typing import Optional

from sqlalchemy import String, ForeignKey, Text, JSON, Enum as SQLEnum, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class GSEAJobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class GSEAJob(Base, TimestampMixin):
    """
    Asynchronous pre-ranked GSEA run for a comparison.

    GSEA over a full gene-set database with permutation-based FDR takes minutes,
    which exceeds HTTP timeouts — so it runs as a Celery task (pure Python, default
    queue). `params` holds the request (comparison_name + GSEA parameters); `result`
    holds the response payload (summary + lean per-set results) once DONE.
    """

    __tablename__ = "gsea_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[GSEAJobStatus] = mapped_column(
        SQLEnum(GSEAJobStatus, name="gsea_job_status"),
        nullable=False,
        default=GSEAJobStatus.PENDING,
    )
    # Request context: {comparison_name, gene_set_database, ranking_metric, ...}
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Full GSEA response payload once DONE: {dataset_id, comparison_name, parameters, summary, results}
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requested_by: Mapped[UUID] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_gsea_jobs_dataset_status", "dataset_id", "status"),
    )
