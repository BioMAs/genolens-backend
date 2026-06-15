import enum
from uuid import UUID, uuid4
from typing import Optional

from sqlalchemy import String, ForeignKey, Text, JSON, Enum as SQLEnum, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class IntersectionEnrichmentStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class IntersectionEnrichmentJob(Base, TimestampMixin):
    """
    Ad-hoc annoDB functional enrichment of a Venn/UpSet intersection's gene set.

    Runs on the r-worker (functional_enrichment.R --gene-list), mirroring the
    report-job pattern. `params` holds the request (label, gene_count, species);
    `result` holds the parsed enrichment rows once DONE.
    """

    __tablename__ = "intersection_enrichment_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[IntersectionEnrichmentStatus] = mapped_column(
        SQLEnum(IntersectionEnrichmentStatus, name="intersection_enrichment_status"),
        nullable=False,
        default=IntersectionEnrichmentStatus.PENDING,
    )
    # Request context: {label, gene_count, species, genes: [...]}
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Parsed enrichment rows once DONE: [{pathway_id, pathway_name, category, ...}]
    result: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requested_by: Mapped[UUID] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_intersection_enrichment_jobs_project_status", "project_id", "status"),
    )
