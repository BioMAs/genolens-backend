"""add analysis_id to report_jobs (analysis-scoped SciLicium report)

Revision ID: report_jobs_analysis_001
Revises: add_user_status_001
Create Date: 2026-06-14 12:00:00
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "report_jobs_analysis_001"
down_revision: Union[str, None] = "add_user_status_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "report_jobs",
        sa.Column("analysis_id", sa.UUID(), nullable=True),
    )
    op.create_index("ix_report_jobs_analysis_id", "report_jobs", ["analysis_id"])
    op.create_foreign_key(
        "fk_report_jobs_analysis_id",
        "report_jobs",
        "self_service_analyses",
        ["analysis_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_report_jobs_analysis_id", "report_jobs", type_="foreignkey")
    op.drop_index("ix_report_jobs_analysis_id", table_name="report_jobs")
    op.drop_column("report_jobs", "analysis_id")
