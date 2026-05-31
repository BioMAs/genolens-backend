"""add_report_jobs_table

Revision ID: report_jobs_001
Revises: bm_plans_001
Create Date: 2026-05-31
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "report_jobs_001"
down_revision: Union[str, None] = "bm_plans_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "report_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "DONE", "FAILED", name="report_job_status"),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("pdf_path", sa.String(1024), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("requested_by", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_report_jobs_project_id", "report_jobs", ["project_id"])
    op.create_index(
        "ix_report_jobs_project_status",
        "report_jobs",
        ["project_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_report_jobs_project_status", table_name="report_jobs")
    op.drop_index("ix_report_jobs_project_id", table_name="report_jobs")
    op.drop_table("report_jobs")
    op.execute("DROP TYPE IF EXISTS report_job_status")
