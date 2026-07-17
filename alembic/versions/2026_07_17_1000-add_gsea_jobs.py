"""add_gsea_jobs_table

Revision ID: gsea_jobs_001
Revises: modal_monthly_costs_001
Create Date: 2026-07-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "gsea_jobs_001"
down_revision: Union[str, None] = "modal_monthly_costs_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gsea_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "DONE", "FAILED",
                name="gsea_job_status",
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("params", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSON(astext_type=sa.Text()), nullable=True),
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
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_gsea_jobs_dataset_id", "gsea_jobs", ["dataset_id"])
    op.create_index("ix_gsea_jobs_project_id", "gsea_jobs", ["project_id"])
    op.create_index("ix_gsea_jobs_dataset_status", "gsea_jobs", ["dataset_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_gsea_jobs_dataset_status", table_name="gsea_jobs")
    op.drop_index("ix_gsea_jobs_project_id", table_name="gsea_jobs")
    op.drop_index("ix_gsea_jobs_dataset_id", table_name="gsea_jobs")
    op.drop_table("gsea_jobs")
    op.execute("DROP TYPE IF EXISTS gsea_job_status")
