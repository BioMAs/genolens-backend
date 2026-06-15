"""add_intersection_enrichment_jobs_table

Revision ID: intersection_enrich_001
Revises: cosmetics_module_001
Create Date: 2026-06-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "intersection_enrich_001"
down_revision: Union[str, None] = "cosmetics_module_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "intersection_enrichment_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "DONE", "FAILED",
                name="intersection_enrichment_status",
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
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_intersection_enrichment_jobs_project_id",
        "intersection_enrichment_jobs",
        ["project_id"],
    )
    op.create_index(
        "ix_intersection_enrichment_jobs_project_status",
        "intersection_enrichment_jobs",
        ["project_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_intersection_enrichment_jobs_project_status",
        table_name="intersection_enrichment_jobs",
    )
    op.drop_index(
        "ix_intersection_enrichment_jobs_project_id",
        table_name="intersection_enrichment_jobs",
    )
    op.drop_table("intersection_enrichment_jobs")
    op.execute("DROP TYPE IF EXISTS intersection_enrichment_status")
