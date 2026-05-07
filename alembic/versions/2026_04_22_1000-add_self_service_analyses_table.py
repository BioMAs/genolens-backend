"""add_self_service_analyses_table

Revision ID: add_self_service_analyses
Revises: a3f8c2e1b4d9
Create Date: 2026-04-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "add_self_service_analyses"
down_revision: Union[str, None] = "a3f8c2e1b4d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum type (idempotent)
    op.execute(sa.text(
        "DO $$ BEGIN "
        "CREATE TYPE self_service_analysis_status AS ENUM "
        "('PENDING', 'RUNNING', 'DONE', 'FAILED', 'CANCELLED'); "
        "EXCEPTION WHEN duplicate_object THEN null; "
        "END $$;"
    ))

    op.create_table(
        "self_service_analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "DONE", "FAILED", "CANCELLED",
                name="self_service_analysis_status",
                create_type=False,
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column(
            "matrix_dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "samples_dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "comparisons_dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("result_dataset_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("current_step", sa.String(100), nullable=True),
        sa.Column("progress_log", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index(
        "ix_self_service_analyses_project_status",
        "self_service_analyses",
        ["project_id", "status"],
    )
    op.create_index(
        "ix_self_service_analyses_user",
        "self_service_analyses",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_self_service_analyses_user", "self_service_analyses")
    op.drop_index("ix_self_service_analyses_project_status", "self_service_analyses")
    op.drop_table("self_service_analyses")
    op.execute(sa.text("DROP TYPE IF EXISTS self_service_analysis_status"))
