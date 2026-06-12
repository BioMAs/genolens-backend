"""create_self_service_analyses_table

Creates the self_service_analyses table that backs the SelfServiceAnalysis model.
This table was introduced with the model but no create-table migration was ever
committed, so a fresh `alembic upgrade head` (e.g. in CI) fails when the later
add_data_type_to_analyses migration tries to ALTER a non-existent table.

The create is guarded by an inspector check so it is a no-op on databases where
the table already exists (e.g. production created it out-of-band).

Revision ID: create_self_service_analyses_001
Revises: add_scilicium_admin_role_001
Create Date: 2026-06-02
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "create_self_service_analyses_001"
down_revision: Union[str, None] = "add_scilicium_admin_role_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    if "self_service_analyses" in sa.inspect(conn).get_table_names():
        # Table already exists (created out-of-band); nothing to do.
        return

    op.create_table(
        "self_service_analyses",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "DONE",
                "FAILED",
                "CANCELLED",
                name="self_service_analysis_status",
            ),
            nullable=False,
        ),
        sa.Column("matrix_dataset_id", sa.UUID(), nullable=True),
        sa.Column("samples_dataset_id", sa.UUID(), nullable=True),
        sa.Column("comparisons_dataset_id", sa.UUID(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("result_dataset_ids", sa.JSON(), nullable=False),
        sa.Column("intermediate_dataset_ids", sa.JSON(), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("current_step", sa.String(100), nullable=True),
        sa.Column("progress_log", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["matrix_dataset_id"],
            ["datasets.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["samples_dataset_id"],
            ["datasets.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["comparisons_dataset_id"],
            ["datasets.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_self_service_analyses_project_id", "self_service_analyses", ["project_id"]
    )
    op.create_index(
        "ix_self_service_analyses_status", "self_service_analyses", ["status"]
    )
    op.create_index(
        "ix_self_service_analyses_user_id", "self_service_analyses", ["user_id"]
    )
    op.create_index(
        "ix_self_service_analyses_project_status",
        "self_service_analyses",
        ["project_id", "status"],
    )
    op.create_index(
        "ix_self_service_analyses_user", "self_service_analyses", ["user_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_self_service_analyses_user", table_name="self_service_analyses")
    op.drop_index(
        "ix_self_service_analyses_project_status", table_name="self_service_analyses"
    )
    op.drop_index("ix_self_service_analyses_user_id", table_name="self_service_analyses")
    op.drop_index("ix_self_service_analyses_status", table_name="self_service_analyses")
    op.drop_index(
        "ix_self_service_analyses_project_id", table_name="self_service_analyses"
    )
    op.drop_table("self_service_analyses")
    op.execute("DROP TYPE IF EXISTS self_service_analysis_status")
