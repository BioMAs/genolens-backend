"""add report customization module + comparison-scoped report fields

- report_jobs: dataset_id, comparison_name, conclusion, materials_methods
  (comparison-scoped report; per-report editable content)
- users.report_customization_module_enabled (per-user unlock toggled by admins)
- user_report_settings (persistent per-user report branding)

Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: report_customization_001
Revises: intersection_enrich_001
Create Date: 2026-06-24 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "report_customization_001"
down_revision: Union[str, None] = "intersection_enrich_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # --- report_jobs: comparison-scoped fields ---------------------------
    report_columns = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "dataset_id" not in report_columns:
        op.add_column("report_jobs", sa.Column("dataset_id", sa.UUID(), nullable=True))
        op.create_index("ix_report_jobs_dataset_id", "report_jobs", ["dataset_id"])
        op.create_foreign_key(
            "fk_report_jobs_dataset_id",
            "report_jobs",
            "datasets",
            ["dataset_id"],
            ["id"],
            ondelete="CASCADE",
        )
        op.create_index(
            "ix_report_jobs_dataset_comparison",
            "report_jobs",
            ["dataset_id", "comparison_name"],
        )
    if "comparison_name" not in report_columns:
        op.add_column(
            "report_jobs", sa.Column("comparison_name", sa.String(length=512), nullable=True)
        )
    if "conclusion" not in report_columns:
        op.add_column("report_jobs", sa.Column("conclusion", sa.Text(), nullable=True))
    if "materials_methods" not in report_columns:
        op.add_column("report_jobs", sa.Column("materials_methods", sa.Text(), nullable=True))

    # --- users.report_customization_module_enabled -----------------------
    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "report_customization_module_enabled" not in user_columns:
        op.add_column(
            "users",
            sa.Column(
                "report_customization_module_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )

    # --- user_report_settings -------------------------------------------
    if "user_report_settings" not in existing_tables:
        op.create_table(
            "user_report_settings",
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("logo_path", sa.String(length=1024), nullable=True),
            sa.Column("institute_name", sa.String(length=255), nullable=True),
            sa.Column("institute_address", sa.String(length=512), nullable=True),
            sa.Column("primary_color", sa.String(length=16), nullable=True),
            sa.Column("secondary_color", sa.String(length=16), nullable=True),
            sa.Column("default_materials_methods", sa.Text(), nullable=True),
            sa.Column("default_conclusion", sa.Text(), nullable=True),
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
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("user_id"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "user_report_settings" in existing_tables:
        op.drop_table("user_report_settings")

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "report_customization_module_enabled" in user_columns:
        op.drop_column("users", "report_customization_module_enabled")

    report_columns = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "dataset_id" in report_columns:
        op.drop_index("ix_report_jobs_dataset_comparison", table_name="report_jobs")
        op.drop_constraint("fk_report_jobs_dataset_id", "report_jobs", type_="foreignkey")
        op.drop_index("ix_report_jobs_dataset_id", table_name="report_jobs")
        op.drop_column("report_jobs", "dataset_id")
    for col in ("comparison_name", "conclusion", "materials_methods"):
        if col in report_columns:
            op.drop_column("report_jobs", col)
