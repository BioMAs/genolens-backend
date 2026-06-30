"""add report page-model + cover-info fields

- user_report_settings: first_page_type, last_page_type, cover_info (defaults)
- report_jobs: first_page_type, last_page_type, cover_info (per-report overrides)

Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: report_page_models_001
Revises: report_customization_001
Create Date: 2026-06-30 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "report_page_models_001"
down_revision: Union[str, None] = "report_customization_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    settings_cols = {c["name"] for c in inspector.get_columns("user_report_settings")}
    if "first_page_type" not in settings_cols:
        op.add_column(
            "user_report_settings",
            sa.Column("first_page_type", sa.String(length=32), nullable=False,
                      server_default="detailed"),
        )
    if "last_page_type" not in settings_cols:
        op.add_column(
            "user_report_settings",
            sa.Column("last_page_type", sa.String(length=32), nullable=False,
                      server_default="color"),
        )
    if "cover_info" not in settings_cols:
        op.add_column("user_report_settings", sa.Column("cover_info", sa.JSON(), nullable=True))

    report_cols = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "first_page_type" not in report_cols:
        op.add_column("report_jobs", sa.Column("first_page_type", sa.String(length=32), nullable=True))
    if "last_page_type" not in report_cols:
        op.add_column("report_jobs", sa.Column("last_page_type", sa.String(length=32), nullable=True))
    if "cover_info" not in report_cols:
        op.add_column("report_jobs", sa.Column("cover_info", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    report_cols = {c["name"] for c in inspector.get_columns("report_jobs")}
    for col in ("cover_info", "last_page_type", "first_page_type"):
        if col in report_cols:
            op.drop_column("report_jobs", col)

    settings_cols = {c["name"] for c in inspector.get_columns("user_report_settings")}
    for col in ("cover_info", "last_page_type", "first_page_type"):
        if col in settings_cols:
            op.drop_column("user_report_settings", col)
