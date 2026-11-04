"""add report_jobs.template (report format: latex | editorial)

Existing jobs are LaTeX reports, hence the server default.
Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: report_template_001
Revises: demo_accounts_001
Create Date: 2026-10-13 10:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "report_template_001"
down_revision: Union[str, None] = "demo_accounts_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "template" not in cols:
        op.add_column(
            "report_jobs",
            sa.Column("template", sa.String(length=16), nullable=False, server_default="latex"),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "template" in cols:
        op.drop_column("report_jobs", "template")
