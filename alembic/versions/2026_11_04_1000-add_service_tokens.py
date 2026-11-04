"""add service_tokens; report_jobs.use_ai and report_jobs.service_token_id

service_tokens: scoped bearer tokens of a machine client (the analysis pipeline), stored as a
sha256 only (docs/SERVICE_TOKENS.md).
report_jobs.use_ai: False gives a report without any AI text; existing jobs had their AI, hence
the server default.
report_jobs.service_token_id: the token a job was requested through (NULL: a user); such a job
is rendered without the owner's saved branding.
Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: service_tokens_001
Revises: report_template_001
Create Date: 2026-11-04 10:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "service_tokens_001"
down_revision: Union[str, None] = "report_template_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "service_tokens" not in inspector.get_table_names():
        op.create_table(
            "service_tokens",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("owner_id", sa.UUID(), nullable=False),
            sa.Column("token_sha256", sa.String(length=64), nullable=False),
            sa.Column("scopes", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("token_sha256", name="uq_service_tokens_token_sha256"),
        )
        op.create_index("ix_service_tokens_owner_id", "service_tokens", ["owner_id"])

    cols = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "use_ai" not in cols:
        op.add_column(
            "report_jobs",
            sa.Column("use_ai", sa.Boolean(), nullable=False, server_default=sa.true()),
        )
    if "service_token_id" not in cols:
        op.add_column("report_jobs", sa.Column("service_token_id", sa.UUID(), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("report_jobs")}
    if "service_token_id" in cols:
        op.drop_column("report_jobs", "service_token_id")
    if "use_ai" in cols:
        op.drop_column("report_jobs", "use_ai")
    if "service_tokens" in inspector.get_table_names():
        op.drop_index("ix_service_tokens_owner_id", table_name="service_tokens")
        op.drop_table("service_tokens")
