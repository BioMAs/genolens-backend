"""add demo-account columns on users

Demo (prospect) accounts used to be faked by hand: a TEAM account whose analyses
counter was pre-filled to `150 - N`, with `quota_reset_at` set so the monthly reset
would not hand the 150 back, and an end date placed before the 1st of the month.
Nothing capped AI usage, and nothing marked what the account generated.

- users.is_demo                  : DEMO watermark on generated reports, and no
                                   monthly reset of the account's counters.
- users.ai_requests_limit        : max AI requests over the account's lifetime
                                   (NULL = no cap).
- users.ai_requests_used         : AI requests consumed (real LLM calls only).
- users.analyses_quota_override  : per-account analyses quota, overriding the
                                   plan's (NULL = plan default).

Purely additive with server defaults, so the old code keeps working while the
deploy job applies migrations before restarting the containers. Idempotent so it
is safe to re-run on partially migrated databases, matching the convention of the
surrounding migrations.

Revision ID: demo_accounts_001
Revises: gene_search_indexes_001
Create Date: 2026-10-12 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "demo_accounts_001"
down_revision: Union[str, None] = "gene_search_indexes_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns() -> list:
    # Built fresh on each call: a Column object can only be attached to one table.
    return [
        sa.Column(
            "is_demo",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            comment=(
                "Demo (prospect) account: DEMO watermark on generated reports, "
                "no monthly reset of its counters"
            ),
        ),
        sa.Column(
            "ai_requests_limit",
            sa.Integer(),
            nullable=True,
            comment="Max AI requests over the account's lifetime. NULL = no cap.",
        ),
        sa.Column(
            "ai_requests_used",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="AI requests consumed (real LLM calls; cached results are free).",
        ),
        sa.Column(
            "analyses_quota_override",
            sa.Integer(),
            nullable=True,
            comment="Per-account analyses quota overriding the plan's. NULL = plan default.",
        ),
    ]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    for column in _columns():
        if column.name not in user_columns:
            op.add_column("users", column)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    for column in reversed(_columns()):
        if column.name in user_columns:
            op.drop_column("users", column.name)
