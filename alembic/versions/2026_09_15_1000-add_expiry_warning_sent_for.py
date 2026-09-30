"""add users.expiry_warning_sent_for

Records the smallest expiry-warning threshold (7, 3 or 1 day) already emailed to
a user, so the daily expiration check can be idempotent.

Without it the task would re-send on every run of the same day — a second beat
worker, a manual trigger or a retry each produce a duplicate "1 day left" email,
which reads as a malfunction to the customer. NULL means nothing has been sent,
and the column is reset to NULL whenever `subscription_ends_at` is changed, so a
renewed account is re-armed for the next cycle.

Idempotent so it is safe to re-run on partially migrated databases, matching the
convention of the surrounding migrations.

Revision ID: expiry_warning_sent_001
Revises: drug_discovery_module_001
Create Date: 2026-09-15 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "expiry_warning_sent_001"
down_revision: Union[str, None] = "drug_discovery_module_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "expiry_warning_sent_for" not in user_columns:
        op.add_column(
            "users",
            sa.Column(
                "expiry_warning_sent_for",
                sa.Integer(),
                nullable=True,
                comment=(
                    "Smallest expiry-warning threshold already emailed (7, 3, 1). "
                    "NULL = none sent. Reset when subscription_ends_at changes."
                ),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "expiry_warning_sent_for" in user_columns:
        op.drop_column("users", "expiry_warning_sent_for")
