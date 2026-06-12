"""add_user_status_replace_is_active

Adds the users.status column (user_status_enum) and migrates the legacy
users.is_active boolean onto it. The User model exposes is_active as a
read-only property derived from status, so the canonical migration chain must
create the status column.

This logic originally lived only on the (unmerged) feature/phase1-account-management
lineage as revision 2026_05_06_0001, which never reached the canonical chain that
production follows. It is re-incorporated here, made idempotent so it is safe on
databases that already migrated (e.g. dev on the old lineage) or that lack the
legacy is_active column.

Revision ID: add_user_status_001
Revises: add_data_type_to_analyses_001
Create Date: 2026-06-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_user_status_001"
down_revision: Union[str, None] = "add_data_type_to_analyses_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {c["name"] for c in sa.inspect(bind).get_columns("users")}

    # Already migrated (e.g. dev on the old lineage) — nothing to do.
    if "status" in columns:
        return

    user_status = sa.Enum(
        "pending", "active", "suspended", "cancelled", name="user_status_enum"
    )
    user_status.create(bind, checkfirst=True)

    op.add_column(
        "users",
        sa.Column("status", user_status, nullable=False, server_default="active"),
    )

    # Migrate the legacy boolean onto the new status column, if it exists.
    if "is_active" in columns:
        op.execute("UPDATE users SET status = 'active' WHERE is_active = TRUE")
        op.execute("UPDATE users SET status = 'cancelled' WHERE is_active = FALSE")
        op.drop_column("users", "is_active")

    op.create_index("ix_users_status", "users", ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    columns = {c["name"] for c in sa.inspect(bind).get_columns("users")}

    if "is_active" not in columns:
        op.add_column(
            "users",
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="TRUE"),
        )
        op.execute("UPDATE users SET is_active = TRUE WHERE status = 'active'")
        op.execute(
            "UPDATE users SET is_active = FALSE "
            "WHERE status IN ('cancelled', 'suspended', 'pending')"
        )
        op.alter_column("users", "is_active", server_default=None)

    if "status" in columns:
        op.drop_index("ix_users_status", table_name="users")
        op.drop_column("users", "status")
        op.execute("DROP TYPE IF EXISTS user_status_enum")
