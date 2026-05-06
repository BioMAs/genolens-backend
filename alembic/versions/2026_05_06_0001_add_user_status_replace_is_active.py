"""add_user_status_replace_is_active

Revision ID: 2026_05_06_0001
Revises: c9e1f3a08b7d
Create Date: 2026-05-06
"""
from alembic import op
import sqlalchemy as sa

revision = "2026_05_06_0001"
down_revision = "c9e1f3a08b7d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    user_status = sa.Enum(
        "pending", "active", "suspended", "cancelled",
        name="user_status_enum"
    )
    user_status.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "users",
        sa.Column(
            "status",
            user_status,
            nullable=False,
            server_default="active",
        ),
    )

    op.execute("UPDATE users SET status = 'active' WHERE is_active = TRUE")
    op.execute("UPDATE users SET status = 'cancelled' WHERE is_active = FALSE")

    op.drop_column("users", "is_active")
    op.create_index("ix_users_status", "users", ["status"])


def downgrade() -> None:
    op.drop_index("ix_users_status", table_name="users")
    op.add_column(
        "users",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="TRUE"),
    )
    op.execute("UPDATE users SET is_active = TRUE WHERE status = 'active'")
    op.execute("UPDATE users SET is_active = FALSE WHERE status IN ('cancelled', 'suspended', 'pending')")
    op.alter_column("users", "is_active", server_default=None)
    op.drop_column("users", "status")
    op.execute("DROP TYPE IF EXISTS user_status_enum")
