"""add SCILICIUM_ADMIN value to user_role_enum

The UserRole.SCILICIUM_ADMIN value was added to the Python enum but the
corresponding PostgreSQL enum type was never updated, causing
InvalidTextRepresentationError on any user whose Supabase role is
SCILICIUM_ADMIN.

Revision ID: add_scilicium_admin_role_001
Revises: merge_heads_001
Create Date: 2026-06-01
"""
from typing import Sequence, Union

from alembic import op

revision: str = "add_scilicium_admin_role_001"
down_revision: Union[str, None] = "merge_heads_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PostgreSQL does not allow removing enum values, so downgrade is a no-op.
    # Use IF NOT EXISTS guard so re-running the migration is safe.
    op.execute("ALTER TYPE user_role_enum ADD VALUE IF NOT EXISTS 'SCILICIUM_ADMIN'")


def downgrade() -> None:
    # Cannot remove a value from a PostgreSQL enum type without recreating it.
    # Downgrade is intentionally a no-op — roll back application code instead.
    pass
