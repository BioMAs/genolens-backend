"""add drug discovery add-on module flag

- users.drug_discovery_module_enabled (per-user unlock toggled by admins)

Replaces the TEAM/ON_PREMISE plan gate on the Drug Discovery endpoints: access
is now an explicit per-user add-on, independent of the subscription plan.
Admins keep access by role.

Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: drug_discovery_module_001
Revises: scientific_module_001
Create Date: 2026-07-18 12:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "drug_discovery_module_001"
down_revision: Union[str, None] = "scientific_module_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "drug_discovery_module_enabled" not in user_columns:
        op.add_column(
            "users",
            sa.Column(
                "drug_discovery_module_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "drug_discovery_module_enabled" in user_columns:
        op.drop_column("users", "drug_discovery_module_enabled")
