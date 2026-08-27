"""add scientific tools add-on module flag

- users.scientific_module_enabled (per-user unlock toggled by admins)

Gates the scientific module: GSEA, two-contrast log2FC scatter, per-sample
signature scoring, custom gene sets and DEG patterns.

Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: scientific_module_001
Revises: scope_custom_gene_sets_001
Create Date: 2026-07-18 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "scientific_module_001"
down_revision: Union[str, None] = "scope_custom_gene_sets_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "scientific_module_enabled" not in user_columns:
        op.add_column(
            "users",
            sa.Column(
                "scientific_module_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "scientific_module_enabled" in user_columns:
        op.drop_column("users", "scientific_module_enabled")
