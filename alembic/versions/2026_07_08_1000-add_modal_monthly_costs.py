"""add modal_monthly_costs (admin-entered Modal spend per month)

Backs per-user AI cost estimation: rate = spend_eur / total_tokens(month).
Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: modal_monthly_costs_001
Revises: agent_chat_models_001
Create Date: 2026-07-08 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "modal_monthly_costs_001"
down_revision: Union[str, None] = "agent_chat_models_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "modal_monthly_costs" not in inspector.get_table_names():
        op.create_table(
            "modal_monthly_costs",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("year", sa.Integer(), nullable=False),
            sa.Column("month", sa.Integer(), nullable=False, comment="1-12"),
            sa.Column("spend_eur", sa.Float(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index(
            "ux_modal_monthly_costs_year_month",
            "modal_monthly_costs",
            ["year", "month"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "modal_monthly_costs" in inspector.get_table_names():
        op.drop_index("ux_modal_monthly_costs_year_month", table_name="modal_monthly_costs")
        op.drop_table("modal_monthly_costs")
