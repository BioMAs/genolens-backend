"""business_model_plans

Rename subscription_plan_enum values BASIC→STARTER, PREMIUM→TEAM, ADVANCED→ON_PREMISE.
Add comparisons_used_this_month and quota_reset_at to users table.

Revision ID: bm_plans_001
Revises: deployment_tables
Create Date: 2026-05-27
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "bm_plans_001"
down_revision: Union[str, None] = "deployment_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Rename existing enum values (PostgreSQL 15+)
    op.execute("ALTER TYPE subscription_plan_enum RENAME VALUE 'BASIC' TO 'STARTER'")
    op.execute("ALTER TYPE subscription_plan_enum RENAME VALUE 'PREMIUM' TO 'TEAM'")
    op.execute("ALTER TYPE subscription_plan_enum RENAME VALUE 'ADVANCED' TO 'ON_PREMISE'")

    # 2. Add comparison quota columns
    op.add_column(
        'users',
        sa.Column(
            'comparisons_used_this_month',
            sa.Integer(),
            nullable=False,
            server_default='0',
            comment='Counter reset on the 1st of each month'
        )
    )
    op.add_column(
        'users',
        sa.Column(
            'quota_reset_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment='Timestamp of last monthly quota reset'
        )
    )


def downgrade() -> None:
    op.drop_column('users', 'quota_reset_at')
    op.drop_column('users', 'comparisons_used_this_month')
    op.execute("ALTER TYPE subscription_plan_enum RENAME VALUE 'ON_PREMISE' TO 'ADVANCED'")
    op.execute("ALTER TYPE subscription_plan_enum RENAME VALUE 'TEAM' TO 'PREMIUM'")
    op.execute("ALTER TYPE subscription_plan_enum RENAME VALUE 'STARTER' TO 'BASIC'")
