"""add_updated_at_to_cached_computations

Revision ID: 919df7c733b3
Revises: 01387da21cce
Create Date: 2026-02-26 09:17:25.598356

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '919df7c733b3'
down_revision: Union[str, None] = '01387da21cce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add updated_at column to cached_computations table
    # Set default to now() so existing rows get a value
    op.add_column(
        'cached_computations',
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
            comment='Last updated timestamp'
        )
    )


def downgrade() -> None:
    # Remove updated_at column
    op.drop_column('cached_computations', 'updated_at')
