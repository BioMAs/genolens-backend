"""merge performance optimizations and regulation branches

Revision ID: 01387da21cce
Revises: a1b2c3d4e5f6, performance_optimizations
Create Date: 2026-02-25 14:31:52.837809

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '01387da21cce'
down_revision: Union[str, None] = ('a1b2c3d4e5f6', 'performance_optimizations')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
