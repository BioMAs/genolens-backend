"""add metadata types

Revision ID: f7a8b9c0d1e2
Revises: d5b243f60608
Create Date: 2025-12-18 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, None] = 'd5b243f60608'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new values to the enum
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE dataset_type ADD VALUE 'METADATA_SAMPLE'")
        op.execute("ALTER TYPE dataset_type ADD VALUE 'METADATA_CONTRAST'")


def downgrade() -> None:
    # Removing values from enum is not supported in Postgres without dropping the type
    pass
