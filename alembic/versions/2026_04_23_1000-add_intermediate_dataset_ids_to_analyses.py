"""add_intermediate_dataset_ids_to_self_service_analyses

Revision ID: add_intermediate_dataset_ids
Revises: add_self_service_analyses
Create Date: 2026-04-23 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "add_intermediate_dataset_ids"
down_revision: Union[str, None] = "add_self_service_analyses"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "self_service_analyses",
        sa.Column(
            "intermediate_dataset_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Keys: 'vst', 'normalized'. Values: dataset UUIDs uploaded alongside DEG results",
        ),
    )
    # Back-fill existing rows with an empty object
    op.execute("UPDATE self_service_analyses SET intermediate_dataset_ids = '{}' WHERE intermediate_dataset_ids IS NULL")
    op.alter_column("self_service_analyses", "intermediate_dataset_ids", nullable=False)


def downgrade() -> None:
    op.drop_column("self_service_analyses", "intermediate_dataset_ids")
