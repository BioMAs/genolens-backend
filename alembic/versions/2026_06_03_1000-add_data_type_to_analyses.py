"""Add data_type column to self_service_analyses

Adds an OmicsDataType enum (transcriptomics, proteomics, lipidomics) and a
data_type column to self_service_analyses. Existing rows default to
'transcriptomics'.

Revision ID: add_data_type_to_analyses_001
Revises: add_scilicium_admin_role_001
Create Date: 2026-06-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_data_type_to_analyses_001"
down_revision: Union[str, None] = "add_scilicium_admin_role_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the PostgreSQL enum type
    op.execute(
        "CREATE TYPE omics_data_type AS ENUM ('transcriptomics', 'proteomics', 'lipidomics')"
    )

    # Add the column with a server default so existing rows get 'transcriptomics'
    op.add_column(
        "self_service_analyses",
        sa.Column(
            "data_type",
            sa.Enum("transcriptomics", "proteomics", "lipidomics", name="omics_data_type"),
            nullable=False,
            server_default="transcriptomics",
        ),
    )


def downgrade() -> None:
    op.drop_column("self_service_analyses", "data_type")
    op.execute("DROP TYPE IF EXISTS omics_data_type")
