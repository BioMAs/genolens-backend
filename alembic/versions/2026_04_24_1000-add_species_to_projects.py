"""add_species_to_projects

Revision ID: add_species_to_projects
Revises: add_intermediate_dataset_ids
Create Date: 2026-04-24 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "add_species_to_projects"
down_revision: Union[str, None] = "add_intermediate_dataset_ids"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "species",
            sa.String(100),
            nullable=True,
            comment="Organism species for functional enrichment (human, mouse, rat, zebrafish, pig)",
        ),
    )
    # Back-fill existing projects with default "human"
    op.execute("UPDATE projects SET species = 'human' WHERE species IS NULL")


def downgrade() -> None:
    op.drop_column("projects", "species")
