"""add_enrichment_ratio_level_params_hash

Revision ID: a49ee4194bbb
Revises: add_species_to_projects
Create Date: 2026-04-26 11:13:40.892348

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a49ee4194bbb'
down_revision: Union[str, None] = 'add_species_to_projects'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('enrichment_pathways', sa.Column('enrichment_ratio', sa.Float(), nullable=True))
    op.add_column('enrichment_pathways', sa.Column('level', sa.Integer(), nullable=True))
    op.add_column('enrichment_pathways', sa.Column('parameters_hash', sa.String(length=16), nullable=True))
    op.create_index('ix_enrichment_pathways_parameters_hash', 'enrichment_pathways', ['parameters_hash'], unique=False)
    op.create_index('ix_enrichment_pathways_params_hash', 'enrichment_pathways', ['dataset_id', 'comparison_name', 'parameters_hash'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_enrichment_pathways_params_hash', table_name='enrichment_pathways')
    op.drop_index('ix_enrichment_pathways_parameters_hash', table_name='enrichment_pathways')
    op.drop_column('enrichment_pathways', 'parameters_hash')
    op.drop_column('enrichment_pathways', 'level')
    op.drop_column('enrichment_pathways', 'enrichment_ratio')
