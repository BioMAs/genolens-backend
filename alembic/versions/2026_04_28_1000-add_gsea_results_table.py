"""add_gsea_results_table

Revision ID: c9e1f3a08b7d
Revises: b3f7d2a91c04
Create Date: 2026-04-28 10:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c9e1f3a08b7d'
down_revision: Union[str, None] = 'b3f7d2a91c04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'gsea_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('comparison_name', sa.String(255), nullable=False),
        sa.Column('parameters_hash', sa.String(16), nullable=True),
        sa.Column('parameters_json', sa.JSON(), nullable=False),
        sa.Column('results_json', sa.JSON(), nullable=False),
        sa.Column('summary_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_gsea_results_dataset_id', 'gsea_results', ['dataset_id'])
    op.create_index('ix_gsea_results_comparison_name', 'gsea_results', ['comparison_name'])
    op.create_index('ix_gsea_results_parameters_hash', 'gsea_results', ['parameters_hash'])
    op.create_index(
        'ix_gsea_results_dataset_comparison',
        'gsea_results',
        ['dataset_id', 'comparison_name'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('ix_gsea_results_dataset_comparison', table_name='gsea_results')
    op.drop_index('ix_gsea_results_parameters_hash', table_name='gsea_results')
    op.drop_index('ix_gsea_results_comparison_name', table_name='gsea_results')
    op.drop_index('ix_gsea_results_dataset_id', table_name='gsea_results')
    op.drop_table('gsea_results')
