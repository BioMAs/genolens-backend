"""add_sample_correlations_table

Revision ID: a0de1acb7e87
Revises: 919df7c733b3
Create Date: 2026-02-26 14:01:39.256615

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a0de1acb7e87'
down_revision: Union[str, None] = '919df7c733b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create sample_correlations table for caching heatmap computations
    op.create_table(
        'sample_correlations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'dataset_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('datasets.id', ondelete='CASCADE'),
            nullable=False
        ),
        sa.Column('sample_a', sa.String(255), nullable=False),
        sa.Column('sample_b', sa.String(255), nullable=False),
        sa.Column('correlation', sa.Float, nullable=True, comment='Correlation coefficient'),
        sa.Column('distance', sa.Float, nullable=True, comment='Distance metric value'),
        sa.Column('method', sa.String(50), nullable=False, comment='clustering method: ward, average, complete, single, kmeans'),
        sa.Column('metric', sa.String(50), nullable=False, comment='distance metric: euclidean, manhattan, correlation, cosine'),
        sa.Column('top_n_genes', sa.Integer, nullable=False, default=2000, comment='Number of genes used for computation'),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()')
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()')
        )
    )

    # Create indexes for efficient queries
    op.create_index(
        'ix_sample_correlations_dataset_method_metric',
        'sample_correlations',
        ['dataset_id', 'method', 'metric', 'top_n_genes']
    )
    op.create_index(
        'ix_sample_correlations_dataset',
        'sample_correlations',
        ['dataset_id']
    )
    op.create_index(
        'ix_sample_correlations_samples',
        'sample_correlations',
        ['dataset_id', 'sample_a', 'sample_b']
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_sample_correlations_samples', table_name='sample_correlations')
    op.drop_index('ix_sample_correlations_dataset', table_name='sample_correlations')
    op.drop_index('ix_sample_correlations_dataset_method_metric', table_name='sample_correlations')
    
    # Drop table
    op.drop_table('sample_correlations')
