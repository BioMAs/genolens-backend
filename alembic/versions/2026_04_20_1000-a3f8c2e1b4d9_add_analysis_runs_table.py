"""add_analysis_runs_table

Revision ID: a3f8c2e1b4d9
Revises: user_login_events
Create Date: 2026-04-20 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a3f8c2e1b4d9'
down_revision: Union[str, None] = 'user_login_events'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'analysis_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'dataset_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('datasets.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'user_id',
            sa.String(255),
            nullable=False,
            comment='Supabase Auth user UUID who triggered the analysis',
        ),
        sa.Column(
            'analysis_type',
            sa.String(100),
            nullable=False,
            comment='Analysis type: VOLCANO, GO_ENRICHMENT, GSEA, SAMPLE_CLUSTERING, PCA, UMAP, HEATMAP',
        ),
        sa.Column(
            'comparison_name',
            sa.String(255),
            nullable=True,
            comment='Comparison name for DEG-linked analyses (volcano, enrichment)',
        ),
        sa.Column(
            'parameters',
            sa.JSON(),
            nullable=False,
            comment='Exact parameters used for the analysis',
        ),
        sa.Column(
            'algorithm_versions',
            sa.JSON(),
            nullable=False,
            comment='Package/algorithm versions used (e.g. {"scipy": "1.12.0"})',
        ),
        sa.Column(
            'reference_db_versions',
            sa.JSON(),
            nullable=True,
            comment='Reference database versions consulted (e.g. {"go_release": "2024-01"})',
        ),
        sa.Column(
            'result_summary',
            sa.JSON(),
            nullable=True,
            comment='High-level summary of results (e.g. {"gene_count": 42, "pathway_count": 15})',
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
        ),
    )

    op.create_index(
        'ix_analysis_runs_dataset_id',
        'analysis_runs',
        ['dataset_id'],
    )
    op.create_index(
        'ix_analysis_runs_user_id',
        'analysis_runs',
        ['user_id'],
    )
    op.create_index(
        'ix_analysis_runs_dataset_type',
        'analysis_runs',
        ['dataset_id', 'analysis_type'],
    )


def downgrade() -> None:
    op.drop_index('ix_analysis_runs_dataset_type', table_name='analysis_runs')
    op.drop_index('ix_analysis_runs_user_id', table_name='analysis_runs')
    op.drop_index('ix_analysis_runs_dataset_id', table_name='analysis_runs')
    op.drop_table('analysis_runs')
