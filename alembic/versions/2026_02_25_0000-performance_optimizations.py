"""performance optimizations: indexes and cached stats

Revision ID: performance_optimizations
Revises: 6783e8d60345
Create Date: 2026-02-25 00:00:00.000000

Adds performance optimizations:
- GIN index on dataset_metadata JSONB column for faster metadata queries
- Pre-calculated statistics columns for DEG datasets
- Persistent cache table for computation results
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'performance_optimizations'
down_revision = '6783e8d60345'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply performance optimizations."""
    
    # 1. Convert JSON column to JSONB for better performance and GIN indexing
    # JSONB is faster for queries and supports GIN indexes with operator classes
    # Note: Only for datasets table which already exists. cached_computations will be created with JSONB.
    op.execute(
        "ALTER TABLE datasets ALTER COLUMN dataset_metadata TYPE jsonb USING dataset_metadata::jsonb"
    )
    
    # 2. Add GIN index on dataset_metadata JSONB column
    # This significantly speeds up queries that filter/search within metadata
    # Example: WHERE dataset_metadata->'comparisons' ? 'ComparisonName'
    # Note: Using raw SQL because Alembic doesn't support operator classes well
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_datasets_metadata_gin ON datasets USING gin (dataset_metadata jsonb_path_ops)"
    )
    
    # 3. Add pre-calculated statistics columns for DEG datasets
    # These avoid the need to fetch and count 100K rows on every page load
    op.add_column(
        'datasets',
        sa.Column(
            'deg_up_count',
            sa.Integer(),
            nullable=True,
            comment='Pre-calculated count of up-regulated genes (padj < 0.05, logFC > 0)'
        )
    )
    
    op.add_column(
        'datasets',
        sa.Column(
            'deg_down_count',
            sa.Integer(),
            nullable=True,
            comment='Pre-calculated count of down-regulated genes (padj < 0.05, logFC < 0)'
        )
    )
    
    op.add_column(
        'datasets',
        sa.Column(
            'deg_significant_count',
            sa.Integer(),
            nullable=True,
            comment='Pre-calculated count of significant genes (padj < 0.05)'
        )
    )
    
    op.add_column(
        'datasets',
        sa.Column(
            'total_genes',
            sa.Integer(),
            nullable=True,
            comment='Total number of genes in dataset'
        )
    )
    
    # 3. Add index on new statistics columns for faster sorting/filtering
    op.create_index(
        'ix_datasets_deg_counts',
        'datasets',
        ['type', 'deg_significant_count'],
        unique=False
    )
    
    # 4. Create cached_computation table for persistent cache
    # This replaces file-based cache with database storage
    op.create_table(
        'cached_computations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('computation_type', sa.String(50), nullable=False, comment='Type: clustering, volcano, stats, etc.'),
        sa.Column('params_hash', sa.String(32), nullable=False, comment='MD5 hash of computation parameters'),
        sa.Column('result_data', postgresql.JSONB(), nullable=False, comment='Cached result data'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True, comment='Optional expiration timestamp'),
        sa.Column('hit_count', sa.Integer(), nullable=False, default=0, comment='Number of cache hits (for monitoring)'),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True, comment='Last access timestamp'),
    )
    
    # 5. Add composite index on cached_computations for fast lookups
    op.create_index(
        'ix_cached_computations_lookup',
        'cached_computations',
        ['dataset_id', 'computation_type', 'params_hash'],
        unique=True
    )
    
    # 6. Add index for cache cleanup (expired entries)
    op.create_index(
        'ix_cached_computations_expires',
        'cached_computations',
        ['expires_at'],
        unique=False,
        postgresql_where=sa.text('expires_at IS NOT NULL')
    )
    
    # 7. Add GIN index on result_data for potential JSONB queries within cached results
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_cached_computations_result_gin ON cached_computations USING gin (result_data jsonb_path_ops)"
    )


def downgrade() -> None:
    """Revert performance optimizations."""
    
    # Drop cached_computations table and indexes
    op.execute("DROP INDEX IF EXISTS ix_cached_computations_result_gin")
    op.drop_index('ix_cached_computations_expires', table_name='cached_computations')
    op.drop_index('ix_cached_computations_lookup', table_name='cached_computations')
    op.drop_table('cached_computations')
    
    # Drop statistics indexes
    op.drop_index('ix_datasets_deg_counts', table_name='datasets')
    
    # Drop statistics columns
    op.drop_column('datasets', 'total_genes')
    op.drop_column('datasets', 'deg_significant_count')
    op.drop_column('datasets', 'deg_down_count')
    op.drop_column('datasets', 'deg_up_count')
    
    # Drop GIN index on metadata
    op.execute("DROP INDEX IF EXISTS ix_datasets_metadata_gin")
    
    # Convert JSONB columns back to JSON
    op.execute(
        "ALTER TABLE datasets ALTER COLUMN dataset_metadata TYPE json USING dataset_metadata::json"
    )
