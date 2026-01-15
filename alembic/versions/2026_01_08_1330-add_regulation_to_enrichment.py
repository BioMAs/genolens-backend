"""add regulation to enrichment pathways

Revision ID: a1b2c3d4e5f6
Revises: 6783e8d60345
Create Date: 2026-01-08 13:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '6783e8d60345'
branch_labels = None
depends_on = None


def upgrade():
    # Add regulation column to enrichment_pathways
    op.add_column('enrichment_pathways', 
        sa.Column('regulation', sa.String(10), nullable=True, server_default='ALL')
    )
    
    # Create index for faster filtering
    op.create_index(
        'ix_enrichment_pathways_regulation',
        'enrichment_pathways',
        ['regulation']
    )
    
    # Update existing records to 'ALL'
    op.execute("UPDATE enrichment_pathways SET regulation = 'ALL' WHERE regulation IS NULL")
    
    # Make column non-nullable after update
    op.alter_column('enrichment_pathways', 'regulation', nullable=False)


def downgrade():
    op.drop_index('ix_enrichment_pathways_regulation', table_name='enrichment_pathways')
    op.drop_column('enrichment_pathways', 'regulation')
