"""
Add gene bookmarks and custom gene lists.

Revision ID: bookmark_gene_lists
Revises: 15f7b63003d2
Create Date: 2026-02-26 14:30:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# revision identifiers
revision: str = 'bookmark_gene_lists'
down_revision: Union[str, None] = '15f7b63003d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create gene_bookmarks table
    op.create_table(
        'gene_bookmarks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, 
                  comment='User who created bookmark'),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False,
                  comment='Project context for bookmark'),
        sa.Column('gene_symbol', sa.String(100), nullable=False,
                  comment='Gene symbol (e.g., TP53)'),
        sa.Column('gene_id', sa.String(100), nullable=True,
                  comment='Gene ID (Entrez, Ensembl)'),
        sa.Column('notes', sa.Text, nullable=True,
                  comment='User notes about this gene'),
        sa.Column('tags', postgresql.JSON, nullable=False, server_default='[]',
                  comment='Custom tags for categorization'),
        sa.Column('color', sa.String(20), nullable=True,
                  comment='Custom color for visualization'),
        sa.Column('is_favorite', sa.Boolean, nullable=False, server_default='true',
                  comment='Quick favorite flag'),
        sa.Column('extra_data', postgresql.JSON, nullable=False, server_default='{}',
                  comment='Additional metadata'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                  server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()'))
    )
    
    # Indexes for gene_bookmarks
    op.create_index('ix_gene_bookmarks_user_project', 'gene_bookmarks', 
                   ['user_id', 'project_id'])
    op.create_index('ix_gene_bookmarks_gene_symbol', 'gene_bookmarks', 
                   ['gene_symbol'])
    op.create_index('ix_gene_bookmarks_project_gene', 'gene_bookmarks',
                   ['project_id', 'gene_symbol'], unique=True)
    
    # Create gene_lists table
    op.create_table(
        'gene_lists',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False,
                  comment='List name'),
        sa.Column('description', sa.Text, nullable=True,
                  comment='List description'),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False,
                  comment='Owner user ID'),
        sa.Column('project_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False,
                  comment='Associated project'),
        sa.Column('genes', postgresql.JSON, nullable=False, server_default='[]',
                  comment='Array of gene symbols'),
        sa.Column('gene_count', sa.Integer, nullable=False, server_default='0',
                  comment='Number of genes in list'),
        sa.Column('color', sa.String(20), nullable=True,
                  comment='Display color'),
        sa.Column('is_public', sa.Boolean, nullable=False, server_default='false',
                  comment='Visible to project members'),
        sa.Column('tags', postgresql.JSON, nullable=False, server_default='[]',
                  comment='Tags for categorization'),
        sa.Column('extra_data', postgresql.JSON, nullable=False, server_default='{}',
                  comment='Additional metadata (source, date, etc.)'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()'))
    )
    
    # Indexes for gene_lists
    op.create_index('ix_gene_lists_user_project', 'gene_lists',
                   ['user_id', 'project_id'])
    op.create_index('ix_gene_lists_project', 'gene_lists', ['project_id'])
    op.create_index('ix_gene_lists_name', 'gene_lists', ['name'])
    
    # Note: GIN index on JSON arrays would require JSONB type
    # For now, we'll rely on standard B-tree indexes


def downgrade() -> None:
    # Drop tables
    op.drop_index('ix_gene_lists_name', table_name='gene_lists')
    op.drop_index('ix_gene_lists_project', table_name='gene_lists')
    op.drop_index('ix_gene_lists_user_project', table_name='gene_lists')
    op.drop_table('gene_lists')
    
    op.drop_index('ix_gene_bookmarks_project_gene', table_name='gene_bookmarks')
    op.drop_index('ix_gene_bookmarks_gene_symbol', table_name='gene_bookmarks')
    op.drop_index('ix_gene_bookmarks_user_project', table_name='gene_bookmarks')
    op.drop_table('gene_bookmarks')
