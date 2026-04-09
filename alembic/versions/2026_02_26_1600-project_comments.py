"""
Add project comments table for collaboration.

Revision ID: project_comments
Revises: bookmark_gene_lists
Create Date: 2026-02-26 16:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = 'project_comments'
down_revision: Union[str, None] = 'bookmark_gene_lists'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create comment_type enum only if it doesn't already exist
    bind = op.get_bind()
    result = bind.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'comment_type_enum')")
    )
    if not result.scalar():
        bind.execute(sa.text(
            "CREATE TYPE comment_type_enum AS ENUM ('GENERAL', 'GENE', 'COMPARISON', 'PATHWAY')"
        ))
    
    # Create project_comments table
    op.create_table(
        'project_comments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False,
                  comment='Project this comment belongs to'),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False,
                  comment='Supabase Auth user UUID who created the comment'),
        sa.Column('comment_type', sa.Enum(
            'GENERAL', 'GENE', 'COMPARISON', 'PATHWAY',
            name='comment_type_enum', 
            create_type=False
        ), nullable=False, server_default='GENERAL',
                  comment='Type of comment (general, gene, comparison, pathway)'),
        sa.Column('target_id', sa.String(255), nullable=True,
                  comment='ID of the target entity (gene_symbol, comparison_name, pathway_id)'),
        sa.Column('content', sa.Text, nullable=False,
                  comment='Comment content in markdown format'),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('project_comments.id', ondelete='CASCADE'), nullable=True,
                  comment='Parent comment ID for threaded discussions'),
        sa.Column('is_resolved', sa.Boolean, nullable=False, server_default='false',
                  comment='Whether this comment thread is resolved'),
        sa.Column('extra_metadata', postgresql.JSON, nullable=False, server_default='{}',
                  comment='Additional metadata (mentions, tags, etc.)'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                  server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()'))
    )
    
    # Indexes for project_comments
    op.create_index('ix_project_comments_project_id', 'project_comments', ['project_id'])
    op.create_index('ix_project_comments_user_id', 'project_comments', ['user_id'])
    op.create_index('ix_project_comments_target_id', 'project_comments', ['target_id'])
    op.create_index('ix_project_comments_parent_id', 'project_comments', ['parent_id'])
    op.create_index('ix_project_comments_project_type', 'project_comments',
                   ['project_id', 'comment_type'])
    op.create_index('ix_project_comments_target', 'project_comments',
                   ['project_id', 'target_id'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_project_comments_target', table_name='project_comments')
    op.drop_index('ix_project_comments_project_type', table_name='project_comments')
    op.drop_index('ix_project_comments_parent_id', table_name='project_comments')
    op.drop_index('ix_project_comments_target_id', table_name='project_comments')
    op.drop_index('ix_project_comments_user_id', table_name='project_comments')
    op.drop_index('ix_project_comments_project_id', table_name='project_comments')
    
    # Drop table
    op.drop_table('project_comments')
    
    # Drop enum
    op.execute(sa.text("DROP TYPE IF EXISTS comment_type_enum"))
