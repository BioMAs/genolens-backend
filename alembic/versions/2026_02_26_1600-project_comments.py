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
    # Create comment_type enum idempotently (works in all PostgreSQL versions)
    op.execute(sa.text("""
        DO $$ BEGIN
            CREATE TYPE comment_type_enum AS ENUM ('GENERAL', 'GENE', 'COMPARISON', 'PATHWAY');
        EXCEPTION WHEN duplicate_object THEN null;
        END $$;
    """))

    # Create project_comments table (idempotent)
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS project_comments (
            id UUID PRIMARY KEY,
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            user_id UUID NOT NULL,
            comment_type comment_type_enum NOT NULL DEFAULT 'GENERAL',
            target_id VARCHAR(255),
            content TEXT NOT NULL,
            parent_id UUID REFERENCES project_comments(id) ON DELETE CASCADE,
            is_resolved BOOLEAN NOT NULL DEFAULT false,
            extra_metadata JSON NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))

    # Indexes (all idempotent)
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_comments_project_id ON project_comments (project_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_comments_user_id ON project_comments (user_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_comments_target_id ON project_comments (target_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_comments_parent_id ON project_comments (parent_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_comments_project_type ON project_comments (project_id, comment_type)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_comments_target ON project_comments (project_id, target_id)"))


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
