"""Add project_activity_log table

Revision ID: project_activity_log
Revises: project_comments
Create Date: 2026-02-27 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'project_activity_log'
down_revision: Union[str, None] = 'project_comments'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the enum type idempotently
    op.execute(sa.text("""
        DO $$ BEGIN
            CREATE TYPE activity_event_type_enum AS ENUM (
                'dataset_uploaded', 'dataset_deleted', 'comparison_created',
                'enrichment_run', 'clustering_run', 'gsea_run', 'go_enrichment_run',
                'bookmark_created', 'bookmark_batch_created', 'bookmark_deleted',
                'gene_list_created', 'comment_added', 'project_shared'
            );
        EXCEPTION WHEN duplicate_object THEN null;
        END $$;
    """))

    # Create the table idempotently
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS project_activity_log (
            id UUID NOT NULL PRIMARY KEY,
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            user_id UUID NOT NULL,
            event_type activity_event_type_enum NOT NULL,
            entity_type VARCHAR(100),
            entity_id VARCHAR(255),
            entity_name VARCHAR(500),
            extra_metadata JSON NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))

    # Indexes (all idempotent)
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_activity_log_project_id ON project_activity_log (project_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_activity_log_user_id ON project_activity_log (user_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_activity_log_event_type ON project_activity_log (event_type)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_project_activity_log_created_at ON project_activity_log (created_at)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_activity_log_project_created ON project_activity_log (project_id, created_at)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_activity_log_project_event ON project_activity_log (project_id, event_type)"))


def downgrade() -> None:
    op.drop_index('ix_activity_log_project_event', table_name='project_activity_log')
    op.drop_index('ix_activity_log_project_created', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_created_at', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_event_type', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_user_id', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_project_id', table_name='project_activity_log')
    op.drop_table('project_activity_log')
    sa.Enum(name='activity_event_type_enum').drop(op.get_bind(), checkfirst=True)
