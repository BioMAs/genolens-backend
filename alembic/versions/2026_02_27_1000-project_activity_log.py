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
    # Create the enum type
    activity_event_type_enum = sa.Enum(
        'dataset_uploaded',
        'dataset_deleted',
        'comparison_created',
        'enrichment_run',
        'clustering_run',
        'gsea_run',
        'go_enrichment_run',
        'bookmark_created',
        'bookmark_batch_created',
        'bookmark_deleted',
        'gene_list_created',
        'comment_added',
        'project_shared',
        name='activity_event_type_enum',
    )
    activity_event_type_enum.create(op.get_bind(), checkfirst=True)

    # Create the table
    op.create_table(
        'project_activity_log',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column(
            'event_type',
            sa.Enum(
                'dataset_uploaded',
                'dataset_deleted',
                'comparison_created',
                'enrichment_run',
                'clustering_run',
                'gsea_run',
                'go_enrichment_run',
                'bookmark_created',
                'bookmark_batch_created',
                'bookmark_deleted',
                'gene_list_created',
                'comment_added',
                'project_shared',
                name='activity_event_type_enum',
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column('entity_type', sa.String(length=100), nullable=True),
        sa.Column('entity_id', sa.String(length=255), nullable=True),
        sa.Column('entity_name', sa.String(length=500), nullable=True),
        sa.Column('extra_metadata', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['project_id'],
            ['projects.id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Simple indexes
    op.create_index('ix_project_activity_log_project_id', 'project_activity_log', ['project_id'], unique=False)
    op.create_index('ix_project_activity_log_user_id', 'project_activity_log', ['user_id'], unique=False)
    op.create_index('ix_project_activity_log_event_type', 'project_activity_log', ['event_type'], unique=False)
    op.create_index('ix_project_activity_log_created_at', 'project_activity_log', ['created_at'], unique=False)

    # Composite indexes
    op.create_index('ix_activity_log_project_created', 'project_activity_log', ['project_id', 'created_at'], unique=False)
    op.create_index('ix_activity_log_project_event', 'project_activity_log', ['project_id', 'event_type'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_activity_log_project_event', table_name='project_activity_log')
    op.drop_index('ix_activity_log_project_created', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_created_at', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_event_type', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_user_id', table_name='project_activity_log')
    op.drop_index('ix_project_activity_log_project_id', table_name='project_activity_log')
    op.drop_table('project_activity_log')
    sa.Enum(name='activity_event_type_enum').drop(op.get_bind(), checkfirst=True)
