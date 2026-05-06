"""add_chart_type_to_ai_tables

Revision ID: b3f7d2a91c04
Revises: a49ee4194bbb
Create Date: 2026-04-27 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'b3f7d2a91c04'
down_revision: Union[str, None] = 'a49ee4194bbb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ai_conversations: add chart_type column
    # server_default='comparison' backfills existing rows (all were comparison-type chats)
    op.add_column('ai_conversations',
        sa.Column('chart_type', sa.String(50), nullable=False, server_default='comparison'))
    op.drop_index('ix_ai_conversations_dataset_comparison', table_name='ai_conversations')
    op.create_index(
        'ix_ai_conversations_dataset_chart',
        'ai_conversations',
        ['dataset_id', 'chart_type', 'comparison_name']
    )

    # ai_interpretations: add chart_type, update unique index, make stat columns nullable
    # server_default='comparison' backfills existing rows
    op.add_column('ai_interpretations',
        sa.Column('chart_type', sa.String(50), nullable=False, server_default='comparison'))
    op.alter_column('ai_interpretations', 'deg_up', existing_type=sa.Integer(), nullable=True)
    op.alter_column('ai_interpretations', 'deg_down', existing_type=sa.Integer(), nullable=True)
    op.alter_column('ai_interpretations', 'pathways_count', existing_type=sa.Integer(), nullable=True)
    op.alter_column('ai_interpretations', 'genes_count', existing_type=sa.Integer(), nullable=True)
    op.drop_index('ix_ai_interpretations_dataset_comparison', table_name='ai_interpretations')
    op.create_index(
        'ix_ai_interpretations_dataset_chart_comparison',
        'ai_interpretations',
        ['dataset_id', 'chart_type', 'comparison_name'],
        unique=True
    )


def downgrade() -> None:
    op.drop_index('ix_ai_interpretations_dataset_chart_comparison', table_name='ai_interpretations')
    op.create_index('ix_ai_interpretations_dataset_comparison', 'ai_interpretations',
                    ['dataset_id', 'comparison_name'], unique=True)
    # Backfill NULLs introduced by non-comparison chart types before restoring NOT NULL
    op.execute("UPDATE ai_interpretations SET deg_up = 0 WHERE deg_up IS NULL")
    op.execute("UPDATE ai_interpretations SET deg_down = 0 WHERE deg_down IS NULL")
    op.execute("UPDATE ai_interpretations SET pathways_count = 0 WHERE pathways_count IS NULL")
    op.execute("UPDATE ai_interpretations SET genes_count = 0 WHERE genes_count IS NULL")
    op.alter_column('ai_interpretations', 'genes_count', existing_type=sa.Integer(), nullable=False)
    op.alter_column('ai_interpretations', 'pathways_count', existing_type=sa.Integer(), nullable=False)
    op.alter_column('ai_interpretations', 'deg_down', existing_type=sa.Integer(), nullable=False)
    op.alter_column('ai_interpretations', 'deg_up', existing_type=sa.Integer(), nullable=False)
    op.drop_column('ai_interpretations', 'chart_type')

    op.drop_index('ix_ai_conversations_dataset_chart', table_name='ai_conversations')
    op.create_index('ix_ai_conversations_dataset_comparison', 'ai_conversations',
                    ['dataset_id', 'comparison_name'])
    op.drop_column('ai_conversations', 'chart_type')
