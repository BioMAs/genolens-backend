"""add agentic chat-mode models (agent_sessions, agent_messages)

Multi-turn agentic chat sessions bound to a selected (project, dataset, comparison)
context, and their per-turn messages (narrative + tool calls + tool results +
figure payloads for reload).

Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: agent_chat_models_001
Revises: report_page_models_001
Create Date: 2026-07-07 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "agent_chat_models_001"
down_revision: Union[str, None] = "report_page_models_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "agent_sessions" not in tables:
        op.create_table(
            "agent_sessions",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("dataset_id", sa.Uuid(), nullable=False),
            sa.Column("comparison_name", sa.String(length=255), nullable=True),
            sa.Column("title", sa.String(length=255), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        )
        op.create_index("ix_agent_sessions_user", "agent_sessions", ["user_id"])
        op.create_index("ix_agent_sessions_project_id", "agent_sessions", ["project_id"])
        op.create_index("ix_agent_sessions_dataset_id", "agent_sessions", ["dataset_id"])

    if "agent_messages" not in tables:
        op.create_table(
            "agent_messages",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("session_id", sa.Uuid(), nullable=False),
            sa.Column("role", sa.String(length=20), nullable=False),
            sa.Column("content", sa.Text(), nullable=True),
            sa.Column("tool_calls", sa.JSON(), nullable=True),
            sa.Column("tool_results", sa.JSON(), nullable=True),
            sa.Column("figures", sa.JSON(), nullable=True),
            sa.Column("model", sa.String(length=100), nullable=True),
            sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"], ondelete="CASCADE"),
        )
        op.create_index("ix_agent_messages_session_id", "agent_messages", ["session_id"])
        op.create_index(
            "ix_agent_messages_session_sequence",
            "agent_messages",
            ["session_id", "sequence"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "agent_messages" in tables:
        op.drop_table("agent_messages")
    if "agent_sessions" in tables:
        op.drop_table("agent_sessions")
