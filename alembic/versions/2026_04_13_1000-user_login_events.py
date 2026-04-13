"""Add user_login_events table

Revision ID: user_login_events
Revises: project_activity_log
Create Date: 2026-04-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'user_login_events'
down_revision: Union[str, None] = 'project_activity_log'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS user_login_events (
            id UUID NOT NULL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))

    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_user_login_events_user_id ON user_login_events (user_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_user_login_events_created_at ON user_login_events (created_at)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_user_login_events_user_created ON user_login_events (user_id, created_at)"))


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_user_login_events_user_created"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_user_login_events_created_at"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_user_login_events_user_id"))
    op.execute(sa.text("DROP TABLE IF EXISTS user_login_events"))
