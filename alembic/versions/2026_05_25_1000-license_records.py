"""Add license_records table

Revision ID: license_records
Revises: user_login_events
Create Date: 2026-05-25 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'license_records'
down_revision: Union[str, None] = 'user_login_events'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS license_records (
            id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
            client_id VARCHAR(255) NOT NULL,
            plan VARCHAR(50) NOT NULL,
            product VARCHAR(100) NOT NULL DEFAULT 'genolens',
            expires_at INTEGER NOT NULL,
            license_key TEXT NOT NULL UNIQUE,
            notes TEXT,
            is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
            created_by VARCHAR(255),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))

    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_license_records_client_id ON license_records (client_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_license_records_expires_at ON license_records (expires_at)"))


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_license_records_expires_at"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_license_records_client_id"))
    op.execute(sa.text("DROP TABLE IF EXISTS license_records"))
