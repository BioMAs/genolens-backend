"""Add deployment tables: server_configs, deployment_secrets, deployment_jobs

Revision ID: deployment_tables
Revises: license_records
Create Date: 2026-05-25 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'deployment_tables'
down_revision: Union[str, None] = 'license_records'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TYPE deployment_status_enum AS ENUM ('pending', 'running', 'success', 'failed')
    """))

    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS server_configs (
            id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            host VARCHAR(255) NOT NULL,
            ssh_port INTEGER NOT NULL DEFAULT 22,
            ssh_user VARCHAR(100) NOT NULL DEFAULT 'ubuntu',
            ssh_key_encrypted TEXT,
            repo_path VARCHAR(500) NOT NULL DEFAULT '/opt/genolens',
            health_url VARCHAR(500),
            notes TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_server_configs_host ON server_configs (host)"))

    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS deployment_secrets (
            id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
            server_id UUID NOT NULL REFERENCES server_configs(id) ON DELETE CASCADE,
            key VARCHAR(255) NOT NULL,
            value_encrypted TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_deployment_secrets_server ON deployment_secrets (server_id)"))
    op.execute(sa.text("CREATE UNIQUE INDEX IF NOT EXISTS ix_deployment_secrets_server_key ON deployment_secrets (server_id, key)"))

    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS deployment_jobs (
            id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
            server_id UUID REFERENCES server_configs(id) ON DELETE SET NULL,
            services JSONB NOT NULL DEFAULT '[]',
            skip_build BOOLEAN NOT NULL DEFAULT FALSE,
            status deployment_status_enum NOT NULL DEFAULT 'pending',
            logs TEXT,
            triggered_by VARCHAR(255),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_deployment_jobs_server ON deployment_jobs (server_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_deployment_jobs_status_created ON deployment_jobs (status, created_at)"))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS deployment_jobs"))
    op.execute(sa.text("DROP TABLE IF EXISTS deployment_secrets"))
    op.execute(sa.text("DROP TABLE IF EXISTS server_configs"))
    op.execute(sa.text("DROP TYPE IF EXISTS deployment_status_enum"))
