"""local authentication: users credential columns and the four auth_* tables

users: password_hash, email_verified_at, password_changed_at, mfa_totp_secret_enc,
mfa_pending_secret_enc, mfa_enabled_at, mfa_last_used_step, failed_login_count (NOT NULL,
server default 0), locked_until, last_login_at, auth_migrated_at. Every one but the counter is
nullable: an account that has never signed in locally keeps NULLs and keeps working through
Supabase until it is migrated.
auth_sessions: one login, with a rotating refresh token stored as a sha256 only.
auth_one_time_tokens: invitation / password reset / e-mail verification / MFA tokens (sha256 only,
purpose stored as VARCHAR, not a native Postgres enum).
auth_mfa_recovery_codes: single-use MFA recovery codes (hash only).
auth_events: audit trail (user_id SET NULL when the account is deleted).
No unique index on lower(users.email) here: the migration script reports duplicates first and the
index comes with the cleanup phase.
Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: local_auth_001
Revises: nova_honesty_001
Create Date: 2026-11-24 10:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "local_auth_001"
down_revision: Union[str, None] = "nova_honesty_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PURPOSES = ("invite", "password_reset", "email_verify", "mfa_challenge", "mfa_enroll")

# users columns added by this revision, in order. The counter is handled apart (NOT NULL).
USER_COLUMNS: tuple[tuple[str, sa.types.TypeEngine], ...] = (
    ("password_hash", sa.Text()),
    ("email_verified_at", sa.DateTime(timezone=True)),
    ("password_changed_at", sa.DateTime(timezone=True)),
    ("mfa_totp_secret_enc", sa.Text()),
    ("mfa_pending_secret_enc", sa.Text()),
    ("mfa_enabled_at", sa.DateTime(timezone=True)),
    ("mfa_last_used_step", sa.BigInteger()),
    ("locked_until", sa.DateTime(timezone=True)),
    ("last_login_at", sa.DateTime(timezone=True)),
    ("auth_migrated_at", sa.DateTime(timezone=True)),
)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    cols = {c["name"] for c in inspector.get_columns("users")}
    for name, type_ in USER_COLUMNS:
        if name not in cols:
            op.add_column("users", sa.Column(name, type_, nullable=True))
    if "failed_login_count" not in cols:
        op.add_column(
            "users",
            sa.Column(
                "failed_login_count", sa.Integer(), nullable=False, server_default=sa.text("0")
            ),
        )

    tables = set(inspector.get_table_names())

    if "auth_sessions" not in tables:
        op.create_table(
            "auth_sessions",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "last_used_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("idle_expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_reason", sa.String(length=64), nullable=True),
            sa.Column("ip", sa.String(length=64), nullable=True),
            sa.Column("user_agent", sa.String(length=512), nullable=True),
            sa.Column("amr", sa.JSON(), nullable=False),
            sa.Column("refresh_token_hash", sa.String(length=64), nullable=False),
            sa.Column("previous_refresh_token_hash", sa.String(length=64), nullable=True),
            sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("refresh_token_hash", name="uq_auth_sessions_refresh_token_hash"),
        )
        op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
        op.create_index(
            "ix_auth_sessions_previous_refresh_token_hash",
            "auth_sessions",
            ["previous_refresh_token_hash"],
        )

    if "auth_one_time_tokens" not in tables:
        op.create_table(
            "auth_one_time_tokens",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column(
                "purpose",
                sa.Enum(*PURPOSES, name="one_time_token_purpose", native_enum=False, length=32),
                nullable=False,
            ),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("created_by", sa.UUID(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("token_hash", name="uq_auth_one_time_tokens_token_hash"),
        )
        op.create_index("ix_auth_one_time_tokens_user_id", "auth_one_time_tokens", ["user_id"])

    if "auth_mfa_recovery_codes" not in tables:
        op.create_table(
            "auth_mfa_recovery_codes",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("code_hash", sa.String(length=64), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_auth_mfa_recovery_codes_user_id", "auth_mfa_recovery_codes", ["user_id"]
        )

    if "auth_events" not in tables:
        op.create_table(
            "auth_events",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("event_type", sa.String(length=64), nullable=False),
            sa.Column("ip", sa.String(length=64), nullable=True),
            sa.Column("user_agent", sa.String(length=512), nullable=True),
            sa.Column("details", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_auth_events_user_id", "auth_events", ["user_id"])
        op.create_index("ix_auth_events_event_type", "auth_events", ["event_type"])
        op.create_index("ix_auth_events_created_at", "auth_events", ["created_at"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    # Dropping a table drops its indexes with it; the order is that of the creation, reversed.
    for table in (
        "auth_events",
        "auth_mfa_recovery_codes",
        "auth_one_time_tokens",
        "auth_sessions",
    ):
        if table in tables:
            op.drop_table(table)

    cols = {c["name"] for c in inspector.get_columns("users")}
    for name in ("failed_login_count", *(name for name, _ in reversed(USER_COLUMNS))):
        if name in cols:
            op.drop_column("users", name)
