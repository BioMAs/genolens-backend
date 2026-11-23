"""
Local authentication: sessions, one-time tokens, MFA recovery codes and the audit trail.

The `users` table carries the credential columns themselves (password hash, MFA secret, lock
state; see `app.models.models.User`). The four tables here hold everything that is issued,
consumed or recorded around them:

- `AuthSession`: one login. Its refresh token is stored as a sha256 only; rotating it keeps the
  previous hash so that replaying a spent refresh token is detected rather than merely refused.
- `AuthOneTimeToken`: an invitation, password-reset or e-mail-verification link, or a short-lived
  MFA challenge. Stored as a sha256 only, single use, with an attempt counter.
- `AuthRecoveryCode`: a single-use MFA recovery code, stored as a hash only.
- `AuthEvent`: the audit trail of everything above.

Column types are portable (JSON, not JSONB) so the same models run on the SQLite test database.
"""

import enum
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, BigInteger, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Integer, String, func
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class OneTimeTokenPurpose(str, enum.Enum):
    """What a one-time token authorises. A token is only ever valid for its own purpose."""

    INVITE = "invite"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFY = "email_verify"
    MFA_CHALLENGE = "mfa_challenge"
    MFA_ENROLL = "mfa_enroll"


class AuthEventType(str, enum.Enum):
    """Kinds of audit event. The value (stored in `auth_events.event_type`) is the lowercase name."""

    LOGIN_SUCCEEDED = "login_succeeded"
    LOGIN_FAILED = "login_failed"
    ACCOUNT_LOCKED = "account_locked"
    MFA_SUCCEEDED = "mfa_succeeded"
    MFA_FAILED = "mfa_failed"
    MFA_ENROLLED = "mfa_enrolled"
    MFA_ROTATED = "mfa_rotated"
    MFA_RESET_BY_ADMIN = "mfa_reset_by_admin"
    RECOVERY_CODE_USED = "recovery_code_used"
    RECOVERY_CODES_REGENERATED = "recovery_codes_regenerated"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    SESSION_REVOKED = "session_revoked"
    SESSIONS_REVOKED_ALL = "sessions_revoked_all"
    REFRESH_REUSE_DETECTED = "refresh_reuse_detected"
    INVITE_SENT = "invite_sent"
    INVITE_ACCEPTED = "invite_accepted"
    SIGNUP = "signup"
    EMAIL_VERIFIED = "email_verified"
    BREAK_GLASS = "break_glass"
    ACCOUNT_UNLOCKED = "account_unlocked"
    SERVICE_TOKENS_REVOKED = "service_tokens_revoked"


class AuthSession(Base):
    """One login of one account, renewable through a rotating refresh token."""

    __tablename__ = "auth_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Inactivity expiry: pushed forward each time the session is used.
    idle_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Absolute expiry, whatever the activity.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Authentication methods used to open the session (RFC 8176), e.g. ["pwd", "otp"].
    amr: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # sha256 (hex) of the refresh token; the token itself is never stored.
    refresh_token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # The hash a rotation replaced. Presented again after a short grace period (concurrent
    # requests), it means the token was reused: the session is revoked.
    previous_refresh_token_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    rotated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<AuthSession(id={self.id}, user_id={self.user_id}, revoked={self.revoked_at is not None})>"


class AuthOneTimeToken(Base):
    """A single-use link or challenge token (invite, reset, e-mail verification, MFA)."""

    __tablename__ = "auth_one_time_tokens"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Stored as VARCHAR holding the enum *value* ("invite", ...), never a native Postgres enum:
    # adding a purpose later is then a code change, not an ALTER TYPE.
    purpose: Mapped[OneTimeTokenPurpose] = mapped_column(
        SQLEnum(
            OneTimeTokenPurpose,
            name="one_time_token_purpose",
            native_enum=False,
            length=32,
            values_callable=lambda members: [m.value for m in members],
        ),
        nullable=False,
    )
    # sha256 (hex) of the token; the token itself is never stored.
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Wrong attempts against this token (the short MFA challenge); bounds brute force.
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=sa_text("0")
    )
    # The administrator who issued it (invitations); NULL when the account itself asked.
    created_by: Mapped[Optional[UUID]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<AuthOneTimeToken(id={self.id}, user_id={self.user_id}, purpose={self.purpose})>"


class AuthRecoveryCode(Base):
    """A single-use MFA recovery code. Only its hash is stored."""

    __tablename__ = "auth_mfa_recovery_codes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<AuthRecoveryCode(id={self.id}, user_id={self.user_id}, used={self.used_at is not None})>"


class AuthEvent(Base):
    """Audit trail. Survives the deletion of the account (`user_id` is set to NULL)."""

    __tablename__ = "auth_events"

    # BIGINT identity on Postgres; SQLite only auto-increments an INTEGER primary key.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=True
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # An `AuthEventType` value. A plain string: the set of events grows without a migration.
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<AuthEvent(id={self.id}, user_id={self.user_id}, event_type={self.event_type!r})>"
