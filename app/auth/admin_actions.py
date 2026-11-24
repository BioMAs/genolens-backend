"""
The security actions an administrator can take on an account, as plain functions.

The admin endpoints (`app/api/endpoints/admin.py`) are thin wrappers around them, and the
break-glass command (the way back in when no administrator can sign in) calls the same ones, so
that "reset the second factor" means exactly one thing wherever it is done.

- `reset_user_mfa`: remove the second factor, so that the next sign-in is an enrolment.
- `revoke_user_sessions`: end every session of the account.
- `revoke_user_service_tokens`: revoke the service tokens the account owns.
- `unlock_user`: clear the brute-force lock of the account.

Conventions shared by all four:

- The caller commits. Each function runs its statements and records its audit event (flushed),
  and nothing more: a caller that fails afterwards rolls the whole action back, and the audit
  line is never committed without the change it reports.
- They never send an e-mail. A notification goes out only once the change is committed (a mail
  about something that was then rolled back would be a lie), so the caller sends it.
- The audit event is recorded against the account (`user_id`), with `actor_id` in the details:
  the administrator, or None when no administrator is behind the action (the command line).
- Service tokens are revoked by `revoke_user_service_tokens` and by nothing else (spec 7.7): a
  machine client must not stop because a person lost a phone or a password. Resetting the second
  factor, ending sessions and unlocking leave them alone.

The changes are written as UPDATE statements rather than attribute assignments: an assignment
that does not change the value the row was loaded with issues no UPDATE, so it would silently
miss a concurrent sign-in that has just changed the row. The session's copy of the row is
brought up to date by the statement itself (SQLAlchemy's "auto" synchronisation).
"""

from typing import Optional
from uuid import UUID

from fastapi import Request
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.audit import record_event
from app.auth.clock import now_utc
from app.auth.sessions import revoke_all_sessions
from app.models.auth import AuthEventType, AuthRecoveryCode
from app.models.models import User
from app.models.service_token import ServiceToken

# `auth_sessions.revoked_reason` of the sessions each action ends.
MFA_RESET_REASON = "mfa_reset"
ADMIN_REVOKED_REASON = "admin_revoked"


async def reset_user_mfa(
    db: AsyncSession,
    user: User,
    *,
    actor_id: Optional[UUID],
    event: AuthEventType,
    request: Optional[Request] = None,
) -> int:
    """Remove the second factor of `user` and end all their sessions. Returns how many sessions
    that ended.

    Cleared: the TOTP secret, the secret of an enrolment in progress, the moment MFA was enabled
    (so that it is off), the replay counter, and every recovery code (spent or not). The password
    is untouched: it is not the second factor. The next sign-in answers `mfa_setup`, and a
    sign-in that was already half done (password given, code pending) can no longer finish with
    the factor that is gone. The sessions go too, since each was opened with that factor.

    `event` is what the audit trail says happened: `MFA_RESET_BY_ADMIN` from the admin endpoint,
    `BREAK_GLASS` from the command. The caller commits, then notifies the account
    (`send_mfa_reset_email`).
    """
    await db.execute(
        update(User)
        .where(User.id == user.id)
        .values(
            mfa_totp_secret_enc=None,
            mfa_pending_secret_enc=None,
            mfa_enabled_at=None,
            mfa_last_used_step=None,
        )
    )
    await db.execute(delete(AuthRecoveryCode).where(AuthRecoveryCode.user_id == user.id))
    revoked = await revoke_all_sessions(db, user.id, MFA_RESET_REASON)
    await record_event(
        db,
        event,
        user_id=user.id,
        request=request,
        actor_id=actor_id,
        sessions_revoked=revoked,
    )
    return revoked


async def revoke_user_sessions(
    db: AsyncSession,
    user_id: UUID,
    *,
    actor_id: Optional[UUID],
    request: Optional[Request] = None,
) -> int:
    """End every session of the account that is still open. Returns how many that was (sessions
    that had already ended keep their own revocation and are not counted). Service tokens are
    not sessions and are not touched."""
    revoked = await revoke_all_sessions(db, user_id, ADMIN_REVOKED_REASON)
    await record_event(
        db,
        AuthEventType.SESSIONS_REVOKED_ALL,
        user_id=user_id,
        request=request,
        actor_id=actor_id,
        count=revoked,
    )
    return revoked


async def revoke_user_service_tokens(
    db: AsyncSession,
    user_id: UUID,
    *,
    actor_id: Optional[UUID],
    request: Optional[Request] = None,
) -> int:
    """Revoke the service tokens owned by `user_id` that are not revoked yet. Returns how many
    that was; a token revoked earlier keeps its own date. The only function that does this."""
    result = await db.execute(
        update(ServiceToken)
        .where(ServiceToken.owner_id == user_id, ServiceToken.revoked_at.is_(None))
        .values(revoked_at=now_utc())
    )
    revoked = result.rowcount
    await record_event(
        db,
        AuthEventType.SERVICE_TOKENS_REVOKED,
        user_id=user_id,
        request=request,
        actor_id=actor_id,
        count=revoked,
    )
    return revoked


async def unlock_user(
    db: AsyncSession,
    user: User,
    *,
    actor_id: Optional[UUID],
    request: Optional[Request] = None,
) -> None:
    """Clear the brute-force lock of the account: no recorded failures, no lock date.

    This is the account row only. An address that has no account is locked by the sign-in
    throttle in the rate limiter's storage (so that unknown addresses cannot be told from known
    ones); that lock belongs to no row, is not touched here, and expires by itself.
    """
    await db.execute(
        update(User).where(User.id == user.id).values(failed_login_count=0, locked_until=None)
    )
    await record_event(
        db,
        AuthEventType.ACCOUNT_UNLOCKED,
        user_id=user.id,
        request=request,
        actor_id=actor_id,
    )
