"""
Server-side sessions of the local authentication, renewed through rotating refresh tokens.

A login opens an `AuthSession` row and hands out two tokens: a short-lived access token (a JWT
bound to the session, see `app.auth.tokens`) and an opaque refresh token. Only the SHA-256 of the
refresh token is stored. Because the access token is checked against the row on every request
(`get_session_user`), revoking a session cuts it off at once rather than when the JWT expires.

Lifetimes. A session dies after `IDLE_TTL` without a refresh, and at `ABSOLUTE_TTL` after the
login whatever the activity. It is valid iff it is not revoked, `now < idle_expires_at` and
`now < expires_at`. A refresh slides the inactivity window forward (never past the absolute one);
reading the session to authenticate a request does not.

Rotation. Every refresh replaces the refresh token. The hash it replaces is kept for one
generation in `previous_refresh_token_hash`, with `rotated_at`:

- presented within `REFRESH_GRACE` of the rotation, the old token is a concurrent request of the
  same client (two tabs, a retry after a lost response): it gets a fresh access token and no new
  refresh token (the client already holds the one the winner got);
- presented later, it is a replay (a stolen token, or a client that kept an old one): the session
  is revoked with the reason `refresh_reuse`, so the thief and the honest holder are both cut off
  and the user has to sign in again.

Only one generation is kept: a token two rotations old matches nothing and is simply "invalid".

Transactions. Nothing here commits; the caller does. That matters in one place: on a reuse,
`rotate_refresh` flushes the revocation and then raises, and the caller must still commit before it
answers 401, or the revocation rolls back with the failed request.

Secrets. Raw tokens exist only in the return values (whose `repr` hides them) and are never logged.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, Optional
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.tokens import hash_token, issue_access_token, new_opaque_token
from app.models.auth import AuthSession
from app.models.models import User

IDLE_TTL = timedelta(days=7)
ABSOLUTE_TTL = timedelta(days=30)
REFRESH_GRACE = timedelta(seconds=30)

REFRESH_REUSE_REASON = "refresh_reuse"

# Widths of the `auth_sessions` columns: Postgres refuses a longer value, and both come straight
# from request headers, so a hostile client would otherwise turn a login into a 500.
_IP_MAX_LENGTH = 64
_USER_AGENT_MAX_LENGTH = 512


@dataclass
class IssuedSession:
    """What a login hands to the caller, to be put in cookies.

    `expires_at` is the absolute end of the session (`ABSOLUTE_TTL` after the login), which bounds
    the lifetime of the refresh cookie.
    """

    session_id: UUID
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: datetime


@dataclass
class RefreshOutcome:
    """The result of a refresh. `refresh_token` is None when the request fell in the grace window:
    only a new access token is issued and the client keeps the refresh token it already has."""

    session_id: UUID
    access_token: str = field(repr=False)
    refresh_token: Optional[str] = field(repr=False)


class RefreshRejected(Exception):
    """A refresh token that cannot be used. The caller answers 401 whatever the reason; the reason
    tells it what to record (`reuse` is a security event) and says nothing about the token."""

    def __init__(self, reason: Literal["invalid", "expired", "reuse"]) -> None:
        super().__init__(reason)
        self.reason = reason


def _now(now: Optional[datetime]) -> datetime:
    """The instant to work at, as an aware UTC datetime (a naive one is taken as UTC)."""
    if now is None:
        return datetime.now(UTC)
    return _aware_utc(now)


def _aware_utc(moment: datetime) -> datetime:
    """SQLite hands back the naive UTC wall clock of what was stored, Postgres an aware value:
    normalise both so that they compare with `now`."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _is_live(session: AuthSession, now: datetime) -> bool:
    return (
        session.revoked_at is None
        and now < _aware_utc(session.idle_expires_at)
        and now < _aware_utc(session.expires_at)
    )


async def create_session(
    db: AsyncSession,
    user: User,
    amr: list[str],
    ip: Optional[str],
    user_agent: Optional[str],
    now: Optional[datetime] = None,
) -> IssuedSession:
    """Open a session for `user` and issue its first access and refresh tokens.

    `amr` records how the user proved who they are (`["pwd"]`, `["pwd", "otp"]`, ...); every access
    token of the session carries it. Also records `user.last_login_at`. Flushes, does not commit.
    """
    now = _now(now)
    session_id = uuid4()
    refresh_token = new_opaque_token()
    expires_at = now + ABSOLUTE_TTL

    db.add(
        AuthSession(
            id=session_id,
            user_id=user.id,
            created_at=now,
            last_used_at=now,
            idle_expires_at=now + IDLE_TTL,
            expires_at=expires_at,
            ip=ip[:_IP_MAX_LENGTH] if ip else None,
            user_agent=user_agent[:_USER_AGENT_MAX_LENGTH] if user_agent else None,
            amr=list(amr),
            refresh_token_hash=hash_token(refresh_token),
        )
    )
    # A statement, not just an attribute: the caller's `user` may belong to no session (a loaded
    # object whose session was closed), in which case assigning it would reach nothing. The
    # statement also refreshes the object when it is attached; the assignment covers the other case.
    await db.execute(update(User).where(User.id == user.id).values(last_login_at=now))
    user.last_login_at = now
    await db.flush()

    return IssuedSession(
        session_id=session_id,
        access_token=issue_access_token(user.id, session_id, list(amr), now=now),
        refresh_token=refresh_token,
        expires_at=expires_at,
    )


async def rotate_refresh(
    db: AsyncSession, raw_refresh: str, now: Optional[datetime] = None
) -> RefreshOutcome:
    """Exchange a refresh token for a new access token, and usually a new refresh token.

    - the current token: rotated (the old hash moves to `previous_refresh_token_hash`,
      `rotated_at` and `last_used_at` become `now`, `idle_expires_at` becomes
      `min(now + IDLE_TTL, expires_at)`) and a new refresh token is returned;
    - the previous token, at most `REFRESH_GRACE` after the rotation: a new access token only,
      `refresh_token` is None and nothing is written;
    - the previous token, later: the session is revoked (`refresh_reuse`) and
      `RefreshRejected("reuse")` is raised. The revocation is flushed, not committed: the caller
      must commit before answering;
    - a session that is revoked or expired, whichever token matches: `RefreshRejected("expired")`.
      This check comes first, so a replay against a session that is already dead is "expired"
      (it is not revoked a second time and raises no second reuse alarm);
    - a token that matches nothing: `RefreshRejected("invalid")`.

    Concurrency. The session row is read `FOR UPDATE` (Postgres honours it, SQLite ignores it), by
    `refresh_token_hash = h OR previous_refresh_token_hash = h` in a single statement, and stays
    locked until the caller commits or rolls back. Two refreshes of one token are therefore
    serialised: the first rotates; the second, which waited on the lock, re-reads the committed row
    (Postgres re-evaluates the WHERE on the updated row, and the token is now its previous hash),
    lands in the grace window and receives an access token only. Without the lock both would read
    the same current hash and both rotate, and the one that wrote second would orphan the other
    client's new refresh token. Because of that, the caller commits right after this returns and
    does nothing slow in between.
    """
    now = _now(now)
    presented = hash_token(raw_refresh)

    session = (
        (
            await db.execute(
                select(AuthSession)
                .where(
                    or_(
                        AuthSession.refresh_token_hash == presented,
                        AuthSession.previous_refresh_token_hash == presented,
                    )
                )
                .with_for_update()
                # An instance already in this session's identity map must be refreshed from the
                # locked row, not served as it was loaded.
                .execution_options(populate_existing=True)
            )
        )
        .scalars()
        .first()
    )
    if session is None:
        raise RefreshRejected("invalid")
    if not _is_live(session, now):
        raise RefreshRejected("expired")

    if session.refresh_token_hash == presented:
        refresh_token = new_opaque_token()
        session.previous_refresh_token_hash = presented
        session.refresh_token_hash = hash_token(refresh_token)
        session.rotated_at = now
        session.last_used_at = now
        session.idle_expires_at = min(now + IDLE_TTL, _aware_utc(session.expires_at))
        await db.flush()
        return RefreshOutcome(
            session_id=session.id,
            access_token=issue_access_token(
                session.user_id, session.id, list(session.amr), now=now
            ),
            refresh_token=refresh_token,
        )

    # The presented token is the previous one. A session with a previous hash always has a
    # rotated_at; if it is missing anyway, fail closed (outside the grace window).
    rotated_at = session.rotated_at
    if rotated_at is not None and now - _aware_utc(rotated_at) <= REFRESH_GRACE:
        return RefreshOutcome(
            session_id=session.id,
            access_token=issue_access_token(
                session.user_id, session.id, list(session.amr), now=now
            ),
            refresh_token=None,
        )

    session.revoked_at = now
    session.revoked_reason = REFRESH_REUSE_REASON
    await db.flush()
    raise RefreshRejected("reuse")


async def revoke_session(db: AsyncSession, session_id: UUID, reason: str) -> None:
    """Revoke one session. A session that is already revoked keeps its first revocation (instant
    and reason), and an unknown id is not an error.

    The session is looked up by id alone: a caller acting on behalf of a user (listing and ending
    "my sessions") must check that the session is theirs first. Does not commit.
    """
    await db.execute(
        update(AuthSession)
        .where(AuthSession.id == session_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC), revoked_reason=reason)
    )


async def revoke_all_sessions(
    db: AsyncSession,
    user_id: UUID,
    reason: str,
    keep_session_id: Optional[UUID] = None,
) -> int:
    """Revoke every session of `user_id` that is not revoked yet, except `keep_session_id` (the one
    making the request, when a user changes their password). Returns how many were revoked: sessions
    that were already revoked keep their own revocation and are not counted. Does not commit."""
    statement = update(AuthSession).where(
        AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None)
    )
    if keep_session_id is not None:
        statement = statement.where(AuthSession.id != keep_session_id)
    result = await db.execute(statement.values(revoked_at=datetime.now(UTC), revoked_reason=reason))
    return result.rowcount


async def get_session_user(
    db: AsyncSession, session_id: UUID, user_id: UUID, now: Optional[datetime] = None
) -> Optional[tuple[User, AuthSession]]:
    """The user and the session behind an access token, or None.

    One query: `users` joined to `auth_sessions`, filtered on the session id AND the user id (the
    token's `sid` and `sub`, so a session id under another account's id finds nothing) and on the
    session being live: not revoked, `now < idle_expires_at` and `now < expires_at`. Read-only: it
    does not slide the inactivity window.
    """
    now = _now(now)
    row = (
        await db.execute(
            select(User, AuthSession)
            .join(AuthSession, AuthSession.user_id == User.id)
            .where(
                AuthSession.id == session_id,
                User.id == user_id,
                AuthSession.revoked_at.is_(None),
                AuthSession.idle_expires_at > now,
                AuthSession.expires_at > now,
            )
        )
    ).first()
    if row is None:
        return None
    return row[0], row[1]
