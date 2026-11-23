"""
One-time tokens of the local authentication: invitation, password-reset and e-mail-verification
links, and the short-lived step tokens of the two-factor flow.

A token is an opaque random string (`new_opaque_token`) of which only the SHA-256 is stored in
`auth_one_time_tokens`, so a leaked table gives nothing that can be replayed. Each token belongs
to one user and to one `OneTimeTokenPurpose`, and is only ever valid for that purpose: an
invitation cannot reset a password, a password-reset link cannot stand in for an MFA challenge.

Lifetimes (`TOKEN_TTL`). A token is valid iff it is unused and `now < expires_at`, so it is refused
at the exact instant its lifetime ends.

Single use. `consume_token` spends a token; a spent token is refused for ever. A new token of a
purpose replaces the user's earlier ones of that purpose (`issue_token` marks them used), so only
the last link sent is live. Other purposes and other users are never touched.

MFA step tokens (`MFA_CHALLENGE` after the password, `MFA_ENROLL` while a user sets a second factor
up) are the same rows with a wrong-attempt counter. The flow is `peek_token` (is this step token
alive?), check the code, then `record_failed_attempt` on a wrong one or `consume_token` on the right
one. The fifth wrong attempt spends the token. The counter is per token: a user who logs in again
gets a new challenge with a fresh counter, so the account-level failure counter (the lock on the
user) has to bound guessing across challenges, not this module.

Refusals. `InvalidOneTimeToken` carries one fixed message whatever went wrong, because the caller
answers the same thing for a token that never existed, belongs to another purpose, expired or was
used. The cause is a machine attribute (`reason`) for logs and tests; it is never put in a response.

Transactions. Nothing here commits; the caller does. A refused `consume_token` writes nothing.

Secrets. Raw tokens exist only in the return value of `issue_token` and in what the caller passes
back; they are never logged, and neither is their hash.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal, Optional
from uuid import UUID

from sqlalchemy import case, func, literal, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.tokens import hash_token, new_opaque_token
from app.models.auth import AuthOneTimeToken, OneTimeTokenPurpose

TOKEN_TTL: dict[OneTimeTokenPurpose, timedelta] = {
    OneTimeTokenPurpose.INVITE: timedelta(days=7),
    OneTimeTokenPurpose.PASSWORD_RESET: timedelta(hours=1),
    OneTimeTokenPurpose.EMAIL_VERIFY: timedelta(hours=24),
    OneTimeTokenPurpose.MFA_CHALLENGE: timedelta(minutes=5),
    OneTimeTokenPurpose.MFA_ENROLL: timedelta(minutes=15),
}

# Wrong codes tolerated against one step token; the last one spends it.
MAX_CHALLENGE_ATTEMPTS = 5

_REFUSAL_MESSAGE = "Invalid or expired token"

RefusalReason = Literal["unknown", "wrong_purpose", "used", "expired"]


class InvalidOneTimeToken(Exception):
    """A token that cannot be used. The caller answers the same thing whatever the cause.

    `str(exc)` is one fixed message. `reason` says why (`unknown`, `wrong_purpose`, `used`,
    `expired`) for logs and tests: it is not part of the exception's arguments, and it must not be
    shown to the client, or it would tell an attacker which tokens exist.
    """

    def __init__(self, reason: RefusalReason) -> None:
        super().__init__(_REFUSAL_MESSAGE)
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


def _hash_presented(raw: str) -> str:
    """The hash of a token the client presented. A string that cannot be encoded (a lone
    surrogate from a JSON `"\\ud800"`) is a token that never existed, not a server error."""
    try:
        return hash_token(raw)
    except UnicodeEncodeError:
        raise InvalidOneTimeToken("unknown") from None


async def _load(db: AsyncSession, token_hash: str) -> Optional[AuthOneTimeToken]:
    """The token with this hash as the database holds it now. `populate_existing` makes an
    instance this session already has (loaded before another transaction spent it) take the
    stored values instead of being served stale."""
    return (
        await db.execute(
            select(AuthOneTimeToken)
            .where(AuthOneTimeToken.token_hash == token_hash)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


def _refusal(
    token: Optional[AuthOneTimeToken], purpose: OneTimeTokenPurpose, now: datetime
) -> Optional[RefusalReason]:
    """Why `token` cannot be used for `purpose` at `now`, or None when it can."""
    if token is None:
        return "unknown"
    if token.purpose is not purpose:
        return "wrong_purpose"
    if token.used_at is not None:
        return "used"
    if now >= _aware_utc(token.expires_at):
        return "expired"
    return None


async def issue_token(
    db: AsyncSession,
    user_id: UUID,
    purpose: OneTimeTokenPurpose,
    created_by: Optional[UUID] = None,
    now: Optional[datetime] = None,
) -> str:
    """Create a token for `user_id` and return it in clear (the only time it exists in clear).

    The token lives `TOKEN_TTL[purpose]` from `now`. The user's earlier tokens of the same purpose
    that are still unused are marked used first (at `now`; a token that was already spent keeps its
    own `used_at`), so that only the newest link works; other purposes and other users are
    untouched. `created_by` is the administrator who issued it (an invitation), None when the
    account itself asked.

    Two concurrent calls for the same user and purpose can both succeed and leave two live tokens
    (each invalidates what it sees, and neither sees the other's); both went to the same account,
    so this is accepted rather than serialised. Callers that throttle ask `count_recent_tokens`.
    Flushes, does not commit.
    """
    now = _now(now)
    await db.execute(
        update(AuthOneTimeToken).where(
            AuthOneTimeToken.user_id == user_id,
            AuthOneTimeToken.purpose == purpose,
            AuthOneTimeToken.used_at.is_(None),
        )
        # No `synchronize_session` override: the criteria are also evaluated against the tokens
        # this session already holds, so one it peeked at earlier does not still look usable.
        .values(used_at=now)
    )

    raw = new_opaque_token()
    db.add(
        AuthOneTimeToken(
            user_id=user_id,
            purpose=purpose,
            token_hash=hash_token(raw),
            expires_at=now + TOKEN_TTL[purpose],
            created_by=created_by,
            created_at=now,
        )
    )
    await db.flush()
    return raw


async def consume_token(
    db: AsyncSession,
    raw: str,
    purpose: OneTimeTokenPurpose,
    now: Optional[datetime] = None,
) -> AuthOneTimeToken:
    """Spend the token `raw` for `purpose` and return it (so the caller reads `user_id`).

    Raises `InvalidOneTimeToken` for a token that is unknown, of another purpose, expired or
    already used, and then writes nothing: a wrong-purpose attempt does not burn the token.

    Race-safety: a single conditional `UPDATE ... SET used_at = now WHERE token_hash = h AND
    purpose = p AND used_at IS NULL AND expires_at > now`, and success means `rowcount == 1`. The
    check and the write are one atomic statement, so two concurrent consumers of one token cannot
    both succeed: on Postgres the second waits for the first's row lock, re-evaluates the WHERE on
    the committed row (now used) and matches nothing; on SQLite, which ignores `FOR UPDATE`,
    statements are serialised anyway. A locked read followed by a write would hold the lock across
    two statements and rely on the dialect honouring it; this does not, so the guarantee is the same
    under the test database and in production. The row is read after the update only to return it;
    that read does not decide anything. Does not commit.
    """
    now = _now(now)
    token_hash = _hash_presented(raw)

    result = await db.execute(
        update(AuthOneTimeToken)
        .where(
            AuthOneTimeToken.token_hash == token_hash,
            AuthOneTimeToken.purpose == purpose,
            AuthOneTimeToken.used_at.is_(None),
            AuthOneTimeToken.expires_at > now,
        )
        .values(used_at=now)
        # The instance is reloaded just below, with the stored values.
        .execution_options(synchronize_session=False)
    )
    token = await _load(db, token_hash)
    if result.rowcount == 1 and token is not None:
        return token

    # Nothing was spent. Look at what is stored to tell the logs why. `None` cannot happen here (a
    # row that is usable would have matched the UPDATE), the fallback only keeps the type honest.
    raise InvalidOneTimeToken(_refusal(token, purpose, now) or "used")


async def peek_token(
    db: AsyncSession,
    raw: str,
    purpose: OneTimeTokenPurpose,
    now: Optional[datetime] = None,
) -> AuthOneTimeToken:
    """The token `raw` for `purpose`, if it can still be used, without spending it.

    Same checks and same refusals as `consume_token`. For the second step of a two-factor login:
    the step token must survive a wrong code so that the user can try again (see
    `record_failed_attempt`), and is spent with `consume_token` once the code is right.
    Read-only; a later `consume_token` still has to win its own conditional update.
    """
    now = _now(now)
    token = await _load(db, _hash_presented(raw))
    reason = _refusal(token, purpose, now)
    if reason is not None or token is None:
        raise InvalidOneTimeToken(reason or "unknown")
    return token


async def record_failed_attempt(
    db: AsyncSession, token: AuthOneTimeToken, now: Optional[datetime] = None
) -> int:
    """Count one wrong attempt against `token` and return how many are left
    (`MAX_CHALLENGE_ATTEMPTS` minus the attempts made, never below 0). The attempt that uses the
    last one spends the token (`used_at = now`), so it is refused from then on.

    The increment is done by the database in one statement (`attempts = attempts + 1`, and `used_at`
    set in the same statement when the new count reaches the maximum), never from the value the
    caller's object holds: two requests that loaded the same challenge each count against the
    stored value, whatever their copies say. A token that is already spent (exhausted, or consumed
    by a request that got there first) is left as it is and has 0 attempts left; failures against
    it are not counted, so the counter stops at the maximum.

    `token` must belong to `db`'s session (it comes from `peek_token` on it); it is refreshed with
    the stored values. Flushes through the statement, does not commit.
    """
    now = _now(now)
    await db.execute(
        update(AuthOneTimeToken)
        .where(AuthOneTimeToken.id == token.id, AuthOneTimeToken.used_at.is_(None))
        .values(
            attempts=AuthOneTimeToken.attempts + 1,
            used_at=case(
                (
                    AuthOneTimeToken.attempts + 1 >= MAX_CHALLENGE_ATTEMPTS,
                    # Declared as timestamptz instead of left to inference: a driver that is strict
                    # about types (asyncpg) refuses an aware datetime for a plain timestamp.
                    literal(now, type_=AuthOneTimeToken.used_at.type),
                ),
                else_=AuthOneTimeToken.used_at,
            ),
        )
        .execution_options(synchronize_session=False)
    )
    stored = (
        await db.execute(
            select(AuthOneTimeToken)
            .where(AuthOneTimeToken.id == token.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()
    if stored.used_at is not None:
        return 0
    return max(0, MAX_CHALLENGE_ATTEMPTS - stored.attempts)


async def count_recent_tokens(
    db: AsyncSession, user_id: UUID, purpose: OneTimeTokenPurpose, since: datetime
) -> int:
    """How many tokens of `purpose` were issued for `user_id` at or after `since` (a naive `since`
    is taken as UTC). Counts every token issued, used or superseded or not: it measures what was
    asked for, which is what a rate limit on "send me a reset link" needs."""
    count = (
        await db.execute(
            select(func.count())
            .select_from(AuthOneTimeToken)
            .where(
                AuthOneTimeToken.user_id == user_id,
                AuthOneTimeToken.purpose == purpose,
                AuthOneTimeToken.created_at >= _aware_utc(since),
            )
        )
    ).scalar_one()
    return int(count)
