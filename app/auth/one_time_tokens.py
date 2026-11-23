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
up) are the same rows with an attempt counter. The flow that checks a code is `reserve_attempt` ->
verify the code -> `consume_token` on a right one. The attempt is taken BEFORE the code is checked,
in one atomic statement, so that the bound counts codes checked, not failures recorded: N parallel
requests with N guesses cannot all get through before the first failure is written down. The fifth
reservation is the last; a wrong code leaves nothing to undo (the attempt is simply spent), a right
one is still consumable (`consume_token` does not look at the counter). `peek_token` is for the
steps that check no code (starting an enrolment, say) and does not count anything. The counter is
per token: a user who logs in again gets a new challenge with a fresh counter, so the account-level
failure counter (the lock on the user) has to bound guessing across challenges, not this module.

Refusals. `InvalidOneTimeToken` carries one fixed message whatever went wrong, because the caller
answers the same thing for a token that never existed, belongs to another purpose, expired or was
used. The cause is a machine attribute (`reason`) for logs and tests; it is never put in a response.

Transactions. Nothing here commits; the caller does. A refused `consume_token` or `reserve_attempt`
writes nothing. One place needs care: a reserved attempt is only spent once it is committed, so the
caller commits right after `reserve_attempt` (which also releases the row lock early) and must not
let a rollback, for example the one that follows an HTTP 401, hand the attempt back.

Secrets. Raw tokens exist only in the return value of `issue_token` and in what the caller passes
back; they are never logged, and neither is their hash.
"""

from datetime import datetime, timedelta
from typing import Literal, Optional
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.clock import aware_utc, now_utc
from app.auth.tokens import hash_token, new_opaque_token
from app.models.auth import AuthOneTimeToken, OneTimeTokenPurpose

TOKEN_TTL: dict[OneTimeTokenPurpose, timedelta] = {
    OneTimeTokenPurpose.INVITE: timedelta(days=7),
    OneTimeTokenPurpose.PASSWORD_RESET: timedelta(hours=1),
    OneTimeTokenPurpose.EMAIL_VERIFY: timedelta(hours=24),
    OneTimeTokenPurpose.MFA_CHALLENGE: timedelta(minutes=5),
    OneTimeTokenPurpose.MFA_ENROLL: timedelta(minutes=15),
}

# Code checks a step token allows (`reserve_attempt`); the sixth is refused.
MAX_CHALLENGE_ATTEMPTS = 5

_REFUSAL_MESSAGE = "Invalid or expired token"

RefusalReason = Literal["unknown", "wrong_purpose", "used", "expired", "exhausted"]


class InvalidOneTimeToken(Exception):
    """A token that cannot be used. The caller answers the same thing whatever the cause.

    `str(exc)` is one fixed message. `reason` says why (`unknown`, `wrong_purpose`, `used`,
    `expired`, and for `reserve_attempt` `exhausted`) for logs and tests: it is not part of the
    exception's arguments, and it must not be shown to the client, or it would tell an attacker
    which tokens exist.
    """

    def __init__(self, reason: RefusalReason) -> None:
        super().__init__(_REFUSAL_MESSAGE)
        self.reason = reason


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
    if now >= aware_utc(token.expires_at):
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
    now = now_utc(now)
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
    now = now_utc(now)
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

    Same checks and same refusals as `consume_token` (it does not look at `attempts`). For the
    steps that check no code, such as starting an enrolment: a step that does check a code must
    use `reserve_attempt`, which counts the attempt, and not this. Read-only and counts nothing; a
    later `consume_token` still has to win its own conditional update.
    """
    now = now_utc(now)
    token = await _load(db, _hash_presented(raw))
    reason = _refusal(token, purpose, now)
    if reason is not None or token is None:
        raise InvalidOneTimeToken(reason or "unknown")
    return token


async def reserve_attempt(
    db: AsyncSession,
    raw: str,
    purpose: OneTimeTokenPurpose,
    now: Optional[datetime] = None,
) -> AuthOneTimeToken:
    """Take one of the `MAX_CHALLENGE_ATTEMPTS` code checks of the step token `raw`, and return the
    token with `attempts` already counting this one (`MAX_CHALLENGE_ATTEMPTS - token.attempts` are
    left). The caller then verifies the code, and calls `consume_token` if it is right.

    Raises `InvalidOneTimeToken` for a token that is unknown, of another purpose, expired, already
    used, or whose attempts are all taken (`exhausted`), and then writes nothing.

    Race-safety: a single conditional `UPDATE ... SET attempts = attempts + 1 WHERE token_hash = h
    AND purpose = p AND used_at IS NULL AND expires_at > now AND attempts < MAX`, and success means
    `rowcount == 1`. Reserving before verifying is the point: if the attempt were counted after a
    failed check, N parallel requests would all pass the check on a counter still at 0 and make N
    guesses. Here the database hands out the attempts one by one (on Postgres a concurrent request
    waits on the row lock and re-evaluates `attempts < MAX` on the committed row), so exactly
    `MAX_CHALLENGE_ATTEMPTS` reservations succeed, however many arrive at once. The row is read
    after the update only to return it and, on a refusal, to tell the logs why.

    The attempt is spent for good, right or wrong, once committed; the fifth, if the code is right,
    can still be consumed (`consume_token` does not look at `attempts`). Commit right after this
    call, before verifying: the attempt must survive a rollback caused by answering a wrong code,
    and the row lock is released early. A token whose attempts are all taken stays unused until it
    expires; nothing can reserve on it any more.
    """
    now = now_utc(now)
    token_hash = _hash_presented(raw)

    result = await db.execute(
        update(AuthOneTimeToken)
        .where(
            AuthOneTimeToken.token_hash == token_hash,
            AuthOneTimeToken.purpose == purpose,
            AuthOneTimeToken.used_at.is_(None),
            AuthOneTimeToken.expires_at > now,
            AuthOneTimeToken.attempts < MAX_CHALLENGE_ATTEMPTS,
        )
        .values(attempts=AuthOneTimeToken.attempts + 1)
        # The instance is reloaded just below, with the stored values.
        .execution_options(synchronize_session=False)
    )
    token = await _load(db, token_hash)
    if result.rowcount == 1 and token is not None:
        return token

    # Nothing was reserved. A token that is of this purpose, unused and unexpired, yet did not
    # match, can only have failed the one condition `_refusal` does not look at: the attempts.
    raise InvalidOneTimeToken(_refusal(token, purpose, now) or "exhausted")


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
                AuthOneTimeToken.created_at >= aware_utc(since),
            )
        )
    ).scalar_one()
    return int(count)
