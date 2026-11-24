"""
Daily retention purge of the local authentication's records.

`purge_auth_records` is declared in the `beat_schedule` of `app/worker/celery_app.py` (daily,
03:15 UTC: after the 02:00 database backup, well before the 06:30 expiry warnings). Nothing else
ever deletes these rows: without it, `auth_events`, `auth_one_time_tokens` and `auth_sessions`
only grow.

What goes, and when (strictly older than the limit: a record exactly at it is kept until the next
run):

- `AuthEvent`: created more than `EVENT_RETENTION` (365 days) ago. The audit trail is the one
  record of who did what; a year is what it is kept for.
- `AuthOneTimeToken`: its `expires_at` OR its `used_at` is more than `TOKEN_RETENTION` (7 days)
  old. A link that expired a week ago, or was spent a week ago (even if it would still have been
  valid), can neither be used nor explain anything the audit trail does not already say.
- `AuthSession`: its `expires_at` OR its `revoked_at` OR its `idle_expires_at` is more than
  `SESSION_RETENTION` (30 days) old. A session is dead as soon as ANY of the three has passed
  (the absolute end, a revocation, or the idle timeout), so each of them counts as "expired or
  revoked" here; a session still live has all three in the future (or `revoked_at` empty) and is
  never touched. The idle timeout is included on purpose: a session nobody refreshed for the idle
  period plus 30 days is long dead, whatever its absolute end says, and leaving it until
  `expires_at` would keep rows for no use. The refresh token of a deleted session is simply
  unknown to the server afterwards, and is refused like any other unknown token.

The rows of one table are deleted by one statement, all three in one transaction: either the
purge happened or it did not, and a failure is retried the next day (nothing is lost by waiting,
the rules are about age only, so the task is idempotent and a double run costs nothing). It
never touches an account, a recovery code or a service token.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Optional, TypeVar

from sqlalchemy import delete, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.clock import now_utc
from app.models.auth import AuthEvent, AuthOneTimeToken, AuthSession
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

T = TypeVar("T")

EVENT_RETENTION = timedelta(days=365)
TOKEN_RETENTION = timedelta(days=7)
SESSION_RETENTION = timedelta(days=30)


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a synchronous Celery task."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def purge_expired_auth_records(
    db: AsyncSession, now: Optional[datetime] = None
) -> dict[str, int]:
    """Delete what the retention rules (see the module docstring) no longer keep, as of `now`
    (default: the current time; a naive `now` is taken as UTC). Returns how many rows went:
    `{"events": ..., "tokens": ..., "sessions": ...}`.

    Does not commit: the caller does (the task does, once the three deletions have run).
    """
    now = now_utc(now)
    event_cutoff = now - EVENT_RETENTION
    token_cutoff = now - TOKEN_RETENTION
    session_cutoff = now - SESSION_RETENTION

    # A NULL `used_at` / `revoked_at` makes its comparison NULL, which `OR` absorbs: a link never
    # used is judged by its expiry only, a session never revoked by its two expiries only.
    events = await db.execute(
        delete(AuthEvent)
        .where(AuthEvent.created_at < event_cutoff)
        .execution_options(synchronize_session=False)
    )
    tokens = await db.execute(
        delete(AuthOneTimeToken)
        .where(
            or_(
                AuthOneTimeToken.expires_at < token_cutoff,
                AuthOneTimeToken.used_at < token_cutoff,
            )
        )
        .execution_options(synchronize_session=False)
    )
    sessions = await db.execute(
        delete(AuthSession)
        .where(
            or_(
                AuthSession.expires_at < session_cutoff,
                AuthSession.revoked_at < session_cutoff,
                AuthSession.idle_expires_at < session_cutoff,
            )
        )
        .execution_options(synchronize_session=False)
    )
    return {"events": events.rowcount, "tokens": tokens.rowcount, "sessions": sessions.rowcount}


@celery_app.task(name="app.worker.tasks.auth_tasks.purge_auth_records")
def purge_auth_records() -> dict[str, int]:
    """Celery Beat entry point — daily."""
    return _run_async(_async_purge())


async def _async_purge() -> dict[str, int]:
    from app.db.session import AsyncSessionLocal

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        counts = await purge_expired_auth_records(db, now)
        await db.commit()

    logger.info(
        "Auth retention purge done: events=%d tokens=%d sessions=%d",
        counts["events"],
        counts["tokens"],
        counts["sessions"],
    )
    return counts
