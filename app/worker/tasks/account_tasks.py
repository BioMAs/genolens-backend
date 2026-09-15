"""
Daily check on subscription expiry — warns before access ends.

Since 2026-09-14 an account whose `subscription_ends_at` has passed drops to
read-only (`app/api/deps/account_state.py`). Enforcement shipped without any
notice, so a user reaches the date and discovers it by being refused. This task
is the missing half: it warns at 7, 3 and 1 days beforehand.

Adapted from `feature/phase1-account-management`, whose version this improves on
in three ways:

1. **It reuses `parse_subscription_end`** instead of re-implementing the date
   parsing. The column is an ISO string written by two callers with different
   formats; having the warning read it one way and the guard another is how an
   email ends up promising a date the guard does not honour. One rule, one place.

2. **It is idempotent.** The original sent on every run where
   `(ends_at - now).days` landed exactly on 7, 3 or 1. Two beat workers, a retry
   or a manual trigger therefore produced duplicate mails, and a day the task did
   not run meant that warning was lost for good. Here the threshold reached is
   recorded on `User.expiry_warning_sent_for`, so each is sent at most once — and
   a missed day sends the warning late rather than never, because the rule is
   "days remaining is at or below the threshold", not "equals it".

3. **It does not cancel anything.** The original auto-cancelled after a 7-day
   grace period. Under the current guard `CANCELLED` is a *hard block* — the user
   loses read access to their own data — where expiry alone is merely read-only.
   Escalating one into the other is a product decision nobody has taken, so it is
   deliberately left out.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

#: Days before expiry at which a warning goes out, largest first.
WARNING_THRESHOLDS = (7, 3, 1)


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task.

    Mirrors `app/worker/tasks/quota_tasks.py` rather than using `asyncio.run`,
    which is what the branch this came from did: the worker may already hold a
    loop, and `asyncio.run` refuses to nest.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def threshold_for(days_remaining: int, already_sent: Optional[int]) -> Optional[int]:
    """The warning threshold to send now, or None when there is nothing to send.

    Returns the smallest threshold at or above `days_remaining` that has not been
    reached yet. Using "at or below the threshold" rather than equality is what
    makes a missed run recoverable: a task that does not run on D-7 and next runs
    on D-5 still sends the 7-day warning, late, instead of skipping it silently.

    `already_sent` shrinks monotonically (7 → 3 → 1), so comparing against it is
    enough to guarantee each threshold goes out at most once.
    """
    if days_remaining < 0:
        # Already expired: the account is read-only and the banner says so. An
        # email announcing "1 day left" would simply be false.
        return None
    candidates = [t for t in WARNING_THRESHOLDS if days_remaining <= t]
    if not candidates:
        return None
    target = min(candidates)
    if already_sent is not None and target >= already_sent:
        return None
    return target


@celery_app.task(name="app.worker.tasks.account_tasks.check_account_expirations")
def check_account_expirations() -> dict:
    """Celery Beat entry point — daily."""
    return _run_async(_async_check_account_expirations())


async def _async_check_account_expirations() -> dict:
    from app.api.deps.account_state import parse_subscription_end
    from app.db.session import AsyncSessionLocal
    from app.models.models import User, UserStatus
    from app.services import email_service

    now = datetime.now(timezone.utc)
    warned = 0
    skipped = 0

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(
                User.status == UserStatus.ACTIVE,
                User.subscription_ends_at.isnot(None),
            )
        )
        users = result.scalars().all()

        for user in users:
            ends_at = parse_subscription_end(user.subscription_ends_at)
            if ends_at is None:
                # Unparsable: `parse_subscription_end` already logged it, and the
                # guard treats it as "no expiry". Warning about a date we cannot
                # read would contradict the access the user actually has.
                skipped += 1
                continue

            days_remaining = (ends_at - now).days
            target = threshold_for(days_remaining, user.expiry_warning_sent_for)
            if target is None:
                continue

            try:
                await email_service.send_expiration_warning_email(
                    to=user.email,
                    days_remaining=max(0, days_remaining),
                    ends_on=ends_at.strftime("%d/%m/%Y"),
                )
            except Exception as exc:
                # Never let one bad address stop the run, and never record a
                # threshold that was not actually delivered — it would be
                # swallowed for good on the next pass.
                logger.warning("Expiry warning failed for %s: %s", user.email, exc)
                continue

            user.expiry_warning_sent_for = target
            warned += 1
            logger.info("Expiry warning sent to %s (J-%d)", user.email, target)

        await db.commit()

    logger.info(
        "Expiration check done: warned=%d skipped=%d over %d account(s) with an end date",
        warned, skipped, len(users),
    )
    return {"warned": warned, "skipped": skipped, "checked": len(users)}
