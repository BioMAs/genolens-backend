"""
Celery Beat task: daily account expiration check.
Sends warning emails at D-7, D-3, D-1 and cancels accounts after 7-day grace period.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List

from celery import shared_task
from sqlalchemy import select

from app.models.models import User, UserStatus
from app.services import email_service

logger = logging.getLogger(__name__)

GRACE_PERIOD_DAYS = 7
WARNING_DAYS = [7, 3, 1]


def _parse_dt(iso_str: str) -> datetime:
    """Parse ISO datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _get_users_to_cancel(users: List[User], now: datetime) -> List[User]:
    """Return users whose grace period has elapsed (7+ days past expiry)."""
    result = []
    for user in users:
        if not user.subscription_ends_at:
            continue
        try:
            ends_at = _parse_dt(user.subscription_ends_at)
        except (ValueError, TypeError):
            logger.error("Malformed subscription_ends_at for user %s: %r", getattr(user, "email", "?"), user.subscription_ends_at)
            continue
        if now >= ends_at + timedelta(days=GRACE_PERIOD_DAYS):
            result.append(user)
    return result


def _get_users_to_warn(users: List[User], now: datetime) -> List[User]:
    """Return users expiring in exactly 7, 3, or 1 day (integer day comparison)."""
    result = []
    for user in users:
        if not user.subscription_ends_at:
            continue
        try:
            ends_at = _parse_dt(user.subscription_ends_at)
        except (ValueError, TypeError):
            logger.error("Malformed subscription_ends_at for user %s: %r", getattr(user, "email", "?"), user.subscription_ends_at)
            continue
        days_remaining = (ends_at - now).days
        if days_remaining in WARNING_DAYS:
            result.append(user)
    return result


@shared_task(name="app.worker.account_tasks.check_account_expirations")
def check_account_expirations() -> dict:
    """Celery Beat entry point — sync wrapper over async implementation."""
    return asyncio.run(_async_check_account_expirations())


async def _async_check_account_expirations() -> dict:
    # Import here to avoid circular import at module load time
    from app.db.session import AsyncSessionLocal

    now = datetime.now(timezone.utc)
    cancelled_count = 0
    warned_count = 0

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(
                User.status == UserStatus.ACTIVE,
                User.subscription_ends_at.isnot(None),
            )
        )
        active_users = result.scalars().all()

        to_cancel = _get_users_to_cancel(active_users, now)
        cancel_ids = set()
        for user in to_cancel:
            user.status = UserStatus.CANCELLED
            cancel_ids.add(user.id)
            try:
                await email_service.send_subscription_cancelled_email(to_email=user.email)
            except Exception as exc:
                logger.warning("Cancellation email failed for %s: %s", user.email, exc)
            cancelled_count += 1
            logger.info("Auto-cancelled account: %s", user.email)

        # Exclude just-cancelled users from the warning list
        to_warn = [u for u in _get_users_to_warn(active_users, now) if u.id not in cancel_ids]
        for user in to_warn:
            try:
                ends_at = _parse_dt(user.subscription_ends_at)
                days_left = (ends_at - now).days
                await email_service.send_expiration_warning_email(
                    to_email=user.email, days_remaining=days_left
                )
            except Exception as exc:
                logger.warning("Warning email failed for %s: %s", user.email, exc)
                continue
            warned_count += 1

        await db.commit()

    logger.info("Expiration check done: cancelled=%d warned=%d", cancelled_count, warned_count)
    return {"cancelled": cancelled_count, "warned": warned_count}
