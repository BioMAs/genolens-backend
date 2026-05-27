"""
Periodic tasks for quota management.
Scheduled via Celery Beat — runs on the 1st of each month at 00:05 UTC.
"""
import asyncio
import logging
from datetime import datetime, timezone

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(name="app.worker.tasks.quota_tasks.reset_monthly_comparison_quotas")
def reset_monthly_comparison_quotas() -> dict:
    """
    Reset comparisons_used_this_month to 0 for all users.
    Updates quota_reset_at to now.
    Runs via async session (asyncpg) — no psycopg2 required.
    """
    return _run_async(_async_reset())


async def _async_reset() -> dict:
    from sqlalchemy import update
    from app.db.session import AsyncSessionLocal
    from app.models.models import User

    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            update(User)
            .values(comparisons_used_this_month=0, quota_reset_at=now)
        )
        await db.commit()
        row_count = result.rowcount

    logger.info("Monthly quota reset: %d users reset at %s", row_count, now.isoformat())
    return {"reset_count": row_count, "reset_at": now.isoformat()}
