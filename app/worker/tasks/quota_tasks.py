"""
Periodic tasks for quota management.

`reset_monthly_analysis_quotas` est declaree dans le `beat_schedule` de
`app/worker/celery_app.py` (1er du mois, 00:05 UTC) et tourne en production
depuis l'ajout du service `beat` a `docker-compose.prod.yml`.

Le chemin paresseux reste en place en filet : `_reset_quota_if_new_month`
(app/api/deps/subscription.py) remet le compteur a zero au premier appel
touchant au quota que fait l'utilisateur dans le mois.

Les deux chemins epargnent les comptes demo (`users.is_demo`) : leur quota
couvre toute la duree de la demo, un nouveau mois ne doit pas le renouveler.
"""
import asyncio
import logging
from datetime import datetime, timezone

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="app.worker.tasks.quota_tasks.reset_monthly_analysis_quotas")
def reset_monthly_analysis_quotas() -> dict:
    """
    Reset analyses_used_this_month to 0 for all users except demo accounts.
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
            .where(User.is_demo.is_(False))
            .values(analyses_used_this_month=0, quota_reset_at=now)
        )
        await db.commit()
        row_count = result.rowcount

    logger.info("Monthly quota reset: %d users reset at %s", row_count, now.isoformat())
    return {"reset_count": row_count, "reset_at": now.isoformat()}
