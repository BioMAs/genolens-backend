"""
Periodic tasks for quota management.

ATTENTION : cette tache est declaree dans le `beat_schedule` de
`app/worker/celery_app.py` (1er du mois, 00:05 UTC) mais **aucun service
`celery beat` n'existe dans `docker-compose.prod.yml`** — seuls `api`, `worker`
et `r-worker` tournent. Elle n'est donc jamais declenchee en production.

La remise a zero effective vient du chemin paresseux :
`_reset_quota_if_new_month` (app/api/deps/subscription.py) remet le compteur a
zero au premier appel touchant au quota que fait l'utilisateur dans le mois.
C'est suffisant fonctionnellement — un compteur n'a d'importance qu'au moment
ou on le lit — mais ca veut dire qu'aucun etat en base n'est remis a zero pour
un utilisateur inactif, et que les statistiques lues directement en base
peuvent montrer un compteur du mois precedent.

Ajouter un service `beat` rendrait cette tache reelle. En attendant, ne pas
compter sur elle.
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
    Reset analyses_used_this_month to 0 for all users.
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
            .values(analyses_used_this_month=0, quota_reset_at=now)
        )
        await db.commit()
        row_count = result.rowcount

    logger.info("Monthly quota reset: %d users reset at %s", row_count, now.isoformat())
    return {"reset_count": row_count, "reset_at": now.isoformat()}
