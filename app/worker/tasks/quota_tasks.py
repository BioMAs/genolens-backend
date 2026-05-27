"""
Periodic tasks for quota management.
Scheduled via Celery Beat — runs on the 1st of each month at 00:05 UTC.
"""
import logging
from datetime import datetime, timezone

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.worker.tasks.quota_tasks.reset_monthly_comparison_quotas")
def reset_monthly_comparison_quotas() -> dict:
    """
    Reset comparisons_used_this_month to 0 for all users.
    Updates quota_reset_at to now.
    Runs synchronously (uses sync SQLAlchemy) — Celery workers are sync.
    """
    from sqlalchemy import create_engine, update
    from app.core.config import settings
    from app.models.models import User

    # Celery tasks use sync DB — convert async URL to sync
    sync_url = settings.DATABASE_URL.replace(
        "postgresql+asyncpg://", "postgresql+psycopg2://"
    )
    engine = create_engine(sync_url)
    now = datetime.now(timezone.utc)

    with engine.begin() as conn:
        result = conn.execute(
            update(User)
            .values(comparisons_used_this_month=0, quota_reset_at=now)
        )
        row_count = result.rowcount

    logger.info("Monthly quota reset: %d users reset at %s", row_count, now.isoformat())
    return {"reset_count": row_count, "reset_at": now.isoformat()}
