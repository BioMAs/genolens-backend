"""
Celery application configuration for background task processing.
"""
from celery import Celery

from app.core.config import settings


# Create Celery app
celery_app = Celery(
    "genolens_worker",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["app.worker.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # Results expire after 1 hour
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Task routing (optional - for multiple queues)
celery_app.conf.task_routes = {
    "app.worker.tasks.process_dataset_upload": {"queue": "data_processing"},
    "app.worker.tasks.report_task.*": {"queue": "default"},
    "app.worker.tasks.*": {"queue": "default"},
}

# ── Periodic tasks (Celery Beat) ──────────────────────────────────────────────
from celery.schedules import crontab  # noqa: E402

celery_app.conf.include = [
    "app.worker.tasks",
    "app.worker.tasks.quota_tasks",
    "app.worker.tasks.deployment_task",
    "app.worker.tasks.report_task",
    "app.worker.tasks.intersection_enrichment_task",
    "app.worker.tasks.gsea_task",
]

celery_app.conf.beat_schedule = {
    "reset-monthly-comparison-quotas": {
        "task": "app.worker.tasks.quota_tasks.reset_monthly_comparison_quotas",
        "schedule": crontab(minute=5, hour=0, day_of_month=1),
        # Runs at 00:05 UTC on the 1st of each month
    },
}
