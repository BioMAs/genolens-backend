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
    "app.worker.tasks.*": {"queue": "default"},
}
