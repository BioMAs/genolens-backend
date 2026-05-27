"""Celery task modules."""
# Import quota tasks so Celery discovers them via include=["app.worker.tasks"]
from app.worker.tasks.quota_tasks import reset_monthly_comparison_quotas  # noqa: F401
