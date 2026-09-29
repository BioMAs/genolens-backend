"""
Every task in the beat schedule must actually be registered with Celery.

This catches a trap the codebase is unusually exposed to. `app/worker/tasks/` is
a package that *shadows* the old `tasks.py`, so it re-exports each task by hand
in its `__init__`. Listing a module in `celery_app.conf.include` and adding an
entry to `beat_schedule` is therefore not enough: unless the task is also
imported in `app/worker/tasks/__init__.py`, beat happily enqueues a name the
worker cannot resolve and answers "Received unregistered task" — at 06:30, in a
log nobody is reading, on a schedule nobody is watching.

That is exactly what happened while adding `check_account_expirations`, and the
only reason it was caught is that the registration was checked rather than
assumed. This test makes the check permanent.
"""
import pytest

from app.worker.celery_app import celery_app

# Importing the package is what performs the registration — the same import the
# worker process does at startup.
import app.worker.tasks  # noqa: F401


def _scheduled_task_names() -> list[str]:
    return [cfg["task"] for cfg in celery_app.conf.beat_schedule.values()]


def test_beat_schedule_is_not_empty():
    """A rename that empties the schedule must not make the test below vacuous."""
    assert _scheduled_task_names(), "beat_schedule is empty — did a key get renamed?"


@pytest.mark.parametrize("task_name", _scheduled_task_names())
def test_scheduled_task_is_registered(task_name):
    assert task_name in celery_app.tasks, (
        f"{task_name} is scheduled in beat_schedule but not registered with Celery. "
        f"Import it in app/worker/tasks/__init__.py — listing the module in "
        f"celery_app.conf.include is not sufficient, because the tasks/ package "
        f"shadows tasks.py and re-exports by hand."
    )


def test_expiry_check_is_scheduled_daily():
    """
    Pinned because the value is load-bearing rather than cosmetic: the warning
    thresholds (7, 3, 1 days) assume one run per day. A weekly schedule would
    step straight over the 3- and 1-day buckets.
    """
    entry = celery_app.conf.beat_schedule["check-account-expirations"]
    schedule = entry["schedule"]
    assert schedule.day_of_month == {*range(1, 32)}, "must run every day of the month"
    assert schedule.month_of_year == {*range(1, 13)}, "must run every month"
    assert len(schedule.hour) == 1 and len(schedule.minute) == 1, "once a day, not hourly"
