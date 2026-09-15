"""
Tests for the daily subscription-expiry warning task.

Enforcement of `subscription_ends_at` shipped on 2026-09-14 with no notice at
all: a user reached the date and found out by being refused a write. This task
is the missing half — warnings at 7, 3 and 1 days.

Most of the risk lives in `threshold_for`, which is why it is tested directly and
exhaustively rather than only through the task. Two properties matter more than
any single case:

- **each threshold goes out at most once**, or a customer gets the same
  "1 day left" mail several times and reads it as a malfunction;
- **a missed run is recoverable**, because a scheduler that skips a day must not
  silently swallow a warning for good. That is the reason the rule is "days
  remaining is at or below the threshold" rather than the equality the original
  branch used.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.models import SubscriptionPlan, User, UserRole, UserStatus
from app.worker.tasks.account_tasks import (
    WARNING_THRESHOLDS,
    _async_check_account_expirations,
    threshold_for,
)


# ── threshold_for ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "days_remaining,expected",
    [
        (30, None),  # far off
        (8, None),   # one day before the first threshold
        (7, 7),
        (3, 3),
        (1, 1),
        (0, 1),      # expires today — still the 1-day bucket
    ],
)
def test_first_pass_picks_the_right_threshold(days_remaining, expected):
    assert threshold_for(days_remaining, None) is expected


@pytest.mark.parametrize("days_remaining", [7, 6, 5, 4])
def test_seven_day_warning_is_not_repeated(days_remaining):
    """Between D-7 and D-4 the 7 bucket is already spent: nothing more to send."""
    assert threshold_for(days_remaining, 7) is None


def test_each_threshold_fires_exactly_once_over_a_full_countdown():
    """
    Walk a user from D-10 to expiry one day at a time, as the daily task would,
    and assert exactly three mails leave — one per threshold, in order.
    """
    sent = None
    fired = []
    for day in range(10, -1, -1):
        target = threshold_for(day, sent)
        if target is not None:
            fired.append((day, target))
            sent = target
    assert fired == [(7, 7), (3, 3), (1, 1)]
    assert sorted(WARNING_THRESHOLDS, reverse=True) == [7, 3, 1]


def test_a_missed_day_still_sends_the_warning_late():
    """
    The scheduler skips D-7 and next runs at D-5. The warning must still go out —
    late — instead of being lost. The original branch compared for equality and
    would have skipped it silently.
    """
    assert threshold_for(5, None) == 7


def test_a_long_outage_collapses_to_the_most_urgent_warning():
    """Down from D-8 to D-1: send the 1-day warning, not a burst of three."""
    assert threshold_for(1, None) == 1


def test_already_expired_sends_nothing():
    """
    The account is already read-only and the banner says so. A mail announcing
    "1 day remaining" would simply be false.
    """
    assert threshold_for(-1, None) is None
    assert threshold_for(-30, None) is None


def test_thresholds_never_go_backwards():
    """Once the 1-day mail is out, nothing re-opens an earlier bucket."""
    for day in range(0, 11):
        assert threshold_for(day, 1) is None


# ── The task ─────────────────────────────────────────────────────────────────


def make_user(days_from_now=None, sent_for=None, status=UserStatus.ACTIVE) -> User:
    u = User()
    u.id = uuid4()
    u.email = f"{uuid4().hex[:6]}@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.STARTER
    u.status = status
    u.expiry_warning_sent_for = sent_for
    u.subscription_ends_at = (
        (datetime.now(timezone.utc) + timedelta(days=days_from_now, hours=1)).isoformat()
        if days_from_now is not None
        else None
    )
    return u


async def run_task(monkeypatch, users, send=None):
    """Drive the task against a fake session, capturing the mails it sends."""
    import app.worker.tasks.account_tasks as mod

    db = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=users)))
    db.execute = AsyncMock(return_value=result)

    class _Session:
        async def __aenter__(self_):
            return db

        async def __aexit__(self_, *exc):
            return False

    monkeypatch.setattr("app.db.session.AsyncSessionLocal", lambda: _Session())

    sent = []

    async def _send(to, days_remaining, ends_on):
        if send is not None:
            return await send(to, days_remaining, ends_on)
        sent.append({"to": to, "days_remaining": days_remaining, "ends_on": ends_on})
        return True

    from app.services import email_service

    monkeypatch.setattr(email_service, "send_expiration_warning_email", _send)
    stats = await mod._async_check_account_expirations()
    return stats, sent, db


async def test_task_warns_and_records_the_threshold(monkeypatch):
    user = make_user(days_from_now=7)
    stats, sent, db = await run_task(monkeypatch, [user])

    assert stats["warned"] == 1
    assert len(sent) == 1
    assert sent[0]["to"] == user.email
    assert user.expiry_warning_sent_for == 7
    db.commit.assert_awaited()


async def test_task_is_idempotent_across_runs(monkeypatch):
    """A second run the same day must not re-send — the whole point of the column."""
    user = make_user(days_from_now=7)
    await run_task(monkeypatch, [user])
    stats, sent, _ = await run_task(monkeypatch, [user])

    assert stats["warned"] == 0
    assert sent == []


async def test_task_skips_users_without_an_end_date(monkeypatch):
    stats, sent, _ = await run_task(monkeypatch, [make_user(days_from_now=None)])
    assert stats["warned"] == 0
    assert sent == []


async def test_task_skips_unparsable_dates_without_crashing(monkeypatch):
    """
    Fails open, like the guard does. Warning about a date we cannot read would
    contradict the access the user actually has.
    """
    user = make_user(days_from_now=7)
    user.subscription_ends_at = "31/12/2026"
    stats, sent, _ = await run_task(monkeypatch, [user])

    assert stats["skipped"] == 1
    assert sent == []
    assert user.expiry_warning_sent_for is None


async def test_a_failed_send_is_not_recorded_as_sent(monkeypatch):
    """
    Otherwise the threshold would be swallowed for good on the next pass and the
    user would never hear about it. One bad address must not stop the run either.
    """
    boom, ok = make_user(days_from_now=3), make_user(days_from_now=3)

    async def _send(to, days_remaining, ends_on):
        if to == boom.email:
            raise RuntimeError("SMTP down")
        return True

    stats, _, _ = await run_task(monkeypatch, [boom, ok], send=_send)

    assert boom.expiry_warning_sent_for is None
    assert ok.expiry_warning_sent_for == 3
    assert stats["warned"] == 1


async def test_the_mail_describes_read_only_not_a_lockout(monkeypatch):
    """
    The copy has to match what `require_active_account` actually does. Promising
    a lock-out would frighten people about data they are not losing.
    """
    from app.services.email_service import _expiry_warning_text

    body = _expiry_warning_text(3, "31/12/2026", "https://example.com/pricing")
    assert "lecture seule" in body
    assert "ne sont ni supprimées ni modifiées" in body
