"""Tests of the shared clock helpers (`app/auth/clock.py`)."""

from datetime import UTC, datetime, timedelta, timezone

from app.auth.clock import aware_utc, now_utc

PLUS_TWO = timezone(timedelta(hours=2))


def test_aware_utc_takes_a_naive_value_as_utc():
    naive = datetime(2026, 11, 24, 12, 0, 0)
    assert aware_utc(naive) == datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
    assert aware_utc(naive).tzinfo is UTC


def test_aware_utc_converts_an_aware_value():
    local = datetime(2026, 11, 24, 14, 0, 0, tzinfo=PLUS_TWO)
    converted = aware_utc(local)
    assert converted == local  # the same instant ...
    assert converted.utcoffset() == timedelta(0)  # ... expressed in UTC
    assert converted.hour == 12


def test_aware_utc_leaves_a_utc_value_alone():
    value = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
    assert aware_utc(value) == value


def test_now_utc_without_argument_is_the_current_aware_utc_time():
    before = datetime.now(UTC)
    now = now_utc()
    after = datetime.now(UTC)
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)
    assert before <= now <= after


def test_now_utc_normalises_the_instant_it_is_given():
    assert now_utc(datetime(2026, 11, 24, 12, 0, 0)) == datetime(2026, 11, 24, 12, tzinfo=UTC)
    shifted = now_utc(datetime(2026, 11, 24, 14, 0, 0, tzinfo=PLUS_TWO))
    assert shifted.hour == 12 and shifted.utcoffset() == timedelta(0)
