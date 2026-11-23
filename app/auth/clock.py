"""
Instants of the local authentication, always as aware UTC datetimes.

Token and session expiry are decided by comparing instants, so every module of `app.auth` reads
"now" and stored datetimes through these two functions rather than each repeating the same
normalisation. Callers take an optional `now=` so that tests can fix the clock.
"""

from datetime import UTC, datetime
from typing import Optional


def aware_utc(value: datetime) -> datetime:
    """`value` as an aware UTC datetime: a naive one is taken to be UTC, an aware one is converted.

    SQLite hands back the naive UTC wall clock of what was stored, Postgres an aware value:
    normalise both so that they compare with `now`.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def now_utc(now: Optional[datetime] = None) -> datetime:
    """The instant to work at, as an aware UTC datetime: the current time when `now` is None,
    otherwise `now` normalised by `aware_utc` (a naive `now` is taken as UTC)."""
    if now is None:
        return datetime.now(UTC)
    return aware_utc(now)
