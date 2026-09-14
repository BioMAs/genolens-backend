"""
Coverage tripwire for `require_active_account`, plus the expiry-parsing rules.

`require_active_account` is attached per-router in `app/main.py`. That buys
blanket coverage — including the thirteen endpoint modules that resolve their
caller through `get_current_user` and never touch `get_or_create_user`, which a
guard living inside the latter would have missed entirely — but it also means a
new `include_router(...)` inherits nothing. This file walks the assembled route
table and fails when a user-facing route is unguarded, so the gap shows up as a
red test rather than as a suspended customer who still has access.

The exemption list is the specification, not a convenience: every entry is a
route that MUST stay reachable for a blocked account (renew, read your own
profile, read the pricing grid, check whether the service is up).
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.routing import APIRoute

from app.api.deps.account_state import (
    parse_subscription_end,
    require_active_account,
    subscription_has_expired,
)
from app.main import app
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

# Prefixes that must answer even when the account is suspended, cancelled or
# expired. Anything else under /api/v1 is expected to carry the guard.
EXEMPT_PREFIXES = (
    # Covers cosmetics.admin_router too — it mounts under /admin.
    "/api/v1/admin",  # admin-only; privileged roles bypass the guard anyway
    "/api/v1/users",  # /users/me feeds the banner explaining the refusals
    "/api/v1/billing",  # a cancelled account must reach checkout to renew
    "/api/v1/pricing",  # public by design, read before sign-in
    "/api/v1/license",  # infrastructure status
)


def _user_routes():
    """Every /api/v1 route that is not explicitly exempt."""
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith("/api/v1"):
            continue
        if route.path.startswith(EXEMPT_PREFIXES):
            continue
        yield route


def _has_guard(route: APIRoute) -> bool:
    """True when require_active_account is reachable from this route's tree."""
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is require_active_account:
            return True
        stack.extend(dep.dependencies)
    return False


def test_every_user_facing_route_is_guarded():
    """A new router added without the guard fails here, not in production."""
    unguarded = sorted(f"{sorted(r.methods)} {r.path}" for r in _user_routes() if not _has_guard(r))
    assert not unguarded, (
        "These routes are reachable by a suspended, cancelled or expired account. "
        "Add dependencies=[Depends(require_active_account)] to their include_router "
        "in app/main.py, or add an entry to EXEMPT_PREFIXES with a reason:\n  "
        + "\n  ".join(unguarded)
    )


def test_the_tripwire_can_actually_fail():
    """
    Guards the guard: if `_has_guard` silently returned True for everything, the
    test above would pass while enforcing nothing. An exempt route must read as
    unguarded, which proves the detector discriminates.
    """
    billing = [
        r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/v1/billing")
    ]
    assert billing, "expected billing routes to exist"
    assert not any(_has_guard(r) for r in billing)


def test_exempt_routes_still_exist():
    """A renamed prefix would silently empty the exemption list and hide gaps."""
    paths = [r.path for r in app.routes if isinstance(r, APIRoute)]
    for prefix in EXEMPT_PREFIXES:
        assert any(p.startswith(prefix) for p in paths), f"no route under {prefix}"


# ── Expiry parsing ────────────────────────────────────────────────────────────
#
# The column is a String(50) written by two callers that disagree on format: the
# admin invite path stores `datetime.isoformat()` (naive when handed a naive
# datetime), the Stripe webhook stores an aware UTC isoformat. String comparison
# would sort those on punctuation rather than time, so they are parsed.


def make_user(ends_at=None, status=UserStatus.ACTIVE, role=UserRole.USER) -> User:
    u = User()
    u.id = uuid4()
    u.email = "expiry@example.com"
    u.role = role
    u.subscription_plan = SubscriptionPlan.STARTER
    u.status = status
    u.subscription_ends_at = ends_at
    return u


def test_naive_iso_is_read_as_utc():
    parsed = parse_subscription_end("2026-01-02T00:00:00")
    assert parsed == datetime(2026, 1, 2, tzinfo=timezone.utc)


def test_aware_iso_is_normalised_to_utc():
    parsed = parse_subscription_end("2026-01-02T01:00:00+01:00")
    assert parsed == datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw", [None, "", "not-a-date", "31/12/2026"])
def test_unparsable_values_mean_no_expiry(raw):
    """
    Fail open, deliberately. Ignoring a malformed date costs a little extra
    access; guessing at one locks a paying customer out of their own data.
    """
    assert parse_subscription_end(raw) is None
    assert subscription_has_expired(make_user(raw)) is False


def test_past_and_future_dates():
    now = datetime.now(timezone.utc)
    past = (now - timedelta(days=1)).isoformat()
    future = (now + timedelta(days=1)).isoformat()
    assert subscription_has_expired(make_user(past)) is True
    assert subscription_has_expired(make_user(future)) is False


def test_expiry_boundary_is_inclusive():
    """The instant the period ends, it has ended."""
    now = datetime.now(timezone.utc)
    assert subscription_has_expired(make_user(now.isoformat()), now=now) is True
