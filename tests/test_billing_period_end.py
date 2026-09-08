"""
Tests for the Stripe renewal-date sync on the subscription webhook.

Nothing populated `User.subscription_ends_at` from Stripe: the only writer was
the admin path (app/services/account_service.py via admin.py). The profile page
renders a "Renewal Date" row from that column, so it was blank for every Stripe
customer unless an admin typed it in.

The abandoned `feature/stripe-integration` branch had this sync, but read only
`subscription["current_period_end"]`. Stripe moved that field onto the
subscription's *items* in API version 2025-03-31.basil, and this codebase pins
no API version — so on a modern account that branch's version would have
silently written nothing, reproducing the very bug it was meant to fix. Both
shapes are covered below.
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.api.endpoints.billing import _subscription_period_end
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

# 2026-10-01T00:00:00+00:00
PERIOD_END = 1790812800
PERIOD_END_ISO = datetime.fromtimestamp(PERIOD_END, tz=timezone.utc).isoformat()
LATER = PERIOD_END + 86_400


def make_user(customer_id: str = "cus_live") -> User:
    u = User()
    u.id = uuid4()
    u.email = "billing@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.STARTER
    u.status = UserStatus.ACTIVE
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = 0
    u.stripe_customer_id = customer_id
    u.subscription_ends_at = None
    return u


# ── The extraction helper ─────────────────────────────────────────────────────

def test_reads_top_level_current_period_end():
    """Legacy API versions put it on the Subscription object."""
    assert _subscription_period_end({"current_period_end": PERIOD_END}) == PERIOD_END


def test_falls_back_to_subscription_items():
    """2025-03-31.basil and later put it on each item instead.

    This is the case the old branch got wrong.
    """
    sub = {"items": {"data": [{"current_period_end": PERIOD_END}]}}
    assert _subscription_period_end(sub) == PERIOD_END


def test_takes_the_latest_period_across_items():
    """A plan plus an add-on can bill on different cycles; access ends last."""
    sub = {"items": {"data": [
        {"current_period_end": PERIOD_END},
        {"current_period_end": LATER},
    ]}}
    assert _subscription_period_end(sub) == LATER


def test_top_level_wins_when_both_present():
    sub = {"current_period_end": PERIOD_END, "items": {"data": [{"current_period_end": LATER}]}}
    assert _subscription_period_end(sub) == PERIOD_END


@pytest.mark.parametrize("sub", [
    {},
    {"current_period_end": None},
    {"items": {"data": []}},
    {"items": {"data": [{"price": {"id": "price_x"}}]}},
])
def test_returns_none_when_stripe_sends_no_period(sub):
    """Absent rather than guessed: the caller logs and leaves the column alone."""
    assert _subscription_period_end(sub) is None


# ── The webhook ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    app.dependency_overrides.clear()


def make_client(user: User | None):
    """Client whose DB returns `user` for every lookup.

    Returns the db mock alongside the client: asserting only on the mutated ORM
    object is not enough. A handler that sets the attribute and then returns
    early without committing would satisfy such an assertion while persisting
    nothing — mutation testing caught exactly that hole here, so the tests below
    assert `commit` was awaited.
    """
    from app.main import app
    from app.api.deps.supabase_deps import get_db

    db = AsyncMock()
    db.execute = AsyncMock(
        return_value=SimpleNamespace(scalar_one_or_none=lambda: user)
    )
    db.add = lambda _obj: None
    db.commit = AsyncMock()

    async def _fake_db():
        yield db

    app.dependency_overrides[get_db] = _fake_db
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
    return client, db


def subscription_event(sub: dict, event_type: str = "customer.subscription.updated") -> dict:
    return {"type": event_type, "data": {"object": sub}}


async def post_event(user: User | None, payload: dict):
    """POST the event; return (response, db_mock)."""
    client, db = make_client(user)
    async with client as c:
        resp = await c.post(
            "/api/v1/billing/webhook",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
    return resp, db


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [
    "customer.subscription.created",
    "customer.subscription.updated",
])
async def test_webhook_writes_renewal_date(event_type):
    user = make_user()
    resp, db = await post_event(user, subscription_event(
        {"customer": "cus_live", "current_period_end": PERIOD_END, "items": {"data": []}},
        event_type,
    ))
    assert resp.status_code == 200, resp.text
    assert user.subscription_ends_at == PERIOD_END_ISO
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_webhook_writes_renewal_date_from_items():
    """The modern Stripe shape must land too."""
    user = make_user()
    resp, db = await post_event(user, subscription_event({
        "customer": "cus_live",
        "items": {"data": [{"current_period_end": PERIOD_END, "price": {"id": "price_unknown"}}]},
    }))
    assert resp.status_code == 200, resp.text
    assert user.subscription_ends_at == PERIOD_END_ISO
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_renewal_date_syncs_even_when_the_price_is_unrecognised():
    """The regression this restructuring fixes.

    An unknown price_id used to `return` before anything was written, so the
    renewal date never synced. An add-on price, or one simply not configured in
    this environment, is enough to hit that path — and a plan we cannot map is
    no reason to lose the date.
    """
    user = make_user()
    resp, db = await post_event(user, subscription_event({
        "customer": "cus_live",
        "current_period_end": PERIOD_END,
        "items": {"data": [{"price": {"id": "price_not_in_config"}}]},
    }))
    assert resp.status_code == 200, resp.text
    assert user.subscription_ends_at == PERIOD_END_ISO
    db.commit.assert_awaited()
    # ...and the plan is left untouched rather than guessed
    assert user.subscription_plan is SubscriptionPlan.STARTER


@pytest.mark.asyncio
async def test_renewal_date_syncs_with_no_items_at_all():
    """An empty item list used to short-circuit the handler entirely."""
    user = make_user()
    resp, db = await post_event(user, subscription_event(
        {"customer": "cus_live", "current_period_end": PERIOD_END, "items": {"data": []}}
    ))
    assert resp.status_code == 200, resp.text
    assert user.subscription_ends_at == PERIOD_END_ISO
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_missing_period_leaves_the_column_alone():
    """Never overwrite a known date with a guess."""
    user = make_user()
    user.subscription_ends_at = "2026-01-01T00:00:00+00:00"
    resp, db = await post_event(user, subscription_event(
        {"customer": "cus_live", "items": {"data": []}}
    ))
    assert resp.status_code == 200, resp.text
    assert user.subscription_ends_at == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_unknown_customer_is_acknowledged_not_retried():
    """Stripe retries on a non-2xx; an unmatchable customer never will match."""
    resp, db = await post_event(None, subscription_event(
        {"customer": "cus_ghost", "current_period_end": PERIOD_END, "items": {"data": []}}
    ))
    assert resp.status_code == 200, resp.text
    assert resp.json().get("note") == "no matching user"
