"""
The Stripe webhook must never process an unsigned payload in production.

The endpoint is unauthenticated by design: Stripe proves itself with a
signature header, not a token. But the fallback that parsed the payload when
STRIPE_WEBHOOK_SECRET was unset applied in *every* environment. The secret was
not configured in production, so an unsigned POST from anywhere on the internet
reached the handler — which writes subscription_plan and subscription_ends_at.
Anyone naming an existing stripe_customer_id could have granted themselves
TEAM or ON_PREMISE. That is worse than the self-service plan endpoint removed
in 3a5a119: this one needs no account at all.

Found by posting an unsigned event at production while verifying an unrelated
fix, and getting it processed.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.api.endpoints.billing import _UNVERIFIED_WEBHOOK_ENVIRONMENTS
from app.core.config import settings
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

EVENT = {
    "type": "customer.subscription.updated",
    "data": {"object": {
        "customer": "cus_forged",
        "current_period_end": 1790812800,
        "items": {"data": [{"price": {"id": "price_x"}}]},
    }},
}


def make_user() -> User:
    u = User()
    u.id = uuid4()
    u.email = "victim@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.STARTER
    u.status = UserStatus.ACTIVE
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.comparisons_used_this_month = 0
    u.stripe_customer_id = "cus_forged"
    u.subscription_ends_at = None
    return u


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    app.dependency_overrides.clear()


def make_client(user: User | None):
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
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver"), db


async def post_unsigned(user: User | None):
    client, db = make_client(user)
    async with client as c:
        resp = await c.post(
            "/api/v1/billing/webhook",
            content=json.dumps(EVENT),
            headers={"Content-Type": "application/json"},
        )
    return resp, db


# ── The allowlist itself ──────────────────────────────────────────────────────

def test_production_is_not_allowed_to_skip_verification():
    assert "production" not in _UNVERIFIED_WEBHOOK_ENVIRONMENTS


def test_staging_is_not_allowed_either():
    """Staging sees real Stripe traffic; it must verify like production."""
    assert "staging" not in _UNVERIFIED_WEBHOOK_ENVIRONMENTS


def test_dev_and_test_are_allowed():
    """Both are needed: local runs as `development`, CI as `test`."""
    assert {"development", "test"} <= set(_UNVERIFIED_WEBHOOK_ENVIRONMENTS)


# ── Fail closed where it matters ──────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["production", "staging", "PRODUCTION", "preprod"])
async def test_unsigned_payload_is_refused_without_a_secret(monkeypatch, environment):
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(settings, "STRIPE_WEBHOOK_SECRET", None)

    user = make_user()
    resp, db = await post_unsigned(user)

    # 503 so Stripe keeps retrying a possibly-genuine event once we are fixed
    assert resp.status_code == 503, resp.text
    # and above all: nothing was written
    assert user.subscription_ends_at is None
    assert user.subscription_plan is SubscriptionPlan.STARTER
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsigned_payload_is_processed_in_development(monkeypatch):
    """The local and CI workflow must keep working."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "STRIPE_WEBHOOK_SECRET", None)

    user = make_user()
    resp, db = await post_unsigned(user)

    assert resp.status_code == 200, resp.text
    assert user.subscription_ends_at is not None
    db.commit.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["Development", "TEST", "development"])
async def test_allowlist_is_case_insensitive(monkeypatch, environment):
    """What the `.lower()` on ENVIRONMENT actually buys.

    Dropping it would make the check *stricter*, not weaker — an oddly-cased
    "Development" would be refused, which fails closed and is safe. So the
    security tests above cannot catch its removal (verified by mutation). This
    one pins the intended convenience instead: casing in a deployment's env var
    must not silently break a developer's local webhook.
    """
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(settings, "STRIPE_WEBHOOK_SECRET", None)

    resp, db = await post_unsigned(make_user())

    assert resp.status_code == 200, resp.text
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_a_bad_signature_is_rejected_when_a_secret_is_set(monkeypatch):
    """With a secret configured, an unsigned payload fails verification."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "STRIPE_WEBHOOK_SECRET", "whsec_test_secret")

    user = make_user()
    resp, db = await post_unsigned(user)

    assert resp.status_code == 400, resp.text
    assert user.subscription_ends_at is None
    db.commit.assert_not_awaited()
