"""
Verrou sur la charge utile de /billing/subscription.

Le type frontend `SubscriptionInfo` déclarait dix champs que cette route ne
renvoie pas, dont `subscription_ends_at` : la carte de plan du dashboard
affichait un bloc « Renews » qui ne pouvait jamais s'afficher. Les colonnes
existent sur le modèle (String(50), ISO 8601) mais n'étaient dans aucun schéma
de réponse.

Ce test fige ce que la route sert, pour qu'un champ ne puisse plus être promis
côté client sans exister côté serveur.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

ENDPOINT = "/api/v1/billing/subscription"

CURRENT_KEYS = {
    "plan",
    "is_active",
    "stripe_customer_id",
    "subscription_starts_at",
    "subscription_ends_at",
    "analyses_used_this_month",
    "analyses_quota",
    "analyses_remaining",
    "can_use_ai",
    "can_use_multi_comparison",
}

#: Anciens noms, servis le temps que le frontend deploye bascule. Ce test est
#: ce qui forcera leur retrait a etre explicite : supprimer les champs sans
#: toucher a cet ensemble fait echouer la suite.
DEPRECATED_KEYS = {
    "comparisons_used_this_month",
    "comparisons_quota",
    "comparisons_remaining",
}

EXPECTED_KEYS = CURRENT_KEYS | DEPRECATED_KEYS


def make_user() -> User:
    u = User()
    u.id = uuid4()
    u.email = "dates@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.TEAM
    u.status = UserStatus.ACTIVE
    u.analyses_used_this_month = 12
    u.stripe_customer_id = "cus_123"
    u.subscription_starts_at = "2026-01-15T00:00:00+00:00"
    u.subscription_ends_at = "2027-01-15T00:00:00+00:00"
    return u


def make_client(as_user: User):
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    async def _fake_db():
        yield AsyncMock()

    app.dependency_overrides[get_or_create_user] = lambda: as_user
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=as_user.id, email=as_user.email
    )
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


async def test_payload_exposes_exactly_the_expected_keys():
    user = make_user()
    async with make_client(user) as client:
        res = await client.get(ENDPOINT)

    assert res.status_code == 200
    assert set(res.json()) == EXPECTED_KEYS


async def test_payload_carries_the_subscription_dates():
    user = make_user()
    async with make_client(user) as client:
        res = await client.get(ENDPOINT)

    body = res.json()
    assert body["subscription_starts_at"] == "2026-01-15T00:00:00+00:00"
    assert body["subscription_ends_at"] == "2027-01-15T00:00:00+00:00"


async def test_dates_are_null_when_absent():
    user = make_user()
    user.subscription_starts_at = None
    user.subscription_ends_at = None
    async with make_client(user) as client:
        res = await client.get(ENDPOINT)

    assert res.json()["subscription_starts_at"] is None
    assert res.json()["subscription_ends_at"] is None


async def test_deprecated_aliases_mirror_the_current_fields():
    """Les deux vocabulaires doivent dire la meme chose.

    Servir deux noms pour une meme valeur n'est tenable que s'ils ne peuvent
    pas divulguer : ce test echoue si un alias est cable sur autre chose que le
    champ qu'il remplace.
    """
    user = make_user()
    async with make_client(user) as client:
        res = await client.get(ENDPOINT)

    body = res.json()
    assert body["comparisons_used_this_month"] == body["analyses_used_this_month"]
    assert body["comparisons_quota"] == body["analyses_quota"]
    assert body["comparisons_remaining"] == body["analyses_remaining"]
