"""
Regression tests for PATCH /admin/users/{id}/role.

The endpoint carried two defects that both made it lie about what it had done.

**The 500.** It returned the raw PostgREST `profiles` row against
`response_model=UserProfile`. That row has no `subscription_plan`, which
`UserProfile` requires, so response validation failed and the caller got a 500 —
*after* the write had already landed in Supabase. The role really did change; the
admin panel just reported failure. That is the nastiest shape a bug can take: it
invites a retry, or the conclusion that the feature is broken, when in fact it
worked. `test_raw_profiles_row_cannot_satisfy_userprofile` pins the root cause
directly, so the reason the endpoint must not return that row survives even if
the endpoint is rewritten again.

**The half-write.** The role went to Supabase only. The local `users.role` syncs
lazily in `get_or_create_user`, which deliberately refuses to downgrade ADMIN and
SCILICIUM_ADMIN from a Supabase claim. That protection is sound — a stale or
forged claim must not strip an admin — but it also meant a *deliberate* demotion
made through this endpoint could never reach the local row. The demoted user lost
the admin panel (whose guard reads Supabase) while keeping every privilege the
local role grants: unlimited analysis quota, AI access, and a free pass through
`require_active_account`. That is a privilege that outlives its revocation.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.api.endpoints.admin import UserProfile
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

TARGET_ID = uuid4()

# Exactly what PostgREST returns for the `profiles` select this endpoint uses:
# "id,full_name,avatar_url,role,created_at,updated_at". Note the absence of
# subscription_plan — that absence IS the bug.
PROFILES_ROW = {
    "id": str(TARGET_ID),
    "full_name": "Lea Martin",
    "avatar_url": None,
    "role": "user",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-09-15T00:00:00Z",
}

AUTH_USER = {
    "id": str(TARGET_ID),
    "email": "lea@example.com",
    "last_sign_in_at": "2026-09-14T10:00:00Z",
    "confirmed_at": "2026-01-01T00:00:00Z",
}


# ── Root cause ───────────────────────────────────────────────────────────────


def test_raw_profiles_row_cannot_satisfy_userprofile():
    """
    The direct cause of the 500, pinned on its own so it cannot silently return.

    If someone later re-introduces `return data[0]`, the endpoint test below goes
    red — but this test explains *why* in one line, without any mocking.
    """
    with pytest.raises(ValidationError) as exc:
        UserProfile.model_validate(PROFILES_ROW)
    assert "subscription_plan" in str(exc.value)


# ── Endpoint ─────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


class _FakeSupabase:
    """Stands in for httpx.AsyncClient inside app.api.endpoints.admin.

    The module builds its client inline (`async with httpx.AsyncClient() as ...`),
    so there is no transport to inject — the class itself is replaced.
    """

    def __init__(self, *args, **kwargs):
        self.patched: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def patch(self, url, **kwargs):
        self.patched.append(kwargs.get("json", {}))
        merged = {**PROFILES_ROW, **kwargs.get("json", {})}
        return _Resp([merged])

    async def get(self, url, **kwargs):
        if "/auth/v1/admin/users/" in url:
            return _Resp(AUTH_USER)
        return _Resp([PROFILES_ROW])


def make_local_user(role=UserRole.ADMIN) -> User:
    u = User()
    u.id = TARGET_ID
    u.email = "lea@example.com"
    u.role = role
    u.subscription_plan = SubscriptionPlan.TEAM
    u.status = UserStatus.ACTIVE
    u.subscription_ends_at = None
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = 0
    u.cosmetics_module_enabled = False
    u.report_customization_module_enabled = False
    u.scientific_module_enabled = False
    u.drug_discovery_module_enabled = False
    return u


@pytest_asyncio.fixture
async def client_and_user(monkeypatch):
    """Admin caller, stubbed Supabase, and a local row the endpoint can mirror onto."""
    from app.api.endpoints import admin as admin_module
    from app.api.deps.supabase_deps import get_current_user, get_db, require_admin
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    monkeypatch.setattr(admin_module.httpx, "AsyncClient", _FakeSupabase)

    local_user = make_local_user()

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=local_user)
    db.execute = AsyncMock(return_value=result)

    async def _fake_db():
        yield db

    caller = SupabaseUser(user_id=uuid4(), email="admin@scilicium.com")
    app.dependency_overrides[require_admin] = lambda: caller
    app.dependency_overrides[get_current_user] = lambda: caller
    app.dependency_overrides[get_db] = _fake_db

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as c:
        yield c, local_user, db

    app.dependency_overrides.clear()


async def test_role_change_returns_a_valid_profile_not_a_500(client_and_user):
    """The regression itself: a successful write must serialise, not 500."""
    client, _, _ = client_and_user
    resp = await client.patch(f"/api/v1/admin/users/{TARGET_ID}/role", json={"role": "user"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Must be a full profile, which is exactly what the raw PostgREST row is not.
    assert body["subscription_plan"] == "TEAM"
    assert "status" in body
    UserProfile.model_validate(body)


async def test_role_change_reaches_the_local_row(client_and_user):
    """
    A demotion must land where the quota and plan guards actually look.

    Without this, an ex-admin kept unlimited quota, AI access and a bypass of
    `require_active_account` indefinitely, because `get_or_create_user` refuses to
    downgrade a protected role from a Supabase claim.
    """
    client, local_user, db = client_and_user
    assert local_user.role is UserRole.ADMIN

    resp = await client.patch(f"/api/v1/admin/users/{TARGET_ID}/role", json={"role": "user"})

    assert resp.status_code == 200, resp.text
    assert local_user.role is UserRole.USER
    db.commit.assert_awaited()


async def test_invalid_role_is_rejected_before_any_write(client_and_user):
    """400, and nothing written anywhere — validation precedes both writes."""
    client, local_user, db = client_and_user
    resp = await client.patch(f"/api/v1/admin/users/{TARGET_ID}/role", json={"role": "wizard"})

    assert resp.status_code == 400, resp.text
    assert "Invalid role" in resp.json()["detail"]
    assert local_user.role is UserRole.ADMIN
    db.commit.assert_not_awaited()
