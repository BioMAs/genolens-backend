"""
Guards against privilege escalation through the plan/subscription surface.

Two regressions are locked here:

1. PATCH /users/me/subscription used to let ANY authenticated caller set their
   own subscription_plan to TEAM or ON_PREMISE. That single endpoint nullified
   every plan-keyed limit at once (comparison quota, project and dataset caps,
   AI access, advanced export), so no pricing grid could be enforced while it
   existed. A plan change must stay privileged: Stripe webhook, or
   PATCH /admin/users/{id}/subscription behind require_admin.

2. Three routes under /datasets/admin/ carried only Depends(get_current_user)
   despite being documented "Admin only", including POST /admin/cache-cleanup
   which mutates server state (it purges the persistent cache). Any logged-in
   user could call them.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

from httpx import AsyncClient, ASGITransport

from app.models.models import User, UserRole, SubscriptionPlan, UserStatus

ADMIN_ROUTES_READ = [
    "/api/v1/datasets/admin/performance-stats",
    "/api/v1/datasets/admin/cache-stats",
]
ADMIN_ROUTE_WRITE = "/api/v1/datasets/admin/cache-cleanup"


def make_user(role: UserRole, plan: SubscriptionPlan = SubscriptionPlan.STARTER) -> User:
    u = User()
    u.id = uuid4()
    u.email = "test@example.com"
    u.role = role
    u.subscription_plan = plan
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = 0
    u.status = UserStatus.ACTIVE
    u.cosmetics_module_enabled = False
    u.report_customization_module_enabled = False
    u.scientific_module_enabled = False
    u.drug_discovery_module_enabled = False
    return u


def make_client(*, as_user: User):
    """Client whose resolved User is `as_user`, bypassing Supabase and the DB.

    Three seams are needed: get_or_create_user (what require_admin reads), plus
    the route's own get_current_user and get_db. Without the latter two an
    admin still gets a 403 "Not authenticated" from the route signature — which
    would make the negative tests below pass for the wrong reason, so
    assert_forbidden_by_role() checks the failure actually comes from the guard.
    """
    from app.main import app
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser

    async def _fake_db():
        yield AsyncMock()

    app.dependency_overrides[get_or_create_user] = lambda: as_user
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=as_user.id, email=as_user.email
    )
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def assert_forbidden_by_role(resp):
    """403 must come from require_admin, not from missing authentication."""
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Admin access required", resp.text


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    app.dependency_overrides.clear()


# ── 1. Self-service plan escalation ──────────────────────────────────────────

def test_self_service_subscription_route_is_not_registered():
    """No route may accept a plan change scoped to the caller themselves."""
    from app.main import app

    offenders = [
        f"{sorted(r.methods)} {r.path}"
        for r in app.routes
        if getattr(r, "path", "").endswith("/me/subscription")
    ]
    assert offenders == [], f"self-service plan change reachable at: {offenders}"


@pytest.mark.asyncio
async def test_patch_own_subscription_is_gone():
    user = make_user(UserRole.USER, SubscriptionPlan.STARTER)
    async with make_client(as_user=user) as c:
        resp = await c.patch("/api/v1/users/me/subscription", json={"plan": "ON_PREMISE"})
    assert resp.status_code in (404, 405), resp.text
    assert user.subscription_plan is SubscriptionPlan.STARTER


@pytest.mark.asyncio
async def test_plan_stays_readable_on_me():
    """Removing the setter must not break the profile read."""
    user = make_user(UserRole.USER, SubscriptionPlan.STARTER)
    async with make_client(as_user=user) as c:
        resp = await c.get("/api/v1/users/me")
    assert resp.status_code == 200, resp.text
    assert resp.json()["subscription_plan"] == "STARTER"


# ── 2. /datasets/admin/* guards ───────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ADMIN_ROUTES_READ)
async def test_admin_read_routes_forbid_plain_user(path):
    async with make_client(as_user=make_user(UserRole.USER)) as c:
        resp = await c.get(path)
    assert_forbidden_by_role(resp)


@pytest.mark.asyncio
async def test_admin_cache_cleanup_forbids_plain_user():
    """The mutating one: a plain user must not be able to purge the cache."""
    async with make_client(as_user=make_user(UserRole.USER)) as c:
        resp = await c.post(ADMIN_ROUTE_WRITE)
    assert_forbidden_by_role(resp)


@pytest.mark.asyncio
async def test_admin_routes_forbid_on_premise_plan_too():
    """The guard is on the ROLE, not the plan — a top-tier plan is not an admin."""
    user = make_user(UserRole.USER, SubscriptionPlan.ON_PREMISE)
    async with make_client(as_user=user) as c:
        resp = await c.post(ADMIN_ROUTE_WRITE)
    assert_forbidden_by_role(resp)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.SCILICIUM_ADMIN])
async def test_admin_read_routes_allow_admins(role):
    async with make_client(as_user=make_user(role)) as c:
        resp = await c.get(ADMIN_ROUTES_READ[0])
    assert resp.status_code == 200, resp.text
