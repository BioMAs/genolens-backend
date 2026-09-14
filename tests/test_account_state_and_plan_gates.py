"""
Behavioural tests for the two controls that had columns but no teeth.

`tests/test_account_state_coverage.py` proves the guard is *attached* to every
user-facing router. This file proves it *fires*, and that the two plan gates
resurrected alongside it (advanced export, multi-comparison) refuse STARTER and
admit TEAM.

Why these assertions are written against the 403 body rather than the status
code: three different guards can answer 403 on the same route
(`require_active_license`, `require_active_account`, the plan gate), so a bare
`status_code == 403` would pass for the wrong reason — the failure mode the
`assert_forbidden_by` / `assert_not_blocked_by` helpers exist to rule out.

The `account_inactive` payload shape is a contract with the frontend, not an
internal detail: `frontend/src/utils/api.ts` keys on `detail.error` and stores
`detail.status` in a cookie that `middleware.ts` and `/suspended` both read.
`test_suspended_payload_matches_frontend_contract` is what makes reshaping it a
test failure instead of a silent redirect loop.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
FUTURE = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()

DATASET_ID = uuid4()
ANALYSIS_ID = uuid4()

# A write, a plain read, and a POST that computes a view rather than creating
# anything — the three shapes the expiry rule has to tell apart.
WRITE_ROUTE = ("POST", "/api/v1/projects/", {"name": "x", "description": "y"})
READ_ROUTE = ("GET", "/api/v1/projects", None)
READONLY_POST_ROUTE = ("POST", f"/api/v1/datasets/{DATASET_ID}/venn-analysis", {})


def make_user(
    *,
    plan: SubscriptionPlan = SubscriptionPlan.STARTER,
    role: UserRole = UserRole.USER,
    status: UserStatus = UserStatus.ACTIVE,
    ends_at: str | None = None,
) -> User:
    u = User()
    u.id = uuid4()
    u.email = "gate@example.com"
    u.role = role
    u.subscription_plan = plan
    u.status = status
    u.subscription_ends_at = ends_at
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = 0
    u.cosmetics_module_enabled = False
    u.report_customization_module_enabled = False
    u.scientific_module_enabled = False
    u.drug_discovery_module_enabled = False
    return u


def make_client(*, as_user: User):
    """Client resolving to `as_user`, with Supabase, the DB and the app-wide
    license stubbed out. The license override matters: several routes under test
    carry `require_active_license`, which answers 403 of its own in a dev
    environment with no key configured, and would mask the guard being tested."""
    from app.main import app
    from app.api.deps.license import require_active_license
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
    app.dependency_overrides[require_active_license] = lambda: None
    return AsyncClient(
        # raise_app_exceptions=False: these routes run against a mocked DB and
        # blow up downstream of the guard. We care only about which guard did or
        # did not answer, so a 500 must come back as a response, not as a raise.
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


def assert_forbidden_by(resp, error: str):
    """403 raised by the specific guard named, not by a neighbouring one."""
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"
    detail = resp.json()["detail"]
    assert isinstance(detail, dict), f"expected structured detail, got {detail!r}"
    assert detail["error"] == error, resp.text


def assert_not_blocked_by(resp, error: str):
    """
    The named guard let this through. The route may still fail downstream on a
    mocked DB — that is fine and expected; what must not appear is this guard's
    own refusal.
    """
    if resp.status_code != 403:
        return
    detail = resp.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("error") != error, f"blocked by {error}: {resp.text}"


async def _call(client, method, url, body):
    return await (client.get(url) if method == "GET" else client.post(url, json=body))


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app

    app.dependency_overrides.clear()


# ── Suspended / cancelled: hard block ────────────────────────────────────────


@pytest.mark.parametrize("bad_status", [UserStatus.SUSPENDED, UserStatus.CANCELLED])
@pytest.mark.parametrize("method,url,body", [WRITE_ROUTE, READ_ROUTE])
async def test_inactive_account_is_blocked_on_reads_and_writes(bad_status, method, url, body):
    """Suspension is a lock-out, not a downgrade — reads are refused too."""
    async with make_client(as_user=make_user(status=bad_status)) as client:
        assert_forbidden_by(await _call(client, method, url, body), "account_inactive")


async def test_suspended_payload_matches_frontend_contract():
    """
    The exact shape `frontend/src/utils/api.ts` has always keyed on and never
    once received. `detail.status` becomes the `account_status` cookie that
    `middleware.ts` and `/suspended` read to choose their copy, so it must carry
    the raw enum value.
    """
    async with make_client(as_user=make_user(status=UserStatus.CANCELLED)) as client:
        resp = await client.get("/api/v1/projects")
    detail = resp.json()["detail"]
    assert detail["error"] == "account_inactive"
    assert detail["status"] == "cancelled"
    assert "message" in detail


async def test_pending_is_not_treated_as_inactive():
    """
    PENDING is cleared by `get_or_create_user` on the invitee's first
    authenticated request. Were it blocked here instead, every invited user would
    be locked out on arrival — the regression this whole change had to avoid.
    """
    async with make_client(as_user=make_user(status=UserStatus.PENDING)) as client:
        resp = await client.get("/api/v1/projects")
    assert_not_blocked_by(resp, "account_inactive")


# ── Expired subscription: read-only ──────────────────────────────────────────


async def test_expired_account_cannot_write():
    async with make_client(as_user=make_user(ends_at=PAST)) as client:
        resp = await client.post("/api/v1/projects/", json={"name": "x", "description": "y"})
    assert_forbidden_by(resp, "subscription_expired")


async def test_expired_account_can_still_read():
    """Read-only, not locked out: the data they paid for stays reachable."""
    async with make_client(as_user=make_user(ends_at=PAST)) as client:
        resp = await client.get("/api/v1/projects")
    assert_not_blocked_by(resp, "subscription_expired")


async def test_expired_account_can_still_use_readonly_posts():
    """
    Venn is a POST that computes a view and creates nothing. Letting the HTTP
    method decide alone would hide an existing result from someone entitled to
    read it, which is why the allow-list is explicit.
    """
    method, url, body = READONLY_POST_ROUTE
    async with make_client(as_user=make_user(plan=SubscriptionPlan.TEAM, ends_at=PAST)) as client:
        resp = await _call(client, method, url, body)
    assert_not_blocked_by(resp, "subscription_expired")


async def test_future_expiry_blocks_nothing():
    async with make_client(as_user=make_user(ends_at=FUTURE)) as client:
        resp = await client.post("/api/v1/projects/", json={"name": "x", "description": "y"})
    assert_not_blocked_by(resp, "subscription_expired")


async def test_no_expiry_blocks_nothing():
    async with make_client(as_user=make_user(ends_at=None)) as client:
        resp = await client.post("/api/v1/projects/", json={"name": "x", "description": "y"})
    assert_not_blocked_by(resp, "subscription_expired")


@pytest.mark.parametrize("bad_status", [UserStatus.SUSPENDED, UserStatus.CANCELLED])
async def test_admins_bypass_both_controls(bad_status):
    """Privileged roles answer to neither control, as with every other guard."""
    admin = make_user(role=UserRole.ADMIN, status=bad_status, ends_at=PAST)
    async with make_client(as_user=admin) as client:
        resp = await client.post("/api/v1/projects/", json={"name": "x", "description": "y"})
    assert_not_blocked_by(resp, "account_inactive")
    assert_not_blocked_by(resp, "subscription_expired")


# ── Plan gates ───────────────────────────────────────────────────────────────

REPORT_TRIGGERS = [
    f"/api/v1/analyses/{ANALYSIS_ID}/report",
    f"/api/v1/datasets/{DATASET_ID}/report?comparison_name=A_vs_B",
]

MULTI_COMPARISON_ROUTES = [
    f"/api/v1/datasets/{DATASET_ID}/venn-analysis",
    f"/api/v1/datasets/{DATASET_ID}/intersection-enrichment",
]


def assert_plan_refusal(resp, expected_fragment: str):
    """The plan gates raise a plain-string detail, unlike the account guard."""
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str), f"expected string detail, got {detail!r}"
    assert expected_fragment in detail, resp.text


@pytest.mark.parametrize("url", REPORT_TRIGGERS)
async def test_starter_cannot_trigger_pdf_reports(url):
    """`advanced_export` reads false for STARTER in the grid; until now nothing
    enforced it and STARTER accounts generated PDF reports freely."""
    async with make_client(as_user=make_user(plan=SubscriptionPlan.STARTER)) as client:
        assert_plan_refusal(await client.post(url, json={}), "PDF report export")


@pytest.mark.parametrize("url", REPORT_TRIGGERS)
async def test_team_can_trigger_pdf_reports(url):
    async with make_client(as_user=make_user(plan=SubscriptionPlan.TEAM)) as client:
        resp = await client.post(url, json={})
    assert_not_blocked_by(resp, "advanced_export")
    if resp.status_code == 403:
        assert "PDF report export" not in resp.text


@pytest.mark.parametrize("url", MULTI_COMPARISON_ROUTES)
async def test_starter_cannot_use_multi_comparison(url):
    async with make_client(as_user=make_user(plan=SubscriptionPlan.STARTER)) as client:
        assert_plan_refusal(await client.post(url, json={}), "TEAM or ON_PREMISE plan")


@pytest.mark.parametrize("url", MULTI_COMPARISON_ROUTES)
async def test_team_passes_multi_comparison_gate(url):
    async with make_client(as_user=make_user(plan=SubscriptionPlan.TEAM)) as client:
        resp = await client.post(url, json={})
    if resp.status_code == 403:
        assert "TEAM or ON_PREMISE plan" not in resp.text


async def test_csv_export_stays_open_to_starter():
    """
    CSV/TSV is included in every plan. Only PDF is `advanced_export` — and no
    Excel export exists in the product, so this is the whole of the open surface.
    """
    async with make_client(as_user=make_user(plan=SubscriptionPlan.STARTER)) as client:
        resp = await client.get(f"/api/v1/datasets/{DATASET_ID}/deg-stats/export")
    if resp.status_code == 403:
        assert "PDF report export" not in resp.text


async def test_comparison_listing_stays_open_to_starter():
    """
    `/comparisons` is a primary sidebar destination, not a multi-contrast
    analysis. `multi_comparison` gates comparing contrasts against each other
    (Venn, intersection enrichment) — gating the listing would strip core
    navigation from every STARTER account.
    """
    async with make_client(as_user=make_user(plan=SubscriptionPlan.STARTER)) as client:
        resp = await client.get("/api/v1/comparisons")
    if resp.status_code == 403:
        assert "TEAM or ON_PREMISE plan" not in resp.text


# ── Admin expiry editing ─────────────────────────────────────────────────────


def test_subscription_update_distinguishes_absent_from_explicit_null():
    """
    The three-way distinction the endpoint relies on. `absent` and `null` both
    leave `subscription_ends_at` equal to None on the model, so only
    `model_fields_set` can tell them apart — and collapsing them would make a
    mistakenly set expiry impossible to lift: either every plan change silently
    clears it, or nothing ever can.
    """
    from app.api.endpoints.admin import SubscriptionUpdate

    absent = SubscriptionUpdate.model_validate({"plan": "TEAM"})
    assert "subscription_ends_at" not in absent.model_fields_set

    explicit_null = SubscriptionUpdate.model_validate(
        {"plan": "TEAM", "subscription_ends_at": None}
    )
    assert "subscription_ends_at" in explicit_null.model_fields_set
    assert explicit_null.subscription_ends_at is None

    with_value = SubscriptionUpdate.model_validate(
        {"plan": "TEAM", "subscription_ends_at": "2026-12-31T23:59:59+00:00"}
    )
    assert "subscription_ends_at" in with_value.model_fields_set
    assert with_value.subscription_ends_at is not None


def test_admin_profile_exposes_the_expiry():
    """
    It was settable at invitation and invisible ever after, so an admin could
    neither review nor correct a limit they had set.
    """
    from app.api.endpoints.admin import UserProfile

    assert "subscription_ends_at" in UserProfile.model_fields
    assert "subscription_starts_at" in UserProfile.model_fields


def test_users_me_schema_carries_account_state():
    """The client needs all three to explain a refusal rather than just fail."""
    from app.schemas.user import UserSelf

    for field in ("status", "subscription_ends_at", "subscription_expired"):
        assert field in UserSelf.model_fields, field
