"""
Tests for the pricing configuration (app/config/pricing.json).

The point of this file is the **neutrality test**: introducing the pricing grid
must not change a single enforced limit. `test_active_grid_matches_user_model`
compares every value in the active grid against the `User` property that
actually enforces it, so the two cannot drift apart silently. When the target
grid is activated (annual quota of interpreted contrasts), that test is the one
that must be updated deliberately — it is the tripwire, not a formality.

It also validates the proposed grid with the same schema without serving it, so
the day it is activated we already know it parses.
"""
import json
import re
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from pydantic import ValidationError
from uuid import uuid4

from app.core.pricing import (
    ACTIVE_GRID,
    ADDON_OWNED_ENTITLEMENTS,
    DRAFT_GRID,
    Plan,
    PricingConfig,
    get_pricing,
    load_pricing,
    resolve_limit,
)
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def make_user(plan: SubscriptionPlan) -> User:
    u = User()
    u.id = uuid4()
    u.email = "test@example.com"
    u.role = UserRole.USER
    u.subscription_plan = plan
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = 0
    u.status = UserStatus.ACTIVE
    return u


# ── Schema plumbing ───────────────────────────────────────────────────────────

def test_resolve_limit_semantics():
    """None/unlimited/custom all mean unlimited; 0 means none included."""
    assert resolve_limit(None) is None
    assert resolve_limit("unlimited") is None
    assert resolve_limit("custom") is None
    assert resolve_limit(0) == 0
    assert resolve_limit(30) == 30


def test_active_grid_loads():
    cfg = get_pricing()
    assert cfg.version
    assert cfg.plans


def test_unknown_entitlement_key_is_rejected():
    """A typo must break CI rather than silently deny a capability."""
    with pytest.raises(ValidationError):
        Plan.model_validate({
            "id": "x", "name_fr": "x", "name_en": "x", "order": 1,
            "entitlements": {"gsae": True},
        })


def test_plan_may_not_claim_addon_owned_entitlements():
    """A plan flag and a *_module_enabled guard must not both own a capability."""
    with pytest.raises(ValidationError, match="add-on owned"):
        Plan.model_validate({
            "id": "lens_12", "name_fr": "x", "name_en": "x", "order": 1,
            "entitlements": {"gsea": True},
        })


def test_addon_module_flag_must_look_like_a_column():
    with pytest.raises(ValidationError):
        PricingConfig.model_validate({
            "version": "t", "status": "t",
            "plans": [{"id": "p", "name_fr": "p", "name_en": "p", "order": 1}],
            "addons": [{"id": "a", "name_fr": "a", "name_en": "a", "order": 1,
                        "module_flag": "scientific"}],
        })


def test_duplicate_plan_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate plan ids"):
        PricingConfig.model_validate({
            "version": "t", "status": "t",
            "plans": [
                {"id": "p", "name_fr": "p", "name_en": "p", "order": 1},
                {"id": "p", "name_fr": "p", "name_en": "p", "order": 2},
            ],
        })


def test_get_plan_accepts_enum_and_reports_known_ids():
    cfg = get_pricing()
    assert cfg.get_plan(SubscriptionPlan.STARTER).id == "STARTER"
    with pytest.raises(KeyError, match="lens_40"):
        cfg.get_plan("lens_40")


# ── Neutrality: the grid must describe what the code enforces ────────────────

def test_every_plan_enum_member_is_in_the_active_grid():
    cfg = get_pricing()
    for plan in SubscriptionPlan:
        cfg.get_plan(plan.value)


def test_active_grid_has_no_extra_plans():
    """A plan in the grid with no enum member could never be assigned."""
    cfg = get_pricing()
    known = {p.value for p in SubscriptionPlan}
    assert {p.id for p in cfg.plans} == known


@pytest.mark.parametrize("plan", list(SubscriptionPlan))
def test_active_grid_matches_user_model(plan):
    """THE tripwire: the grid must equal what `User` actually enforces.

    If this fails, either the grid drifted from the code or a limit changed
    without the grid being updated. Both are bugs; pick the intended value and
    make the two agree.
    """
    cfg = get_pricing()
    p = cfg.get_plan(plan.value)
    u = make_user(plan)

    assert p.resolved_contrast_quota == u.analyses_quota, "analyses_quota"
    assert p.resolved_max_projects == u.max_projects, "max_projects"
    assert p.resolved_datasets_limit_per_project == u.max_datasets_per_project, \
        "max_datasets_per_project"

    ai = p.entitlements.ai_interpretation
    assert ai in ("none", "quota"), f"unexpected active AI mode {ai!r}"
    assert (ai != "none") == u.can_use_ai, "can_use_ai"

    assert p.entitlements.multi_comparison == u.can_use_multi_comparison, \
        "can_use_multi_comparison"
    assert p.entitlements.advanced_export == u.can_export_advanced, \
        "can_export_advanced"


def test_active_quota_period_is_monthly_everywhere():
    """The counter resets on the 1st of each month; the grid must say so.

    Flipping this to "annual" is a deliberate act that belongs with the
    contrast-ledger work, not a silent config edit.
    """
    for p in get_pricing().plans:
        assert p.quota_period == "monthly", p.id


def test_active_overage_is_blocking():
    """check_analysis_quota raises 429 today. The grid must not promise
    otherwise while that is still true."""
    overage = get_pricing().overage
    assert overage.hard_block is True
    assert overage.policy == "block"


def test_active_addons_cover_every_user_module_column():
    """Every *_module_enabled column must be described and priced somewhere."""
    cfg = get_pricing()
    columns = {c.name for c in User.__table__.columns if c.name.endswith("_module_enabled")}
    assert {a.module_flag for a in cfg.addons} == columns


def test_addon_module_flags_are_real_columns():
    cfg = get_pricing()
    for addon in cfg.addons:
        assert hasattr(User, addon.module_flag), addon.module_flag


def test_addons_carry_the_entitlements_plans_may_not():
    """The three keys moved off the plans must be owned by an add-on."""
    cfg = get_pricing()
    owned = set()
    for addon in cfg.addons:
        owned |= set(addon.entitlements.stated())
    for key in ("gsea", "gsea_leading_edge", "custom_gene_sets"):
        assert key in owned, f"{key} is owned by no add-on"
    assert owned <= ADDON_OWNED_ENTITLEMENTS | {"support"}


def test_active_grid_carries_the_public_card_copy():
    """The pricing page renders from the grid, so the copy must live here.

    Locked as a count rather than as text: the point is that porting the page
    to the grid did not silently drop bullets. The wording itself is expected
    to change when the unenforced promises are corrected.

    Counts bumped on 2026-09-08 when the cards were aligned on
    genolens.com/pricing: Starter gained the "Priority support" exclusion, Pro
    gained "SSO / SAML" and "On-premise deployment", and Enterprise dropped
    "SSO / SAML" — which the public card does not advertise and nothing
    implements.
    """
    cfg = get_pricing()
    expected = {"STARTER": 12, "TEAM": 13, "ON_PREMISE": 5}
    for plan_id, count in expected.items():
        plan = cfg.get_plan(plan_id)
        assert len(plan.marketing_features) == count, plan_id
        assert plan.description_en
        assert plan.cta_label_en
        assert plan.engagement_en


def test_marketing_bullets_are_not_entitlements():
    """A bullet is a commercial promise; access comes from `entitlements`.

    Starter advertises "5 datasets" as an account total while the enforced cap
    is 5 per project, and Pro advertises "50 datasets" and "50 reports / month"
    that nothing enforces. This test exists so nobody wires access decisions to
    the marketing list by mistake.
    """
    cfg = get_pricing()
    starter = cfg.get_plan("STARTER")
    assert any(f.label == "5 datasets" for f in starter.marketing_features)
    # ...while the enforced limit is per project, not per account
    assert starter.resolved_datasets_limit_per_project == 5
    assert starter.resolved_datasets_limit is None

    team = cfg.get_plan("TEAM")
    assert any("50 datasets" in f.label for f in team.marketing_features)
    assert team.resolved_datasets_limit_per_project is None, (
        "Pro advertises 50 datasets but enforces no cap; if that changes, "
        "update the grid and the page together"
    )

    # Aligning the cards on the website widened this gap rather than closing
    # it: the site sells Team collaboration and Custom gene sets as excluded
    # from Pro, while neither is gated by the plan — the first is gated by
    # nothing at all, the second by the `scientific` add-on. Asserted so the
    # bullets cannot be mistaken for the guard.
    excluded = {f.label for f in team.marketing_features if not f.included}
    assert {"Team collaboration", "Custom gene sets"} <= excluded
    assert "custom_gene_sets" not in team.entitlements.stated()


# ── No price may live outside the grid ───────────────────────────────────────

def test_no_hard_coded_plan_prices_in_app_code():
    plan_ids = "|".join(p.value for p in SubscriptionPlan)
    pattern = re.compile(rf'"(?:{plan_ids})"\s*:\s*\d')
    offenders = [
        f"{path.relative_to(APP_DIR.parent)}:{i}"
        for path in APP_DIR.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), start=1)
        if pattern.search(line)
    ]
    assert offenders == [], f"plan-keyed numeric literals outside the grid: {offenders}"


def test_grid_files_are_tracked_by_git():
    """The grid must ship with the code.

    It first lived under app/data/, which .gitignore excludes wholesale for
    runtime data artifacts. It loaded fine locally and would have passed every
    other test here — then gone missing on deploy, taking every price and
    quota with it.

    Asserts the files are *tracked*, not merely "not ignored": `git
    check-ignore` skips anything already in the index, so an ignore-based check
    would have gone inert the moment these files were first committed and would
    never catch the next grid file added to an excluded directory.

    Repo hygiene, so it only runs in a checkout. CI executes pytest *inside the
    container*, where .dockerignore has stripped .git and git is not installed.
    That the grid is present and parses wherever the app runs — container
    included — is covered by the other tests in this file.
    """
    import shutil
    import subprocess

    repo = APP_DIR.parent
    if shutil.which("git") is None or not (repo / ".git").exists():
        pytest.skip("not a git checkout (running inside the container)")

    for path in (ACTIVE_GRID, DRAFT_GRID):
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=repo, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{path.relative_to(repo)} is not tracked by git; it would not "
            f"ship on deploy. Is it under an ignored directory? "
            f"({result.stderr.strip()})"
        )


# ── The proposed grid must stay loadable ─────────────────────────────────────

def test_draft_grid_parses_with_the_same_schema():
    cfg = load_pricing(DRAFT_GRID)
    assert cfg.version == "2026.09-draft"
    assert [p.id for p in cfg.plans_ordered] == [
        "access", "lens_12", "lens_40", "portfolio", "enterprise",
    ]


def test_draft_grid_is_the_target_model_not_the_current_one():
    """Guards the two things that make the draft a different business model."""
    cfg = load_pricing(DRAFT_GRID)
    assert cfg.billable_unit.id == "interpreted_contrast"
    assert all(p.quota_period == "annual" for p in cfg.plans)
    assert cfg.overage.hard_block is False
    assert cfg.overage.policy == "allow_and_bill"
    # Access sells AI per act rather than denying it — impossible with a bool.
    assert cfg.get_plan("access").entitlements.ai_interpretation == "metered_a_la_carte"
    assert cfg.get_plan("access").resolved_contrast_quota == 0
    # cross_project_comparison is the differentiator: lens_40 and up only.
    assert cfg.get_plan("lens_12").entitlements.cross_project_comparison is False
    assert cfg.get_plan("lens_40").entitlements.cross_project_comparison is True


def test_draft_grid_is_not_served():
    """The active endpoint must never hand out the unvalidated proposal."""
    assert get_pricing().version != load_pricing(DRAFT_GRID).version


def test_draft_grid_prices_all_shipped_addons():
    cfg = load_pricing(DRAFT_GRID)
    columns = {c.name for c in User.__table__.columns if c.name.endswith("_module_enabled")}
    assert {a.module_flag for a in cfg.addons} == columns


def test_both_grids_are_valid_json_and_utf8():
    for path in (ACTIVE_GRID, DRAFT_GRID):
        json.loads(path.read_text(encoding="utf-8"))


# ── The endpoint ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    app.dependency_overrides.clear()


def _client():
    from app.main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
async def test_pricing_endpoint_is_public():
    """No Authorization header: the grid is a public commercial promise."""
    async with _client() as c:
        resp = await c.get("/api/v1/pricing")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {p["id"] for p in body["plans"]} == {p.value for p in SubscriptionPlan}
    assert len(body["addons"]) == 4


@pytest.mark.asyncio
async def test_pricing_endpoint_sets_cache_headers():
    async with _client() as c:
        resp = await c.get("/api/v1/pricing")
    assert resp.headers["etag"].startswith('W/"')
    assert "max-age" in resp.headers["cache-control"]


@pytest.mark.asyncio
async def test_pricing_endpoint_answers_head():
    """Proxies and uptime checks issue HEAD against a public cacheable URL."""
    async with _client() as c:
        resp = await c.head("/api/v1/pricing")
    assert resp.status_code == 200, resp.text
    assert resp.headers["etag"].startswith('W/"')


@pytest.mark.asyncio
async def test_pricing_endpoint_honours_if_none_match():
    async with _client() as c:
        first = await c.get("/api/v1/pricing")
        again = await c.get(
            "/api/v1/pricing", headers={"If-None-Match": first.headers["etag"]}
        )
    assert again.status_code == 304
