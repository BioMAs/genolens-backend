"""
Comptes démo : plafond de requêtes IA, quota d'analyses par compte, filigrane.

Ce que ces tests verrouillent, et pourquoi :

- La réservation d'une requête IA est atomique et précède l'appel au LLM ; un
  refus porte `detail.error == "ai_quota_exhausted"`, la clé sur laquelle le
  frontend branche (même contrat que `account_inactive`).
- `/ask` enveloppe tout son corps dans un `except Exception` qui renvoie 500 :
  un 429 levé dedans arriverait au client comme une panne serveur. Le test
  HTTP sur `/ask` existe pour ça.
- Un résultat servi depuis le cache ne coûte rien, et un échec du LLM rend la
  requête réservée.
- Le quota d'un compte démo couvre toute la démo : ni le chemin paresseux ni la
  tâche mensuelle ne le remettent à zéro.
- Le filigrane DEMO n'apparaît dans le LaTeX que pour un demandeur démo.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.api.deps import subscription as sub
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()


def make_user(
    *,
    plan: SubscriptionPlan = SubscriptionPlan.TEAM,
    role: UserRole = UserRole.USER,
    is_demo: bool = False,
    ai_limit: int | None = None,
    ai_used: int = 0,
    quota_override: int | None = None,
    analyses_used: int = 0,
    ends_at: str | None = None,
) -> User:
    u = User()
    u.id = uuid4()
    u.email = "demo@example.com"
    u.role = role
    u.subscription_plan = plan
    u.status = UserStatus.ACTIVE
    u.subscription_ends_at = ends_at
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = analyses_used
    u.quota_reset_at = None
    u.cosmetics_module_enabled = False
    u.report_customization_module_enabled = False
    u.scientific_module_enabled = False
    u.drug_discovery_module_enabled = False
    u.is_demo = is_demo
    u.ai_requests_limit = ai_limit
    u.ai_requests_used = ai_used
    u.analyses_quota_override = quota_override
    return u


def db_returning(new_count: int | None) -> AsyncMock:
    """AsyncSession dont l'UPDATE … RETURNING rend `new_count` (None = bloqué)."""
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=new_count)
    db.execute = AsyncMock(return_value=result)
    return db


class FakeSessionFactory:
    """Remplace AsyncSessionLocal : enregistre les requêtes de la session à part."""

    def __init__(self, fail: bool = False):
        self.session = AsyncMock()
        self.fail = fail

    def __call__(self):
        factory = self

        class _Ctx:
            async def __aenter__(self_inner):
                if factory.fail:
                    raise RuntimeError("database unreachable")
                return factory.session

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


# ── Modèle ──────────────────────────────────────────────────────────────────


def test_quota_override_takes_precedence_over_plan():
    assert make_user(quota_override=3).analyses_quota == 3
    assert make_user(quota_override=3, analyses_used=1).analyses_remaining == 2


def test_quota_override_of_zero_is_a_real_cap():
    # `is not None`, pas un test de vérité : 0 ne doit pas retomber sur les 150 du plan.
    assert make_user(quota_override=0).analyses_quota == 0


def test_quota_override_does_not_cap_privileged_roles():
    assert make_user(role=UserRole.ADMIN, quota_override=3).analyses_quota is None


def test_ai_requests_remaining():
    assert make_user().ai_requests_remaining is None
    assert make_user(ai_limit=5, ai_used=2).ai_requests_remaining == 3
    assert make_user(ai_limit=5, ai_used=9).ai_requests_remaining == 0
    assert make_user(role=UserRole.SCILICIUM_ADMIN, ai_limit=5).ai_requests_remaining is None


# ── Réservation / remboursement d'une requête IA ────────────────────────────


async def test_reserve_without_cap_writes_nothing():
    db = db_returning(None)
    reservation = await sub.reserve_ai_request(make_user(), db)
    assert reservation.reserved is False
    db.execute.assert_not_awaited()


async def test_reserve_skips_privileged_roles_even_with_a_cap():
    db = db_returning(None)
    user = make_user(role=UserRole.ADMIN, ai_limit=1, ai_used=1)
    reservation = await sub.reserve_ai_request(user, db)
    assert reservation.reserved is False
    db.execute.assert_not_awaited()


async def test_reserve_under_cap_consumes_and_commits():
    db = db_returning(3)
    user = make_user(ai_limit=5, ai_used=2)
    reservation = await sub.reserve_ai_request(user, db)
    assert reservation.reserved is True
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()
    assert user.ai_requests_used == 3


async def test_reserve_at_cap_refuses_with_structured_429():
    db = db_returning(None)
    user = make_user(ai_limit=5, ai_used=5)
    with pytest.raises(HTTPException) as exc:
        await sub.reserve_ai_request(user, db)
    assert exc.value.status_code == 429
    detail = exc.value.detail
    assert detail["error"] == "ai_quota_exhausted"
    assert detail["limit"] == 5 and detail["used"] == 5
    assert isinstance(detail["message"], str)
    db.commit.assert_not_awaited()


async def test_refund_runs_once_on_a_separate_session():
    factory = FakeSessionFactory()
    user = make_user(ai_limit=5, ai_used=2)
    with patch.object(sub, "AsyncSessionLocal", factory):
        reservation = await sub.reserve_ai_request(user, db_returning(3))
        await reservation.refund()
        await reservation.refund()  # idempotent
    factory.session.execute.assert_awaited_once()
    factory.session.commit.assert_awaited_once()


async def test_refund_of_nothing_reserved_touches_no_database():
    factory = FakeSessionFactory()
    with patch.object(sub, "AsyncSessionLocal", factory):
        reservation = await sub.reserve_ai_request(make_user(), db_returning(None))
        await reservation.refund()
    factory.session.execute.assert_not_awaited()


async def test_refund_failure_never_masks_the_original_error():
    with patch.object(sub, "AsyncSessionLocal", FakeSessionFactory(fail=True)):
        await sub.refund_ai_request(uuid4())  # ne lève pas


# ── Quota d'analyses : pas de remise à zéro mensuelle pour une démo ─────────


def test_lazy_monthly_reset_spares_demo_accounts():
    last_month = datetime.now(timezone.utc) - timedelta(days=40)
    demo = make_user(is_demo=True, quota_override=3, analyses_used=3)
    demo.quota_reset_at = last_month
    assert sub._reset_quota_if_new_month(demo) is False
    assert demo.analyses_used_this_month == 3

    regular = make_user(analyses_used=3)
    regular.quota_reset_at = last_month
    assert sub._reset_quota_if_new_month(regular) is True
    assert regular.analyses_used_this_month == 0


async def test_monthly_reset_task_excludes_demo_accounts():
    from app.worker.tasks import quota_tasks

    factory = FakeSessionFactory()
    factory.session.execute = AsyncMock(return_value=MagicMock(rowcount=4))
    with patch("app.db.session.AsyncSessionLocal", factory):
        await quota_tasks._async_reset()
    stmt = factory.session.execute.await_args.args[0]
    assert "is_demo" in str(stmt)


def test_quota_refusal_does_not_promise_a_renewal_to_a_demo():
    assert "1st" not in sub.analysis_quota_renewal_hint(make_user(is_demo=True))
    assert "1st" in sub.analysis_quota_renewal_hint(make_user())


async def test_check_analysis_quota_message_for_an_exhausted_demo():
    user = make_user(is_demo=True, quota_override=3, analyses_used=3)
    with pytest.raises(HTTPException) as exc:
        await sub.check_analysis_quota(user, AsyncMock())
    assert exc.value.status_code == 429
    assert "(3/3)" in exc.value.detail and "demo" in exc.value.detail


# ── HTTP : endpoints IA ─────────────────────────────────────────────────────


def make_client(as_user: User, db: AsyncMock):
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    async def _fake_db():
        yield db

    app.dependency_overrides[get_or_create_user] = lambda: as_user
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=as_user.id, email=as_user.email
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


def dataset_db(*, update_returns: int | None) -> AsyncMock:
    """Session dont la recherche du dataset réussit et dont l'UPDATE de
    réservation rend `update_returns` (None = plafond atteint)."""
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(
        return_value=SimpleNamespace(id=uuid4(), project_id=uuid4())
    )
    result.scalar = MagicMock(return_value=update_returns)
    db.execute = AsyncMock(return_value=result)
    return db


async def test_ask_at_cap_is_429_not_500():
    user = make_user(ai_limit=2, ai_used=2, ends_at=FUTURE)
    db = dataset_db(update_returns=None)
    with patch("app.api.endpoints.datasets.assert_project_read_access", AsyncMock()):
        async with make_client(user, db) as client:
            resp = await client.post(
                f"/api/v1/datasets/{uuid4()}/comparisons/A_vs_B/ask",
                json={"question": "Which pathways?"},
            )
    assert resp.status_code == 429, resp.text
    assert resp.json()["detail"]["error"] == "ai_quota_exhausted"


async def test_ask_llm_failure_refunds_the_reservation():
    user = make_user(ai_limit=2, ai_used=0, ends_at=FUTURE)
    db = dataset_db(update_returns=1)
    interpreter = MagicMock()
    interpreter.generate_simple_answer = AsyncMock(side_effect=RuntimeError("LLM down"))
    factory = FakeSessionFactory()
    with patch("app.api.endpoints.datasets.assert_project_read_access", AsyncMock()), \
         patch("app.api.endpoints.datasets.LocalAIInterpreter", return_value=interpreter), \
         patch.object(sub, "AsyncSessionLocal", factory):
        async with make_client(user, db) as client:
            resp = await client.post(
                f"/api/v1/datasets/{uuid4()}/comparisons/A_vs_B/ask",
                json={"question": "Which pathways?"},
            )
    assert resp.status_code == 500
    factory.session.execute.assert_awaited_once()  # le remboursement
    factory.session.commit.assert_awaited_once()


async def test_cached_interpretation_costs_nothing_even_at_cap():
    user = make_user(ai_limit=2, ai_used=2, ends_at=FUTURE)
    db = dataset_db(update_returns=None)
    db.scalar = AsyncMock(return_value=SimpleNamespace(
        interpretation="cached text",
        created_at=datetime.now(timezone.utc),
        model="gemma",
        deg_up=1, deg_down=2, pathways_count=3, genes_count=4,
    ))
    with patch("app.api.endpoints.datasets.assert_project_read_access", AsyncMock()):
        async with make_client(user, db) as client:
            resp = await client.post(
                f"/api/v1/datasets/{uuid4()}/comparisons/A_vs_B/interpret"
            )
    assert resp.status_code == 200, resp.text
    assert resp.json()["cached"] is True


async def test_interpret_at_cap_keeps_the_existing_interpretation():
    """force_regenerate au plafond : refusé AVANT de supprimer l'existante."""
    user = make_user(ai_limit=2, ai_used=2, ends_at=FUTURE)
    db = dataset_db(update_returns=None)
    db.scalar = AsyncMock(return_value=SimpleNamespace(interpretation="old"))
    with patch("app.api.endpoints.datasets.assert_project_read_access", AsyncMock()):
        async with make_client(user, db) as client:
            resp = await client.post(
                f"/api/v1/datasets/{uuid4()}/comparisons/A_vs_B/interpret",
                params={"force_regenerate": "true"},
            )
    assert resp.status_code == 429
    db.delete.assert_not_awaited()


async def test_chat_message_at_cap_is_refused_before_the_stream_opens():
    user = make_user(ai_limit=1, ai_used=1, ends_at=FUTURE)
    db = dataset_db(update_returns=None)
    session = SimpleNamespace(
        title="t", dataset_id=uuid4(), project_id=uuid4(), comparison_name="A_vs_B"
    )
    with patch("app.api.endpoints.chat._load_owned_session", AsyncMock(return_value=session)):
        async with make_client(user, db) as client:
            resp = await client.post(
                f"/api/v1/chat/sessions/{uuid4()}/message", json={"message": "hello"}
            )
    assert resp.status_code == 429, resp.text
    assert resp.json()["detail"]["error"] == "ai_quota_exhausted"
    db.add.assert_not_called()  # le message utilisateur n'a pas été enregistré


# ── /users/me ───────────────────────────────────────────────────────────────


async def test_users_me_exposes_demo_fields():
    user = make_user(is_demo=True, ai_limit=10, ai_used=4, quota_override=3, analyses_used=1)
    db = AsyncMock()
    db.scalar = AsyncMock(return_value=0)
    async with make_client(user, db) as client:
        resp = await client.get("/api/v1/users/me")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_demo"] is True
    assert body["ai_requests_limit"] == 10
    assert body["ai_requests_used"] == 4
    assert body["ai_requests_remaining"] == 6
    assert body["analyses_quota"] == 3
    assert body["analyses_remaining"] == 2


# ── Admin ───────────────────────────────────────────────────────────────────


def make_admin_client(target: User):
    from app.api.deps import require_admin
    from app.api.deps.license import require_active_license
    from app.api.deps.supabase_deps import get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    db = AsyncMock()
    db.scalar = AsyncMock(return_value=target)

    async def _fake_db():
        yield db

    app.dependency_overrides[require_admin] = lambda: SupabaseUser(
        user_id=uuid4(), email="admin@example.com"
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None
    client = AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )
    return client, db


def _profile_stub(user: User):
    return {"id": str(user.id), "role": "user", "subscription_plan": "TEAM"}


async def test_admin_demo_patch_respects_absent_versus_null():
    target = make_user(ai_limit=10, quota_override=3, ends_at=FUTURE)
    client, db = make_admin_client(target)
    with patch("app.api.endpoints.admin.get_user_details",
               AsyncMock(return_value=_profile_stub(target))):
        async with client:
            resp = await client.patch(f"/api/v1/admin/users/{target.id}/demo",
                                      json={"is_demo": True})
            assert resp.status_code == 200, resp.text
            assert target.is_demo is True
            assert target.ai_requests_limit == 10      # absent → inchangé
            assert target.analyses_quota_override == 3

            resp = await client.patch(f"/api/v1/admin/users/{target.id}/demo",
                                      json={"ai_requests_limit": None})
            assert resp.status_code == 200, resp.text
            assert target.ai_requests_limit is None    # null → effacé
            assert target.analyses_quota_override == 3
    db.commit.assert_awaited()


async def test_admin_demo_patch_reset_usage_zeroes_both_counters():
    target = make_user(is_demo=True, ai_limit=10, ai_used=7, analyses_used=2, ends_at=FUTURE)
    client, _ = make_admin_client(target)
    with patch("app.api.endpoints.admin.get_user_details",
               AsyncMock(return_value=_profile_stub(target))):
        async with client:
            resp = await client.patch(f"/api/v1/admin/users/{target.id}/demo",
                                      json={"reset_usage": True})
    assert resp.status_code == 200, resp.text
    assert target.ai_requests_used == 0
    assert target.analyses_used_this_month == 0
    assert target.quota_reset_at is not None


async def test_admin_demo_requires_an_end_date():
    target = make_user(ends_at=None)
    client, db = make_admin_client(target)
    async with client:
        resp = await client.patch(f"/api/v1/admin/users/{target.id}/demo",
                                  json={"is_demo": True})
    assert resp.status_code == 422
    db.commit.assert_not_awaited()


async def test_admin_demo_rejects_a_negative_cap():
    target = make_user(ends_at=FUTURE)
    client, _ = make_admin_client(target)
    async with client:
        resp = await client.patch(f"/api/v1/admin/users/{target.id}/demo",
                                  json={"ai_requests_limit": -1})
    assert resp.status_code == 422


async def test_admin_invite_demo_without_end_date_sends_no_invitation():
    target = make_user()
    client, _ = make_admin_client(target)
    with patch("app.api.endpoints.admin.AccountService") as service:
        async with client:
            resp = await client.post("/api/v1/admin/users/invite", json={
                "email": "prospect@example.com", "plan": "TEAM",
                "is_demo": True, "ai_requests_limit": 20,
            })
    assert resp.status_code == 422
    service.assert_not_called()


# ── Filigrane du rapport PDF ────────────────────────────────────────────────


def _render_template(**overrides) -> str:
    from jinja2 import meta

    from app.services import report_latex_service as rl

    env = rl.report_latex_service.env
    source = env.loader.get_source(env, rl.MAIN_TEMPLATE)[0]
    values = {k: "" for k in meta.find_undeclared_variables(env.parse(source))}
    values.update(overrides)
    return env.get_template(rl.MAIN_TEMPLATE).render(**values)


def test_report_template_stamps_demo_watermark_only_when_asked():
    stamped = _render_template(jj_demo_watermark=True)
    assert "AddToShipoutPictureFG" in stamped and "DEMO" in stamped

    clean = _render_template(jj_demo_watermark=False)
    assert "AddToShipoutPictureFG" not in clean


async def test_is_demo_requester():
    from app.services.report_latex_service import report_latex_service as svc

    db = AsyncMock()
    db.get = AsyncMock(return_value=make_user(is_demo=True))
    assert await svc.is_demo_requester(db, uuid4()) is True

    db.get = AsyncMock(return_value=make_user(is_demo=False))
    assert await svc.is_demo_requester(db, uuid4()) is False

    db.get = AsyncMock(return_value=None)
    assert await svc.is_demo_requester(db, uuid4()) is False
    assert await svc.is_demo_requester(db, None) is False
