"""
Who is calling: one resolution of the caller for every protected route, behind `AUTH_MODE`.

`resolve_principal` reads the credential (an `Authorization: Bearer` header, else the `gl_at`
cookie), then resolves, in this order: a service token (`glst_`), a local ES256 access token (only
when local tokens are accepted: `AUTH_MODE=local` or `AUTH_LOCAL_ENABLED`), a hosted-provider
token (only in `AUTH_MODE=supabase`).

The default configuration (`AUTH_MODE=supabase`, local tokens off) must answer exactly as before
this module existed: same status codes, same details, same headers. Every API test here runs the
real chain against an in-memory SQLite database; nothing hosted is reached.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from starlette.requests import Request

from app.models.auth import AuthSession
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus
from tests.auth_support import CSRF, access_cookie, bearer, seed_user

API = "/api/v1"
ME = f"{API}/users/me"
PROJECTS = f"{API}/projects/"
NEW_PROJECT = {"name": "TST01 v1", "description": "test"}
BEARER_CHALLENGE = "Bearer"


def make_request(method: str = "GET", headers: dict | None = None, route=None) -> Request:
    """A bare request, for calling the resolution directly."""
    scope = {
        "type": "http",
        "method": method,
        "path": "/",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    if route is not None:
        scope["route"] = route
    return Request(scope)


async def users_count(auth_db) -> int:
    async with auth_db() as s:
        return await s.scalar(select(func.count()).select_from(User))


# ── Default configuration: unchanged ─────────────────────────────────────────


async def test_supabase_mode_unchanged(auth_client, auth_db, supabase_token_for):
    user = await seed_user(auth_db)

    resp = await auth_client.get(ME, headers=bearer(supabase_token_for(user)))

    assert resp.status_code == 200, resp.text
    assert resp.json()["email"] == user.email


async def test_supabase_mode_refusals_are_byte_for_byte_those_of_before(
    auth_client, auth_db, supabase_token_for, local_session_for
):
    user = await seed_user(auth_db)
    local = await local_session_for(user)

    # What `HTTPBearer(auto_error=True)` answered: 403, no challenge header.
    for headers in ({}, {"Authorization": "Bearer "}, access_cookie(local.access_token)):
        resp = await auth_client.get(ME, headers=headers)
        assert resp.status_code == 403, headers
        assert resp.json() == {"detail": "Not authenticated"}
        assert "www-authenticate" not in resp.headers
    resp = await auth_client.get(ME, headers={"Authorization": "Basic dXNlcjpwdw=="})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "Invalid authentication credentials"}

    # A token the hosted provider did not sign: 401 with the Bearer challenge.
    forged = supabase_token_for(user) + "x"
    resp = await auth_client.get(ME, headers=bearer(forged))
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid authentication credentials"}
    assert resp.headers["www-authenticate"] == BEARER_CHALLENGE

    # The scheme is case-insensitive, as it was.
    resp = await auth_client.get(
        ME, headers={"Authorization": f"bearer {supabase_token_for(user)}"}
    )
    assert resp.status_code == 200, resp.text


async def test_local_token_rejected_unless_enabled(
    auth_client, auth_db, local_session_for, settings_override
):
    user = await seed_user(auth_db)
    local = await local_session_for(user)

    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=False)
    resp = await auth_client.get(ME, headers=bearer(local.access_token))
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid authentication credentials"}

    # Migration window: both providers answer.
    settings_override(AUTH_LOCAL_ENABLED=True)
    resp = await auth_client.get(ME, headers=bearer(local.access_token))
    assert resp.status_code == 200, resp.text


async def test_local_token_never_reaches_the_keys_while_local_is_off(
    auth_client, auth_db, local_session_for, settings_override
):
    """Production without signing keys must not turn an ES256-headed token into a 500."""
    from app.auth.keys import reset_key_cache

    user = await seed_user(auth_db)
    local = await local_session_for(user)
    settings_override(ENVIRONMENT="production", AUTH_JWT_PRIVATE_KEYS="")
    reset_key_cache()

    resp = await auth_client.get(ME, headers=bearer(local.access_token))

    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid authentication credentials"}


# ── Local mode ────────────────────────────────────────────────────────────────


async def test_local_mode_cookie_and_header(
    auth_client, auth_db, local_session_for, settings_override
):
    settings_override(AUTH_MODE="local")
    user = await seed_user(
        auth_db, role=UserRole.ADMIN, subscription_plan=SubscriptionPlan.ON_PREMISE
    )
    s = await local_session_for(user)

    for headers in (access_cookie(s.access_token), bearer(s.access_token)):
        resp = await auth_client.get(ME, headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == str(user.id)
        assert body["role"] == UserRole.ADMIN.value
        assert body["subscription_plan"] == SubscriptionPlan.ON_PREMISE.value


async def test_local_mode_without_credentials_gets_401(auth_client, settings_override):
    settings_override(AUTH_MODE="local")

    for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Basic dXNlcjpwdw=="}):
        resp = await auth_client.get(ME, headers=headers)
        assert resp.status_code == 401, headers
        assert resp.json() == {"detail": "not_authenticated"}
        assert resp.headers["www-authenticate"] == BEARER_CHALLENGE


async def test_header_wins_over_cookie(auth_client, auth_db, local_session_for, settings_override):
    settings_override(AUTH_MODE="local")
    user = await seed_user(auth_db)
    s = await local_session_for(user)

    resp = await auth_client.get(
        ME, headers={**bearer("not-a-token"), **access_cookie(s.access_token)}
    )

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid_token"}


async def test_principal_of_a_local_token_comes_from_the_users_row(
    auth_db, local_session_for, settings_override
):
    from app.auth.principal import resolve_principal

    settings_override(AUTH_MODE="local")
    user = await seed_user(
        auth_db, role=UserRole.ANALYST, subscription_plan=SubscriptionPlan.STARTER
    )
    s = await local_session_for(user, amr=("pwd", "otp"))

    async with auth_db() as db:
        principal = await resolve_principal(make_request(headers=bearer(s.access_token)), db)

    assert principal.user_id == user.id
    assert principal.email == user.email
    assert principal.role == UserRole.ANALYST
    assert principal.subscription_tier == SubscriptionPlan.STARTER
    assert principal.session_id == s.session_id
    assert principal.auth_source == "local"
    assert principal.service_token_id is None


async def test_a_session_without_a_second_factor_is_refused(
    auth_client, auth_db, local_session_for, settings_override
):
    """Every session is opened after a second factor (`otp` or `recovery` in its `amr`): one
    without is refused wherever it comes from, as a structural invariant, not only because the
    routes that open sessions happen to check a code first."""
    settings_override(AUTH_MODE="local")
    user = await seed_user(auth_db)

    for amr in (("pwd",), (), ("pwd", "mfa")):
        s = await local_session_for(user, amr=amr)
        resp = await auth_client.get(ME, headers=bearer(s.access_token))
        assert resp.status_code == 401, amr
        assert resp.json() == {"detail": "invalid_token"}
        assert resp.headers["www-authenticate"] == BEARER_CHALLENGE

    for amr in (("pwd", "otp"), ("pwd", "recovery")):
        s = await local_session_for(user, amr=amr)
        resp = await auth_client.get(ME, headers=bearer(s.access_token))
        assert resp.status_code == 200, (amr, resp.text)


# ── CSRF ──────────────────────────────────────────────────────────────────────


async def test_csrf_header_required_for_cookie_writes(
    auth_client, auth_db, local_session_for, settings_override
):
    settings_override(AUTH_MODE="local")
    user = await seed_user(auth_db)
    token = (await local_session_for(user)).access_token

    for extra in ({}, {"X-GenoLens-Client": ""}):
        r = await auth_client.post(
            PROJECTS, json=NEW_PROJECT, headers={**access_cookie(token), **extra}
        )
        assert r.status_code == 403 and r.json()["detail"] == "csrf_header_missing", extra

    r = await auth_client.post(PROJECTS, json=NEW_PROJECT, headers={**access_cookie(token), **CSRF})
    assert r.status_code == 201, r.text

    # Reads need no header; a Bearer request is exempt (a page cannot set Authorization).
    r = await auth_client.get(PROJECTS, headers=access_cookie(token))
    assert r.status_code == 200, r.text
    r = await auth_client.post(PROJECTS, json={"name": "TST02"}, headers=bearer(token))
    assert r.status_code == 201, r.text


async def test_csrf_applies_in_the_migration_window_too(
    auth_client, auth_db, local_session_for, settings_override
):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=True)
    user = await seed_user(auth_db)
    token = (await local_session_for(user)).access_token

    r = await auth_client.post(PROJECTS, json=NEW_PROJECT, headers=access_cookie(token))

    assert r.status_code == 403 and r.json()["detail"] == "csrf_header_missing"


# ── Sessions and accounts ─────────────────────────────────────────────────────


async def test_revoked_session_and_deleted_user_get_401(
    auth_client, auth_db, local_session_for, settings_override
):
    from app.auth.sessions import revoke_session
    from app.auth.tokens import issue_access_token

    settings_override(AUTH_MODE="local")

    def refused(resp) -> bool:
        return (
            resp.status_code == 401
            and resp.json() == {"detail": "invalid_token"}
            and resp.headers["www-authenticate"] == BEARER_CHALLENGE
        )

    # Revoked: cut off at once, though the JWT itself is still valid.
    user = await seed_user(auth_db)
    s = await local_session_for(user)
    async with auth_db() as db:
        await revoke_session(db, s.session_id, reason="logout")
        await db.commit()
    assert refused(await auth_client.get(ME, headers=bearer(s.access_token)))

    # Idle past its window.
    s = await local_session_for(user)
    async with auth_db() as db:
        await db.execute(
            update(AuthSession)
            .where(AuthSession.id == s.session_id)
            .values(idle_expires_at=AuthSession.created_at)
        )
        await db.commit()
    assert refused(await auth_client.get(ME, headers=bearer(s.access_token)))

    # A session id presented under another account's id.
    other = await seed_user(auth_db)
    s = await local_session_for(user)
    crossed = issue_access_token(other.id, s.session_id, ["pwd"])
    assert refused(await auth_client.get(ME, headers=bearer(crossed)))

    # Not a token of ours (tampered signature), and not a token at all.
    assert refused(await auth_client.get(ME, headers=bearer(s.access_token[:-4] + "AAAA")))
    assert refused(await auth_client.get(ME, headers=bearer("not-a-token")))

    # Deleted account.
    async with auth_db() as db:
        await db.execute(delete(User).where(User.id == user.id))
        await db.commit()
    assert refused(await auth_client.get(ME, headers=bearer(s.access_token)))

    # Suspended: authenticated, then refused by the account guard as before.
    suspended = await seed_user(auth_db, status=UserStatus.SUSPENDED)
    s = await local_session_for(suspended)
    resp = await auth_client.get(PROJECTS, headers=bearer(s.access_token))
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "account_inactive"


async def test_local_mode_rejects_supabase_tokens_and_never_creates_users(
    auth_client, auth_db, local_session_for, supabase_token_for, settings_override
):
    settings_override(AUTH_MODE="local")
    user = await seed_user(auth_db)
    before = await users_count(auth_db)

    resp = await auth_client.get(ME, headers=bearer(supabase_token_for(user)))
    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid_token"}

    # A hosted token of someone unknown would have created the account in supabase mode.
    stranger = User(id=uuid4(), email="stranger@example.test", role=UserRole.USER)
    resp = await auth_client.get(ME, headers=bearer(supabase_token_for(stranger)))
    assert resp.status_code == 401

    # A local token whose user row is absent.
    ghost = User(id=uuid4(), email="ghost@example.test", role=UserRole.USER)
    s = await local_session_for(ghost)
    resp = await auth_client.get(ME, headers=bearer(s.access_token))
    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid_token"}

    assert await users_count(auth_db) == before


async def test_get_or_create_user_in_local_mode(auth_db, settings_override):
    """No creation, no role taken from the caller, invitation activation kept."""
    from app.api.deps.subscription import get_or_create_user
    from app.core.supabase_auth import SupabaseUser

    settings_override(AUTH_MODE="local")

    absent = SupabaseUser(user_id=uuid4(), email="absent@example.test", auth_source="local")
    async with auth_db() as db:
        with pytest.raises(HTTPException) as refused:
            await get_or_create_user(absent, db)
    assert refused.value.status_code == 401
    assert refused.value.detail == "user_not_found"
    assert await users_count(auth_db) == 0

    user = await seed_user(auth_db, status=UserStatus.PENDING)
    claims_admin = SupabaseUser(
        user_id=user.id, email=user.email, role=UserRole.ADMIN, auth_source="local"
    )
    async with auth_db() as db:
        row = await get_or_create_user(claims_admin, db)
        assert row.role == UserRole.USER
        assert row.status == UserStatus.ACTIVE
    async with auth_db() as db:
        stored = await db.get(User, user.id)
        assert stored.role == UserRole.USER
        assert stored.status == UserStatus.ACTIVE


async def test_get_or_create_user_in_supabase_mode_is_unchanged(auth_db):
    from app.api.deps.subscription import get_or_create_user
    from app.core.supabase_auth import SupabaseUser

    newcomer = SupabaseUser(user_id=uuid4(), email="newcomer@example.test", role=UserRole.ANALYST)
    async with auth_db() as db:
        created = await get_or_create_user(newcomer, db)
    assert created.id == newcomer.user_id
    assert created.role == UserRole.ANALYST
    assert created.subscription_plan == SubscriptionPlan.STARTER

    synced = SupabaseUser(user_id=newcomer.user_id, email=newcomer.email, role=UserRole.USER)
    async with auth_db() as db:
        assert (await get_or_create_user(synced, db)).role == UserRole.USER


# ── Service tokens ────────────────────────────────────────────────────────────


async def test_service_tokens_work_in_both_modes(auth_client, auth_db, settings_override):
    from app.api.endpoints.projects import create_project
    from app.auth.principal import resolve_principal
    from app.scripts.service_token import create_service_token

    owner = await seed_user(auth_db)  # no mfa_enabled_at
    assert owner.mfa_enabled_at is None
    async with auth_db() as db:
        row, token = await create_service_token(
            db, owner=owner.email, name="pipeline", scopes=("projects:create",)
        )

    for mode in ("supabase", "local"):
        settings_override(AUTH_MODE=mode)
        resp = await auth_client.post(PROJECTS, json={"name": f"TST-{mode}"}, headers=bearer(token))
        assert resp.status_code == 201, f"{mode}: {resp.text}"
        # Still refused where its scopes do not open the route.
        resp = await auth_client.get(ME, headers=bearer(token))
        assert resp.status_code == 403, mode

    route = SimpleNamespace(endpoint=create_project)
    async with auth_db() as db:
        principal = await resolve_principal(make_request("POST", bearer(token), route), db)
    assert principal.auth_source == "service_token"
    assert principal.service_token_id == row.id
    assert principal.user_id == owner.id


# ── One database session per request ─────────────────────────────────────────


async def test_one_db_session_per_request(
    auth_client,
    auth_db,
    db_sessions_opened,
    local_session_for,
    supabase_token_for,
    settings_override,
):
    user = await seed_user(auth_db)

    settings_override(AUTH_MODE="local")
    s = await local_session_for(user)
    db_sessions_opened.opened = 0
    resp = await auth_client.get(ME, headers=bearer(s.access_token))
    assert resp.status_code == 200, resp.text
    assert db_sessions_opened.opened == 1

    db_sessions_opened.opened = 0
    resp = await auth_client.post(
        PROJECTS, json=NEW_PROJECT, headers={**access_cookie(s.access_token), **CSRF}
    )
    assert resp.status_code == 201, resp.text
    assert db_sessions_opened.opened == 1

    settings_override(AUTH_MODE="supabase")
    db_sessions_opened.opened = 0
    resp = await auth_client.get(ME, headers=bearer(supabase_token_for(user)))
    assert resp.status_code == 200, resp.text
    assert db_sessions_opened.opened == 1


# ── The other dependency (app.api.deps.auth) ─────────────────────────────────


async def test_deps_auth_get_current_user_in_supabase_mode_is_unchanged(
    auth_db, supabase_token_for
):
    from datetime import datetime, timedelta, timezone

    from app.api.deps.auth import get_current_user

    user = await seed_user(auth_db)

    async def call(headers):
        async with auth_db() as db:
            return await get_current_user(make_request(headers=headers), None, db)

    current = await call(bearer(supabase_token_for(user)))
    assert current.id == user.id and current.email == user.email

    cases = [
        ({}, 403, "Not authenticated", None),
        ({"Authorization": "Basic dXNlcjpwdw=="}, 403, "Invalid authentication credentials", None),
        (
            bearer(supabase_token_for(user, exp=datetime.now(timezone.utc) - timedelta(hours=1))),
            401,
            "Token has expired",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            bearer("not-a-token"),
            401,
            "Could not validate credentials",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            bearer(
                supabase_token_for(User(id=uuid4(), email="x@example.test", role=UserRole.USER))
            ),
            401,
            "User not found",
            None,
        ),
    ]
    for headers, code, detail, challenge in cases:
        with pytest.raises(HTTPException) as refused:
            await call(headers)
        assert refused.value.status_code == code, headers
        assert refused.value.detail == detail, headers
        assert refused.value.headers == challenge, headers

    suspended = await seed_user(auth_db, status=UserStatus.SUSPENDED)
    with pytest.raises(HTTPException) as refused:
        await call(bearer(supabase_token_for(suspended)))
    assert (refused.value.status_code, refused.value.detail) == (403, "User account is inactive")


async def test_deps_auth_dependencies_in_local_mode(
    auth_db, local_session_for, supabase_token_for, settings_override
):
    from app.api.deps.auth import get_current_user, get_current_user_optional

    settings_override(AUTH_MODE="local")
    user = await seed_user(auth_db)
    s = await local_session_for(user)

    async with auth_db() as db:
        current = await get_current_user(
            make_request(headers=access_cookie(s.access_token)), None, db
        )
        assert current.id == user.id
        with pytest.raises(HTTPException) as refused:
            await get_current_user(make_request(headers=bearer(supabase_token_for(user))), None, db)
        assert (refused.value.status_code, refused.value.detail) == (401, "invalid_token")

        optional = await get_current_user_optional(
            make_request(headers=bearer(s.access_token)), None, db
        )
        assert optional is not None and optional.id == user.id
        for headers in ({}, bearer("not-a-token"), bearer("glst_" + "a" * 43)):
            assert await get_current_user_optional(make_request(headers=headers), None, db) is None
        # A cookie write without the CSRF header stays anonymous.
        assert (
            await get_current_user_optional(
                make_request("POST", access_cookie(s.access_token)), None, db
            )
            is None
        )


def test_openapi_still_declares_the_bearer_scheme():
    from app.main import app

    schema = app.openapi()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    assert {"HTTPBearer": []} in schema["paths"][ME]["get"]["security"]


async def test_cors_preflight_allows_the_csrf_header(auth_client):
    from app.core.config import settings

    origin = settings.CORS_ORIGINS[0]
    resp = await auth_client.options(
        PROJECTS,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-genolens-client",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.headers["access-control-allow-origin"] == origin
    assert resp.headers["access-control-allow-credentials"] == "true"
    allowed = resp.headers["access-control-allow-headers"].lower()
    assert "x-genolens-client" in allowed


# ── Login tracking ────────────────────────────────────────────────────────────


async def test_login_tracking_reads_the_same_credentials(
    auth_db, local_session_for, supabase_token_for, settings_override
):
    from app.auth.keys import reset_key_cache
    from app.middleware.login_tracking import _extract_user_id

    user = await seed_user(auth_db)
    s = await local_session_for(user)
    hosted = supabase_token_for(user)

    def extract(headers):
        return _extract_user_id(make_request(headers=headers))

    # Default configuration: hosted tokens from the header only, as before.
    assert extract(bearer(hosted)) == str(user.id)
    assert extract(access_cookie(hosted)) is None
    assert extract(bearer(s.access_token)) is None
    assert extract(bearer("glst_" + "a" * 43)) is None
    assert extract({}) is None

    settings_override(AUTH_MODE="local")
    assert extract(bearer(s.access_token)) == str(user.id)
    assert extract(access_cookie(s.access_token)) == str(user.id)
    assert extract(bearer(hosted)) is None
    assert extract(bearer("not-a-token")) is None

    # Local accepted but no keys in production: never raises.
    settings_override(ENVIRONMENT="production", AUTH_JWT_PRIVATE_KEYS="")
    reset_key_cache()
    assert extract(bearer(s.access_token)) is None


async def test_login_tracking_records_cookie_sessions(
    auth_client, auth_db, local_session_for, settings_override
):
    from app.middleware import login_tracking

    settings_override(AUTH_MODE="local")
    user = await seed_user(auth_db)
    s = await local_session_for(user)
    login_tracking._last_recorded.clear()
    recorder = AsyncMock()
    try:
        with patch.object(login_tracking, "_record_login_event", recorder):
            resp = await auth_client.get(ME, headers=access_cookie(s.access_token))
    finally:
        login_tracking._last_recorded.clear()

    assert resp.status_code == 200, resp.text
    recorder.assert_called_once_with(str(user.id))
