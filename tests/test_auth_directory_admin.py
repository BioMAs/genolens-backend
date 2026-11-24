"""
Account administration in both authentication modes, through the identity directory
(`app/auth/directory.py`).

`AUTH_MODE=supabase` is what production runs until the cutover: every admin endpoint must keep
making the same calls to the hosted provider and answering the same thing. `AUTH_MODE=local` is
the self-hosted mode: the same endpoints work on the `users` table alone and never call the
provider (`fake_supabase` fails the test if they do).

What is pinned down, per endpoint and per mode:
- invitations (and `create_user`, which becomes one in local mode) create a PENDING account; in
  local mode the link is a one-time INVITE token issued by the admin, e-mailed after the commit,
  audited, and a failed e-mail is reported to the admin (502) with the account and the link kept
  so that a resend works;
- deletion is local only in local mode (sessions, links and recovery codes go with the account)
  and never touches the Sequencing tables;
- role changes write `profiles` and drop the role cache only while roles come from the provider;
- listing, details, stats, projects and members read identities locally in local mode;
- adding a project member by e-mail looks the address up locally, ignoring case;
- the user list and details expose MFA, verification, last login and lock from the local row in
  both modes, and never a hash or a secret.

Everything runs against an in-memory SQLite database; no address is real (`@example.test`).
"""

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from urllib.parse import unquote
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, func, select, text

from app.auth.tokens import hash_token
from app.models.auth import (
    AuthEvent,
    AuthEventType,
    AuthOneTimeToken,
    AuthRecoveryCode,
    AuthSession,
)
from app.models.auth import OneTimeTokenPurpose as Purpose
from app.models.models import Project, ProjectMember, SubscriptionPlan, User, UserRole, UserStatus
from tests.auth_support import seed_user

API = "/api/v1"
ADMIN = f"{API}/admin"
MODES = pytest.mark.parametrize("mode", ["supabase", "local"])

LAST_LOGIN = datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc)
VERIFIED = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
MFA_ON = datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc)
LOCKED = datetime(2026, 11, 24, 13, 0, tzinfo=timezone.utc)

# What the hosted provider says, verbatim: supabase mode must pass these strings through as is.
SB_LAST_SIGN_IN = "2026-09-14T10:00:00.123456+00:00"
SB_CONFIRMED = "2026-01-01T00:00:00+00:00"

# No response may carry a key naming any of these, in any mode...
FORBIDDEN_KEY_PARTS = ("password", "hash", "secret")
# ...nor the stored values of `target`'s password hash and TOTP secret.
STORED_HASH = "$argon2id$placeholder"
STORED_SECRET = "encrypted-placeholder"


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def target(auth_db) -> User:
    """The account the admin acts on: verified, with MFA, a past login, a lock and secrets."""
    return await seed_user(
        auth_db,
        email=f"lea-{uuid4().hex[:6]}@example.test",
        full_name="Lea Martin",
        password_hash=STORED_HASH,
        mfa_totp_secret_enc=STORED_SECRET,
        email_verified_at=VERIFIED,
        mfa_enabled_at=MFA_ON,
        last_login_at=LAST_LOGIN,
        locked_until=LOCKED,
    )


@pytest.fixture
async def project(auth_db, admin_user) -> Project:
    async with auth_db() as s:
        row = Project(name="Shared project", owner_id=admin_user.id)
        s.add(row)
        await s.commit()
        return row


async def reload(auth_db, user_id) -> User | None:
    async with auth_db() as s:
        return await s.get(User, user_id)


async def scalars(auth_db, stmt) -> list:
    async with auth_db() as s:
        return list((await s.execute(stmt)).scalars().all())


async def count(auth_db, model, **where) -> int:
    stmt = select(func.count()).select_from(model)
    for column, value in where.items():
        stmt = stmt.where(getattr(model, column) == value)
    async with auth_db() as s:
        return (await s.execute(stmt)).scalar_one()


def same_instant(value: str, expected: datetime) -> bool:
    """A datetime the API rendered, compared to the one stored (SQLite drops the offset)."""

    def naive_utc(moment: datetime) -> datetime:
        return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo else moment

    return naive_utc(datetime.fromisoformat(value)) == naive_utc(expected)


def invite_link_token(email_text: str) -> str:
    match = re.search(r"/auth/invite\?token=(\S+)", email_text)
    assert match, email_text
    return unquote(match.group(1))


def profile_row(user: User, role: str = "user") -> dict:
    """A `profiles` row as PostgREST returns it for the admin selects."""
    return {
        "id": str(user.id),
        "full_name": "Name From Profiles",
        "avatar_url": None,
        "role": role,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-09-15T00:00:00+00:00",
    }


def auth_user(user_id, email: str, full_name: str = "Name From Auth") -> dict:
    """A user of the provider's admin API."""
    return {
        "id": str(user_id),
        "email": email,
        "last_sign_in_at": SB_LAST_SIGN_IN,
        "confirmed_at": SB_CONFIRMED,
        "user_metadata": {"full_name": full_name},
    }


def assert_no_secret(response) -> None:
    assert STORED_HASH not in response.text and STORED_SECRET not in response.text
    payload = response.json()
    for item in payload if isinstance(payload, list) else [payload]:
        for key in item:
            assert not any(part in key.lower() for part in FORBIDDEN_KEY_PARTS), key


def assert_no_local_invitation(smtp_capture) -> None:
    assert smtp_capture.messages == []


# ── The directory itself ─────────────────────────────────────────────────────


def test_role_source_follows_the_mode(settings_override):
    from app.auth.directory import role_source_is_supabase

    settings_override(AUTH_MODE="supabase")
    assert role_source_is_supabase() is True
    settings_override(AUTH_MODE="local")
    assert role_source_is_supabase() is False


async def test_local_create_identity_is_a_fresh_id_without_any_call(
    settings_override, fake_supabase
):
    from app.auth.directory import create_identity

    settings_override(AUTH_MODE="local")
    first = await create_identity("new@example.test", "New Person")
    second = await create_identity("new@example.test", "New Person")
    assert isinstance(first, UUID) and first != second
    assert fake_supabase.calls == []


async def test_supabase_create_identity_is_the_current_invitation(settings_override, fake_supabase):
    from app.auth.directory import create_identity

    settings_override(AUTH_MODE="supabase")
    identity_id = await create_identity("new@example.test", "New Person")
    assert str(identity_id) == fake_supabase.issued_ids[0]
    [call] = fake_supabase.calls
    assert (call.method, call.path) == ("POST", "/auth/v1/invite")
    assert call.json == {"email": "new@example.test", "data": {"full_name": "New Person"}}


async def test_supabase_create_identity_refuses_an_answer_without_id(
    settings_override, fake_supabase
):
    from app.auth.directory import create_identity

    settings_override(AUTH_MODE="supabase")
    fake_supabase.respond("POST", "/auth/v1/invite", 200, {})
    with pytest.raises(RuntimeError, match="no user id"):
        await create_identity("new@example.test", None)


async def test_local_find_identity_reads_the_users_table_ignoring_case(
    auth_db, target, settings_override, fake_supabase
):
    from app.auth.directory import find_identity_by_email, find_identity_by_id, list_identities

    settings_override(AUTH_MODE="local")
    async with auth_db() as db:
        by_email = await find_identity_by_email(db, f"  {target.email.upper()} ")
        by_id = await find_identity_by_id(db, target.id)
        listed = await list_identities(db)
        missing = await find_identity_by_email(db, "nobody@example.test")
    assert by_email == by_id == listed[target.id]
    assert by_email.id == target.id and by_email.email == target.email
    assert by_email.full_name == "Lea Martin"
    assert same_instant(by_email.last_sign_in_at.isoformat(), LAST_LOGIN)
    assert same_instant(by_email.email_confirmed_at.isoformat(), VERIFIED)
    assert missing is None
    assert fake_supabase.calls == []


async def test_local_find_identity_refuses_an_address_two_accounts_share(
    auth_db, settings_override
):
    from app.auth.directory import find_identity_by_email

    settings_override(AUTH_MODE="local")
    await seed_user(auth_db, email="Twin@example.test")
    await seed_user(auth_db, email="twin@example.test")
    async with auth_db() as db:
        assert await find_identity_by_email(db, "twin@example.test") is None


async def test_supabase_find_identity_uses_the_current_lookups(
    auth_db, settings_override, fake_supabase
):
    from app.auth.directory import find_identity_by_email, find_identity_by_id, list_identities

    settings_override(AUTH_MODE="supabase")
    uid = uuid4()
    record = auth_user(uid, "hosted@example.test", "Hosted Person")
    fake_supabase.respond("GET", "/auth/v1/admin/users", 200, {"users": [record]})
    fake_supabase.respond("GET", f"/auth/v1/admin/users/{uid}", 200, record)
    async with auth_db() as db:
        by_email = await find_identity_by_email(db, "hosted@example.test")
        by_id = await find_identity_by_id(db, uid)
        listed = await list_identities(db)
    assert by_email == by_id == listed[uid]
    assert (by_email.id, by_email.email, by_email.full_name) == (
        uid,
        "hosted@example.test",
        "Hosted Person",
    )
    assert by_email.last_sign_in_at == datetime.fromisoformat(SB_LAST_SIGN_IN)
    assert by_email.email_confirmed_at == datetime.fromisoformat(SB_CONFIRMED)
    # The lookup as fixed on main: GoTrue filters on `filter` (it ignores `email`), one page.
    assert fake_supabase.calls[0].params == {
        "filter": "hosted@example.test",
        "page": "1",
        "per_page": "1000",
    }
    assert fake_supabase.paths("GET") == [
        "/auth/v1/admin/users",
        f"/auth/v1/admin/users/{uid}",
        "/auth/v1/admin/users",
    ]


# ── Invitations ──────────────────────────────────────────────────────────────


@MODES
async def test_invite(
    admin_client, admin_user, auth_db, mode, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE=mode)
    r = await admin_client.post(
        f"{ADMIN}/users/invite",
        json={"email": "i@example.test", "full_name": "Ines", "plan": "TEAM"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "pending"
    invited = await reload(auth_db, UUID(r.json()["id"]))
    assert invited.status == UserStatus.PENDING
    assert invited.subscription_plan == SubscriptionPlan.TEAM
    assert invited.password_hash is None
    tokens = await scalars(
        auth_db, select(AuthOneTimeToken).where(AuthOneTimeToken.user_id == invited.id)
    )
    sent = await scalars(
        auth_db, select(AuthEvent).where(AuthEvent.event_type == AuthEventType.INVITE_SENT.value)
    )

    if mode == "local":
        assert fake_supabase.calls == [] and "/auth/invite?token=" in smtp_capture.last_text
        assert smtp_capture.last.to == "i@example.test"
        [token] = tokens
        assert token.purpose is Purpose.INVITE
        assert token.created_by == admin_user.id
        assert token.used_at is None
        assert token.token_hash == hash_token(invite_link_token(smtp_capture.last_text))
        [audit] = sent
        assert audit.user_id == invited.id
    else:
        assert fake_supabase.calls[0].path == "/auth/v1/invite"
        assert [c.path for c in fake_supabase.calls] == ["/auth/v1/invite"]
        assert str(invited.id) == fake_supabase.issued_ids[0]
        assert tokens == [] and sent == []
        assert_no_local_invitation(smtp_capture)


async def test_local_invite_normalises_the_address_and_refuses_a_case_variant(
    admin_client, auth_db, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE="local")
    await seed_user(auth_db, email="taken@example.test")

    r = await admin_client.post(f"{ADMIN}/users/invite", json={"email": "Taken@Example.test"})
    assert r.status_code == 409

    r = await admin_client.post(f"{ADMIN}/users/invite", json={"email": " New@Example.TEST "})
    assert r.status_code == 201, r.text
    assert r.json()["email"] == "new@example.test"
    assert smtp_capture.last.to == "new@example.test"


@pytest.mark.parametrize("failure", ["smtp_down", "smtp_not_configured"])
async def test_local_invite_email_failure_is_reported_and_resend_recovers(
    admin_client, auth_db, failure, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE="local")
    if failure == "smtp_down":
        smtp_capture.fail_with = ConnectionRefusedError("down")
    else:
        settings_override(SMTP_HOST=None)

    r = await admin_client.post(f"{ADMIN}/users/invite", json={"email": "late@example.test"})
    assert r.status_code == 502
    assert r.json()["detail"] == "invitation_email_failed"

    [pending] = await scalars(auth_db, select(User).where(User.email == "late@example.test"))
    assert pending.status == UserStatus.PENDING
    [first] = await scalars(
        auth_db, select(AuthOneTimeToken).where(AuthOneTimeToken.user_id == pending.id)
    )
    assert first.used_at is None  # kept: the admin can resend
    assert await count(auth_db, AuthEvent, event_type=AuthEventType.INVITE_SENT.value) == 0

    smtp_capture.fail_with = None
    settings_override(SMTP_HOST="smtp.example.test")
    r = await admin_client.post(f"{ADMIN}/users/{pending.id}/resend-invite")
    assert r.status_code == 200, r.text
    tokens = await scalars(
        auth_db,
        select(AuthOneTimeToken)
        .where(AuthOneTimeToken.user_id == pending.id)
        .order_by(AuthOneTimeToken.created_at),
    )
    live = [t for t in tokens if t.used_at is None]
    assert len(tokens) == 2 and len(live) == 1
    assert live[0].id != first.id
    assert live[0].token_hash == hash_token(invite_link_token(smtp_capture.last_text))
    assert await count(auth_db, AuthEvent, event_type=AuthEventType.INVITE_SENT.value) == 1
    assert fake_supabase.calls == []


@MODES
async def test_resend_invite(
    admin_client, admin_user, auth_db, mode, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE=mode)
    pending = await seed_user(auth_db, status=UserStatus.PENDING, full_name="Pending Person")
    active = await seed_user(auth_db)

    r = await admin_client.post(f"{ADMIN}/users/{active.id}/resend-invite")
    assert r.status_code == 400

    r = await admin_client.post(f"{ADMIN}/users/{pending.id}/resend-invite")
    assert r.status_code == 200, r.text
    assert r.json() == {"detail": "Invitation email resent", "email": pending.email}
    if mode == "local":
        assert fake_supabase.calls == []
        [token] = await scalars(
            auth_db, select(AuthOneTimeToken).where(AuthOneTimeToken.user_id == pending.id)
        )
        assert token.created_by == admin_user.id
        assert token.token_hash == hash_token(invite_link_token(smtp_capture.last_text))
    else:
        [call] = fake_supabase.calls
        assert (call.method, call.path) == ("POST", "/auth/v1/invite")
        assert call.json == {"email": pending.email, "data": {"full_name": "Pending Person"}}
        assert_no_local_invitation(smtp_capture)


# ── create_user ──────────────────────────────────────────────────────────────


@MODES
async def test_create_user(
    admin_client, admin_user, auth_db, mode, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE=mode)
    r = await admin_client.post(
        f"{ADMIN}/users",
        json={
            "email": "created@example.test",
            "password": "Chosen-By-The-Admin-91",
            "full_name": "Created Person",
            "role": "admin",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    created = await reload(auth_db, UUID(body["id"]))
    assert created.role == UserRole.ADMIN
    assert created.password_hash is None  # a password chosen by the admin is never stored

    if mode == "local":
        assert fake_supabase.calls == []
        assert body["invitation_sent"] is True
        assert body["status"] == "pending"
        assert created.status == UserStatus.PENDING
        assert smtp_capture.last.to == "created@example.test"
        [token] = await scalars(
            auth_db, select(AuthOneTimeToken).where(AuthOneTimeToken.user_id == created.id)
        )
        assert token.purpose is Purpose.INVITE and token.created_by == admin_user.id
    else:
        [call] = fake_supabase.calls
        assert (call.method, call.path) == ("POST", "/auth/v1/admin/users")
        assert call.json == {
            "email": "created@example.test",
            "password": "Chosen-By-The-Admin-91",
            "email_confirm": True,
            "user_metadata": {"full_name": "Created Person"},
        }
        assert body["id"] == fake_supabase.issued_ids[0]
        assert "invitation_sent" not in body
        assert body["role"] == "ADMIN"
        assert_no_local_invitation(smtp_capture)


async def test_local_create_user_needs_no_password(
    admin_client, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE="local")
    r = await admin_client.post(f"{ADMIN}/users", json={"email": "nopass@example.test"})
    assert r.status_code == 200, r.text
    assert r.json()["invitation_sent"] is True
    assert r.json()["role"] == "user"
    assert smtp_capture.last.to == "nopass@example.test"


async def test_local_create_user_reports_a_failed_invitation_email(
    admin_client, auth_db, fake_supabase, smtp_capture, settings_override
):
    settings_override(AUTH_MODE="local")
    smtp_capture.fail_with = ConnectionRefusedError("down")
    r = await admin_client.post(f"{ADMIN}/users", json={"email": "unlucky@example.test"})
    assert r.status_code == 502
    assert r.json()["detail"] == "invitation_email_failed"
    assert await count(auth_db, User, email="unlucky@example.test") == 1


async def test_supabase_create_user_still_requires_a_password(
    admin_client, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="supabase")
    r = await admin_client.post(f"{ADMIN}/users", json={"email": "x@example.test", "role": "nope"})
    assert r.status_code == 422
    assert r.json()["detail"] == [
        {"loc": ["body", "password"], "msg": "Field required", "type": "missing"}
    ]
    assert fake_supabase.calls == []


# ── Listing and details ──────────────────────────────────────────────────────


@MODES
async def test_list_users_exposes_mfa_and_last_login(
    admin_client, admin_user, auth_db, target, mode, fake_supabase, settings_override
):
    settings_override(AUTH_MODE=mode)
    fake_supabase.respond(
        "GET", "/rest/v1/profiles", 200, [profile_row(target), profile_row(admin_user, "admin")]
    )
    fake_supabase.respond(
        "GET", "/auth/v1/admin/users", 200, {"users": [auth_user(target.id, target.email)]}
    )

    r = await admin_client.get(f"{ADMIN}/users")
    assert r.status_code == 200, r.text
    listed = {item["id"]: item for item in r.json()}
    assert set(listed) == {str(target.id), str(admin_user.id)}
    assert_no_secret(r)

    row = listed[str(target.id)]
    assert row["mfa_enabled"] is True
    assert row["email_verified"] is True
    assert same_instant(row["last_login_at"], LAST_LOGIN)
    assert same_instant(row["locked_until"], LOCKED)
    assert row["email"] == target.email
    other = listed[str(admin_user.id)]
    assert other["mfa_enabled"] is False and other["email_verified"] is False
    assert other["locked_until"] is None
    # The admin's own session (`admin_client`) recorded a login on its row.
    assert same_instant(
        other["last_login_at"], (await reload(auth_db, admin_user.id)).last_login_at
    )

    if mode == "local":
        assert fake_supabase.calls == []
        assert row["full_name"] == "Lea Martin"
        assert same_instant(row["last_sign_in_at"], LAST_LOGIN)
        assert same_instant(row["confirmed_at"], VERIFIED)
        assert other["role"] == "scilicium_admin"
    else:
        assert fake_supabase.paths("GET") == ["/rest/v1/profiles", "/auth/v1/admin/users"]
        assert row["full_name"] == "Name From Profiles"
        assert row["last_sign_in_at"] == SB_LAST_SIGN_IN  # the provider's string, untouched
        assert row["confirmed_at"] == SB_CONFIRMED
        assert other["email"] is None  # absent from the provider's list, as before


@MODES
async def test_get_user_details(admin_client, target, mode, fake_supabase, settings_override):
    settings_override(AUTH_MODE=mode)
    fake_supabase.respond("GET", "/rest/v1/profiles", 200, [profile_row(target)])
    fake_supabase.respond(
        "GET", f"/auth/v1/admin/users/{target.id}", 200, auth_user(target.id, target.email)
    )

    r = await admin_client.get(f"{ADMIN}/users/{target.id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert_no_secret(r)
    assert body["mfa_enabled"] is True and body["email_verified"] is True
    assert same_instant(body["last_login_at"], LAST_LOGIN)
    assert same_instant(body["locked_until"], LOCKED)
    assert body["status"] == "active" and body["subscription_plan"] == "TEAM"
    if mode == "local":
        assert fake_supabase.calls == []
        assert body["email"] == target.email and body["full_name"] == "Lea Martin"
        assert same_instant(body["last_sign_in_at"], LAST_LOGIN)
    else:
        assert fake_supabase.paths("GET") == [
            "/rest/v1/profiles",
            f"/auth/v1/admin/users/{target.id}",
        ]
        assert body["last_sign_in_at"] == SB_LAST_SIGN_IN
        assert body["confirmed_at"] == SB_CONFIRMED


async def test_local_user_details_of_an_unknown_account_is_404(
    admin_client, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="local")
    r = await admin_client.get(f"{ADMIN}/users/{uuid4()}")
    assert r.status_code == 404


@MODES
async def test_system_stats(admin_client, auth_db, target, mode, fake_supabase, settings_override):
    from sqlalchemy.schema import CreateTable

    from app.models.models import Dataset

    settings_override(AUTH_MODE=mode)
    async with auth_db.kw["bind"].begin() as conn:  # the stats count datasets too
        await conn.execute(CreateTable(Dataset.__table__))
    fake_supabase.respond("GET", "/rest/v1/profiles", 200, [{"id": str(uuid4())}] * 3)
    r = await admin_client.get(f"{ADMIN}/stats")
    assert r.status_code == 200, r.text
    if mode == "local":
        assert fake_supabase.calls == []
        assert r.json()["total_users"] == 2  # the admin and the target
    else:
        assert fake_supabase.paths() == ["/rest/v1/profiles"]
        assert r.json()["total_users"] == 3
        assert r.json()["users_by_plan"]["STARTER"] == 1  # 3 hosted, 2 local


# ── Subscription of an account without a local row ───────────────────────────


@MODES
async def test_subscription_of_an_account_without_local_row(
    admin_client, auth_db, mode, fake_supabase, settings_override
):
    settings_override(AUTH_MODE=mode)
    ghost = uuid4()
    fake_supabase.respond(
        "GET", f"/auth/v1/admin/users/{ghost}", 200, auth_user(ghost, "ghost@example.test")
    )
    fake_supabase.respond(
        "GET",
        "/rest/v1/profiles",
        200,
        [{**profile_row(User(id=ghost)), "id": str(ghost)}],
    )
    r = await admin_client.patch(f"{ADMIN}/users/{ghost}/subscription", json={"plan": "TEAM"})
    if mode == "local":
        assert r.status_code == 404
        assert await reload(auth_db, ghost) is None
        assert fake_supabase.calls == []
    else:
        assert r.status_code == 200, r.text
        created = await reload(auth_db, ghost)
        assert created.email == "ghost@example.test"
        assert created.subscription_plan == SubscriptionPlan.TEAM


# ── Role and profile changes ─────────────────────────────────────────────────


@pytest.fixture
def role_cache_drops(monkeypatch) -> list:
    dropped: list = []
    monkeypatch.setattr(
        "app.api.endpoints.admin.invalidate_cached_role", lambda user_id: dropped.append(user_id)
    )
    return dropped


async def test_local_role_change_does_not_write_profiles(
    admin_client, auth_db, target, fake_supabase, role_cache_drops, settings_override
):
    settings_override(AUTH_MODE="local")
    r = await admin_client.patch(f"{ADMIN}/users/{target.id}/role", json={"role": "admin"})
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "admin"
    assert (await reload(auth_db, target.id)).role == UserRole.ADMIN
    assert fake_supabase.calls == []
    assert role_cache_drops == []

    r = await admin_client.patch(f"{ADMIN}/users/{uuid4()}/role", json={"role": "admin"})
    assert r.status_code == 404


async def test_supabase_role_change_writes_profiles_and_the_local_row(
    admin_client, auth_db, target, fake_supabase, role_cache_drops, settings_override
):
    settings_override(AUTH_MODE="supabase")
    fake_supabase.respond("GET", "/rest/v1/profiles", 200, [profile_row(target, "admin")])
    r = await admin_client.patch(f"{ADMIN}/users/{target.id}/role", json={"role": "admin"})
    assert r.status_code == 200, r.text
    patch_call = fake_supabase.calls[0]
    assert (patch_call.method, patch_call.path) == ("PATCH", "/rest/v1/profiles")
    assert patch_call.json == {"role": "admin"}
    assert patch_call.params == {"id": f"eq.{target.id}"}
    assert role_cache_drops == [target.id]
    assert (await reload(auth_db, target.id)).role == UserRole.ADMIN


@MODES
async def test_update_user(
    admin_client, auth_db, target, mode, fake_supabase, role_cache_drops, settings_override
):
    settings_override(AUTH_MODE=mode)
    fake_supabase.respond("GET", "/rest/v1/profiles", 200, [profile_row(target)])
    r = await admin_client.patch(
        f"{ADMIN}/users/{target.id}", json={"full_name": "Lea M.", "role": "admin"}
    )
    assert r.status_code == 200, r.text
    if mode == "local":
        assert fake_supabase.calls == []
        assert role_cache_drops == []
        stored = await reload(auth_db, target.id)
        assert (stored.full_name, stored.role) == ("Lea M.", UserRole.ADMIN)
        assert r.json()["full_name"] == "Lea M." and r.json()["role"] == "admin"
    else:
        patch_call = fake_supabase.calls[0]
        assert (patch_call.method, patch_call.path) == ("PATCH", "/rest/v1/profiles")
        assert patch_call.json == {"full_name": "Lea M.", "role": "admin"}
        assert role_cache_drops == [target.id]


async def test_local_update_user_of_an_unknown_account_is_404(
    admin_client, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="local")
    r = await admin_client.patch(f"{ADMIN}/users/{uuid4()}", json={"full_name": "Nobody"})
    assert r.status_code == 404


# ── Deletion ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def statements(auth_db):
    """Every SQL statement the test database runs from now on, as text."""
    engine = auth_db.kw["bind"].sync_engine
    seen: list[str] = []

    def record(_conn, _cursor, statement, *_args):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    yield seen
    event.remove(engine, "before_cursor_execute", record)


async def give_auth_state(auth_db, local_session_for, user: User) -> None:
    """A session, a pending link and a recovery code: what must go with the account."""
    from app.auth.one_time_tokens import issue_token

    await local_session_for(user)
    async with auth_db() as s:
        await issue_token(s, user.id, Purpose.PASSWORD_RESET)
        s.add(AuthRecoveryCode(user_id=user.id, code_hash="0" * 64))
        s.add(Project(name="Owned", owner_id=user.id))
        await s.commit()


async def test_local_delete_never_touches_supabase(
    admin_client, auth_db, target, local_session_for, statements, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="local")
    async with auth_db() as s:  # prove the cascade: SQLite only enforces it when asked to
        await s.execute(text("PRAGMA foreign_keys=ON"))
        assert (await s.execute(text("PRAGMA foreign_keys"))).scalar() == 1
    await give_auth_state(auth_db, local_session_for, target)

    r = await admin_client.delete(f"{ADMIN}/users/{target.id}")
    assert r.status_code == 200, r.text
    assert r.json() == {"message": "User deleted successfully"}
    assert await reload(auth_db, target.id) is None
    for model in (AuthSession, AuthOneTimeToken, AuthRecoveryCode):
        assert await count(auth_db, model, user_id=target.id) == 0, model
    assert await count(auth_db, Project, owner_id=target.id) == 0
    assert fake_supabase.calls == []
    touched = " ".join(statements).lower()
    assert "sequencing_projects" not in touched and "status_history" not in touched


async def test_local_delete_of_an_unknown_account_is_404(
    admin_client, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="local")
    r = await admin_client.delete(f"{ADMIN}/users/{uuid4()}")
    assert r.status_code == 404


@pytest.mark.parametrize(
    "provider_status, expected",
    [
        (200, {"message": "User deleted successfully"}),
        (404, {"message": "User deleted (was not found in Supabase Auth)"}),
    ],
)
async def test_supabase_delete_keeps_the_hosted_cleanup(
    admin_client, auth_db, target, provider_status, expected, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="supabase")
    fake_supabase.respond("DELETE", f"/auth/v1/admin/users/{target.id}", provider_status, {})
    r = await admin_client.delete(f"{ADMIN}/users/{target.id}")
    assert r.status_code == 200, r.text
    assert r.json() == expected
    assert await reload(auth_db, target.id) is None
    assert [(c.method, c.path, c.params) for c in fake_supabase.calls] == [
        ("DELETE", "/rest/v1/sequencing_projects", {"user_id": f"eq.{target.id}"}),
        ("DELETE", "/rest/v1/status_history", {"changed_by": f"eq.{target.id}"}),
        ("DELETE", "/rest/v1/status_history", {"user_id": f"eq.{target.id}"}),
        ("DELETE", f"/auth/v1/admin/users/{target.id}", {}),
    ]


async def test_supabase_delete_reports_a_provider_refusal(
    admin_client, target, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="supabase")
    fake_supabase.respond("DELETE", f"/auth/v1/admin/users/{target.id}", 500, {"msg": "boom"})
    r = await admin_client.delete(f"{ADMIN}/users/{target.id}")
    assert r.status_code == 500
    assert r.json()["detail"].startswith("Failed to delete user from Supabase: ")
    assert "boom" in r.json()["detail"]


# ── Projects and members (admin) ─────────────────────────────────────────────


@MODES
async def test_list_all_projects(
    admin_client, auth_db, target, mode, fake_supabase, settings_override
):
    settings_override(AUTH_MODE=mode)
    async with auth_db() as s:
        s.add(Project(name="Lea's", owner_id=target.id))
        await s.commit()
    fake_supabase.respond(
        "GET",
        "/auth/v1/admin/users",
        200,
        {"users": [auth_user(target.id, "hosted-lea@example.test", "Hosted Lea")]},
    )
    r = await admin_client.get(f"{ADMIN}/projects")
    assert r.status_code == 200, r.text
    [row] = r.json()
    if mode == "local":
        assert fake_supabase.calls == []
        assert (row["owner_email"], row["owner_full_name"]) == (target.email, "Lea Martin")
    else:
        assert fake_supabase.paths() == ["/auth/v1/admin/users"]
        assert (row["owner_email"], row["owner_full_name"]) == (
            "hosted-lea@example.test",
            "Hosted Lea",
        )


@MODES
async def test_list_project_members(
    admin_client, auth_db, target, project, mode, fake_supabase, settings_override
):
    settings_override(AUTH_MODE=mode)
    async with auth_db() as s:
        s.add(ProjectMember(project_id=project.id, user_id=target.id, access_level=UserRole.USER))
        await s.commit()
    fake_supabase.respond(
        "GET",
        "/auth/v1/admin/users",
        200,
        {"users": [auth_user(target.id, "hosted-lea@example.test", "Hosted Lea")]},
    )
    r = await admin_client.get(f"{ADMIN}/projects/{project.id}/members")
    assert r.status_code == 200, r.text
    [member] = r.json()
    if mode == "local":
        assert fake_supabase.calls == []
        assert (member["user_email"], member["user_full_name"]) == (target.email, "Lea Martin")
    else:
        assert fake_supabase.paths() == ["/auth/v1/admin/users"]
        assert member["user_email"] == "hosted-lea@example.test"


@MODES
async def test_admin_add_project_member(
    admin_client, auth_db, target, project, mode, fake_supabase, settings_override
):
    settings_override(AUTH_MODE=mode)
    fake_supabase.respond(
        "GET",
        f"/auth/v1/admin/users/{target.id}",
        200,
        auth_user(target.id, "hosted-lea@example.test", "Hosted Lea"),
    )
    r = await admin_client.post(
        f"{ADMIN}/projects/{project.id}/members",
        json={"user_id": str(target.id), "access_level": "USER"},
    )
    assert r.status_code == 200, r.text
    if mode == "local":
        assert fake_supabase.calls == []
        assert (r.json()["user_email"], r.json()["user_full_name"]) == (target.email, "Lea Martin")
    else:
        assert fake_supabase.paths() == [f"/auth/v1/admin/users/{target.id}"]
        assert r.json()["user_email"] == "hosted-lea@example.test"


async def test_local_admin_add_project_member_refuses_an_unknown_account(
    admin_client, auth_db, project, fake_supabase, settings_override
):
    settings_override(AUTH_MODE="local")
    r = await admin_client.post(
        f"{ADMIN}/projects/{project.id}/members",
        json={"user_id": str(uuid4()), "access_level": "USER"},
    )
    assert r.status_code == 404
    assert await count(auth_db, ProjectMember, project_id=project.id) == 0


@MODES
async def test_update_project_owner_without_local_row(
    admin_client, auth_db, project, mode, fake_supabase, settings_override
):
    settings_override(AUTH_MODE=mode)
    newcomer = uuid4()
    fake_supabase.respond(
        "GET",
        f"/auth/v1/admin/users/{newcomer}",
        200,
        auth_user(newcomer, "newcomer@example.test", "New Comer"),
    )
    r = await admin_client.patch(f"{ADMIN}/projects/{project.id}", json={"owner_id": str(newcomer)})
    if mode == "local":
        assert r.status_code == 404
        assert fake_supabase.calls == []
        assert await reload(auth_db, newcomer) is None
    else:
        assert r.status_code == 200, r.text
        assert fake_supabase.paths() == [f"/auth/v1/admin/users/{newcomer}"]
        assert (await reload(auth_db, newcomer)).email == "newcomer@example.test"


# ── Project members by e-mail (owner route) ──────────────────────────────────


@MODES
async def test_add_member_by_email_uses_local_lookup(
    admin_client, auth_db, target, project, mode, fake_supabase, settings_override, monkeypatch
):
    settings_override(AUTH_MODE=mode)
    monkeypatch.setattr("app.services.history_service.log_activity", AsyncMock())
    monkeypatch.setattr("app.services.email_service.send_project_invitation", AsyncMock())
    fake_supabase.respond(
        "GET", "/auth/v1/admin/users", 200, {"users": [auth_user(target.id, target.email)]}
    )

    r = await admin_client.post(
        f"{API}/projects/{project.id}/members",
        json={"user_email": target.email.upper(), "access_level": "USER"},
    )
    assert r.status_code == 201, r.text
    [member] = await scalars(
        auth_db, select(ProjectMember).where(ProjectMember.project_id == project.id)
    )
    assert member.user_id == target.id
    if mode == "local":
        assert fake_supabase.calls == []
    else:
        [call] = fake_supabase.calls
        assert (call.method, call.path) == ("GET", "/auth/v1/admin/users")
        # The address as the request schema hands it over (its domain lowercased).
        assert call.params["filter"].lower() == target.email


async def test_local_add_member_by_unknown_email_is_404(
    admin_client, project, fake_supabase, settings_override, monkeypatch
):
    settings_override(AUTH_MODE="local")
    monkeypatch.setattr("app.services.email_service.send_project_invitation", AsyncMock())
    r = await admin_client.post(
        f"{API}/projects/{project.id}/members",
        json={"user_email": "stranger@example.test", "access_level": "USER"},
    )
    assert r.status_code == 404
    assert fake_supabase.calls == []


# ── Comment replies notify the parent author found in the directory ──────────


@MODES
async def test_reply_notification_finds_the_parent_author_through_the_directory(
    admin_client, target, project, mode, fake_supabase, settings_override, monkeypatch
):
    from tests.conftest import make_comment

    settings_override(AUTH_MODE=mode)
    parent = make_comment(comment_id=uuid4(), project_id=project.id, user_id=target.id)
    reply = make_comment(comment_id=uuid4(), project_id=project.id, parent_id=parent.id)
    send_reply = AsyncMock()
    svc = "app.api.endpoints.comments.comments_service"
    monkeypatch.setattr(f"{svc}.get_comment_by_id", AsyncMock(return_value=parent))
    monkeypatch.setattr(f"{svc}.create_comment", AsyncMock(return_value=reply))
    monkeypatch.setattr("app.services.history_service.log_activity", AsyncMock())
    monkeypatch.setattr("app.services.email_service.send_reply_notification", send_reply)
    fake_supabase.respond(
        "GET",
        f"/auth/v1/admin/users/{target.id}",
        200,
        auth_user(target.id, "hosted-lea@example.test"),
    )

    r = await admin_client.post(
        f"{API}/projects/{project.id}/comments",
        json={"content": "A reply", "parent_id": str(parent.id)},
    )
    assert r.status_code == 201, r.text
    notified = send_reply.await_args.kwargs["parent_author_email"]
    if mode == "local":
        assert fake_supabase.calls == []
        assert notified == target.email
    else:
        assert fake_supabase.paths() == [f"/auth/v1/admin/users/{target.id}"]
        assert notified == "hosted-lea@example.test"
