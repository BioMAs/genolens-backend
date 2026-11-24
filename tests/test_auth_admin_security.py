"""
The security actions an administrator takes on an account (`/api/v1/admin/users/{id}/...`):
reset the second factor, end the sessions (and, only when asked, the service tokens), unlock,
send a password-reset link, and read the audit trail and the service tokens.

What is pinned down:
- who may do what: administrators only; resetting the second factor of an ADMIN or SCILICIUM_ADMIN
  (the caller's own included), ending their sessions (with or without the service tokens) and
  sending them a password link need a SCILICIUM_ADMIN, and a refusal changes nothing; any
  administrator unlocks any account;
- the reset leaves nothing of the second factor behind (secret, pending secret, recovery codes,
  replay counter), ends every session, and forces an enrolment at the next sign-in; the
  notification e-mail is best effort and never undoes the reset;
- a service token is only ever revoked by the explicit `revoke_service_tokens` box, never by the
  reset, the unlock, the password link or a plain session revocation (spec 7.7);
- the password link is the one the account itself would ask for, issued by the administrator, and
  an e-mail that cannot leave is reported (502) with the link kept; it is refused (409) until
  AUTH_MODE=local, the pre-launch included, since its page answers 404 before the switch;
- every action is audited against the account, with the administrator as actor;
- the audit trail and the token list are bounded and never carry a secret or a token hash;
- the routes work on local data in both modes (the password link aside) and never call the
  hosted provider;
- the shared logic (`app.auth.admin_actions`), which the break-glass command reuses, does not
  commit.

Everything runs against an in-memory SQLite database; no address is real (`@example.test`).
"""

import re
from collections import Counter
from datetime import timedelta
from urllib.parse import unquote
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.auth.admin_actions import (
    reset_user_mfa,
    revoke_user_service_tokens,
    revoke_user_sessions,
    unlock_user,
)
from app.auth.audit import record_event
from app.auth.clock import aware_utc, now_utc
from app.auth.mfa import encrypt_secret, new_totp_secret
from app.auth.passwords import hash_password
from app.auth.tokens import hash_token
from app.models.auth import (
    AuthEvent,
    AuthEventType,
    AuthOneTimeToken,
    AuthRecoveryCode,
    AuthSession,
)
from app.models.auth import OneTimeTokenPurpose as Purpose
from app.models.models import User, UserRole
from app.models.service_token import ServiceToken
from tests.auth_support import CSRF, NOW, PW, bearer, seed_user

API = "/api/v1"
ADMIN = f"{API}/admin"
ME = f"{API}/users/me"
MODES = pytest.mark.parametrize("mode", ["supabase", "local"])

# (method, path under /users/{id}/) of every route this module adds.
ROUTES = [
    ("POST", "mfa/reset"),
    ("POST", "sessions/revoke"),
    ("POST", "unlock"),
    ("POST", "password-reset"),
    ("GET", "auth-events"),
    ("GET", "service-tokens"),
]

SECRET_PLACEHOLDER = "encrypted-secret-placeholder"
PENDING_PLACEHOLDER = "encrypted-pending-placeholder"


# ── Helpers ──────────────────────────────────────────────────────────────────


def url(user_id, suffix: str) -> str:
    return f"{ADMIN}/users/{user_id}/{suffix}"


async def enrolled(auth_db, **fields) -> User:
    """An account with the second factor on, a half-finished rotation, and a replay counter."""
    values = dict(
        mfa_totp_secret_enc=SECRET_PLACEHOLDER,
        mfa_pending_secret_enc=PENDING_PLACEHOLDER,
        mfa_enabled_at=NOW,
        mfa_last_used_step=12345,
        email_verified_at=NOW,
    )
    values.update(fields)
    return await seed_user(auth_db, **values)


async def give_recovery_codes(auth_db, user: User, count: int = 3) -> None:
    async with auth_db() as s:
        s.add_all(
            AuthRecoveryCode(user_id=user.id, code_hash=uuid4().hex + uuid4().hex)
            for _ in range(count)
        )
        await s.commit()


async def mint_token(auth_db, owner: User, scopes=("projects:create",), name="pipeline"):
    """A service token of `owner`: its row and the token itself."""
    from app.scripts.service_token import create_service_token

    async with auth_db() as s:
        return await create_service_token(s, owner=owner.email, name=name, scopes=scopes)


async def token_status(client, token: str) -> int:
    """What the token gets on the one route its scope opens: 201 while valid, 401 once revoked."""
    r = await client.post(f"{API}/projects/", headers=bearer(token), json={"name": "TST01 v1"})
    return r.status_code


async def caller(local_session_for, auth_db, role: UserRole) -> tuple[User, dict]:
    """A signed-in administrator of `role`, and the header that carries their session."""
    user = await seed_user(
        auth_db, email=f"{role.value.lower()}-{uuid4().hex[:6]}@example.test", role=role
    )
    return user, bearer((await local_session_for(user)).access_token)


async def reload(auth_db, user_id) -> User:
    async with auth_db() as s:
        return (
            await s.execute(
                select(User).where(User.id == user_id).execution_options(populate_existing=True)
            )
        ).scalar_one()


async def sessions_of(auth_db, user_id) -> list[AuthSession]:
    async with auth_db() as s:
        rows = await s.execute(select(AuthSession).where(AuthSession.user_id == user_id))
        return list(rows.scalars())


async def events_of(auth_db, user_id, event_type: AuthEventType) -> list[AuthEvent]:
    async with auth_db() as s:
        rows = await s.execute(
            select(AuthEvent)
            .where(AuthEvent.user_id == user_id, AuthEvent.event_type == event_type.value)
            .order_by(AuthEvent.id)
        )
        return list(rows.scalars())


async def count(auth_db, model, **where) -> int:
    stmt = select(func.count()).select_from(model)
    for column, value in where.items():
        stmt = stmt.where(getattr(model, column) == value)
    async with auth_db() as s:
        return (await s.execute(stmt)).scalar_one()


async def _all(auth_db, model) -> list:
    async with auth_db() as s:
        return list((await s.execute(select(model))).scalars())


async def token_rows(auth_db, owner_id) -> list[ServiceToken]:
    async with auth_db() as s:
        rows = await s.execute(select(ServiceToken).where(ServiceToken.owner_id == owner_id))
        return list(rows.scalars())


def reset_link_token(email_text: str) -> str:
    match = re.search(r"/auth/reset\?token=(\S+)", email_text)
    assert match, email_text
    return unquote(match.group(1))


def assert_second_factor_intact(user: User) -> None:
    assert user.mfa_totp_secret_enc == SECRET_PLACEHOLDER
    assert user.mfa_pending_secret_enc == PENDING_PLACEHOLDER
    assert user.mfa_enabled_at is not None
    assert user.mfa_last_used_step == 12345


# ── Who may call, and on whom ────────────────────────────────────────────────


@pytest.mark.parametrize("method,suffix", ROUTES)
async def test_security_routes_are_for_administrators_only(
    admin_client, auth_db, local_session_for, method, suffix
):
    target = await enrolled(auth_db)
    plain = await seed_user(auth_db, role=UserRole.USER)
    headers = bearer((await local_session_for(plain)).access_token)

    r = await admin_client.request(method, url(target.id, suffix), headers=headers)
    assert r.status_code == 403, r.text

    # Without credentials the admin guard answers "not authenticated" (401, or 403 from the
    # bearer scheme): what matters is that it is a refusal.
    r = await admin_client.request(method, url(target.id, suffix), headers={"Authorization": ""})
    assert r.status_code in (401, 403), r.text

    # Nothing was done to the account by either refusal.
    assert_second_factor_intact(await reload(auth_db, target.id))
    assert await count(auth_db, AuthEvent, user_id=target.id) == 0


@pytest.mark.parametrize("method,suffix", ROUTES)
async def test_unknown_account_is_404_on_every_route(admin_client, auth_db, method, suffix):
    r = await admin_client.request(method, url(uuid4(), suffix))
    assert r.status_code == 404, r.text
    assert r.json() == {"detail": "User not found"}


# ── Resetting the second factor ──────────────────────────────────────────────


async def test_mfa_reset_clears_the_second_factor_and_audits_it(
    admin_client, admin_user, auth_db, local_session_for, smtp_capture
):
    target = await enrolled(auth_db, password_hash="$argon2id$placeholder")
    other = await enrolled(auth_db)
    await give_recovery_codes(auth_db, target, 3)
    await give_recovery_codes(auth_db, other, 2)
    await local_session_for(target)
    await local_session_for(target)
    other_session = await local_session_for(other)

    r = await admin_client.post(url(target.id, "mfa/reset"))

    assert r.status_code == 200, r.text
    assert r.json() == {"sessions_revoked": 2, "notification_sent": True}
    after = await reload(auth_db, target.id)
    assert after.mfa_totp_secret_enc is None
    assert after.mfa_pending_secret_enc is None
    assert after.mfa_enabled_at is None
    assert after.mfa_last_used_step is None
    assert after.password_hash == "$argon2id$placeholder"  # the password is not the second factor
    assert await count(auth_db, AuthRecoveryCode, user_id=target.id) == 0
    assert await count(auth_db, AuthRecoveryCode, user_id=other.id) == 2

    ended = await sessions_of(auth_db, target.id)
    assert len(ended) == 2 and all(s.revoked_at is not None for s in ended)
    assert {s.revoked_reason for s in ended} == {"mfa_reset"}
    [theirs] = await sessions_of(auth_db, other.id)
    assert theirs.revoked_at is None
    assert (
        await admin_client.get(ME, headers=bearer(other_session.access_token))
    ).status_code == 200

    [event] = await events_of(auth_db, target.id, AuthEventType.MFA_RESET_BY_ADMIN)
    assert event.details == {"actor_id": str(admin_user.id), "sessions_revoked": 2}
    assert event.ip == "127.0.0.1"
    assert (await reload(auth_db, other.id)).mfa_enabled_at is not None

    [mail] = smtp_capture.messages
    assert mail.to == target.email
    assert "Two-factor authentication was reset" in mail.subject


async def test_mfa_reset_forces_reenrollment(
    admin_client, auth_db, local_session_for, settings_override
):
    # A regular account signs in: AUTH_MODE=local (the pre-launch admits administrators only).
    settings_override(AUTH_MODE="local", AUTH_COOKIE_SECURE=False, AUTH_COOKIE_DOMAIN=None)
    target = await enrolled(auth_db, password_hash=hash_password(PW))
    issued = await local_session_for(target)
    me = bearer(issued.access_token)
    assert (await admin_client.get(ME, headers=me)).status_code == 200

    r = await admin_client.post(url(target.id, "mfa/reset"))
    assert r.status_code == 200, r.text

    # The session that existed is over...
    assert (await admin_client.get(ME, headers=me)).status_code == 401
    # ...and the next sign-in is an enrolment, not a code prompt.
    login = await admin_client.post(
        f"{API}/auth/login", json={"email": target.email, "password": PW}, headers=CSRF
    )
    assert login.status_code == 200, login.text
    assert login.json() == {"next": "mfa_setup"}


async def test_mfa_reset_cancels_a_challenge_that_was_already_open(
    admin_client, auth_db, settings_override, enc_keys
):
    """Someone half-way through a sign-in (password done, code pending) cannot finish it with the
    factor the administrator just removed."""
    import pyotp

    # A regular account signs in: AUTH_MODE=local (the pre-launch admits administrators only).
    settings_override(AUTH_MODE="local", AUTH_COOKIE_SECURE=False, AUTH_COOKIE_DOMAIN=None)
    secret = new_totp_secret()
    target = await enrolled(
        auth_db, password_hash=hash_password(PW), mfa_totp_secret_enc=encrypt_secret(secret)
    )
    login = await admin_client.post(
        f"{API}/auth/login", json={"email": target.email, "password": PW}, headers=CSRF
    )
    assert login.json() == {"next": "mfa"}

    assert (await admin_client.post(url(target.id, "mfa/reset"))).status_code == 200

    verify = await admin_client.post(
        f"{API}/auth/mfa/verify", json={"code": pyotp.TOTP(secret).now()}, headers=CSRF
    )
    assert verify.status_code == 401
    assert verify.json() == {"detail": "challenge_expired"}


@pytest.mark.parametrize(
    "caller_role,target_role,allowed",
    [
        (UserRole.ADMIN, UserRole.ADMIN, False),
        (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN, False),
        (UserRole.ADMIN, UserRole.USER, True),
        (UserRole.ADMIN, UserRole.ANALYST, True),
        (UserRole.SCILICIUM_ADMIN, UserRole.ADMIN, True),
        (UserRole.SCILICIUM_ADMIN, UserRole.SCILICIUM_ADMIN, True),
        (UserRole.SCILICIUM_ADMIN, UserRole.USER, True),
    ],
)
async def test_only_a_scilicium_admin_resets_an_admins_mfa(
    admin_client, auth_db, local_session_for, smtp_capture, caller_role, target_role, allowed
):
    _, headers = await caller(local_session_for, auth_db, caller_role)
    target = await enrolled(auth_db, role=target_role)
    await local_session_for(target)

    r = await admin_client.post(url(target.id, "mfa/reset"), headers=headers)

    after = await reload(auth_db, target.id)
    [session] = await sessions_of(auth_db, target.id)
    recorded = await events_of(auth_db, target.id, AuthEventType.MFA_RESET_BY_ADMIN)
    if allowed:
        assert r.status_code == 200, r.text
        assert after.mfa_enabled_at is None and session.revoked_at is not None
        assert len(recorded) == 1 and len(smtp_capture.messages) == 1
    else:
        assert r.status_code == 403
        assert r.json() == {"detail": "scilicium_admin_required"}
        assert_second_factor_intact(after)
        assert session.revoked_at is None
        assert recorded == [] and smtp_capture.messages == []


async def test_admin_cannot_reset_another_admins_mfa(admin_client, auth_db, local_session_for):
    _, headers = await caller(local_session_for, auth_db, UserRole.ADMIN)
    target = await enrolled(auth_db, role=UserRole.ADMIN)

    r = await admin_client.post(url(target.id, "mfa/reset"), headers=headers)

    assert r.status_code == 403
    assert r.json() == {"detail": "scilicium_admin_required"}
    assert_second_factor_intact(await reload(auth_db, target.id))


@pytest.mark.parametrize(
    "role,allowed", [(UserRole.ADMIN, False), (UserRole.SCILICIUM_ADMIN, True)]
)
async def test_an_administrator_resetting_their_own_mfa_follows_the_same_rule(
    admin_client, auth_db, local_session_for, role, allowed
):
    me = await enrolled(auth_db, role=role)
    headers = bearer((await local_session_for(me)).access_token)

    r = await admin_client.post(url(me.id, "mfa/reset"), headers=headers)

    assert r.status_code == (200 if allowed else 403), r.text
    assert ((await reload(auth_db, me.id)).mfa_enabled_at is None) is allowed


# ── The same rule for the sessions and the password link ─────────────────────

PRIVILEGE_CASES = [
    (UserRole.ADMIN, UserRole.ADMIN, False),
    (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN, False),
    (UserRole.ADMIN, UserRole.USER, True),
    (UserRole.SCILICIUM_ADMIN, UserRole.ADMIN, True),
    (UserRole.SCILICIUM_ADMIN, UserRole.SCILICIUM_ADMIN, True),
    (UserRole.SCILICIUM_ADMIN, UserRole.USER, True),
]


async def plant_service_token(auth_db, owner: User) -> None:
    """A live service token of `owner`, written directly: the command refuses an administrator as
    owner, but a token minted before a promotion still exists."""
    async with auth_db() as s:
        s.add(
            ServiceToken(
                name="before-promotion",
                owner_id=owner.id,
                token_sha256=uuid4().hex + uuid4().hex,
                scopes=["projects:create"],
            )
        )
        await s.commit()


@pytest.mark.parametrize("box", [False, True], ids=["sessions", "sessions_and_tokens"])
@pytest.mark.parametrize("caller_role,target_role,allowed", PRIVILEGE_CASES)
async def test_only_a_scilicium_admin_ends_an_admins_sessions(
    admin_client, auth_db, local_session_for, caller_role, target_role, allowed, box
):
    _, headers = await caller(local_session_for, auth_db, caller_role)
    target = await seed_user(auth_db, role=target_role)
    await local_session_for(target)
    await plant_service_token(auth_db, target)

    r = await admin_client.post(
        url(target.id, "sessions/revoke"), headers=headers, json={"revoke_service_tokens": box}
    )

    [session] = await sessions_of(auth_db, target.id)
    [token] = await token_rows(auth_db, target.id)
    if allowed:
        assert r.status_code == 200, r.text
        assert r.json() == {"sessions_revoked": 1, "service_tokens_revoked": int(box)}
        assert session.revoked_at is not None
        assert (token.revoked_at is not None) is box
    else:
        assert r.status_code == 403
        assert r.json() == {"detail": "scilicium_admin_required"}
        assert session.revoked_at is None and token.revoked_at is None
        assert await count(auth_db, AuthEvent, user_id=target.id) == 0


@pytest.mark.parametrize("caller_role,target_role,allowed", PRIVILEGE_CASES)
async def test_only_a_scilicium_admin_sends_an_admin_a_password_link(
    admin_client,
    auth_db,
    local_session_for,
    smtp_capture,
    settings_override,
    caller_role,
    target_role,
    allowed,
):
    settings_override(AUTH_MODE="local")  # a password link is sent from the switch on
    _, headers = await caller(local_session_for, auth_db, caller_role)
    target = await seed_user(auth_db, role=target_role)

    r = await admin_client.post(url(target.id, "password-reset"), headers=headers)

    links = [t for t in await _all(auth_db, AuthOneTimeToken) if t.user_id == target.id]
    if allowed:
        assert r.status_code == 200 and r.json() == {"email_sent": True}, r.text
        assert len(links) == 1 and [m.to for m in smtp_capture.messages] == [target.email]
    else:
        assert r.status_code == 403
        assert r.json() == {"detail": "scilicium_admin_required"}
        assert links == [] and smtp_capture.messages == []
        assert await count(auth_db, AuthEvent, user_id=target.id) == 0


@pytest.mark.parametrize("suffix", ["sessions/revoke", "password-reset"])
async def test_an_admin_acting_on_their_own_account_follows_the_same_rule(
    admin_client, auth_db, local_session_for, smtp_capture, suffix
):
    me, headers = await caller(local_session_for, auth_db, UserRole.ADMIN)

    r = await admin_client.post(url(me.id, suffix), headers=headers)

    assert r.status_code == 403 and r.json() == {"detail": "scilicium_admin_required"}
    assert [s.revoked_at for s in await sessions_of(auth_db, me.id)] == [None]
    assert smtp_capture.messages == []


async def test_any_administrator_unlocks_an_administrator(admin_client, auth_db, local_session_for):
    _, headers = await caller(local_session_for, auth_db, UserRole.ADMIN)
    target = await seed_user(
        auth_db,
        role=UserRole.SCILICIUM_ADMIN,
        failed_login_count=10,
        locked_until=now_utc() + timedelta(minutes=15),
    )

    r = await admin_client.post(url(target.id, "unlock"), headers=headers)

    assert r.status_code == 200, r.text
    after = await reload(auth_db, target.id)
    assert after.failed_login_count == 0 and after.locked_until is None


@pytest.mark.parametrize("failure", ["send_fails", "not_configured"])
async def test_mfa_reset_stands_when_the_notification_cannot_be_sent(
    admin_client, auth_db, smtp_capture, settings_override, failure
):
    if failure == "send_fails":
        smtp_capture.fail_with = OSError("smtp down")
    else:
        settings_override(SMTP_HOST=None)
    target = await enrolled(auth_db)
    await give_recovery_codes(auth_db, target)

    r = await admin_client.post(url(target.id, "mfa/reset"))

    assert r.status_code == 200, r.text
    assert r.json()["notification_sent"] is False
    after = await reload(auth_db, target.id)
    assert after.mfa_enabled_at is None and after.mfa_totp_secret_enc is None
    assert await count(auth_db, AuthRecoveryCode, user_id=target.id) == 0
    unsent = [
        e
        for e in await events_of(auth_db, target.id, AuthEventType.MFA_RESET_BY_ADMIN)
        if e.details.get("notification") == "not_sent"
    ]
    assert len(unsent) == 1 and unsent[0].details["error_kind"].startswith("Email")


# ── Ending sessions, and service tokens only when asked ──────────────────────


async def test_revoke_sessions_keeps_service_tokens_unless_asked(
    admin_client, auth_db, local_session_for
):
    target = await seed_user(auth_db)
    live = await local_session_for(target)
    row, token = await mint_token(auth_db, target)
    me = bearer(live.access_token)
    assert (await admin_client.get(ME, headers=me)).status_code == 200
    assert await token_status(admin_client, token) == 201

    r = await admin_client.post(url(target.id, "sessions/revoke"))

    assert r.status_code == 200, r.text
    assert r.json() == {"sessions_revoked": 1, "service_tokens_revoked": 0}
    assert (await admin_client.get(ME, headers=me)).status_code == 401
    assert await token_status(admin_client, token) == 201  # the pipeline keeps running
    assert [t.revoked_at for t in await token_rows(auth_db, target.id)] == [None]

    r = await admin_client.post(
        url(target.id, "sessions/revoke"), json={"revoke_service_tokens": True}
    )

    assert r.status_code == 200, r.text
    assert r.json() == {"sessions_revoked": 0, "service_tokens_revoked": 1}
    assert await token_status(admin_client, token) == 401


async def test_revoke_sessions_counts_audits_and_spares_other_accounts(
    admin_client, admin_user, auth_db, local_session_for
):
    target = await seed_user(auth_db)
    other = await seed_user(auth_db)
    for _ in range(3):
        await local_session_for(target)
    other_session = await local_session_for(other)
    first, _ = await mint_token(auth_db, target, name="first")
    await mint_token(auth_db, target, name="second")
    await mint_token(auth_db, other, name="theirs")
    # A token that was revoked long ago keeps its own date and is not counted again.
    long_ago = now_utc() - timedelta(days=30)
    async with auth_db() as s:
        (await s.get(ServiceToken, first.id)).revoked_at = long_ago
        await s.commit()
    # So does a session that already ended.
    async with auth_db() as s:
        ended = (await sessions_of(auth_db, target.id))[0]
        (await s.get(AuthSession, ended.id)).revoked_at = long_ago
        await s.commit()

    r = await admin_client.post(
        url(target.id, "sessions/revoke"), json={"revoke_service_tokens": True}
    )

    assert r.json() == {"sessions_revoked": 2, "service_tokens_revoked": 1}
    sessions = await sessions_of(auth_db, target.id)
    assert all(s.revoked_at is not None for s in sessions)
    assert Counter(s.revoked_reason for s in sessions) == {"admin_revoked": 2, None: 1}
    [theirs] = await sessions_of(auth_db, other.id)
    assert theirs.revoked_at is None
    assert (
        await admin_client.get(ME, headers=bearer(other_session.access_token))
    ).status_code == 200
    by_name = {t.name: t for t in await token_rows(auth_db, target.id)}
    assert by_name["second"].revoked_at is not None
    assert aware_utc(by_name["first"].revoked_at) == long_ago
    assert all(t.revoked_at is None for t in await token_rows(auth_db, other.id))

    [ended_event] = await events_of(auth_db, target.id, AuthEventType.SESSIONS_REVOKED_ALL)
    assert ended_event.details == {"actor_id": str(admin_user.id), "count": 2}
    [token_event] = await events_of(auth_db, target.id, AuthEventType.SERVICE_TOKENS_REVOKED)
    assert token_event.details == {"actor_id": str(admin_user.id), "count": 1}


async def test_revoke_sessions_without_the_box_records_no_token_event(admin_client, auth_db):
    target = await seed_user(auth_db)
    r = await admin_client.post(url(target.id, "sessions/revoke"), json={})
    assert r.status_code == 200 and r.json() == {"sessions_revoked": 0, "service_tokens_revoked": 0}
    assert await events_of(auth_db, target.id, AuthEventType.SERVICE_TOKENS_REVOKED) == []
    assert len(await events_of(auth_db, target.id, AuthEventType.SESSIONS_REVOKED_ALL)) == 1


async def test_revoke_sessions_refuses_a_box_that_is_not_a_boolean(admin_client, auth_db):
    target = await seed_user(auth_db)
    r = await admin_client.post(
        url(target.id, "sessions/revoke"), json={"revoke_service_tokens": "banana"}
    )
    assert r.status_code == 422


@pytest.mark.parametrize("suffix", ["mfa/reset", "unlock", "password-reset", "sessions/revoke"])
async def test_nothing_but_the_explicit_box_revokes_a_service_token(
    admin_client, auth_db, smtp_capture, settings_override, suffix
):
    settings_override(AUTH_MODE="local")  # where every one of these actions is open
    target = await enrolled(auth_db, locked_until=now_utc() + timedelta(minutes=10))
    _, token = await mint_token(auth_db, target)

    r = await admin_client.post(url(target.id, suffix))

    assert r.status_code == 200, r.text
    assert [t.revoked_at for t in await token_rows(auth_db, target.id)] == [None]
    assert await token_status(admin_client, token) == 201
    assert await events_of(auth_db, target.id, AuthEventType.SERVICE_TOKENS_REVOKED) == []


# ── Unlocking ────────────────────────────────────────────────────────────────


async def test_unlock_clears_the_lock_and_audits_it(
    admin_client, admin_user, auth_db, local_session_for, settings_override
):
    settings_override(AUTH_MODE="local")  # a regular account signs in (not in the pre-launch)
    target = await seed_user(
        auth_db,
        password_hash=hash_password(PW),
        email_verified_at=NOW,
        failed_login_count=10,
        locked_until=now_utc() + timedelta(minutes=15),
    )
    session = await local_session_for(target)
    credentials = {"email": target.email, "password": PW}
    refused = await admin_client.post(f"{API}/auth/login", json=credentials, headers=CSRF)
    assert refused.status_code == 429 and refused.json() == {"detail": "too_many_attempts"}

    r = await admin_client.post(url(target.id, "unlock"))

    assert r.status_code == 200, r.text
    after = await reload(auth_db, target.id)
    assert after.failed_login_count == 0 and after.locked_until is None
    [event] = await events_of(auth_db, target.id, AuthEventType.ACCOUNT_UNLOCKED)
    assert event.details == {"actor_id": str(admin_user.id)}
    # The sign-in works again, and the unlock ended nothing that was open.
    retry = await admin_client.post(f"{API}/auth/login", json=credentials, headers=CSRF)
    assert retry.status_code == 200 and retry.json() == {"next": "mfa_setup"}
    [open_session] = await sessions_of(auth_db, target.id)
    assert open_session.revoked_at is None
    assert (await admin_client.get(ME, headers=bearer(session.access_token))).status_code == 200


# ── Sending a password-reset link ────────────────────────────────────────────


async def test_password_reset_sends_the_link_the_account_would_ask_for(
    admin_client, admin_user, auth_db, local_session_for, smtp_capture, settings_override
):
    settings_override(AUTH_MODE="local")
    target = await enrolled(auth_db, password_hash="$argon2id$placeholder")
    live = await local_session_for(target)

    r = await admin_client.post(url(target.id, "password-reset"))

    assert r.status_code == 200, r.text
    assert r.json() == {"email_sent": True}
    [token] = [
        t
        for t in await _all(auth_db, AuthOneTimeToken)
        if t.user_id == target.id and t.purpose is Purpose.PASSWORD_RESET
    ]
    assert token.created_by == admin_user.id and token.used_at is None
    assert aware_utc(token.expires_at) > now_utc()
    [mail] = smtp_capture.messages
    assert mail.to == target.email
    assert token.token_hash == hash_token(reset_link_token(mail.text))
    [event] = await events_of(auth_db, target.id, AuthEventType.PASSWORD_RESET_REQUESTED)
    assert event.details == {"requested_by": str(admin_user.id)}
    # Sending a link changes nothing about the account until it is used.
    after = await reload(auth_db, target.id)
    assert after.password_hash == "$argon2id$placeholder" and after.mfa_enabled_at is not None
    [open_session] = await sessions_of(auth_db, target.id)
    assert open_session.revoked_at is None
    assert (await admin_client.get(ME, headers=bearer(live.access_token))).status_code == 200


async def test_password_reset_needs_local_authentication(
    auth_client, auth_db, admin_user, supabase_token_for, smtp_capture, settings_override
):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=False)
    target = await seed_user(auth_db)
    headers = bearer(supabase_token_for(admin_user))

    r = await auth_client.post(url(target.id, "password-reset"), headers=headers)

    assert r.status_code == 409
    assert r.json() == {"detail": "local_auth_disabled"}
    assert await _all(auth_db, AuthOneTimeToken) == []
    assert smtp_capture.messages == []
    # Every other action still works on the local data.
    assert (await auth_client.post(url(target.id, "unlock"), headers=headers)).status_code == 200


async def test_password_reset_is_refused_before_the_launch(
    admin_client, auth_db, smtp_capture, settings_override
):
    """Pre-launch (AUTH_MODE=supabase, AUTH_LOCAL_ENABLED=true): the link would lead to
    `/auth/password/reset`, which answers 404 until AUTH_MODE=local. Refused, nothing issued."""
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=True)
    target = await seed_user(auth_db)

    r = await admin_client.post(url(target.id, "password-reset"))

    assert r.status_code == 409
    assert r.json() == {"detail": "local_auth_disabled"}
    assert await _all(auth_db, AuthOneTimeToken) == []
    assert smtp_capture.messages == []
    assert await count(auth_db, AuthEvent, user_id=target.id) == 0


@pytest.mark.parametrize("failure", ["send_fails", "not_configured"])
async def test_password_reset_reports_an_email_that_cannot_leave_and_keeps_the_link(
    admin_client, auth_db, smtp_capture, settings_override, failure
):
    settings_override(AUTH_MODE="local")
    target = await seed_user(auth_db)
    if failure == "send_fails":
        smtp_capture.fail_with = OSError("smtp down")
    else:
        settings_override(SMTP_HOST=None)

    r = await admin_client.post(url(target.id, "password-reset"))

    assert r.status_code == 502
    assert r.json() == {"detail": "reset_email_failed"}
    [kept] = await _all(auth_db, AuthOneTimeToken)
    assert kept.purpose is Purpose.PASSWORD_RESET and kept.used_at is None
    unsent = [
        e
        for e in await events_of(auth_db, target.id, AuthEventType.PASSWORD_RESET_REQUESTED)
        if e.details.get("notification") == "not_sent"
    ]
    assert len(unsent) == 1

    # The mail server is back: a retry sends a fresh link, and only that one works.
    smtp_capture.fail_with = None
    settings_override(SMTP_HOST="smtp.example.test")
    again = await admin_client.post(url(target.id, "password-reset"))
    assert again.status_code == 200, again.text
    tokens = await _all(auth_db, AuthOneTimeToken)
    assert [t.used_at is None for t in tokens].count(True) == 1
    live = next(t for t in tokens if t.used_at is None)
    assert live.token_hash == hash_token(reset_link_token(smtp_capture.last_text))


# ── Reading the audit trail and the service tokens ───────────────────────────


async def test_auth_events_are_bounded_newest_first_and_the_account_s_own(admin_client, auth_db):
    target = await seed_user(auth_db)
    other = await seed_user(auth_db)
    async with auth_db() as s:
        for i in range(60):
            await record_event(s, AuthEventType.LOGIN_FAILED, user_id=target.id, n=i)
        await record_event(s, AuthEventType.LOGIN_FAILED, user_id=other.id, n=-1)
        await s.commit()

    r = await admin_client.get(url(target.id, "auth-events"))
    assert r.status_code == 200, r.text
    rows = r.json()
    assert [row["details"]["n"] for row in rows] == list(range(59, 9, -1))  # 50, newest first
    assert set(rows[0]) == {"event_type", "created_at", "ip", "user_agent", "details"}
    assert rows[0]["event_type"] == "login_failed"

    r = await admin_client.get(url(target.id, "auth-events"), params={"limit": 5})
    assert [row["details"]["n"] for row in r.json()] == [59, 58, 57, 56, 55]
    r = await admin_client.get(url(target.id, "auth-events"), params={"limit": 200})
    assert r.status_code == 200 and len(r.json()) == 60
    for bad in (201, 0, -1):
        r = await admin_client.get(url(target.id, "auth-events"), params={"limit": bad})
        assert r.status_code == 422, bad


async def test_auth_events_carry_the_ip_and_user_agent_of_the_request(admin_client, auth_db):
    target = await enrolled(auth_db)
    await admin_client.post(url(target.id, "unlock"), headers={"User-Agent": "admin-panel/1.0"})

    [row] = (await admin_client.get(url(target.id, "auth-events"))).json()

    assert row["event_type"] == "account_unlocked"
    assert row["ip"] == "127.0.0.1" and row["user_agent"] == "admin-panel/1.0"


async def test_events_and_tokens_listing_never_leak_secrets(admin_client, auth_db, smtp_capture):
    target = await enrolled(auth_db, password_hash="$argon2id$placeholder")
    other = await seed_user(auth_db)
    row, token = await mint_token(auth_db, target, name="pipeline")
    revoked_row, revoked_token = await mint_token(
        auth_db, target, name="old", scopes=("datasets:upload",)
    )
    await mint_token(auth_db, other, name="not-theirs")
    async with auth_db() as s:
        (await s.get(ServiceToken, revoked_row.id)).revoked_at = now_utc()
        (await s.get(ServiceToken, row.id)).last_used_at = now_utc()
        await record_event(
            s,
            AuthEventType.LOGIN_FAILED,
            user_id=target.id,
            password="hunter2",
            mfa_code="otp-914273",
            raw_token=token,
            note="kept",
        )
        await s.commit()
    await admin_client.post(url(target.id, "mfa/reset"))

    events = await admin_client.get(url(target.id, "auth-events"))
    tokens = await admin_client.get(url(target.id, "service-tokens"))

    assert events.status_code == 200 and tokens.status_code == 200
    for text in (events.text, tokens.text):
        assert "token_sha256" not in text and "glst_" not in text
        assert token not in text and revoked_token not in text
        assert row.token_sha256 not in text and revoked_row.token_sha256 not in text
        assert "hunter2" not in text and "otp-914273" not in text
        assert SECRET_PLACEHOLDER not in text and PENDING_PLACEHOLDER not in text
        assert "$argon2id" not in text
    [with_note] = [e for e in events.json() if e["details"].get("note") == "kept"]
    assert set(with_note["details"]) == {"note"}

    listed = tokens.json()
    assert {t["name"] for t in listed} == {"pipeline", "old"}
    for t in listed:
        assert set(t) == {"id", "name", "scopes", "created_at", "last_used_at", "revoked_at"}
    by_name = {t["name"]: t for t in listed}
    assert by_name["pipeline"]["scopes"] == ["projects:create"]
    assert by_name["pipeline"]["revoked_at"] is None and by_name["pipeline"]["last_used_at"]
    assert by_name["old"]["scopes"] == ["datasets:upload"] and by_name["old"]["revoked_at"]


async def test_service_token_list_of_an_account_without_any_is_empty(admin_client, auth_db):
    target = await seed_user(auth_db)
    r = await admin_client.get(url(target.id, "service-tokens"))
    assert r.status_code == 200 and r.json() == []


# ── Both modes, and never the hosted provider ────────────────────────────────


@MODES
async def test_security_actions_work_on_local_data_and_never_call_the_provider(
    admin_client, auth_db, local_session_for, smtp_capture, fake_supabase, settings_override, mode
):
    settings_override(AUTH_MODE=mode)
    target = await enrolled(auth_db, locked_until=now_utc() + timedelta(minutes=5))
    await local_session_for(target)
    await mint_token(auth_db, target)

    for method, suffix in ROUTES:
        r = await admin_client.request(method, url(target.id, suffix))
        # The password link waits for AUTH_MODE=local (its page answers 404 before the switch).
        expected = 409 if (mode, suffix) == ("supabase", "password-reset") else 200
        assert r.status_code == expected, (mode, suffix, r.text)

    assert fake_supabase.calls == []
    after = await reload(auth_db, target.id)
    assert after.mfa_enabled_at is None and after.locked_until is None


# ── The shared logic (also called by the break-glass command) ────────────────


async def test_reset_user_mfa_records_the_given_event_and_leaves_the_commit_to_the_caller(auth_db):
    target = await enrolled(auth_db)
    await give_recovery_codes(auth_db, target)
    async with auth_db() as s:
        s.add(
            AuthSession(
                user_id=target.id,
                idle_expires_at=NOW,
                expires_at=NOW,
                refresh_token_hash=uuid4().hex + uuid4().hex,
                amr=["pwd"],
            )
        )
        await s.commit()

    async with auth_db() as s:
        user = await s.get(User, target.id)
        revoked = await reset_user_mfa(s, user, actor_id=None, event=AuthEventType.BREAK_GLASS)
        assert revoked == 1
        assert user.mfa_enabled_at is None and user.mfa_totp_secret_enc is None
        await s.rollback()  # the caller did not commit: nothing may have been persisted
    assert_second_factor_intact(await reload(auth_db, target.id))
    assert await count(auth_db, AuthRecoveryCode, user_id=target.id) == 3
    assert await events_of(auth_db, target.id, AuthEventType.BREAK_GLASS) == []

    async with auth_db() as s:
        user = await s.get(User, target.id)
        await reset_user_mfa(s, user, actor_id=None, event=AuthEventType.BREAK_GLASS)
        await s.commit()
    [event] = await events_of(auth_db, target.id, AuthEventType.BREAK_GLASS)
    assert event.details["actor_id"] is None
    assert await events_of(auth_db, target.id, AuthEventType.MFA_RESET_BY_ADMIN) == []
    assert (await reload(auth_db, target.id)).mfa_enabled_at is None


async def test_unlock_user_leaves_the_commit_to_the_caller(auth_db):
    target = await seed_user(
        auth_db,
        failed_login_count=7,
        mfa_failed_count=12,
        locked_until=now_utc() + timedelta(minutes=5),
    )
    actor = uuid4()

    async with auth_db() as s:
        user = await s.get(User, target.id)
        await unlock_user(s, user, actor_id=actor)
        assert user.failed_login_count == 0 and user.locked_until is None
        assert user.mfa_failed_count == 0
        await s.rollback()
    assert (await reload(auth_db, target.id)).failed_login_count == 7

    async with auth_db() as s:
        user = await s.get(User, target.id)
        await unlock_user(s, user, actor_id=actor)
        await s.commit()
    after = await reload(auth_db, target.id)
    assert after.failed_login_count == 0 and after.locked_until is None
    assert after.mfa_failed_count == 0  # the administrator lets the owner try codes again
    [event] = await events_of(auth_db, target.id, AuthEventType.ACCOUNT_UNLOCKED)
    assert event.details == {"actor_id": str(actor)}


async def test_revoke_helpers_count_only_what_was_live_and_do_not_commit(auth_db):
    target = await seed_user(auth_db)
    await mint_token(auth_db, target, name="a")
    await mint_token(auth_db, target, name="b")

    async with auth_db() as s:
        assert await revoke_user_service_tokens(s, target.id, actor_id=None) == 2
        assert await revoke_user_service_tokens(s, target.id, actor_id=None) == 0
        assert await revoke_user_sessions(s, target.id, actor_id=None) == 0
        await s.rollback()
    assert [t.revoked_at for t in await token_rows(auth_db, target.id)] == [None, None]
    assert await events_of(auth_db, target.id, AuthEventType.SERVICE_TOKENS_REVOKED) == []
