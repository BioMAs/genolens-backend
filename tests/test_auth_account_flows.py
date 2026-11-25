"""
Account flows of the local authentication (`/api/v1/auth`), end to end: sign-up, e-mail
verification, forgotten and changed passwords, invitations, the caller's own sessions, the rotation
of the second factor, and what `/users/me` says about it.

Every API test runs the real chain (`auth_client`) against an in-memory SQLite database, with
freshly generated signing and encryption keys and every e-mail captured (`local_auth`).

What is pinned down:
- anti-enumeration: sign-up, forgot and verification resend answer the same `202` whether the
  address has an account or not, even when SMTP is down; both sign-up branches hash the password;
- at most three link e-mails per address and per hour, and five requests per IP and per hour;
- a password reset revokes every session and opens none: the second factor is still asked at the
  next sign-in;
- an invitation is accepted once, activates a pending account and leads to the enrolment;
- the session routes need a local session (never a service token nor a hosted-provider token), a
  wrong current password or code counts toward the account lock, and a session of another account
  is invisible.
"""

import re
from datetime import timedelta
from urllib.parse import unquote

import pytest
from sqlalchemy import func, select

from app.auth.clock import now_utc
from app.auth.mfa import recovery_code_matches
from app.auth.one_time_tokens import issue_token
from app.auth.passwords import hash_password, verify_password
from app.models.auth import AuthEventType, AuthOneTimeToken, AuthRecoveryCode, AuthSession
from app.models.auth import OneTimeTokenPurpose as Purpose
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus
from tests.auth_support import CSRF, PW, access_cookie, bearer, call_app_directly, seed_user
from tests.test_auth_login_flow import (
    CHALLENGE_EXPIRED,
    SETUP_CONFIRM,
    SETUP_START,
    code_at,
    events,
    expire_lock,
    fail_codes,
    give_recovery_codes,
    login,
    mfa_user,
    reload,
    sessions_of,
    set_cookies,
    verified_user,
    verify,
    wrong_code,
)

AUTH = "/api/v1/auth"
SIGNUP = f"{AUTH}/signup"
VERIFY_EMAIL = f"{AUTH}/verify-email"
RESEND = f"{AUTH}/verify-email/resend"
FORGOT = f"{AUTH}/password/forgot"
RESET = f"{AUTH}/password/reset"
INVITE_ACCEPT = f"{AUTH}/invite/accept"
CHANGE = f"{AUTH}/password/change"
SESSIONS = f"{AUTH}/sessions"
REVOKE_ALL = f"{AUTH}/sessions/revoke-all"
ROTATE_START = f"{AUTH}/mfa/rotate/start"
ROTATE_CONFIRM = f"{AUTH}/mfa/rotate/confirm"
REGENERATE = f"{AUTH}/mfa/recovery-codes/regenerate"
ME = "/api/v1/users/me"

NEW_PW = "Quarry-Heron-Lattice-58"
WRONG_PW = "Wrong-Password-Again-42"
ACCEPTED = {"status": "accepted"}
INVALID_TOKEN = {"detail": "invalid_token"}
INVALID_CREDENTIALS = {"detail": "invalid_credentials"}
TOO_MANY_ATTEMPTS = {"detail": "too_many_attempts"}
LOCAL_SESSION_REQUIRED = {"detail": "local_session_required"}

PUBLIC_ROUTES = (SIGNUP, VERIFY_EMAIL, RESEND, FORGOT, RESET, INVITE_ACCEPT)
SESSION_ROUTES = (
    ("POST", CHANGE),
    ("GET", SESSIONS),
    ("DELETE", f"{SESSIONS}/00000000-0000-0000-0000-000000000000"),
    ("POST", REVOKE_ALL),
    ("POST", ROTATE_START),
    ("POST", ROTATE_CONFIRM),
    ("POST", REGENERATE),
)


@pytest.fixture
def local_auth(settings_override, local_keys, enc_keys, smtp_capture):
    """Local authentication on, cookies usable over plain HTTP, every e-mail captured."""
    settings_override(AUTH_MODE="local", AUTH_COOKIE_SECURE=False, AUTH_COOKIE_DOMAIN=None)
    return smtp_capture


@pytest.fixture
def no_ip_limit(monkeypatch):
    from app.middleware.rate_limit import limiter

    monkeypatch.setattr(limiter, "enabled", False)


# ── Helpers ──────────────────────────────────────────────────────────────────


def link_token(text: str, path: str) -> str:
    """The token of the `<path>#token=…` link in an e-mail body."""
    match = re.search(re.escape(path) + r"#token=([^\s\"<]+)", text)
    assert match, f"no {path} link in the e-mail"
    return unquote(match.group(1))


async def post(client, path: str, body: dict, headers: dict = CSRF):
    return await client.post(path, json=body, headers=headers)


async def call(client, method: str, path: str, headers: dict, body: dict | None = None):
    return await client.request(method, path, json=body, headers=headers)


async def user_by_email(auth_db, email: str):
    async with auth_db() as s:
        return (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()


async def live_sessions(auth_db, user_id) -> list[AuthSession]:
    return [s for s in await sessions_of(auth_db, user_id) if s.revoked_at is None]


async def stored_codes(auth_db, user_id) -> list[AuthRecoveryCode]:
    async with auth_db() as s:
        rows = await s.execute(select(AuthRecoveryCode).where(AuthRecoveryCode.user_id == user_id))
        return list(rows.scalars())


async def token_used(auth_db, user_id, purpose) -> bool:
    """Is every token of `purpose` issued to the account spent?"""
    async with auth_db() as s:
        rows = await s.execute(
            select(AuthOneTimeToken.used_at).where(
                AuthOneTimeToken.user_id == user_id, AuthOneTimeToken.purpose == purpose
            )
        )
        used = list(rows.scalars())
    return bool(used) and all(at is not None for at in used)


async def invite_token(auth_db, user) -> str:
    async with auth_db() as s:
        raw = await issue_token(s, user.id, Purpose.INVITE)
        await s.commit()
    return raw


# ── Sign-up ──────────────────────────────────────────────────────────────────


async def test_signup_anti_enumeration(auth_client, auth_db, local_auth):
    existing = await verified_user(auth_db)

    new = await post(auth_client, SIGNUP, {"email": "n@example.test", "password": PW})
    dup = await post(auth_client, SIGNUP, {"email": existing.email, "password": NEW_PW})

    assert new.status_code == dup.status_code == 202, (new.text, dup.text)
    assert new.json() == dup.json() == ACCEPTED
    assert [m.to for m in local_auth.messages] == ["n@example.test", existing.email]
    assert "/auth/verify-email#token=" in local_auth.messages[0].text
    assert "/auth/reset#token=" in local_auth.messages[1].text
    assert "already have" in local_auth.messages[1].text

    created = await user_by_email(auth_db, "n@example.test")
    assert created.role == UserRole.USER
    assert created.subscription_plan == SubscriptionPlan.STARTER
    assert created.status == UserStatus.ACTIVE
    assert created.email_verified_at is None and created.mfa_enabled_at is None
    assert verify_password(PW, created.password_hash).ok
    # The existing account is untouched: its password is not the one typed at sign-up.
    assert verify_password(PW, (await reload(auth_db, existing.id)).password_hash).ok
    signups = await events(auth_db, AuthEventType.SIGNUP)
    assert {e.user_id for e in signups} == {created.id, existing.id}


async def test_signup_hashes_the_password_in_both_branches(
    auth_client, auth_db, local_auth, monkeypatch
):
    """The same work whether the address is known or not, and off the event loop."""
    from starlette.concurrency import run_in_threadpool

    import app.api.endpoints.auth_account as account_endpoints

    seen = []

    async def spy(func, *args, **kwargs):
        seen.append(func)
        return await run_in_threadpool(func, *args, **kwargs)

    monkeypatch.setattr(account_endpoints, "run_in_threadpool", spy)
    existing = await verified_user(auth_db)

    await post(auth_client, SIGNUP, {"email": "fresh@example.test", "password": PW})
    first = list(seen)
    await post(auth_client, SIGNUP, {"email": existing.email, "password": PW})

    assert first == [hash_password]
    assert seen == [hash_password, hash_password]


async def test_signup_refuses_a_weak_password_whether_the_address_is_known_or_not(
    auth_client, auth_db, local_auth
):
    existing = await verified_user(auth_db)

    for email in ("weak@example.test", existing.email):
        r = await post(auth_client, SIGNUP, {"email": email, "password": "short"})
        assert r.status_code == 400 and r.json() == {"detail": "password_policy:too_short"}
    r = await post(auth_client, SIGNUP, {"email": "same@example.test", "password": "password1234"})
    assert r.json() == {"detail": "password_policy:too_common"}

    assert local_auth.messages == []
    assert await user_by_email(auth_db, "weak@example.test") is None


async def test_signup_answers_the_same_while_smtp_is_down(auth_client, auth_db, local_auth):
    existing = await verified_user(auth_db)
    local_auth.fail_with = ConnectionRefusedError("down")

    new = await post(auth_client, SIGNUP, {"email": "down@example.test", "password": PW})
    dup = await post(auth_client, SIGNUP, {"email": existing.email, "password": PW})

    assert new.status_code == dup.status_code == 202
    assert new.json() == dup.json() == ACCEPTED
    unsent = [
        e for e in await events(auth_db, AuthEventType.SIGNUP) if e.details.get("notification")
    ]
    assert [e.details["notification"] for e in unsent] == ["not_sent", "not_sent"]
    # The account exists: the link can be asked again once SMTP is back.
    assert await user_by_email(auth_db, "down@example.test") is not None


async def test_signup_normalises_the_address(auth_client, auth_db, local_auth):
    r = await post(auth_client, SIGNUP, {"email": " Mixed.Case@Example.test ", "password": PW})
    assert r.status_code == 202, r.text
    assert await user_by_email(auth_db, "mixed.case@example.test") is not None

    again = await post(auth_client, SIGNUP, {"email": "MIXED.case@example.TEST", "password": PW})
    assert again.status_code == 202
    assert "already have" in local_auth.last_text
    async with auth_db() as s:
        assert await s.scalar(select(func.count()).select_from(User)) == 1


async def test_signup_never_adds_a_third_account_to_a_shared_address(
    auth_client, auth_db, local_auth
):
    """Two migrated accounts differing only by case: neither is told apart, none is added."""
    await seed_user(auth_db, email="Twin@example.test")
    await seed_user(auth_db, email="TWIN@example.test")

    r = await post(auth_client, SIGNUP, {"email": "twin@example.test", "password": PW})

    assert r.status_code == 202 and r.json() == ACCEPTED
    assert local_auth.messages == []
    async with auth_db() as s:
        assert await s.scalar(select(func.count()).select_from(User)) == 2


async def test_signup_then_verify_then_sign_in(auth_client, auth_db, local_auth):
    await post(
        auth_client,
        SIGNUP,
        {"email": "flow@example.test", "password": PW, "full_name": "Ada Example"},
    )
    raw = link_token(local_auth.last_text, "/auth/verify-email")
    assert (await login(auth_client, "flow@example.test")).json() == INVALID_CREDENTIALS

    r = await post(auth_client, VERIFY_EMAIL, {"token": raw})
    assert r.status_code == 204, r.text
    user = await user_by_email(auth_db, "flow@example.test")
    assert user.email_verified_at is not None and user.full_name == "Ada Example"
    assert len(await events(auth_db, AuthEventType.EMAIL_VERIFIED)) == 1

    assert (await post(auth_client, VERIFY_EMAIL, {"token": raw})).json() == INVALID_TOKEN
    r = await login(auth_client, "flow@example.test")
    assert r.status_code == 200 and r.json() == {"next": "mfa_setup"}


async def test_signup_then_login_tells_nothing_about_the_address(auth_client, auth_db, local_auth):
    """An unverified account signs in exactly like a wrong password: sign-up then login with the
    same password cannot tell a new address from a known one."""
    existing = await verified_user(auth_db, password_hash=hash_password(NEW_PW))

    await post(auth_client, SIGNUP, {"email": "unknown@example.test", "password": PW})
    fresh = await login(auth_client, "unknown@example.test")
    await post(auth_client, SIGNUP, {"email": existing.email, "password": PW})
    known = await login(auth_client, existing.email)

    assert fresh.status_code == known.status_code == 401
    assert fresh.json() == known.json() == INVALID_CREDENTIALS
    assert "set-cookie" not in fresh.headers and "set-cookie" not in known.headers


async def test_verify_email_refuses_a_bad_token(auth_client, auth_db, local_auth):
    user = await seed_user(auth_db, password_hash=hash_password(PW))
    async with auth_db() as s:
        reset = await issue_token(s, user.id, Purpose.PASSWORD_RESET)
        await s.commit()

    for token in ("not-a-token", reset):
        r = await post(auth_client, VERIFY_EMAIL, {"token": token})
        assert r.status_code == 400 and r.json() == INVALID_TOKEN
    # Text that is not valid Unicode is refused as malformed, before any lookup.
    assert (await post(auth_client, VERIFY_EMAIL, {"token": "\ud800"})).status_code == 422
    assert (await reload(auth_db, user.id)).email_verified_at is None


async def test_verification_resend(auth_client, auth_db, local_auth):
    unverified = await seed_user(auth_db, password_hash=hash_password(PW))
    verified = await verified_user(auth_db)

    answers = [
        await post(auth_client, RESEND, {"email": email})
        for email in (unverified.email.upper(), verified.email, "nobody@example.test")
    ]

    assert all(r.status_code == 202 and r.json() == ACCEPTED for r in answers)
    [message] = local_auth.messages
    assert message.to == unverified.email
    raw = link_token(message.text, "/auth/verify-email")
    assert (await post(auth_client, VERIFY_EMAIL, {"token": raw})).status_code == 204


# ── Forgotten password, reset ────────────────────────────────────────────────


async def test_forgot_answers_the_same_for_an_unknown_address(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)

    known = await post(auth_client, FORGOT, {"email": user.email})
    unknown = await post(auth_client, FORGOT, {"email": "nobody@example.test"})

    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json() == ACCEPTED
    [message] = local_auth.messages
    assert message.to == user.email and "/auth/reset#token=" in message.text
    [requested] = await events(auth_db, AuthEventType.PASSWORD_RESET_REQUESTED)
    assert requested.user_id == user.id


async def test_forgot_sends_at_most_three_per_hour(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)

    answers = [await post(auth_client, FORGOT, {"email": user.email}) for _ in range(4)]

    assert all(r.status_code == 202 and r.json() == ACCEPTED for r in answers)
    assert len(local_auth.messages) == 3
    throttled = [
        e
        for e in await events(auth_db, AuthEventType.PASSWORD_RESET_REQUESTED)
        if e.details.get("throttled")
    ]
    assert len(throttled) == 1


async def test_link_emails_share_one_hourly_budget_per_address(auth_client, auth_db, local_auth):
    """Sign-up, forgot and resend all fill the inbox of the same address."""
    user = await seed_user(auth_db, password_hash=hash_password(PW))

    await post(auth_client, RESEND, {"email": user.email})
    await post(auth_client, FORGOT, {"email": user.email})
    await post(auth_client, SIGNUP, {"email": user.email, "password": PW})
    fourth = await post(auth_client, FORGOT, {"email": user.email})

    assert fourth.status_code == 202 and fourth.json() == ACCEPTED
    assert len(local_auth.messages) == 3


async def test_forgot_answers_the_same_while_smtp_is_down(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    local_auth.fail_with = ConnectionRefusedError("down")

    r = await post(auth_client, FORGOT, {"email": user.email})

    assert r.status_code == 202 and r.json() == ACCEPTED
    requested = await events(auth_db, AuthEventType.PASSWORD_RESET_REQUESTED)
    assert [e.details.get("notification") for e in requested] == [None, "not_sent"]


# ── E-mails leave after the response ─────────────────────────────────────────


async def direct(local_auth, path: str, body: dict, headers: dict = CSRF, method: str = "POST"):
    """`path` driven straight through the ASGI app, with an SMTP server that answers only once the
    response is sent (`call_app_directly`): a route that waited for SMTP would never finish."""
    return await call_app_directly(local_auth, method, path, body, headers)


@pytest.mark.parametrize("case", ["forgot", "resend", "signup_new", "signup_existing"])
async def test_link_e_mails_leave_after_the_response(auth_client, auth_db, local_auth, case):
    verified = await verified_user(auth_db)
    unverified = await seed_user(auth_db, password_hash=hash_password(PW))
    path, body, to = {
        "forgot": (FORGOT, {"email": verified.email}, verified.email),
        "resend": (RESEND, {"email": unverified.email}, unverified.email),
        "signup_new": (SIGNUP, {"email": "new@example.test", "password": PW}, "new@example.test"),
        "signup_existing": (SIGNUP, {"email": verified.email, "password": PW}, verified.email),
    }[case]

    exchange = await direct(local_auth, path, body)

    assert exchange.status == 202 and exchange.json() == ACCEPTED
    assert exchange.emails_attempted == 1
    [message] = local_auth.messages  # sent by the background job, once the response was out
    assert message.to == to


async def test_known_and_unknown_addresses_get_the_same_answer_before_any_e_mail(
    auth_client, auth_db, local_auth, no_ip_limit
):
    """Same status, body and headers, and none of them waits for SMTP: the e-mail of a known
    address leaves after the response, so the response time does not tell the two apart. The
    throttled sign-up (an address that already had its three links this hour) sends nothing and is
    answered the same way too."""
    verified = await verified_user(auth_db)
    unverified = await seed_user(auth_db, password_hash=hash_password(PW))
    throttled = await verified_user(auth_db)
    async with auth_db() as s:
        for _ in range(3):
            await issue_token(s, throttled.id, Purpose.PASSWORD_RESET)
        await s.commit()

    cases = [
        (FORGOT, {"email": verified.email}, 1),
        (FORGOT, {"email": "nobody@example.test"}, 0),
        (RESEND, {"email": unverified.email}, 1),
        (RESEND, {"email": verified.email}, 0),
        (RESEND, {"email": "nobody@example.test"}, 0),
        (SIGNUP, {"email": verified.email, "password": PW}, 1),
        (SIGNUP, {"email": "fresh@example.test", "password": PW}, 1),
        (SIGNUP, {"email": throttled.email, "password": PW}, 0),
    ]
    answers = []
    for path, body, emails in cases:
        sent_before = len(local_auth.messages)
        exchange = await direct(local_auth, path, body)
        assert exchange.emails_attempted == emails, (path, body)
        assert len(local_auth.messages) - sent_before == emails, (path, body)
        answers.append((exchange.status, exchange.body, exchange.headers))

    assert answers[0][0] == 202
    assert all(answer == answers[0] for answer in answers), answers


async def test_an_e_mail_that_fails_after_the_response_is_audited(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    local_auth.fail_with = ConnectionRefusedError("down")

    exchange = await direct(local_auth, FORGOT, {"email": user.email})

    assert exchange.status == 202 and exchange.json() == ACCEPTED
    assert exchange.emails_attempted == 1 and local_auth.messages == []
    requested = await events(auth_db, AuthEventType.PASSWORD_RESET_REQUESTED)
    assert [e.details.get("notification") for e in requested] == [None, "not_sent"]
    assert requested[1].details["error_kind"] == "EmailSendError"
    assert requested[1].user_id == user.id and requested[1].ip == "203.0.113.9"


async def test_password_change_notices_leave_after_the_response(
    auth_client, auth_db, local_auth, local_session_for
):
    user = await verified_user(auth_db)
    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")
    local_auth.messages.clear()

    reset = await direct(local_auth, RESET, {"token": raw, "new_password": NEW_PW})
    assert reset.status == 204 and reset.emails_attempted == 1
    assert "was just changed" in local_auth.last_text

    session = await local_session_for(user)
    change = await direct(
        local_auth,
        CHANGE,
        {"current_password": NEW_PW, "new_password": PW},
        headers=bearer(session.access_token),
    )
    assert change.status == 204 and change.emails_attempted == 1
    assert len(local_auth.messages) == 2 and "was just changed" in local_auth.last_text


async def test_public_routes_are_limited_to_five_per_hour_per_ip(auth_client, local_auth):
    for path, body in (
        (SIGNUP, lambda n: {"email": f"ip-{n}@example.test", "password": PW}),
        (FORGOT, lambda n: {"email": f"ip-{n}@example.test"}),
        (RESEND, lambda n: {"email": f"ip-{n}@example.test"}),
    ):
        for n in range(5):
            assert (await post(auth_client, path, body(n))).status_code == 202, path
        r = await post(auth_client, path, body(99))
        assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS, path


async def test_reset_revokes_every_session_and_still_requires_mfa(
    auth_client, auth_db, local_auth, local_session_for
):
    user, secret = await mfa_user(auth_db)
    first = await local_session_for(user)
    await local_session_for(user)

    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")
    r = await post(auth_client, RESET, {"token": raw, "new_password": NEW_PW})

    assert r.status_code == 204, r.text
    assert not {"gl_at", "gl_rt", "gl_session", "gl_mfa"} & set(r.cookies)
    assert await live_sessions(auth_db, user.id) == []
    assert {s.revoked_reason for s in await sessions_of(auth_db, user.id)} == {"password_reset"}
    stored = await reload(auth_db, user.id)
    assert verify_password(NEW_PW, stored.password_hash).ok
    assert stored.password_changed_at is not None
    assert local_auth.last.to == user.email and "was just changed" in local_auth.last_text
    [done] = await events(auth_db, AuthEventType.PASSWORD_RESET_COMPLETED)
    assert done.details["sessions_revoked"] == 2

    assert (await auth_client.get(ME, headers=bearer(first.access_token))).status_code == 401
    assert (await post(auth_client, RESET, {"token": raw, "new_password": PW})).json() == (
        INVALID_TOKEN
    )
    assert (await login(auth_client, user.email)).status_code == 401  # the old password
    r = await login(auth_client, user.email, NEW_PW)
    assert r.status_code == 200 and r.json() == {"next": "mfa"}
    assert (await verify(auth_client, code=code_at(secret))).status_code == 200


async def test_reset_refused_by_the_policy_keeps_the_link(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")

    r = await post(auth_client, RESET, {"token": raw, "new_password": user.email})
    assert r.status_code == 400 and r.json() == {"detail": "password_policy:same_as_email"}
    r = await post(auth_client, RESET, {"token": raw, "new_password": NEW_PW})
    assert r.status_code == 204, r.text


async def test_reset_refuses_a_bad_token(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    invitation = await invite_token(auth_db, user)  # an invitation is not a reset link

    for token in ("not-a-token", invitation):
        r = await post(auth_client, RESET, {"token": token, "new_password": NEW_PW})
        assert r.status_code == 400 and r.json() == INVALID_TOKEN
    assert verify_password(PW, (await reload(auth_db, user.id)).password_hash).ok


async def test_reset_proves_the_address(auth_client, auth_db, local_auth):
    """A reset link reached the inbox, exactly as a verification link would have."""
    user = await seed_user(auth_db, password_hash=hash_password(PW))
    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")

    await post(auth_client, RESET, {"token": raw, "new_password": NEW_PW})

    assert (await reload(auth_db, user.id)).email_verified_at is not None


async def test_a_reset_kills_the_enrolment_step_of_whoever_had_the_old_password(
    auth_client, auth_db, local_auth
):
    """Someone signs in with a stolen password (no second factor yet: enrolment) and starts the
    enrolment; the owner resets the password. The step token dies with the old password: neither
    enrolment step works any more, even with a right code, and no session is opened."""
    user = await verified_user(auth_db)
    assert (await login(auth_client, user.email)).json() == {"next": "mfa_setup"}
    secret = (await auth_client.post(SETUP_START, headers=CSRF)).json()["secret"]

    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")
    reset = await post(auth_client, RESET, {"token": raw, "new_password": NEW_PW})
    assert reset.status_code == 204, reset.text

    # The client still holds its `gl_mfa` cookie (the reset answered with no cookie).
    start = await auth_client.post(SETUP_START, headers=CSRF)
    assert start.status_code == 401 and start.json() == CHALLENGE_EXPIRED
    confirm = await auth_client.post(SETUP_CONFIRM, json={"code": code_at(secret)}, headers=CSRF)
    assert confirm.status_code == 401 and confirm.json() == CHALLENGE_EXPIRED
    assert await sessions_of(auth_db, user.id) == []
    assert (await reload(auth_db, user.id)).mfa_enabled_at is None
    async with auth_db() as s:
        live = await s.scalar(
            select(func.count())
            .select_from(AuthOneTimeToken)
            .where(AuthOneTimeToken.user_id == user.id, AuthOneTimeToken.used_at.is_(None))
        )
    assert live == 0


async def test_a_password_change_kills_a_pending_challenge(
    auth_client, auth_db, local_auth, local_session_for
):
    """The thief passed the password step and holds a challenge; the owner, signed in elsewhere,
    changes the password. The challenge dies: even a right code opens nothing."""
    user, secret = await mfa_user(auth_db)
    owner = await local_session_for(user)
    assert (await login(auth_client, user.email)).json() == {"next": "mfa"}

    changed = await post(
        auth_client,
        CHANGE,
        {"current_password": PW, "new_password": NEW_PW},
        bearer(owner.access_token),
    )
    assert changed.status_code == 204, changed.text

    r = await verify(auth_client, code=code_at(secret))
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED
    assert [s.id for s in await live_sessions(auth_db, user.id)] == [owner.session_id]
    assert len(await sessions_of(auth_db, user.id)) == 1


async def test_a_reset_lifts_the_account_lock_and_still_requires_mfa(
    auth_client, auth_db, local_auth, no_ip_limit
):
    """Whoever proves the inbox is not kept out by ten failures of somebody else: a completed
    reset clears the count and the lock. The second factor is still asked."""
    user, secret = await mfa_user(auth_db)
    for _ in range(10):
        await login(auth_client, user.email, WRONG_PW)
    assert (await login(auth_client, user.email)).json() == TOO_MANY_ATTEMPTS
    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")

    refused = await post(auth_client, RESET, {"token": raw, "new_password": user.email})
    assert refused.status_code == 400  # the policy: nothing changed, the lock included
    assert (await reload(auth_db, user.id)).locked_until is not None
    r = await post(auth_client, RESET, {"token": raw, "new_password": NEW_PW})

    assert r.status_code == 204, r.text
    stored = await reload(auth_db, user.id)
    assert stored.failed_login_count == 0 and stored.locked_until is None
    r = await login(auth_client, user.email, NEW_PW)
    assert r.status_code == 200 and r.json() == {"next": "mfa"}
    assert await sessions_of(auth_db, user.id) == []
    assert (await verify(auth_client, code=code_at(secret))).status_code == 200


async def test_a_reset_does_not_refill_the_code_guesses(
    auth_client, auth_db, local_auth, no_ip_limit
):
    """Whoever holds the inbox could otherwise reset, sign in and guess ten more codes, again
    and again. The code count survives the reset, and so does the lock it holds; only a completed
    sign-in clears it."""
    user, secret = await mfa_user(auth_db)
    await fail_codes(auth_client, user, secret, 10)
    await post(auth_client, FORGOT, {"email": user.email})
    raw = link_token(local_auth.last_text, "/auth/reset")

    r = await post(auth_client, RESET, {"token": raw, "new_password": NEW_PW})

    assert r.status_code == 204, r.text
    stored = await reload(auth_db, user.id)
    assert stored.mfa_failed_count == 10 and stored.failed_login_count == 0
    assert (await login(auth_client, user.email, NEW_PW)).json() == TOO_MANY_ATTEMPTS
    await expire_lock(auth_db, user.id)
    assert (await login(auth_client, user.email, NEW_PW)).json() == {"next": "mfa"}
    assert (await verify(auth_client, code=wrong_code(secret))).status_code == 401
    assert (await login(auth_client, user.email, NEW_PW)).json() == TOO_MANY_ATTEMPTS


# ── Invitation ───────────────────────────────────────────────────────────────


async def test_invite_accept_activates_pending_and_requires_enrollment(
    auth_client, auth_db, local_auth
):
    user = await seed_user(auth_db, status=UserStatus.PENDING)
    raw = await invite_token(auth_db, user)

    r = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": PW})

    assert r.status_code == 200, r.text
    assert r.json() == {"next": "mfa_setup"}
    assert "gl_mfa" in r.cookies
    assert not {"gl_at", "gl_rt", "gl_session"} & set(r.cookies)
    stored = await reload(auth_db, user.id)
    assert stored.status == UserStatus.ACTIVE
    assert stored.email_verified_at is not None and stored.password_changed_at is not None
    assert verify_password(PW, stored.password_hash).ok
    assert await sessions_of(auth_db, user.id) == []
    [accepted] = await events(auth_db, AuthEventType.INVITE_ACCEPTED)
    assert accepted.user_id == user.id

    start = await auth_client.post(SETUP_START, headers=CSRF)
    assert start.status_code == 200, start.text
    assert start.json()["otpauth_uri"].startswith("otpauth://totp/")

    again = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": NEW_PW})
    assert again.status_code == 400 and again.json() == INVALID_TOKEN
    assert verify_password(PW, (await reload(auth_db, user.id)).password_hash).ok


async def test_invite_accept_checks_the_policy_before_spending_the_link(
    auth_client, auth_db, local_auth
):
    user = await seed_user(auth_db, status=UserStatus.PENDING)
    raw = await invite_token(auth_db, user)

    r = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": "short"})
    assert r.status_code == 400 and r.json() == {"detail": "password_policy:too_short"}
    assert "gl_mfa" not in r.cookies
    assert (await reload(auth_db, user.id)).status == UserStatus.PENDING

    r = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": PW})
    assert r.status_code == 200 and r.json() == {"next": "mfa_setup"}


async def test_invite_accept_on_an_account_with_2fa_asks_for_a_code(
    auth_client, auth_db, local_auth
):
    user, secret = await mfa_user(auth_db, password_hash=None, status=UserStatus.PENDING)
    raw = await invite_token(auth_db, user)

    r = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": PW})

    assert r.status_code == 200 and r.json() == {"next": "mfa"}
    assert (await verify(auth_client, code=code_at(secret))).status_code == 200


async def test_invite_for_an_account_with_a_password_is_refused_and_spent(
    auth_client, auth_db, local_auth
):
    """A stale or forwarded invitation must not become a silent password reset."""
    user = await seed_user(auth_db, status=UserStatus.PENDING, password_hash=hash_password(PW))
    raw = await invite_token(auth_db, user)

    r = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": NEW_PW})

    assert r.status_code == 400 and r.json() == INVALID_TOKEN
    assert "gl_mfa" not in r.cookies
    stored = await reload(auth_db, user.id)
    assert verify_password(PW, stored.password_hash).ok and stored.status == UserStatus.PENDING
    assert await token_used(auth_db, user.id, Purpose.INVITE)


async def test_invite_for_an_active_account_is_refused_and_spent(auth_client, auth_db, local_auth):
    user = await seed_user(auth_db, status=UserStatus.ACTIVE)  # no password
    raw = await invite_token(auth_db, user)

    r = await post(auth_client, INVITE_ACCEPT, {"token": raw, "password": PW})

    assert r.status_code == 400 and r.json() == INVALID_TOKEN
    assert (await reload(auth_db, user.id)).password_hash is None
    assert await token_used(auth_db, user.id, Purpose.INVITE)


async def test_invite_accept_refuses_a_bad_token(auth_client, auth_db, local_auth):
    user = await seed_user(auth_db, status=UserStatus.PENDING)
    async with auth_db() as s:
        reset = await issue_token(s, user.id, Purpose.PASSWORD_RESET)
        await s.commit()

    for token in ("not-a-token", reset):
        r = await post(auth_client, INVITE_ACCEPT, {"token": token, "password": PW})
        assert r.status_code == 400 and r.json() == INVALID_TOKEN
    assert (await reload(auth_db, user.id)).status == UserStatus.PENDING


# ── Password change ──────────────────────────────────────────────────────────


async def test_password_change_keeps_this_session_and_revokes_the_others(
    auth_client, auth_db, local_auth, local_session_for
):
    user, _ = await mfa_user(auth_db)
    current = await local_session_for(user)
    other = await local_session_for(user)

    r = await post(
        auth_client,
        CHANGE,
        {"current_password": PW, "new_password": NEW_PW},
        bearer(current.access_token),
    )

    assert r.status_code == 204, r.text
    [kept] = await live_sessions(auth_db, user.id)
    assert kept.id == current.session_id
    revoked = [s for s in await sessions_of(auth_db, user.id) if s.id == other.session_id]
    assert revoked[0].revoked_reason == "password_change"
    stored = await reload(auth_db, user.id)
    assert verify_password(NEW_PW, stored.password_hash).ok
    assert stored.password_changed_at is not None
    assert local_auth.last.to == user.email and "was just changed" in local_auth.last_text
    [changed] = await events(auth_db, AuthEventType.PASSWORD_CHANGED)
    assert changed.details["sessions_revoked"] == 1
    assert (await auth_client.get(ME, headers=bearer(current.access_token))).status_code == 200
    assert (await auth_client.get(ME, headers=bearer(other.access_token))).status_code == 401


async def test_password_change_refusals(auth_client, auth_db, local_auth, local_session_for):
    user, _ = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(
        auth_client, CHANGE, {"current_password": WRONG_PW, "new_password": NEW_PW}, headers
    )
    assert r.status_code == 400 and r.json() == {"detail": "invalid_password"}
    assert (await reload(auth_db, user.id)).failed_login_count == 1

    r = await post(auth_client, CHANGE, {"current_password": PW, "new_password": "short"}, headers)
    assert r.status_code == 400 and r.json() == {"detail": "password_policy:too_short"}
    assert verify_password(PW, (await reload(auth_db, user.id)).password_hash).ok
    assert local_auth.messages == []


async def test_password_change_failures_count_toward_the_lockout(
    auth_client, auth_db, local_auth, local_session_for
):
    user, _ = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)
    body = {"current_password": WRONG_PW, "new_password": NEW_PW}

    for _ in range(10):
        r = await post(auth_client, CHANGE, body, headers)
        assert r.status_code == 400

    r = await post(auth_client, CHANGE, {"current_password": PW, "new_password": NEW_PW}, headers)
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS
    assert verify_password(PW, (await reload(auth_db, user.id)).password_hash).ok
    [locked] = await events(auth_db, AuthEventType.ACCOUNT_LOCKED)
    assert locked.user_id == user.id
    assert (await login(auth_client, user.email)).json() == TOO_MANY_ATTEMPTS


async def test_password_change_hashes_off_the_event_loop(
    auth_client, auth_db, local_auth, local_session_for, monkeypatch
):
    from starlette.concurrency import run_in_threadpool

    import app.api.endpoints.auth_account as account_endpoints

    seen = []

    async def spy(func, *args, **kwargs):
        seen.append(func)
        return await run_in_threadpool(func, *args, **kwargs)

    monkeypatch.setattr(account_endpoints, "run_in_threadpool", spy)
    user, _ = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)

    await post(auth_client, CHANGE, {"current_password": PW, "new_password": NEW_PW}, headers)

    assert seen == [verify_password, hash_password]


# ── Sessions ─────────────────────────────────────────────────────────────────


async def test_session_list_and_revoke_other(auth_client, auth_db, local_auth, local_session_for):
    user, _ = await mfa_user(auth_db)
    stranger, _ = await mfa_user(auth_db)
    current = await local_session_for(user)
    other = await local_session_for(user)
    foreign = await local_session_for(stranger)
    headers = bearer(current.access_token)

    r = await auth_client.get(SESSIONS, headers=headers)
    assert r.status_code == 200, r.text
    listed = {item["id"]: item for item in r.json()}
    assert set(listed) == {str(current.session_id), str(other.session_id)}
    assert listed[str(current.session_id)]["current"] is True
    assert listed[str(other.session_id)]["current"] is False
    item = listed[str(other.session_id)]
    assert set(item) == {"id", "created_at", "last_used_at", "ip", "user_agent", "current"}
    assert item["ip"] == "203.0.113.7" and item["user_agent"] == "pytest"

    for missing in (foreign.session_id, "00000000-0000-0000-0000-000000000000"):
        r = await auth_client.delete(f"{SESSIONS}/{missing}", headers=headers)
        assert r.status_code == 404 and r.json() == {"detail": "session_not_found"}
    assert len(await live_sessions(auth_db, stranger.id)) == 1

    r = await auth_client.delete(f"{SESSIONS}/{other.session_id}", headers=headers)
    assert r.status_code == 204, r.text
    [kept] = await live_sessions(auth_db, user.id)
    assert kept.id == current.session_id
    [revoked] = await events(auth_db, AuthEventType.SESSION_REVOKED)
    assert revoked.details["session_id"] == str(other.session_id)
    r = await auth_client.get(SESSIONS, headers=headers)
    assert [item["id"] for item in r.json()] == [str(current.session_id)]


async def test_revoke_all_keeps_the_current_session(
    auth_client, auth_db, local_auth, local_session_for
):
    user, _ = await mfa_user(auth_db)
    current = await local_session_for(user)
    for _ in range(2):
        await local_session_for(user)

    r = await auth_client.post(REVOKE_ALL, headers=bearer(current.access_token))

    assert r.status_code == 200, r.text
    assert r.json() == {"revoked": 2}
    [kept] = await live_sessions(auth_db, user.id)
    assert kept.id == current.session_id
    [event] = await events(auth_db, AuthEventType.SESSIONS_REVOKED_ALL)
    assert event.details["count"] == 2


async def test_revoking_the_current_session_clears_its_cookies(
    auth_client, auth_db, local_auth, local_session_for
):
    user, _ = await mfa_user(auth_db)
    current = await local_session_for(user)

    r = await auth_client.delete(
        f"{SESSIONS}/{current.session_id}", headers=bearer(current.access_token)
    )

    assert r.status_code == 204
    assert set_cookies(r)["gl_at"]["max-age"] == "0"
    assert await live_sessions(auth_db, user.id) == []


async def test_service_token_cannot_call_session_endpoints(auth_client, auth_db, local_auth):
    from app.scripts.service_token import create_service_token

    owner, _ = await mfa_user(auth_db)
    async with auth_db() as db:
        _, token = await create_service_token(
            db, owner=owner.email, name="pipeline", scopes=("projects:create",)
        )

    for method, path in SESSION_ROUTES:
        r = await call(auth_client, method, path, bearer(token), {})
        assert r.status_code == 403, (method, path, r.text)


async def test_hosted_provider_token_cannot_call_session_endpoints(
    auth_client, auth_db, local_auth, settings_override, supabase_token_for
):
    """During the migration window the routes are served, but only to a local session."""
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=True)
    user, _ = await mfa_user(auth_db)
    headers = bearer(supabase_token_for(user))

    for method, path in SESSION_ROUTES:
        r = await call(auth_client, method, path, headers, {})
        assert r.status_code == 403, (method, path, r.text)
        assert r.json() == LOCAL_SESSION_REQUIRED


async def test_session_endpoints_need_a_session(auth_client, local_auth):
    for method, path in SESSION_ROUTES:
        r = await call(auth_client, method, path, CSRF, {})
        assert r.status_code == 401, (method, path)


async def test_cookie_carried_session_writes_need_the_csrf_header(
    auth_client, auth_db, local_auth, local_session_for
):
    user, _ = await mfa_user(auth_db)
    cookie = access_cookie((await local_session_for(user)).access_token)

    assert (await auth_client.get(SESSIONS, headers=cookie)).status_code == 200
    r = await auth_client.post(REVOKE_ALL, headers=cookie)
    assert r.status_code == 403 and r.json() == {"detail": "csrf_header_missing"}
    r = await auth_client.post(REVOKE_ALL, headers={**cookie, **CSRF})
    assert r.status_code == 200


# ── Second factor: rotation and new recovery codes ───────────────────────────


async def test_mfa_rotation_requires_password_and_current_code(
    auth_client, auth_db, local_auth, local_session_for
):
    user, old_secret = await mfa_user(auth_db)
    old_codes = await give_recovery_codes(auth_db, user)
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(
        auth_client, ROTATE_START, {"password": WRONG_PW, "code": code_at(old_secret)}, headers
    )
    assert r.status_code == 400 and r.json() == {"detail": "invalid_password"}
    r = await post(
        auth_client, ROTATE_START, {"password": PW, "code": wrong_code(old_secret)}, headers
    )
    assert r.status_code == 400 and r.json() == {"detail": "invalid_code"}
    stored = await reload(auth_db, user.id)
    assert (stored.failed_login_count, stored.mfa_failed_count) == (1, 1)

    r = await post(
        auth_client, ROTATE_START, {"password": PW, "code": code_at(old_secret)}, headers
    )
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    new_secret = r.json()["secret"]
    assert r.json()["otpauth_uri"].startswith("otpauth://totp/")
    stored = await reload(auth_db, user.id)
    assert stored.mfa_pending_secret_enc and stored.mfa_totp_secret_enc == user.mfa_totp_secret_enc

    # The code of the same 30-second step as the one just used with the old secret.
    r = await post(auth_client, ROTATE_CONFIRM, {"code": code_at(new_secret)}, headers)
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    new_codes = r.json()["recovery_codes"]
    assert len(new_codes) == 10 and not set(new_codes) & set(old_codes)
    stored = await reload(auth_db, user.id)
    assert stored.mfa_pending_secret_enc is None and stored.mfa_enabled_at is not None
    assert stored.mfa_totp_secret_enc != user.mfa_totp_secret_enc
    kept = await stored_codes(auth_db, user.id)
    assert len(kept) == 10
    assert not any(recovery_code_matches(old_codes[0], c.code_hash) for c in kept)
    [rotated] = await events(auth_db, AuthEventType.MFA_ROTATED)
    assert rotated.user_id == user.id

    # The new secret signs in, the old one and the old recovery codes no longer do.
    await login(auth_client, user.email)
    r = await verify(auth_client, code=code_at(old_secret, 1))
    assert r.status_code == 401
    r = await verify(auth_client, recovery_code=old_codes[1])
    assert r.status_code == 401
    r = await verify(auth_client, code=code_at(new_secret, 1))
    assert r.status_code == 200, r.text


async def test_a_wrong_rotation_code_drops_the_pending_secret(
    auth_client, auth_db, local_auth, local_session_for
):
    """Otherwise a session holder could guess the code of a rotation the owner started."""
    user, old_secret = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)
    r = await post(
        auth_client, ROTATE_START, {"password": PW, "code": code_at(old_secret)}, headers
    )
    new_secret = r.json()["secret"]

    r = await post(auth_client, ROTATE_CONFIRM, {"code": wrong_code(new_secret)}, headers)

    assert r.status_code == 400 and r.json() == {"detail": "invalid_code"}
    stored = await reload(auth_db, user.id)
    assert stored.mfa_pending_secret_enc is None
    assert stored.mfa_totp_secret_enc == user.mfa_totp_secret_enc
    assert stored.mfa_failed_count == 1
    r = await post(auth_client, ROTATE_CONFIRM, {"code": code_at(new_secret)}, headers)
    assert r.status_code == 400 and r.json() == {"detail": "no_rotation_pending"}
    assert (await reload(auth_db, user.id)).mfa_totp_secret_enc == user.mfa_totp_secret_enc


async def test_wrong_rotation_codes_lock_the_account(
    auth_client, auth_db, local_auth, local_session_for
):
    user, old_secret = await mfa_user(auth_db, mfa_failed_count=9)
    headers = bearer((await local_session_for(user)).access_token)
    r = await post(
        auth_client, ROTATE_START, {"password": PW, "code": code_at(old_secret)}, headers
    )
    new_secret = r.json()["secret"]

    r = await post(auth_client, ROTATE_CONFIRM, {"code": wrong_code(new_secret)}, headers)
    assert r.status_code == 400

    stored = await reload(auth_db, user.id)
    assert stored.mfa_failed_count == 10 and stored.locked_until is not None
    [locked] = await events(auth_db, AuthEventType.ACCOUNT_LOCKED)
    assert locked.user_id == user.id
    r = await post(
        auth_client, ROTATE_START, {"password": PW, "code": code_at(old_secret, 1)}, headers
    )
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS


async def test_a_locked_account_cannot_confirm_a_rotation(
    auth_client, auth_db, local_auth, local_session_for
):
    from app.auth.mfa import encrypt_secret, new_totp_secret

    pending = new_totp_secret()
    user, _ = await mfa_user(
        auth_db,
        failed_login_count=10,
        locked_until=now_utc() + timedelta(minutes=10),
        mfa_pending_secret_enc=encrypt_secret(pending),
    )
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(auth_client, ROTATE_CONFIRM, {"code": code_at(pending)}, headers)

    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS
    assert (await reload(auth_db, user.id)).mfa_totp_secret_enc == user.mfa_totp_secret_enc


async def test_rotation_confirm_is_limited_per_ip(
    auth_client, auth_db, local_auth, local_session_for
):
    user, _ = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)

    for _ in range(30):
        r = await post(auth_client, ROTATE_CONFIRM, {"code": "123456"}, headers)
        assert r.status_code == 400  # nothing pending

    r = await post(auth_client, ROTATE_CONFIRM, {"code": "123456"}, headers)
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS


async def test_rotation_confirm_without_a_pending_secret(
    auth_client, auth_db, local_auth, local_session_for
):
    user, secret = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(auth_client, ROTATE_CONFIRM, {"code": code_at(secret)}, headers)

    assert r.status_code == 400 and r.json() == {"detail": "no_rotation_pending"}


async def test_rotation_and_regeneration_need_an_enabled_factor(
    auth_client, auth_db, local_auth, local_session_for
):
    user = await verified_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(auth_client, ROTATE_START, {"password": PW, "code": "123456"}, headers)
    assert r.status_code == 400 and r.json() == {"detail": "mfa_not_enabled"}
    r = await post(auth_client, REGENERATE, {"code": "123456"}, headers)
    assert r.status_code == 400 and r.json() == {"detail": "mfa_not_enabled"}


async def test_mfa_code_failures_on_session_routes_lock_the_account(
    auth_client, auth_db, local_auth, local_session_for
):
    """They share the code count of `/mfa/verify`: the tenth wrong code locks, and tells the
    owner, from a session route too."""
    user, secret = await mfa_user(auth_db, mfa_failed_count=9)
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(auth_client, REGENERATE, {"code": wrong_code(secret)}, headers)
    assert r.status_code == 400

    assert (await reload(auth_db, user.id)).mfa_failed_count == 10
    assert [m.to for m in local_auth.messages if "locked" in m.subject] == [user.email]
    r = await post(auth_client, REGENERATE, {"code": code_at(secret)}, headers)
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS
    r = await post(auth_client, ROTATE_START, {"password": PW, "code": code_at(secret)}, headers)
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS


@pytest.mark.parametrize("step", ["rotate_start", "rotate_confirm", "regenerate"])
async def test_every_wrong_code_on_a_session_route_counts_as_a_code_failure(
    auth_client, auth_db, local_auth, local_session_for, step
):
    """The password count is left alone: the code was wrong, not the password."""
    user, secret = await mfa_user(auth_db)
    headers = bearer((await local_session_for(user)).access_token)
    if step == "rotate_start":
        r = await post(
            auth_client, ROTATE_START, {"password": PW, "code": wrong_code(secret)}, headers
        )
    elif step == "rotate_confirm":
        started = await post(
            auth_client, ROTATE_START, {"password": PW, "code": code_at(secret)}, headers
        )
        r = await post(
            auth_client, ROTATE_CONFIRM, {"code": wrong_code(started.json()["secret"])}, headers
        )
    else:
        r = await post(auth_client, REGENERATE, {"code": wrong_code(secret)}, headers)

    assert r.status_code == 400 and r.json() == {"detail": "invalid_code"}
    stored = await reload(auth_db, user.id)
    assert (stored.failed_login_count, stored.mfa_failed_count) == (0, 1)


async def test_regenerate_recovery_codes(auth_client, auth_db, local_auth, local_session_for):
    user, secret = await mfa_user(auth_db)
    old_codes = await give_recovery_codes(auth_db, user)
    headers = bearer((await local_session_for(user)).access_token)

    r = await post(auth_client, REGENERATE, {"code": wrong_code(secret)}, headers)
    assert r.status_code == 400 and r.json() == {"detail": "invalid_code"}
    assert (await reload(auth_db, user.id)).mfa_failed_count == 1

    code = code_at(secret)
    r = await post(auth_client, REGENERATE, {"code": code}, headers)
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    new_codes = r.json()["recovery_codes"]
    assert len(new_codes) == 10 and not set(new_codes) & set(old_codes)
    kept = await stored_codes(auth_db, user.id)
    assert len(kept) == 10
    assert all(any(recovery_code_matches(c, row.code_hash) for row in kept) for c in new_codes)
    assert len(await events(auth_db, AuthEventType.RECOVERY_CODES_REGENERATED)) == 1

    r = await post(auth_client, REGENERATE, {"code": code}, headers)  # the same code, replayed
    assert r.status_code == 400 and r.json() == {"detail": "invalid_code"}


# ── /users/me ────────────────────────────────────────────────────────────────


async def test_me_exposes_mfa_state(auth_client, auth_db, local_auth, local_session_for):
    plain = await verified_user(auth_db)
    r = await auth_client.get(ME, headers=bearer((await local_session_for(plain)).access_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mfa_enabled"] is False and body["recovery_codes_remaining"] is None
    assert body["email_verified"] is True

    user, _ = await mfa_user(auth_db)
    codes = await give_recovery_codes(auth_db, user)
    r = await auth_client.get(ME, headers=bearer((await local_session_for(user)).access_token))
    body = r.json()
    assert body["mfa_enabled"] is True and body["recovery_codes_remaining"] == 10

    await login(auth_client, user.email)
    assert (await verify(auth_client, recovery_code=codes[0])).status_code == 200
    r = await auth_client.get(ME)  # the session cookies the recovery code opened
    assert r.status_code == 200 and r.json()["recovery_codes_remaining"] == 9


async def test_me_for_an_unverified_address(auth_client, auth_db, local_auth, local_session_for):
    user = await seed_user(auth_db, password_hash=hash_password(PW))
    r = await auth_client.get(ME, headers=bearer((await local_session_for(user)).access_token))
    assert r.status_code == 200 and r.json()["email_verified"] is False


async def test_me_in_supabase_mode_reports_no_second_factor(
    auth_client, auth_db, settings_override, supabase_token_for
):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=False)
    user = await seed_user(auth_db)

    r = await auth_client.get(ME, headers=bearer(supabase_token_for(user)))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mfa_enabled"] is False and body["recovery_codes_remaining"] is None
    assert body["email_verified"] is False


# ── Switches and CSRF ────────────────────────────────────────────────────────


async def test_account_endpoints_404_when_disabled(auth_client, settings_override):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=False)

    for path in PUBLIC_ROUTES:
        r = await auth_client.post(path, json={}, headers=CSRF)
        assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, path
    for method, path in SESSION_ROUTES:
        r = await call(auth_client, method, path, CSRF, {})
        assert r.status_code == 404, (method, path)


async def test_before_the_launch_the_public_routes_answer_404(
    auth_client, auth_db, local_auth, settings_override
):
    """Pre-launch (AUTH_MODE=supabase with AUTH_LOCAL_ENABLED): the self-service routes answer
    exactly as when local authentication is off, with or without the header, and do nothing,
    even with a valid link."""
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=True)
    user = await verified_user(auth_db)
    pending = await seed_user(auth_db, status=UserStatus.PENDING)
    async with auth_db() as s:
        reset = await issue_token(s, user.id, Purpose.PASSWORD_RESET)
        check = await issue_token(s, user.id, Purpose.EMAIL_VERIFY)
        await s.commit()
    invitation = await invite_token(auth_db, pending)
    bodies = {
        SIGNUP: {"email": "new@example.test", "password": PW},
        VERIFY_EMAIL: {"token": check},
        RESEND: {"email": user.email},
        FORGOT: {"email": user.email},
        RESET: {"token": reset, "new_password": NEW_PW},
        INVITE_ACCEPT: {"token": invitation, "password": NEW_PW},
    }

    for path in PUBLIC_ROUTES:
        for headers in (CSRF, {}):
            r = await post(auth_client, path, bodies[path], headers)
            assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, path
            assert "set-cookie" not in r.headers

    assert local_auth.messages == []
    assert await user_by_email(auth_db, "new@example.test") is None
    assert verify_password(PW, (await reload(auth_db, user.id)).password_hash).ok
    assert (await reload(auth_db, pending.id)).status == UserStatus.PENDING
    async with auth_db() as s:
        spent = await s.scalar(
            select(func.count())
            .select_from(AuthOneTimeToken)
            .where(AuthOneTimeToken.used_at.is_not(None))
        )
    assert spent == 0


@pytest.mark.parametrize("mode, expected", [("supabase", 404), ("local", 422)])
async def test_the_public_routes_open_with_the_switch_to_local(
    auth_client, local_auth, settings_override, mode, expected
):
    """Local authentication on in both cases: the public routes exist only in AUTH_MODE=local
    (an empty body is then refused by validation, 422)."""
    settings_override(AUTH_MODE=mode, AUTH_LOCAL_ENABLED=True)
    for path in PUBLIC_ROUTES:
        r = await post(auth_client, path, {})
        assert r.status_code == expected, (mode, path)


async def test_before_the_launch_the_session_routes_keep_working(
    auth_client, auth_db, local_auth, settings_override, local_session_for
):
    """An administrator signed in during the pre-launch manages their sessions, password and
    second factor as after the launch."""
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=True)
    admin, secret = await mfa_user(auth_db, role=UserRole.ADMIN)
    current = await local_session_for(admin)
    await local_session_for(admin)
    headers = bearer(current.access_token)

    listed = await call(auth_client, "GET", SESSIONS, headers)
    assert listed.status_code == 200 and len(listed.json()) == 2
    changed = await post(
        auth_client, CHANGE, {"current_password": PW, "new_password": NEW_PW}, headers
    )
    assert changed.status_code == 204, changed.text
    regenerated = await post(auth_client, REGENERATE, {"code": code_at(secret)}, headers)
    assert regenerated.status_code == 200 and len(regenerated.json()["recovery_codes"]) == 10


async def test_public_routes_need_the_csrf_header(auth_client, auth_db, local_auth):
    for path in PUBLIC_ROUTES:
        r = await auth_client.post(path, json={})
        assert r.status_code == 403 and r.json() == {"detail": "csrf_header_missing"}, path
    assert local_auth.messages == []


async def test_signup_records_no_secret(auth_client, auth_db, local_auth):
    await post(auth_client, SIGNUP, {"email": "audit@example.test", "password": PW})
    [event] = await events(auth_db, AuthEventType.SIGNUP)
    assert PW not in str(event.details)
