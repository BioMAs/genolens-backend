"""
Sign-in, second factor, session renewal and logout (`/api/v1/auth`), end to end.

Every API test runs the real chain (`auth_client`) against an in-memory SQLite database, with
freshly generated signing and encryption keys and every e-mail captured (`local_auth`). Cookies are
not `Secure` here because the test client speaks plain HTTP; their exact attributes are checked
separately, on a bare response.

What is pinned down:
- the path to a session: password, then a second factor (a TOTP code, a recovery code, or the
  enrolment of an authenticator); `/login` alone never opens a session;
- anti-enumeration: an unknown address and a wrong password get the same answer, and an unknown
  address is locked after ten failures exactly like an account;
- the account lock counts 2FA failures too, so a password holder cannot guess codes across fresh
  challenges;
- renewal: two tabs refreshing with one refresh token both stay signed in; a replay later revokes
  the session, and the revocation outlives the refusal.
"""

import time
from datetime import timedelta
from uuid import uuid4

import bcrypt
import pyotp
import pytest
from sqlalchemy import func, select, update
from starlette.responses import Response

from app.auth.clock import aware_utc, now_utc
from app.auth.mfa import (
    encrypt_secret,
    generate_recovery_codes,
    hash_recovery_code,
    new_totp_secret,
)
from app.auth.passwords import hash_password, verify_password
from app.auth.sessions import IssuedSession, RefreshOutcome
from app.auth.tokens import issue_access_token
from app.models.auth import (
    AuthEvent,
    AuthEventType,
    AuthOneTimeToken,
    AuthRecoveryCode,
    AuthSession,
)
from app.models.models import User
from tests.auth_support import CSRF, NOW, PW, access_cookie, bearer, seed_user

AUTH = "/api/v1/auth"
LOGIN = f"{AUTH}/login"
VERIFY = f"{AUTH}/mfa/verify"
SETUP_START = f"{AUTH}/mfa/setup/start"
SETUP_CONFIRM = f"{AUTH}/mfa/setup/confirm"
REFRESH = f"{AUTH}/refresh"
LOGOUT = f"{AUTH}/logout"
JWKS = f"{AUTH}/jwks.json"
ME = "/api/v1/users/me"

INVALID_CREDENTIALS = {"detail": "invalid_credentials"}
TOO_MANY_ATTEMPTS = {"detail": "too_many_attempts"}
CHALLENGE_EXPIRED = {"detail": "challenge_expired"}


@pytest.fixture
def local_auth(settings_override, local_keys, enc_keys, smtp_capture):
    """Local authentication on, cookies usable over plain HTTP, every e-mail captured."""
    settings_override(AUTH_MODE="local", AUTH_COOKIE_SECURE=False, AUTH_COOKIE_DOMAIN=None)
    return smtp_capture


@pytest.fixture
def no_ip_limit(monkeypatch):
    """Turn the per-IP limits off, for tests that need more attempts than they allow."""
    from app.middleware.rate_limit import limiter

    monkeypatch.setattr(limiter, "enabled", False)


# ── Helpers ──────────────────────────────────────────────────────────────────


async def verified_user(auth_db, **fields) -> User:
    values = dict(password_hash=hash_password(PW), email_verified_at=NOW)
    values.update(fields)
    return await seed_user(auth_db, **values)


async def mfa_user(auth_db, **fields) -> tuple[User, str]:
    secret = new_totp_secret()
    user = await verified_user(
        auth_db, mfa_enabled_at=NOW, mfa_totp_secret_enc=encrypt_secret(secret), **fields
    )
    return user, secret


async def give_recovery_codes(auth_db, user: User, count: int = 10) -> list[str]:
    codes = generate_recovery_codes(count)
    async with auth_db() as s:
        s.add_all(AuthRecoveryCode(user_id=user.id, code_hash=hash_recovery_code(c)) for c in codes)
        await s.commit()
    return codes


def code_at(secret: str, steps_ahead: int = 0) -> str:
    """The code of the current step, or of the next one (`steps_ahead=1`, inside the window)."""
    return pyotp.TOTP(secret).at(time.time() + 30 * steps_ahead)


def wrong_code(secret: str) -> str:
    """Six digits that match none of the steps the server tries."""
    totp = pyotp.TOTP(secret)
    now = time.time()
    taken = {totp.at(now + 30 * offset) for offset in (-2, -1, 0, 1, 2)}
    return next(code for code in (f"{n:06d}" for n in range(10**6)) if code not in taken)


async def login(client, email: str, password: str = PW):
    return await client.post(LOGIN, json={"email": email, "password": password}, headers=CSRF)


async def verify(client, **body):
    return await client.post(VERIFY, json=body, headers=CSRF)


async def reload(auth_db, user_id) -> User:
    async with auth_db() as s:
        return (
            await s.execute(
                select(User).where(User.id == user_id).execution_options(populate_existing=True)
            )
        ).scalar_one()


async def events(auth_db, event_type: AuthEventType) -> list[AuthEvent]:
    async with auth_db() as s:
        rows = await s.execute(
            select(AuthEvent).where(AuthEvent.event_type == event_type.value).order_by(AuthEvent.id)
        )
        return list(rows.scalars())


async def sessions_of(auth_db, user_id) -> list[AuthSession]:
    async with auth_db() as s:
        rows = await s.execute(select(AuthSession).where(AuthSession.user_id == user_id))
        return list(rows.scalars())


def set_cookies(response) -> dict[str, dict]:
    """The cookies a response sets, by name: `{"value": ..., "<attribute>": value or True}`."""
    headers = response.headers
    raw = (
        headers.get_list("set-cookie")
        if hasattr(headers, "get_list")
        else headers.getlist("set-cookie")
    )
    found: dict[str, dict] = {}
    for header in raw:
        parts = [part.strip() for part in header.split(";")]
        name, _, value = parts[0].partition("=")
        attributes: dict = {"value": value.strip('"')}
        for part in parts[1:]:
            key, sep, attr = part.partition("=")
            attributes[key.lower()] = attr if sep else True
        found[name] = attributes
    return found


def is_cleared(cookie: dict) -> bool:
    return cookie.get("max-age") == "0"


# ── Cookies: the attributes of spec §7.3 ─────────────────────────────────────


def test_session_cookie_attributes_follow_the_spec(settings_override):
    from app.auth import cookies

    settings_override(AUTH_COOKIE_SECURE=True, AUTH_COOKIE_DOMAIN=".example.test")
    response = Response()
    cookies.set_session_cookies(
        response,
        IssuedSession(session_id=uuid4(), access_token="a.b.c", refresh_token="rt", expires_at=NOW),
    )
    found = set_cookies(response)

    at = found["gl_at"]
    assert at["value"] == "a.b.c"
    assert at["httponly"] is True and at["secure"] is True and at["samesite"].lower() == "lax"
    assert at["domain"] == ".example.test" and at["path"] == "/" and at["max-age"] == "900"

    rt = found["gl_rt"]
    assert rt["value"] == "rt"
    assert rt["httponly"] is True and rt["secure"] is True and rt["samesite"].lower() == "lax"
    assert "domain" not in rt  # the API host only
    assert rt["path"] == "/api/v1/auth" and rt["max-age"] == str(30 * 86400)

    hint = found["gl_session"]
    assert hint["value"] == "1"
    assert "httponly" not in hint  # read by the web app's middleware
    assert hint["domain"] == ".example.test" and hint["path"] == "/"
    assert hint["max-age"] == str(30 * 86400)

    assert cookies.AUTH_COOKIE_PATH == "/api/v1/auth"


def test_step_cookie_attributes_and_clearing(settings_override):
    from app.auth import cookies

    settings_override(AUTH_COOKIE_SECURE=True, AUTH_COOKIE_DOMAIN=".example.test")
    response = Response()
    cookies.set_step_cookie(response, "step")
    mfa = set_cookies(response)["gl_mfa"]
    assert mfa["value"] == "step"
    assert mfa["httponly"] is True and mfa["secure"] is True and mfa["samesite"].lower() == "strict"
    assert "domain" not in mfa and mfa["path"] == "/api/v1/auth" and mfa["max-age"] == "900"

    response = Response()
    cookies.clear_step_cookie(response)
    cleared = set_cookies(response)["gl_mfa"]
    assert is_cleared(cleared) and cleared["path"] == "/api/v1/auth" and "domain" not in cleared

    # Clearing names the Path and Domain each cookie was set with, or the browser keeps it.
    response = Response()
    cookies.clear_session_cookies(response)
    found = set_cookies(response)
    assert all(is_cleared(found[name]) for name in ("gl_at", "gl_rt", "gl_session"))
    assert found["gl_at"]["domain"] == ".example.test" and found["gl_at"]["path"] == "/"
    assert "domain" not in found["gl_rt"] and found["gl_rt"]["path"] == "/api/v1/auth"
    assert found["gl_session"]["domain"] == ".example.test"


def test_access_cookie_alone_and_host_only_in_local_development(settings_override):
    from app.auth import cookies

    settings_override(AUTH_COOKIE_SECURE=False, AUTH_COOKIE_DOMAIN="")
    response = Response()
    cookies.set_access_cookie(response, "x.y.z")
    found = set_cookies(response)
    assert set(found) == {"gl_at"}
    assert "domain" not in found["gl_at"] and "secure" not in found["gl_at"]

    # A grace-window refresh has no refresh token to set: only the access cookie can be.
    with pytest.raises(ValueError):
        cookies.set_session_cookies(
            Response(), RefreshOutcome(session_id=uuid4(), access_token="a", refresh_token=None)
        )


# ── Login, enrolment, second factor ──────────────────────────────────────────


async def test_full_login_with_enrollment_then_mfa(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)

    r = await login(auth_client, user.email)
    assert r.status_code == 200, r.text
    assert r.json() == {"next": "mfa_setup"} and "gl_mfa" in r.cookies
    assert not {"gl_at", "gl_rt", "gl_session"} & set(r.cookies)

    start = await auth_client.post(SETUP_START, headers=CSRF)
    assert start.status_code == 200, start.text
    secret = start.json()["secret"]
    assert start.json()["otpauth_uri"].startswith("otpauth://totp/")
    assert start.headers["cache-control"] == "no-store"
    assert (await reload(auth_db, user.id)).mfa_pending_secret_enc

    done = await auth_client.post(SETUP_CONFIRM, json={"code": code_at(secret)}, headers=CSRF)
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["next"] == "app" and len(body["recovery_codes"]) == 10
    assert {"gl_at", "gl_rt", "gl_session"} <= set(done.cookies)
    assert is_cleared(set_cookies(done)["gl_mfa"])
    assert done.headers["cache-control"] == "no-store"
    assert (await auth_client.get(ME)).status_code == 200

    stored = await reload(auth_db, user.id)
    assert stored.mfa_enabled_at is not None and stored.mfa_totp_secret_enc
    assert stored.mfa_pending_secret_enc is None and stored.mfa_last_used_step is not None
    async with auth_db() as s:
        stored_codes = await s.scalar(
            select(func.count())
            .select_from(AuthRecoveryCode)
            .where(AuthRecoveryCode.user_id == user.id)
        )
    assert stored_codes == 10
    [session] = await sessions_of(auth_db, user.id)
    assert session.amr == ["pwd", "otp"]
    assert local_auth.last.to == user.email
    assert "turned on" in local_auth.last_text
    assert len(await events(auth_db, AuthEventType.MFA_ENROLLED)) == 1

    # The second login asks for a code, and a fresh one opens a second session.
    r = await login(auth_client, user.email)
    assert r.json() == {"next": "mfa"}
    r = await verify(auth_client, code=code_at(secret, 1))
    assert r.status_code == 200, r.text
    assert r.json() == {"next": "app"}
    assert is_cleared(set_cookies(r)["gl_mfa"])
    assert len(await sessions_of(auth_db, user.id)) == 2
    assert len(await events(auth_db, AuthEventType.MFA_SUCCEEDED)) == 1


async def test_login_never_opens_a_session(auth_client, auth_db, local_auth):
    plain = await verified_user(auth_db)
    enrolled, _ = await mfa_user(auth_db)

    for user in (plain, enrolled):
        r = await login(auth_client, user.email)
        assert r.status_code == 200
        assert not {"gl_at", "gl_rt", "gl_session"} & set(r.cookies)
        assert await sessions_of(auth_db, user.id) == []
    assert len(await events(auth_db, AuthEventType.LOGIN_SUCCEEDED)) == 2


async def test_email_case_and_spaces(auth_client, auth_db, local_auth):
    """Review Focus 2: an address typed with other capitals and surrounding spaces signs in."""
    await verified_user(auth_db, email="Mixed@Example.test")

    r = await login(auth_client, "  mixed@example.TEST ")

    assert r.status_code == 200, r.text
    assert r.json() == {"next": "mfa_setup"}


async def test_same_answer_for_unknown_email_and_wrong_password(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    no_password = await seed_user(auth_db, email_verified_at=NOW)

    answers = [
        await login(auth_client, "nobody@example.test"),
        await login(auth_client, user.email, "Wrong-Password-Again-42"),
        await login(auth_client, no_password.email),
    ]

    for r in answers:
        assert r.status_code == 401
        assert r.json() == INVALID_CREDENTIALS
        assert "set-cookie" not in r.headers
    assert (await reload(auth_db, user.id)).failed_login_count == 1
    failed = await events(auth_db, AuthEventType.LOGIN_FAILED)
    assert [e.user_id for e in failed] == [None, user.id, no_password.id]


async def test_password_check_runs_off_the_event_loop(
    auth_client, auth_db, local_auth, monkeypatch
):
    """Hashing takes tens of milliseconds of CPU: it must not stall every other request."""
    from starlette.concurrency import run_in_threadpool

    import app.api.endpoints.auth as auth_endpoints

    seen = []

    async def spy(func, *args, **kwargs):
        seen.append(func)
        return await run_in_threadpool(func, *args, **kwargs)

    monkeypatch.setattr(auth_endpoints, "run_in_threadpool", spy)
    user = await verified_user(auth_db)

    await login(auth_client, user.email)
    await login(auth_client, "nobody@example.test")

    assert seen.count(verify_password) == 2


async def test_lockout_after_ten_failures(auth_client, auth_db, local_auth, no_ip_limit):
    user = await verified_user(auth_db)

    for _ in range(10):
        r = await login(auth_client, user.email, "Wrong-Password-Again-42")
        assert r.status_code == 401 and r.json() == INVALID_CREDENTIALS

    r = await login(auth_client, user.email)  # the right password, too late
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS
    assert "gl_mfa" not in r.cookies

    stored = await reload(auth_db, user.id)
    assert stored.failed_login_count == 10
    locked_for = aware_utc(stored.locked_until) - now_utc()
    assert timedelta(minutes=14) < locked_for <= timedelta(minutes=15)
    [locked] = await events(auth_db, AuthEventType.ACCOUNT_LOCKED)
    assert locked.user_id == user.id


async def test_unknown_email_is_locked_like_an_account(
    auth_client, auth_db, local_auth, no_ip_limit
):
    """Otherwise the eleventh attempt would tell whether the address has an account."""
    user = await verified_user(auth_db)
    answers = {}
    for email in ("nobody@example.test", user.email):
        for _ in range(10):
            r = await login(auth_client, email, "Wrong-Password-Again-42")
            assert r.status_code == 401 and r.json() == INVALID_CREDENTIALS
        answers[email] = await login(auth_client, email)

    unknown, known = answers["nobody@example.test"], answers[user.email]
    assert unknown.status_code == known.status_code == 429
    assert unknown.json() == known.json() == TOO_MANY_ATTEMPTS
    # The same address in other capitals is the same address.
    r = await login(auth_client, " NOBODY@example.test")
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS


async def test_an_expired_lock_lets_the_right_password_through(auth_client, auth_db, local_auth):
    user = await verified_user(
        auth_db, failed_login_count=10, locked_until=now_utc() - timedelta(seconds=1)
    )

    r = await login(auth_client, user.email)

    assert r.status_code == 200 and r.json() == {"next": "mfa_setup"}


async def test_legacy_hash_upgraded_on_login(auth_client, auth_db, local_auth):
    legacy = bcrypt.hashpw(PW.encode(), bcrypt.gensalt(rounds=4, prefix=b"2a")).decode()
    user = await verified_user(auth_db, password_hash=legacy)

    r = await login(auth_client, user.email)

    assert r.status_code == 200, r.text
    stored = await reload(auth_db, user.id)
    assert stored.password_hash.startswith("$argon2id$")
    assert verify_password(PW, stored.password_hash).ok


async def test_unverified_email_blocked(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db, email_verified_at=None)

    r = await login(auth_client, user.email)
    assert r.status_code == 403 and r.json() == {"detail": "email_not_verified"}
    assert "gl_mfa" not in r.cookies

    # Only the right password learns that the address is unverified.
    r = await login(auth_client, user.email, "Wrong-Password-Again-42")
    assert r.status_code == 401 and r.json() == INVALID_CREDENTIALS


async def test_csrf_header_required(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    body = {"email": user.email, "password": PW}

    for path, payload in ((LOGIN, body), (VERIFY, {"code": "123456"}), (REFRESH, None)):
        r = await auth_client.post(path, json=payload)
        assert r.status_code == 403, path
        assert r.json() == {"detail": "csrf_header_missing"}


async def test_unreadable_input_is_refused_before_any_lookup(auth_client, local_auth):
    for body in (
        {"email": "a\u0000b@example.test", "password": PW},
        {"email": "nobody@example.test", "password": "\ud800" + PW},
        {"email": "nobody@example.test"},
    ):
        r = await auth_client.post(LOGIN, json=body, headers=CSRF)
        assert r.status_code == 422, body


async def test_challenge_five_attempts_then_expired(auth_client, auth_db, local_auth):
    user, secret = await mfa_user(auth_db)
    assert (await login(auth_client, user.email)).json() == {"next": "mfa"}

    for left in (4, 3, 2, 1, 0):
        r = await verify(auth_client, code=wrong_code(secret))
        assert r.status_code == 401
        assert r.json() == {"detail": "invalid_code", "attempts_left": left}

    r = await verify(auth_client, code=code_at(secret))  # right, but the challenge is spent
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED
    assert is_cleared(set_cookies(r)["gl_mfa"])
    assert await sessions_of(auth_db, user.id) == []
    assert len(await events(auth_db, AuthEventType.MFA_FAILED)) >= 5


async def test_fifth_attempt_can_still_succeed(auth_client, auth_db, local_auth):
    user, secret = await mfa_user(auth_db)
    await login(auth_client, user.email)
    for _ in range(4):
        await verify(auth_client, code=wrong_code(secret))

    r = await verify(auth_client, code=code_at(secret))

    assert r.status_code == 200 and r.json() == {"next": "app"}
    assert (await reload(auth_db, user.id)).failed_login_count == 0  # success resets the count


async def test_verify_without_a_challenge(auth_client, auth_db, local_auth):
    r = await verify(auth_client, code="123456")
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED

    r = await auth_client.post(
        VERIFY, json={"code": "123456"}, headers={**CSRF, "Cookie": "gl_mfa=forged"}
    )
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED


async def test_verify_needs_exactly_one_of_code_and_recovery_code(auth_client, auth_db, local_auth):
    user, _ = await mfa_user(auth_db)
    await login(auth_client, user.email)

    for body in ({}, {"code": "123456", "recovery_code": "ABCDE-FGHJK"}, {"code": ""}):
        r = await verify(auth_client, **body)
        assert r.status_code == 422, body


async def test_step_tokens_only_open_their_own_step(auth_client, auth_db, local_auth):
    """An enrolment token cannot answer a challenge, and a challenge cannot enrol a new phone."""
    plain = await verified_user(auth_db)
    await login(auth_client, plain.email)  # enrolment token
    r = await verify(auth_client, code="123456")
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED

    auth_client.cookies.clear()
    enrolled, _ = await mfa_user(auth_db)
    await login(auth_client, enrolled.email)  # challenge token
    r = await auth_client.post(SETUP_START, headers=CSRF)
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED
    r = await auth_client.post(SETUP_CONFIRM, json={"code": "123456"}, headers=CSRF)
    assert r.status_code == 401 and r.json() == CHALLENGE_EXPIRED
    assert (await reload(auth_db, enrolled.id)).mfa_pending_secret_enc is None


async def test_mfa_failures_count_toward_the_lockout(auth_client, auth_db, local_auth):
    """A password holder gets five codes per challenge; fresh challenges must not reset that."""
    user, secret = await mfa_user(auth_db)

    for _ in range(2):
        assert (await login(auth_client, user.email)).json() == {"next": "mfa"}
        for _ in range(5):
            r = await verify(auth_client, code=wrong_code(secret))
            assert r.status_code == 401 and r.json()["detail"] == "invalid_code"

    r = await login(auth_client, user.email)
    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS
    stored = await reload(auth_db, user.id)
    assert stored.failed_login_count == 10 and stored.locked_until is not None
    assert len(await events(auth_db, AuthEventType.ACCOUNT_LOCKED)) == 1


async def test_a_locked_account_cannot_answer_its_challenge(auth_client, auth_db, local_auth):
    user, secret = await mfa_user(auth_db)
    await login(auth_client, user.email)
    async with auth_db() as s:
        await s.execute(
            update(User)
            .where(User.id == user.id)
            .values(locked_until=now_utc() + timedelta(minutes=15))
        )
        await s.commit()

    r = await verify(auth_client, code=code_at(secret))

    assert r.status_code == 429 and r.json() == TOO_MANY_ATTEMPTS
    assert await sessions_of(auth_db, user.id) == []


async def test_totp_code_cannot_be_replayed(auth_client, auth_db, local_auth):
    user, secret = await mfa_user(auth_db)
    code = code_at(secret)
    await login(auth_client, user.email)
    assert (await verify(auth_client, code=code)).status_code == 200

    auth_client.cookies.clear()
    await login(auth_client, user.email)
    r = await verify(auth_client, code=code)

    assert r.status_code == 401 and r.json()["detail"] == "invalid_code"


async def test_recovery_code_login_sends_alert_and_is_single_use(auth_client, auth_db, local_auth):
    user, _ = await mfa_user(auth_db)
    codes = await give_recovery_codes(auth_db, user)
    typed = codes[3].replace("-", "").lower()  # Review Focus 3: case and hyphen do not matter

    await login(auth_client, user.email)
    r = await verify(auth_client, recovery_code=typed)
    assert r.status_code == 200, r.text
    assert r.json() == {"next": "app"}
    [session] = await sessions_of(auth_db, user.id)
    assert session.amr == ["pwd", "recovery"]
    assert local_auth.last.to == user.email
    assert "9 recovery codes" in local_auth.last_text
    [used] = await events(auth_db, AuthEventType.RECOVERY_CODE_USED)
    assert used.details["remaining"] == 9

    auth_client.cookies.clear()
    await login(auth_client, user.email)
    r = await verify(auth_client, recovery_code=codes[3])
    assert r.status_code == 401 and r.json() == {"detail": "invalid_code", "attempts_left": 4}


async def test_recovery_alert_failure_does_not_fail_the_login(auth_client, auth_db, local_auth):
    user, _ = await mfa_user(auth_db)
    codes = await give_recovery_codes(auth_db, user, count=2)
    local_auth.fail_with = ConnectionRefusedError("down")

    await login(auth_client, user.email)
    r = await verify(auth_client, recovery_code=codes[0])

    assert r.status_code == 200 and r.json() == {"next": "app"}
    assert "gl_at" in r.cookies
    used = await events(auth_db, AuthEventType.RECOVERY_CODE_USED)
    assert [e.details.get("notification") for e in used] == [None, "not_sent"]


async def test_enrolment_completes_without_mail(
    auth_client, auth_db, local_auth, settings_override
):
    """The recovery codes are shown once: an SMTP outage must not swallow them."""
    settings_override(SMTP_HOST=None)
    user = await verified_user(auth_db)
    await login(auth_client, user.email)
    secret = (await auth_client.post(SETUP_START, headers=CSRF)).json()["secret"]

    done = await auth_client.post(SETUP_CONFIRM, json={"code": code_at(secret)}, headers=CSRF)

    assert done.status_code == 200, done.text
    assert len(done.json()["recovery_codes"]) == 10
    enrolled = await events(auth_db, AuthEventType.MFA_ENROLLED)
    assert [e.details.get("notification") for e in enrolled] == [None, "not_sent"]


async def test_setup_confirm_with_a_wrong_code(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    await login(auth_client, user.email)
    secret = (await auth_client.post(SETUP_START, headers=CSRF)).json()["secret"]

    r = await auth_client.post(SETUP_CONFIRM, json={"code": wrong_code(secret)}, headers=CSRF)

    assert r.status_code == 401 and r.json() == {"detail": "invalid_code", "attempts_left": 4}
    stored = await reload(auth_db, user.id)
    assert stored.mfa_enabled_at is None and stored.mfa_pending_secret_enc
    assert await sessions_of(auth_db, user.id) == []

    # Still enrollable with the right code.
    r = await auth_client.post(SETUP_CONFIRM, json={"code": code_at(secret)}, headers=CSRF)
    assert r.status_code == 200, r.text


async def test_setup_confirm_without_start(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    await login(auth_client, user.email)

    r = await auth_client.post(SETUP_CONFIRM, json={"code": "123456"}, headers=CSRF)

    assert r.status_code == 401 and r.json()["detail"] == "invalid_code"
    assert (await reload(auth_db, user.id)).mfa_enabled_at is None


# ── Renewal and logout ───────────────────────────────────────────────────────


async def signed_in(client, auth_db) -> tuple[User, str]:
    """A user who went through login and a TOTP code; the client holds their cookies."""
    user, secret = await mfa_user(auth_db)
    await login(client, user.email)
    r = await verify(client, code=code_at(secret))
    assert r.status_code == 200, r.text
    return user, r.cookies["gl_rt"]


async def test_refresh_rotates(auth_client, auth_db, local_auth):
    user, first = await signed_in(auth_client, auth_db)

    r = await auth_client.post(REFRESH, headers=CSRF)

    assert r.status_code == 200, r.text
    found = set_cookies(r)
    assert {"gl_at", "gl_rt", "gl_session"} <= set(found)
    assert found["gl_rt"]["value"] != first
    assert (
        await auth_client.get(ME, headers=access_cookie(found["gl_at"]["value"]))
    ).status_code == 200


async def test_refresh_concurrent_tabs_stay_logged_in(auth_client, auth_db, local_auth):
    """Review Focus 5: two tabs refresh with the same refresh token within 30 s."""
    user, refresh_token = await signed_in(auth_client, auth_db)
    auth_client.cookies.clear()
    same_cookie = {**CSRF, "Cookie": f"gl_rt={refresh_token}"}

    first = await auth_client.post(REFRESH, headers=same_cookie)
    second = await auth_client.post(REFRESH, headers=same_cookie)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert "gl_rt" in set_cookies(first)  # the winner rotated
    assert set(set_cookies(second)) == {"gl_at"}  # the other tab keeps the winner's refresh token
    for r in (first, second):
        token = set_cookies(r)["gl_at"]["value"]
        assert (await auth_client.get(ME, headers=bearer(token))).status_code == 200
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is None


async def test_refresh_reuse_revokes_the_session(auth_client, auth_db, local_auth):
    user, old = await signed_in(auth_client, auth_db)
    rotated = await auth_client.post(REFRESH, headers=CSRF)
    new = set_cookies(rotated)["gl_rt"]["value"]
    async with auth_db() as s:  # the grace window is long gone
        await s.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user.id)
            .values(rotated_at=now_utc() - timedelta(seconds=60))
        )
        await s.commit()
    auth_client.cookies.clear()

    r = await auth_client.post(REFRESH, headers={**CSRF, "Cookie": f"gl_rt={old}"})

    assert r.status_code == 401 and r.json() == {"detail": "invalid_refresh"}
    found = set_cookies(r)
    assert all(is_cleared(found[name]) for name in ("gl_at", "gl_rt", "gl_session"))
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is not None  # committed despite the refusal
    [reuse] = await events(auth_db, AuthEventType.REFRESH_REUSE_DETECTED)
    assert reuse.user_id == user.id
    # The thief and the honest holder are both out.
    r = await auth_client.post(REFRESH, headers={**CSRF, "Cookie": f"gl_rt={new}"})
    assert r.status_code == 401


async def test_refresh_without_a_usable_cookie(auth_client, local_auth):
    for headers in (CSRF, {**CSRF, "Cookie": "gl_rt=unknown"}):
        r = await auth_client.post(REFRESH, headers=headers)
        assert r.status_code == 401 and r.json() == {"detail": "invalid_refresh"}
        found = set_cookies(r)
        assert all(is_cleared(found[name]) for name in ("gl_at", "gl_rt", "gl_session"))


ALL_AUTH_COOKIES = ("gl_at", "gl_rt", "gl_session", "gl_mfa")


def assert_all_cleared_with_their_scopes(response, domain=None) -> None:
    """Every auth cookie removed, each with the Path and Domain it was set with."""
    found = set_cookies(response)
    assert all(is_cleared(found[name]) for name in ALL_AUTH_COOKIES)
    for name in ("gl_at", "gl_session"):
        assert found[name]["path"] == "/" and found[name].get("domain") == domain
    for name in ("gl_rt", "gl_mfa"):
        assert found[name]["path"] == "/api/v1/auth" and "domain" not in found[name]


async def test_logout_revokes(auth_client, auth_db, local_auth):
    user, _ = await signed_in(auth_client, auth_db)
    access = auth_client.cookies["gl_at"]

    r = await auth_client.post(LOGOUT)  # a cookie write without the CSRF header
    assert r.status_code == 403 and r.json() == {"detail": "csrf_header_missing"}
    assert "set-cookie" not in r.headers  # cookies untouched
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is None

    r = await auth_client.post(LOGOUT, headers=CSRF)
    assert r.status_code == 204
    assert_all_cleared_with_their_scopes(r)
    assert (await auth_client.get(ME, headers=bearer(access))).status_code == 401
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is not None and session.revoked_reason == "logout"
    [revoked] = await events(auth_db, AuthEventType.SESSION_REVOKED)
    assert revoked.user_id == user.id and revoked.details["session_id"] == str(session.id)


async def test_logout_with_an_expired_access_cookie_revokes_through_the_refresh_cookie(
    auth_client, auth_db, local_auth, settings_override
):
    """On a shared computer: the 15-minute cookie is gone, the 30-day one must not survive."""
    user, refresh_token = await signed_in(auth_client, auth_db)
    [session] = await sessions_of(auth_db, user.id)
    expired = issue_access_token(
        user.id, session.id, ["pwd", "otp"], now=now_utc() - timedelta(minutes=20)
    )
    settings_override(AUTH_COOKIE_SECURE=True, AUTH_COOKIE_DOMAIN=".example.test")
    auth_client.cookies.clear()

    r = await auth_client.post(
        LOGOUT, headers={**CSRF, "Cookie": f"gl_at={expired}; gl_rt={refresh_token}"}
    )

    assert r.status_code == 204
    assert_all_cleared_with_their_scopes(r, domain=".example.test")
    found = set_cookies(r)
    for name in ALL_AUTH_COOKIES:
        assert found[name]["secure"] is True
    assert all(found[name]["httponly"] is True for name in ("gl_at", "gl_rt", "gl_mfa"))
    assert found["gl_mfa"]["samesite"].lower() == "strict"
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is not None and session.revoked_reason == "logout"
    assert len(await events(auth_db, AuthEventType.SESSION_REVOKED)) == 1
    r = await auth_client.post(REFRESH, headers={**CSRF, "Cookie": f"gl_rt={refresh_token}"})
    assert r.status_code == 401 and r.json() == {"detail": "invalid_refresh"}


async def test_logout_with_the_previous_refresh_token(auth_client, auth_db, local_auth):
    """A tab that still holds the refresh token another tab just rotated logs the session out."""
    user, old = await signed_in(auth_client, auth_db)
    assert (await auth_client.post(REFRESH, headers=CSRF)).status_code == 200
    auth_client.cookies.clear()

    r = await auth_client.post(LOGOUT, headers={**CSRF, "Cookie": f"gl_rt={old}"})

    assert r.status_code == 204
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is not None and session.revoked_reason == "logout"


async def test_logout_without_credentials_still_clears(auth_client, auth_db, local_auth):
    for headers in (
        CSRF,
        {**CSRF, "Cookie": "gl_at=garbage; gl_rt=unknown"},
        {**CSRF, "Authorization": "Bearer garbage"},
    ):
        r = await auth_client.post(LOGOUT, headers=headers)
        assert r.status_code == 204, headers
        assert_all_cleared_with_their_scopes(r)
    assert await events(auth_db, AuthEventType.SESSION_REVOKED) == []


async def test_logout_with_a_bearer_token_needs_no_csrf_header(auth_client, auth_db, local_auth):
    user, _ = await signed_in(auth_client, auth_db)
    access = auth_client.cookies["gl_at"]
    auth_client.cookies.clear()

    r = await auth_client.post(LOGOUT, headers=bearer(access))

    assert r.status_code == 204
    [session] = await sessions_of(auth_db, user.id)
    assert session.revoked_at is not None


async def test_logout_of_an_ended_session_records_nothing_more(auth_client, auth_db, local_auth):
    user, refresh_token = await signed_in(auth_client, auth_db)
    assert (await auth_client.post(LOGOUT, headers=CSRF)).status_code == 204

    r = await auth_client.post(LOGOUT, headers={**CSRF, "Cookie": f"gl_rt={refresh_token}"})

    assert r.status_code == 204
    assert len(await events(auth_db, AuthEventType.SESSION_REVOKED)) == 1


# ── Switches and limits ──────────────────────────────────────────────────────


async def test_endpoints_404_when_disabled(auth_client, settings_override):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=False, AUTH_JWT_PRIVATE_KEYS="")

    for path in (LOGIN, VERIFY, SETUP_START, SETUP_CONFIRM, REFRESH, LOGOUT):
        r = await auth_client.post(path, json={}, headers=CSRF)
        assert r.status_code == 404, path
        assert r.json() == {"detail": "Not Found"}
    r = await auth_client.get(JWKS)
    assert r.status_code == 200
    assert r.json() == {"keys": []}  # nothing configured, nothing signs


async def test_migration_window_serves_the_endpoints(
    auth_client, auth_db, local_auth, settings_override
):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=True)
    user = await verified_user(auth_db)

    r = await login(auth_client, user.email)

    assert r.status_code == 200 and r.json() == {"next": "mfa_setup"}


async def test_jwks_publishes_the_signing_keys(auth_client, local_keys, settings_override):
    settings_override(AUTH_MODE="supabase", AUTH_LOCAL_ENABLED=False)

    r = await auth_client.get(JWKS)

    assert r.status_code == 200
    keys = r.json()["keys"]
    assert len(keys) == 2 and all(k["kty"] == "EC" and k["kid"] for k in keys)
    assert all("d" not in k for k in keys)


async def test_login_rate_limit(auth_client, local_auth):
    for _ in range(10):
        r = await login(auth_client, f"nobody-{uuid4().hex[:6]}@example.test")
        assert r.status_code == 401

    r = await login(auth_client, "someone-else@example.test")

    assert r.status_code == 429
    assert r.json() == TOO_MANY_ATTEMPTS  # the same generic answer as a locked account


async def test_step_tokens_are_stored_hashed(auth_client, auth_db, local_auth):
    user = await verified_user(auth_db)
    r = await login(auth_client, user.email)
    raw = r.cookies["gl_mfa"]

    async with auth_db() as s:
        stored = (await s.execute(select(AuthOneTimeToken.token_hash))).scalars().all()
    assert stored and raw not in stored


# ── The compare-and-set guards behind a second factor ────────────────────────


async def test_a_concurrent_replay_loses_the_compare_and_set(auth_db, enc_keys):
    """A request that read the account before another one used the same code is refused by the
    database, not by what it read."""
    from app.api.endpoints.auth import _accept_totp

    user, secret = await mfa_user(auth_db)
    code = code_at(secret)
    async with auth_db() as s:
        stale = await s.get(User, user.id)  # "no step used yet"
    async with auth_db() as s:
        assert await _accept_totp(s, await s.get(User, user.id), code, now_utc())
        await s.commit()
    async with auth_db() as s:
        assert not await _accept_totp(s, stale, code, now_utc())


async def test_a_recovery_code_is_spent_once(auth_db, enc_keys):
    from app.api.endpoints.auth import _spend_recovery_code

    user, _ = await mfa_user(auth_db)
    [code] = await give_recovery_codes(auth_db, user, count=1)
    for expected in (True, False):
        async with auth_db() as s:
            assert await _spend_recovery_code(s, user.id, code, now_utc()) is expected
            await s.commit()
