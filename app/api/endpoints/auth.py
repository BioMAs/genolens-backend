"""
Sign-in with a password and a second factor, session renewal and logout (`/api/v1/auth`).

The path to a session (spec §7.4):

    POST /login {email, password}
        2FA on   -> `gl_mfa` (challenge: 5 min, 5 codes)  -> {"next": "mfa"}
        2FA off  -> `gl_mfa` (enrolment: 15 min)          -> {"next": "mfa_setup"}
    POST /mfa/verify {code} | {recovery_code}             -> session, {"next": "app"}
    POST /mfa/setup/start                                 -> {"otpauth_uri", "secret"}
    POST /mfa/setup/confirm {code}                        -> session, {"recovery_codes", "next"}

A session is opened in exactly two places, both after a second factor was checked: a right code at
`/mfa/verify`, and the confirmation of an enrolment. `/login` never opens one; it only hands out
the step token that unlocks the next step, and a step token only opens its own step.

Every route but `/jwks.json` answers 404 while local authentication is off (`AUTH_MODE=supabase`
and `AUTH_LOCAL_ENABLED` false). Cookie-carried requests need the `X-GenoLens-Client` header (the
login too, spec §7.3); `/logout` exempts a Bearer caller, as the principal resolution does.
`/logout` always answers 204 and clears every auth cookie, whatever credentials it got.

Failures and the account lock. A wrong password and a wrong 2FA code at `/mfa/verify` both count
toward the account lock: from the tenth consecutive failure the account is locked for 15 minutes,
and every further failure locks it again, until a sign-in completes. The count is reset only when a
session opens, NOT after the password alone: otherwise a password holder could log in again
between challenges and guess five codes per challenge for ever. A locked account gets
`429 too_many_attempts` at `/login`, `/mfa/verify` and `/mfa/setup/confirm`, whatever it sends.
Wrong codes while confirming an enrolment spend the step token's attempts but not the account's:
the caller is confirming a secret it was just given, there is nothing to guess.

Anti-enumeration. An unknown address, an account without a password and a wrong password get the
same `401 invalid_credentials`, after the same work (`verify_password` pays for one argon2 and one
bcrypt check whatever the case) and the same database writes (the counter of an account, or an
audit row). An unknown address is locked like an account too: ten failures lock it for 15 minutes
and every further one locks it again, so the eleventh attempt does not tell whether the address
has an account. That state lives in the rate limiter's storage (Redis in production), keyed by a
SHA-256 of the address and forgotten one day after its first failure; an account's count waits
for a successful sign-in instead, so the two can still be told apart by someone who fails ten
times, waits a day, and fails again. The per-IP limits answer with the same `too_many_attempts`
body (`app.main`).

An account whose address is not verified yet is answered exactly as a wrong password, even for the
right one: `401 invalid_credentials`, the same failure count and lock, the same audit row (reason
`email_not_verified`, never in the response), and the same cost (the bcrypt check a refusal pays
is added to the right argon2 password). Otherwise sign-up then login would tell addresses apart:
sign-up makes an unverified account with the password typed for a new address, and only e-mails an
existing one. The client offers to resend the verification link after a failed sign-in instead.

Transactions. Each route commits its own writes before it answers. `reserve_attempt` is committed
at once, before the code is checked, so a refusal cannot hand the attempt back. A refresh commits
on every outcome: the row lock `rotate_refresh` takes must not be held, and the revocation of a
reused session must survive the 401. The notification e-mails (2FA turned on, recovery code used)
are sent after the commit and are best effort: an SMTP failure is logged and audited, the request
still succeeds (the recovery codes are shown only once).
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from slowapi.util import get_remote_address
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_db
from app.auth import cookies
from app.auth.audit import record_event
from app.auth.clock import aware_utc, now_utc
from app.auth.emails import send_mfa_enabled_email, send_recovery_code_used_email
from app.auth.keys import jwks
from app.auth.mfa import (
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    generate_recovery_codes,
    hash_recovery_code,
    new_totp_secret,
    provisioning_uri,
    recovery_code_matches,
    verify_totp,
)
from app.auth.one_time_tokens import (
    MAX_CHALLENGE_ATTEMPTS,
    InvalidOneTimeToken,
    consume_token,
    issue_token,
    peek_token,
    reserve_attempt,
)
from app.auth.passwords import _Spent, _top_up, verify_password
from app.auth.principal import CSRF_HEADER, local_tokens_accepted, read_credentials
from app.auth.sessions import (
    IssuedSession,
    RefreshRejected,
    create_session,
    get_session_user,
    revoke_session,
    rotate_refresh,
)
from app.auth.tokens import InvalidAccessToken, decode_access_token, hash_token, is_local_token
from app.core.config import settings
from app.middleware.rate_limit import limiter
from app.models.auth import AuthEventType, AuthOneTimeToken, AuthRecoveryCode, AuthSession
from app.models.auth import OneTimeTokenPurpose as Purpose
from app.models.models import User
from app.services.email_service import EmailNotConfiguredError, EmailSendError

logger = logging.getLogger(__name__)

LOCK_THRESHOLD = 10
LOCK_DURATION = timedelta(minutes=15)
UNKNOWN_ADDRESS_MEMORY = timedelta(days=1)

LOGIN_LIMIT = "10/minute;50/hour"
MFA_LIMIT = "30/hour"
REFRESH_LIMIT = "60/minute"

LOGOUT_REASON = "logout"


# ── Switches, CSRF, refusals ─────────────────────────────────────────────────


def require_local_auth() -> None:
    """404, as for a route that does not exist, while local authentication is off."""
    if not local_tokens_accepted():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


def require_csrf_header(request: Request) -> None:
    """The header a page of another site cannot set without a CORS preflight (spec §7.3)."""
    if not request.headers.get(CSRF_HEADER, "").strip():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="csrf_header_missing")


router = APIRouter(prefix="/auth", tags=["auth"])
_local = APIRouter(dependencies=[Depends(require_local_auth)])
_csrf = [Depends(require_csrf_header)]


def _refusal(status_code: int, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, **extra})


def _too_many_attempts() -> JSONResponse:
    return _refusal(status.HTTP_429_TOO_MANY_REQUESTS, "too_many_attempts")


def _invalid_code(attempts_left: int) -> JSONResponse:
    return _refusal(status.HTTP_401_UNAUTHORIZED, "invalid_code", attempts_left=attempts_left)


def _challenge_expired() -> JSONResponse:
    response = _refusal(status.HTTP_401_UNAUTHORIZED, "challenge_expired")
    cookies.clear_step_cookie(response)
    return response


def _invalid_refresh() -> JSONResponse:
    response = _refusal(status.HTTP_401_UNAUTHORIZED, "invalid_refresh")
    cookies.clear_session_cookies(response)
    return response


def _client(request: Request) -> tuple[Optional[str], Optional[str]]:
    ip = request.client.host if request.client is not None else None
    return ip, request.headers.get("user-agent")


# ── Request bodies ───────────────────────────────────────────────────────────


def _storable(value: str) -> str:
    """Text the database and the hashes accept: no NUL (Postgres refuses it in text) and no lone
    surrogate (not encodable as UTF-8). Refused as malformed input, before any account lookup."""
    if "\x00" in value:
        raise ValueError("must not contain NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("must be valid text") from None
    return value


class LoginRequest(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)

    @field_validator("email", "password")
    @classmethod
    def _text(cls, value: str) -> str:
        return _storable(value)


class MfaVerifyRequest(BaseModel):
    code: Optional[str] = Field(default=None, max_length=64)
    recovery_code: Optional[str] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _exactly_one(self) -> "MfaVerifyRequest":
        if bool(self.code) == bool(self.recovery_code):
            raise ValueError("send either code or recovery_code")
        return self


class MfaConfirmRequest(BaseModel):
    code: str = Field(max_length=64)


# ── The account lock ─────────────────────────────────────────────────────────


def _is_locked(user: User, now: datetime) -> bool:
    return user.locked_until is not None and aware_utc(user.locked_until) > now


async def _count_failure(db: AsyncSession, user_id: UUID, request: Request, now: datetime) -> None:
    """One more consecutive failure on the account; from the tenth, a 15-minute lock."""
    failures = (
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(failed_login_count=User.failed_login_count + 1)
            .returning(User.failed_login_count)
            .execution_options(synchronize_session=False)
        )
    ).scalar_one_or_none()
    if failures is not None and failures >= LOCK_THRESHOLD:
        locked_until = now + LOCK_DURATION
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(locked_until=locked_until)
            .execution_options(synchronize_session=False)
        )
        await record_event(
            db,
            AuthEventType.ACCOUNT_LOCKED,
            user_id=user_id,
            request=request,
            failures=failures,
            locked_until=locked_until,
        )


async def _clear_failures(db: AsyncSession, user_id: UUID) -> None:
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(failed_login_count=0, locked_until=None)
        .execution_options(synchronize_session=False)
    )


def _unknown_address_keys(email: str) -> tuple[str, str]:
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()
    return f"auth-unknown-failures/{digest}", f"auth-unknown-lock/{digest}"


def _unknown_address_locked(email: str) -> bool:
    """Is this address, which has no account, locked? A storage outage fails open (logged): a 500
    for unknown addresses only would itself tell which addresses have an account."""
    try:
        return limiter.limiter.storage.get(_unknown_address_keys(email)[1]) > 0
    except Exception as exc:
        logger.warning("Unknown-address lock not checked (%s)", type(exc).__name__)
        return False


def _count_unknown_address_failure(email: str) -> None:
    """The account lock, played for an address that has no account (see the module docstring)."""
    failures_key, lock_key = _unknown_address_keys(email)
    try:
        storage = limiter.limiter.storage
        failures = storage.incr(failures_key, int(UNKNOWN_ADDRESS_MEMORY.total_seconds()))
        if failures >= LOCK_THRESHOLD:
            storage.incr(lock_key, int(LOCK_DURATION.total_seconds()))
    except Exception as exc:
        logger.warning("Unknown-address failure not counted (%s)", type(exc).__name__)


# ── Accounts, step tokens, second factors ────────────────────────────────────


async def _find_account(db: AsyncSession, email: str) -> Optional[User]:
    """The account of a normalised address. Two accounts that differ only by case (possible until
    the migration's duplicates are merged) are both refused: neither can be told from the other."""
    rows = (
        (await db.execute(select(User).where(func.lower(User.email) == email).limit(2)))
        .scalars()
        .all()
    )
    if len(rows) > 1:
        logger.error("Sign-in refused: several accounts share one address, ignoring case")
        return None
    return rows[0] if rows else None


async def _load_user(db: AsyncSession, user_id: UUID) -> Optional[User]:
    return (
        await db.execute(
            select(User).where(User.id == user_id).execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def _reserve(
    db: AsyncSession, raw: str, purpose: Purpose, request: Request
) -> Optional[AuthOneTimeToken]:
    """Take one code check of the step token `raw`, committed at once; None when it is missing or
    can no longer be used (the caller answers `challenge_expired`)."""
    if not raw:
        return None
    try:
        token = await reserve_attempt(db, raw, purpose)
    except InvalidOneTimeToken as exc:
        await record_event(
            db, AuthEventType.MFA_FAILED, request=request, step=purpose.value, reason=exc.reason
        )
        await db.commit()
        return None
    await db.commit()
    return token


async def _spend(db: AsyncSession, raw: str, purpose: Purpose, now: datetime) -> bool:
    try:
        await consume_token(db, raw, purpose, now=now)
    except InvalidOneTimeToken:
        return False
    return True


def _decrypt(secret_enc: Optional[str]) -> Optional[str]:
    if not secret_enc:
        return None
    try:
        return decrypt_secret(secret_enc)
    except SecretDecryptionError:
        logger.error("A stored TOTP secret could not be decrypted; the code is refused")
        return None


async def _claim_step(
    db: AsyncSession, user_id: UUID, step: int, *conditions: Any, **values: Any
) -> bool:
    """Record `step` as the last TOTP step used, only if it is later than the stored one: two
    requests with the same code cannot both pass (the second matches no row)."""
    result = await db.execute(
        update(User)
        .where(
            User.id == user_id,
            or_(User.mfa_last_used_step.is_(None), User.mfa_last_used_step < step),
            *conditions,
        )
        .values(mfa_last_used_step=step, **values)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


async def _accept_totp(db: AsyncSession, user: User, code: str, now: datetime) -> bool:
    secret = _decrypt(user.mfa_totp_secret_enc)
    step = verify_totp(secret, code, user.mfa_last_used_step, now=now) if secret else None
    return step is not None and await _claim_step(db, user.id, step)


async def _spend_recovery_code(db: AsyncSession, user_id: UUID, typed: str, now: datetime) -> bool:
    """Spend the unused recovery code `typed` matches. Every stored code is compared (no early
    exit); the conditional update lets only one of two concurrent uses through."""
    rows = (
        await db.execute(
            select(AuthRecoveryCode.id, AuthRecoveryCode.code_hash).where(
                AuthRecoveryCode.user_id == user_id, AuthRecoveryCode.used_at.is_(None)
            )
        )
    ).all()
    match: Optional[UUID] = None
    for code_id, code_hash in rows:
        if recovery_code_matches(typed, code_hash) and match is None:
            match = code_id
    if match is None:
        return False
    result = await db.execute(
        update(AuthRecoveryCode)
        .where(AuthRecoveryCode.id == match, AuthRecoveryCode.used_at.is_(None))
        .values(used_at=now)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


async def _recovery_codes_left(db: AsyncSession, user_id: UUID) -> int:
    count = await db.scalar(
        select(func.count())
        .select_from(AuthRecoveryCode)
        .where(AuthRecoveryCode.user_id == user_id, AuthRecoveryCode.used_at.is_(None))
    )
    return int(count or 0)


async def _notify(
    db: AsyncSession,
    request: Request,
    user_id: UUID,
    event_type: AuthEventType,
    send: Callable[[], Awaitable[None]],
) -> None:
    """Send a security notification once the event it reports is committed. Best effort: the
    session is already open, so a failure is logged and audited (`notification: not_sent`), never
    raised. Anything else that goes wrong is caught too, after a traceback is logged: answering 500
    now would lose what the response carries (the recovery codes, shown once)."""
    try:
        await send()
        return
    except (EmailSendError, EmailNotConfiguredError) as exc:
        logger.warning("Security notification not sent (%s)", type(exc).__name__)
        error_kind = type(exc).__name__
    except Exception as exc:
        logger.exception("Security notification failed unexpectedly")
        error_kind = type(exc).__name__
    try:
        await record_event(
            db,
            event_type,
            user_id=user_id,
            request=request,
            notification="not_sent",
            error_kind=error_kind,
        )
        await db.commit()
    except Exception:
        logger.exception("Could not audit an unsent security notification")
        await db.rollback()


async def _open_session(
    db: AsyncSession, request: Request, user: User, amr: list[str], now: datetime
) -> IssuedSession:
    """The second factor passed: clear the failure count and open the session."""
    await _clear_failures(db, user.id)
    ip, user_agent = _client(request)
    return await create_session(db, user, amr, ip=ip, user_agent=user_agent, now=now)


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("/jwks.json")
async def get_jwks() -> dict:
    """The public keys of the access tokens. Always served; while local authentication is off and
    no key is configured nothing signs, so the set is empty (rather than a 500 in production)."""
    if not local_tokens_accepted() and not settings.AUTH_JWT_PRIVATE_KEYS.strip():
        return {"keys": []}
    return jwks()


@_local.post("/login", dependencies=_csrf)
@limiter.limit(LOGIN_LIMIT, key_func=get_remote_address)
async def login(
    request: Request, response: Response, body: LoginRequest, db: AsyncSession = Depends(get_db)
) -> Any:
    now = now_utc()
    email = body.email.strip().lower()
    user = await _find_account(db, email)
    user_id = user.id if user is not None else None

    locked = _is_locked(user, now) if user is not None else _unknown_address_locked(email)
    if locked:
        await record_event(
            db, AuthEventType.LOGIN_FAILED, user_id=user_id, request=request, reason="locked"
        )
        await db.commit()
        return _too_many_attempts()

    stored_hash = user.password_hash if user is not None else None
    check = await run_in_threadpool(verify_password, body.password, stored_hash)
    unverified = check.ok and user is not None and user.email_verified_at is None
    if unverified:
        # The right password of an unverified account is answered as a wrong one (see the module
        # docstring). A right argon2 password paid one argon2 check; a refusal also pays a bcrypt
        # one: added here so that the time does not tell them apart either.
        legacy = not (stored_hash or "").startswith("$argon2")
        await run_in_threadpool(_top_up, body.password, _Spent(argon2=True, bcrypt=legacy))
    if not check.ok or user is None or unverified:
        if user is not None:
            await _count_failure(db, user.id, request, now)
        else:
            _count_unknown_address_failure(email)
        await record_event(
            db,
            AuthEventType.LOGIN_FAILED,
            user_id=user_id,
            request=request,
            reason="email_not_verified" if unverified else "invalid_credentials",
        )
        await db.commit()
        return _refusal(status.HTTP_401_UNAUTHORIZED, "invalid_credentials")

    if check.upgraded_hash:
        user.password_hash = check.upgraded_hash

    if user.mfa_enabled_at is not None:
        purpose, next_step = Purpose.MFA_CHALLENGE, "mfa"
    else:
        purpose, next_step = Purpose.MFA_ENROLL, "mfa_setup"
    raw = await issue_token(db, user.id, purpose, now=now)
    await record_event(
        db,
        AuthEventType.LOGIN_SUCCEEDED,
        user_id=user.id,
        request=request,
        step="password",
        next=next_step,
    )
    await db.commit()
    cookies.set_step_cookie(response, raw)
    return {"next": next_step}


@_local.post("/mfa/verify", dependencies=_csrf)
@limiter.limit(MFA_LIMIT, key_func=get_remote_address)
async def verify_mfa(
    request: Request,
    response: Response,
    body: MfaVerifyRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    now = now_utc()
    raw = request.cookies.get(cookies.STEP_COOKIE) or ""
    token = await _reserve(db, raw, Purpose.MFA_CHALLENGE, request)
    user = await _load_user(db, token.user_id) if token is not None else None
    if token is None or user is None or user.mfa_enabled_at is None:
        return _challenge_expired()
    if _is_locked(user, now):
        await record_event(
            db, AuthEventType.MFA_FAILED, user_id=user.id, request=request, reason="locked"
        )
        await db.commit()
        return _too_many_attempts()

    if body.code:
        method, accepted = "otp", await _accept_totp(db, user, body.code, now)
    else:
        typed = body.recovery_code or ""
        method, accepted = "recovery", await _spend_recovery_code(db, user.id, typed, now)
    if not accepted:
        attempts_left = MAX_CHALLENGE_ATTEMPTS - token.attempts
        await _count_failure(db, user.id, request, now)
        await record_event(
            db,
            AuthEventType.MFA_FAILED,
            user_id=user.id,
            request=request,
            method=method,
            attempts_left=attempts_left,
        )
        await db.commit()
        return _invalid_code(attempts_left)

    if not await _spend(db, raw, Purpose.MFA_CHALLENGE, now):
        await db.rollback()  # the code is not spent either: another request won the challenge
        return _challenge_expired()
    issued = await _open_session(db, request, user, ["pwd", method], now)
    await record_event(
        db,
        AuthEventType.MFA_SUCCEEDED,
        user_id=user.id,
        request=request,
        method=method,
        session_id=issued.session_id,
    )
    remaining: Optional[int] = None
    if method == "recovery":
        remaining = await _recovery_codes_left(db, user.id)
        await record_event(
            db,
            AuthEventType.RECOVERY_CODE_USED,
            user_id=user.id,
            request=request,
            remaining=remaining,
        )
    user_id, email = user.id, user.email
    await db.commit()

    cookies.set_session_cookies(response, issued)
    cookies.clear_step_cookie(response)
    if remaining is not None:
        left = remaining
        await _notify(
            db,
            request,
            user_id,
            AuthEventType.RECOVERY_CODE_USED,
            lambda: send_recovery_code_used_email(email, left),
        )
    return {"next": "app"}


@_local.post("/mfa/setup/start", dependencies=_csrf)
async def start_mfa_setup(
    request: Request, response: Response, db: AsyncSession = Depends(get_db)
) -> Any:
    raw = request.cookies.get(cookies.STEP_COOKIE) or ""
    user: Optional[User] = None
    if raw:
        try:
            token = await peek_token(db, raw, Purpose.MFA_ENROLL)
        except InvalidOneTimeToken:
            pass
        else:
            user = await _load_user(db, token.user_id)
    if user is None or user.mfa_enabled_at is not None:
        return _challenge_expired()

    secret = new_totp_secret()
    user.mfa_pending_secret_enc = encrypt_secret(secret)
    uri = provisioning_uri(secret, user.email)
    await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return {"otpauth_uri": uri, "secret": secret}


@_local.post("/mfa/setup/confirm", dependencies=_csrf)
async def confirm_mfa_setup(
    request: Request,
    response: Response,
    body: MfaConfirmRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    now = now_utc()
    raw = request.cookies.get(cookies.STEP_COOKIE) or ""
    token = await _reserve(db, raw, Purpose.MFA_ENROLL, request)
    user = await _load_user(db, token.user_id) if token is not None else None
    if token is None or user is None or user.mfa_enabled_at is not None:
        return _challenge_expired()
    if _is_locked(user, now):
        await record_event(
            db,
            AuthEventType.MFA_FAILED,
            user_id=user.id,
            request=request,
            step="enroll",
            reason="locked",
        )
        await db.commit()
        return _too_many_attempts()

    pending = user.mfa_pending_secret_enc
    secret = _decrypt(pending)
    step = verify_totp(secret, body.code, user.mfa_last_used_step, now=now) if secret else None
    # One statement turns the factor on, and only for the secret this code was checked against.
    enabled = step is not None and await _claim_step(
        db,
        user.id,
        step,
        User.mfa_enabled_at.is_(None),
        User.mfa_pending_secret_enc == pending,
        mfa_totp_secret_enc=pending,
        mfa_pending_secret_enc=None,
        mfa_enabled_at=now,
    )
    if not enabled:
        attempts_left = MAX_CHALLENGE_ATTEMPTS - token.attempts
        await record_event(
            db,
            AuthEventType.MFA_FAILED,
            user_id=user.id,
            request=request,
            step="enroll",
            attempts_left=attempts_left,
        )
        await db.commit()
        return _invalid_code(attempts_left)

    codes = generate_recovery_codes()
    await db.execute(delete(AuthRecoveryCode).where(AuthRecoveryCode.user_id == user.id))
    db.add_all(AuthRecoveryCode(user_id=user.id, code_hash=hash_recovery_code(c)) for c in codes)
    if not await _spend(db, raw, Purpose.MFA_ENROLL, now):
        await db.rollback()
        return _challenge_expired()
    issued = await _open_session(db, request, user, ["pwd", "otp"], now)
    await record_event(
        db,
        AuthEventType.MFA_ENROLLED,
        user_id=user.id,
        request=request,
        session_id=issued.session_id,
    )
    user_id, email = user.id, user.email
    await db.commit()

    cookies.set_session_cookies(response, issued)
    cookies.clear_step_cookie(response)
    response.headers["Cache-Control"] = "no-store"
    await _notify(
        db, request, user_id, AuthEventType.MFA_ENROLLED, lambda: send_mfa_enabled_email(email)
    )
    return {"recovery_codes": codes, "next": "app"}


@_local.post("/refresh", dependencies=_csrf)
@limiter.limit(REFRESH_LIMIT, key_func=get_remote_address)
async def refresh(request: Request, response: Response, db: AsyncSession = Depends(get_db)) -> Any:
    raw = request.cookies.get(cookies.REFRESH_COOKIE)
    if not raw:
        return _invalid_refresh()
    try:
        outcome = await rotate_refresh(db, raw)
    except RefreshRejected as exc:
        if exc.reason == "reuse":
            session = (
                await db.execute(
                    select(AuthSession).where(
                        AuthSession.previous_refresh_token_hash == hash_token(raw)
                    )
                )
            ).scalar_one_or_none()
            await record_event(
                db,
                AuthEventType.REFRESH_REUSE_DETECTED,
                user_id=session.user_id if session is not None else None,
                request=request,
                session_id=session.id if session is not None else None,
            )
        await db.commit()  # the revocation of a reused session must outlive this refusal
        return _invalid_refresh()
    await db.commit()  # releases the session row `rotate_refresh` locked

    if outcome.refresh_token is not None:
        cookies.set_session_cookies(response, outcome)
    else:
        cookies.set_access_cookie(response, outcome.access_token)
    return {"refreshed": True}


async def _session_to_end(
    db: AsyncSession, access_token: Optional[str], refresh_token: Optional[str]
) -> Optional[AuthSession]:
    """The session a logout ends: the one of a valid local access token, else the one whose current
    or previous refresh hash is the `gl_rt` presented (looked up only: no rotation, no reuse
    check), else None."""
    if access_token and is_local_token(access_token):
        try:
            claims = decode_access_token(access_token)
        except InvalidAccessToken:
            claims = None  # typically expired: the refresh cookie still names the session
        if claims is not None:
            found = await get_session_user(db, claims.sid, claims.sub)
            if found is not None:
                return found[1]
    if refresh_token:
        hashed = hash_token(refresh_token)
        return (
            (
                await db.execute(
                    select(AuthSession).where(
                        or_(
                            AuthSession.refresh_token_hash == hashed,
                            AuthSession.previous_refresh_token_hash == hashed,
                        )
                    )
                )
            )
            .scalars()
            .first()
        )
    return None


@_local.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    """Always 204, and always clears every auth cookie, credentials or not: on a shared computer an
    expired `gl_at` must not leave a 30-day `gl_rt` behind. The session ended is the access
    token's, else the refresh cookie's (`_session_to_end`).

    A cookie-carried request needs the CSRF header (403, cookies untouched); a request with an
    `Authorization: Bearer` header is exempt, as everywhere (a page cannot set that header).
    """
    access_token, source = read_credentials(request)
    if source != "header":
        require_csrf_header(request)
    session = await _session_to_end(db, access_token, request.cookies.get(cookies.REFRESH_COOKIE))
    if session is not None and session.revoked_at is None:
        await revoke_session(db, session.id, LOGOUT_REASON)
        await record_event(
            db,
            AuthEventType.SESSION_REVOKED,
            user_id=session.user_id,
            request=request,
            session_id=session.id,
            reason=LOGOUT_REASON,
        )
    await db.commit()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    cookies.clear_session_cookies(response)
    cookies.clear_step_cookie(response)
    return response


router.include_router(_local)
