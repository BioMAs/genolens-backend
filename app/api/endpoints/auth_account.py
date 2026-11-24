"""
Account flows of the local authentication (`/api/v1/auth`, spec §7.4): sign-up, e-mail
verification, forgotten and changed passwords, invitations, the caller's own sessions, and the
rotation of the second factor. The sign-in itself (login, 2FA, refresh, logout) and the helpers
reused here live in `app.api.endpoints.auth`.

Public routes (CSRF header required, like the login):

    POST /signup {email, password, full_name?}      -> 202, always
    POST /verify-email {token}                      -> 204
    POST /verify-email/resend {email}               -> 202, always
    POST /password/forgot {email}                   -> 202, always
    POST /password/reset {token, new_password}      -> 204, every session revoked, none opened
    POST /invite/accept {token, password}           -> `gl_mfa` (enrolment), {"next": "mfa_setup"};
                                                       a pending account without a password only

Session routes (a local session only: a service token or a hosted-provider token gets 403):

    POST   /password/change {current_password, new_password}  -> 204, the other sessions revoked
    GET    /sessions                                          -> the live sessions of the caller
    DELETE /sessions/{id}                                     -> 204; 404 for another account's
    POST   /sessions/revoke-all                               -> {"revoked": n}, keeps this one
    POST   /mfa/rotate/start {password, code}                 -> {"otpauth_uri", "secret"}
    POST   /mfa/rotate/confirm {code}                         -> {"recovery_codes"}
    POST   /mfa/recovery-codes/regenerate {code}              -> {"recovery_codes"}

Every route answers 404 while local authentication is off. The public routes also answer 404, in
the same way, during the pre-launch (`AUTH_MODE=supabase` with `AUTH_LOCAL_ENABLED` true, see
`app.api.endpoints.auth`): no self-service sign-up, link or invitation before `AUTH_MODE=local`,
while the administrators who signed in keep every session route. A refusal on a session route is
never a 401 (except for a missing or dead session): the web client reads 401 as "signed out".

A new password kills what the old one opened. `/password/reset` and `/password/change` revoke the
sessions (all of them, or all but the caller's) and, in the same transaction, spend every unused
one-time token of the account (`invalidate_user_tokens`): an MFA challenge or enrolment step won
with the old password, and any link mailed before the change, stop working with it.

Anti-enumeration. `/signup`, `/password/forgot` and `/verify-email/resend` answer the same `202`
body whether the address has an account or not, whatever happens next. Every e-mail of this module
leaves after the response, from a FastAPI background job (`_notify` in `app.api.endpoints.auth`):
the SMTP exchange is in no response's time, so a known address is not answered more slowly than an
unknown one, and an SMTP outage is logged and audited (`notification: not_sent`) by the job, never
answered (a 500 on the known addresses only would tell which ones have an account). The password
policy is checked before the address is looked up, so a weak password gets the same 400 for a known
and an unknown address. Both sign-up branches pay for one argon2 hash in the threadpool (the new
account stores it, the existing one throws it away), write a token and an audit row, and schedule
exactly one e-mail. What remains unequal between a known and an unknown address at
`/password/forgot` and `/verify-email/resend` is a few database statements (the link issued, the
audit row): no network exchange.

Limits. Five requests per IP and per hour on each of the three routes that send an e-mail (the
limiter's 429 is the generic `too_many_attempts` of the sign-in routes, `app.main`). Per address,
at most `LINK_EMAILS_PER_HOUR` link e-mails (verification and reset links together, all three
routes counted) within the hour, counted in the database from the tokens issued
(`count_recent_tokens`); beyond that the answer is the same 202 and nothing is sent.

The account lock. A wrong current password (`/password/change`, `/mfa/rotate/start`) and a wrong
current TOTP code (`/mfa/rotate/start`, `/mfa/recovery-codes/regenerate`) count toward the lock
exactly as a failed sign-in, and a locked account gets `429 too_many_attempts` on those routes. As
at sign-in, the session routes never reset the count: an opened session does, and so does a
completed `/password/reset`, which clears the count and the lock in the transaction that changes
the password (whoever proves the inbox must not stay locked out by somebody else's failures; the
next sign-in still asks for the second factor). A refused reset (bad link, password policy) leaves
both as they were. A wrong code while confirming a rotation counts too, and also drops the
pending secret: the rotation starts again from `/mfa/rotate/start` (password and current code), so
a session holder gets one guess per rotation, never a free run at the code of one the owner began.
`/mfa/rotate/confirm` also has the per-IP limit of `/mfa/verify`.

Transactions. Each route commits before it answers; its e-mails leave after the response, from a
background job, best effort (an SMTP failure shows in the audit log, not in the HTTP response).
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, NamedTuple, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic.networks import validate_email
from slowapi.util import get_remote_address
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_db
from app.api.endpoints.auth import (
    MFA_LIMIT,
    MfaConfirmRequest,
    _accept_totp,
    _clear_failures,
    _count_failure,
    _decrypt,
    _find_account,
    _is_locked,
    _load_user,
    _notify,
    _refusal,
    _spend,
    _storable,
    _too_many_attempts,
    require_csrf_header,
    require_local_auth,
    require_local_mode,
)
from app.auth import cookies
from app.auth.audit import record_event
from app.auth.clock import aware_utc, now_utc
from app.auth.emails import (
    send_account_exists_email,
    send_password_changed_email,
    send_password_reset_email,
    send_verify_email,
)
from app.auth.mfa import (
    encrypt_secret,
    generate_recovery_codes,
    hash_recovery_code,
    new_totp_secret,
    provisioning_uri,
    verify_totp,
)
from app.auth.one_time_tokens import (
    InvalidOneTimeToken,
    consume_token,
    count_recent_tokens,
    invalidate_user_tokens,
    issue_token,
    peek_token,
)
from app.auth.passwords import (
    PasswordPolicyError,
    hash_password,
    validate_new_password,
    verify_password,
)
from app.auth.principal import resolve_caller
from app.auth.sessions import revoke_all_sessions, revoke_session
from app.middleware.rate_limit import limiter
from app.models.auth import AuthEventType, AuthRecoveryCode, AuthSession
from app.models.auth import OneTimeTokenPurpose as Purpose
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

logger = logging.getLogger(__name__)

EMAIL_LIMIT = "5/hour"  # signup, forgot, verification resend: per IP (spec §7.6)
TOKEN_LIMIT = "30/hour"  # the routes that take a link token: per IP
LINK_EMAILS_PER_HOUR = 3
LINK_EMAIL_WINDOW = timedelta(hours=1)
_LINK_PURPOSES = (Purpose.EMAIL_VERIFY, Purpose.PASSWORD_RESET)

RESET_REASON = "password_reset"
CHANGE_REASON = "password_change"
USER_REVOKED_REASON = "user_revoked"
USER_REVOKED_ALL_REASON = "user_revoked_all"

ACCEPTED = {"status": "accepted"}

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(require_local_auth)])
# The public routes: closed until AUTH_MODE=local (checked first, so that a closed route answers 404
# whatever the request carries, as when local authentication is off), then the CSRF header.
_public = [Depends(require_local_mode), Depends(require_csrf_header)]


# ── Refusals ─────────────────────────────────────────────────────────────────


def _invalid_token() -> JSONResponse:
    return _refusal(status.HTTP_400_BAD_REQUEST, "invalid_token")


def _policy_refusal(exc: PasswordPolicyError) -> JSONResponse:
    return _refusal(status.HTTP_400_BAD_REQUEST, f"password_policy:{exc.code}")


def _bad_request(detail: str) -> JSONResponse:
    return _refusal(status.HTTP_400_BAD_REQUEST, detail)


def _no_content() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Request bodies ───────────────────────────────────────────────────────────


def _address(value: str) -> str:
    return _storable(value).strip().lower()


class SignupRequest(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)
    full_name: Optional[str] = Field(default=None, max_length=255)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        address = _address(value)
        validate_email(address)  # a syntactically valid address, or a 422
        return address

    @field_validator("password", "full_name")
    @classmethod
    def _text(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _storable(value)


class EmailRequest(BaseModel):
    email: str = Field(max_length=320)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return _address(value)


class TokenRequest(BaseModel):
    token: str = Field(max_length=512)


class PasswordResetRequest(BaseModel):
    token: str = Field(max_length=512)
    new_password: str = Field(max_length=1024)

    @field_validator("new_password")
    @classmethod
    def _text(cls, value: str) -> str:
        return _storable(value)


class InviteAcceptRequest(BaseModel):
    token: str = Field(max_length=512)
    password: str = Field(max_length=1024)

    @field_validator("password")
    @classmethod
    def _text(cls, value: str) -> str:
        return _storable(value)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(max_length=1024)
    new_password: str = Field(max_length=1024)

    @field_validator("current_password", "new_password")
    @classmethod
    def _text(cls, value: str) -> str:
        return _storable(value)


class MfaRotateStartRequest(BaseModel):
    password: str = Field(max_length=1024)
    code: str = Field(max_length=64)

    @field_validator("password")
    @classmethod
    def _text(cls, value: str) -> str:
        return _storable(value)


class SessionOut(BaseModel):
    id: UUID
    created_at: datetime
    last_used_at: datetime
    ip: Optional[str]
    user_agent: Optional[str]
    current: bool


# ── The caller of a session route ────────────────────────────────────────────


class LocalCaller(NamedTuple):
    user: User
    session_id: UUID


async def require_local_session(
    request: Request, db: AsyncSession = Depends(get_db)
) -> LocalCaller:
    """The account and session behind a local access token (cookie or Bearer). Anything else that
    authenticates elsewhere, a service token or a hosted-provider token, is 403 here: these routes
    act on local sessions and passwords. The CSRF header is enforced by the resolution itself, for
    a cookie-carried request."""
    principal, user = await resolve_caller(request, db)
    if principal.auth_source != "local" or principal.session_id is None or user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="local_session_required")
    return LocalCaller(user, principal.session_id)


# ── Links sent by e-mail ─────────────────────────────────────────────────────


async def _link_emails_sent(db: AsyncSession, user_id: UUID, now: datetime) -> int:
    since = now - LINK_EMAIL_WINDOW
    return sum([await count_recent_tokens(db, user_id, p, since) for p in _LINK_PURPOSES])


async def _send_link(
    db: AsyncSession,
    background: BackgroundTasks,
    request: Request,
    account: User,
    purpose: Purpose,
    event_type: AuthEventType,
    send: Callable[[str, str], Awaitable[None]],
    now: datetime,
    **details: Any,
) -> None:
    """Issue a `purpose` link for `account`, audit `event_type`, commit, then have the link
    e-mailed with `send(to, raw_token)` after the response (`_notify`). Within the hourly budget of
    the address only: beyond it, the request is audited (`throttled`) and nothing is issued nor
    sent. An SMTP failure is audited by the background job and never reaches the response."""
    if await _link_emails_sent(db, account.id, now) >= LINK_EMAILS_PER_HOUR:
        await record_event(
            db, event_type, user_id=account.id, request=request, throttled=True, **details
        )
        await db.commit()
        return
    raw = await issue_token(db, account.id, purpose, now=now)
    await record_event(db, event_type, user_id=account.id, request=request, **details)
    user_id, to = account.id, account.email
    await db.commit()
    _notify(background, request, user_id, event_type, lambda: send(to, raw))


async def _address_in_use(db: AsyncSession, email: str) -> bool:
    """Does any account have this address, ignoring case? `_find_account` refuses an address that
    several accounts share (until the migration's duplicates are merged): sign-up must not add one
    more."""
    count = await db.scalar(
        select(func.count()).select_from(User).where(func.lower(User.email) == email)
    )
    return bool(count)


# ── Public routes ────────────────────────────────────────────────────────────


@router.post("/signup", status_code=status.HTTP_202_ACCEPTED, dependencies=_public)
@limiter.limit(EMAIL_LIMIT, key_func=get_remote_address)
async def signup(
    request: Request,
    response: Response,
    body: SignupRequest,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Any:
    now = now_utc()
    email = body.email
    try:
        validate_new_password(body.password, email)
    except PasswordPolicyError as exc:
        return _policy_refusal(exc)
    # Paid in both branches, before the lookup: a known address costs what a new one does.
    password_hash = await run_in_threadpool(hash_password, body.password)

    account = await _find_account(db, email)
    if account is not None:
        await _send_link(
            db,
            background,
            request,
            account,
            Purpose.PASSWORD_RESET,
            AuthEventType.SIGNUP,
            send_account_exists_email,
            now,
            outcome="existing_account",
        )
        return ACCEPTED
    if await _address_in_use(db, email):
        logger.error("Sign-up ignored: several accounts share one address, ignoring case")
        return ACCEPTED

    user = User(
        id=uuid4(),
        email=email,
        full_name=body.full_name,
        role=UserRole.USER,
        subscription_plan=SubscriptionPlan.STARTER,
        status=UserStatus.ACTIVE,
        password_hash=password_hash,
        password_changed_at=now,
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        # Another sign-up for the same address won the race: that one sends the e-mail.
        await db.rollback()
        logger.info("Concurrent sign-up for one address; the second one sends nothing")
        return ACCEPTED
    await _send_link(
        db,
        background,
        request,
        user,
        Purpose.EMAIL_VERIFY,
        AuthEventType.SIGNUP,
        send_verify_email,
        now,
        outcome="created",
    )
    return ACCEPTED


@router.post("/verify-email", status_code=status.HTTP_204_NO_CONTENT, dependencies=_public)
@limiter.limit(TOKEN_LIMIT, key_func=get_remote_address)
async def verify_email(
    request: Request, response: Response, body: TokenRequest, db: AsyncSession = Depends(get_db)
) -> Response:
    now = now_utc()
    try:
        token = await consume_token(db, body.token, Purpose.EMAIL_VERIFY, now=now)
    except InvalidOneTimeToken:
        return _invalid_token()
    await db.execute(
        update(User)
        .where(User.id == token.user_id, User.email_verified_at.is_(None))
        .values(email_verified_at=now)
        .execution_options(synchronize_session=False)
    )
    await record_event(db, AuthEventType.EMAIL_VERIFIED, user_id=token.user_id, request=request)
    await db.commit()
    return _no_content()


@router.post("/verify-email/resend", status_code=status.HTTP_202_ACCEPTED, dependencies=_public)
@limiter.limit(EMAIL_LIMIT, key_func=get_remote_address)
async def resend_verification(
    request: Request,
    response: Response,
    body: EmailRequest,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Any:
    now = now_utc()
    account = await _find_account(db, body.email)
    if account is not None and account.email_verified_at is None:
        await _send_link(
            db,
            background,
            request,
            account,
            Purpose.EMAIL_VERIFY,
            AuthEventType.SIGNUP,
            send_verify_email,
            now,
            outcome="verification_resent",
        )
    return ACCEPTED


@router.post("/password/forgot", status_code=status.HTTP_202_ACCEPTED, dependencies=_public)
@limiter.limit(EMAIL_LIMIT, key_func=get_remote_address)
async def forgot_password(
    request: Request,
    response: Response,
    body: EmailRequest,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Any:
    now = now_utc()
    account = await _find_account(db, body.email)
    if account is not None:
        await _send_link(
            db,
            background,
            request,
            account,
            Purpose.PASSWORD_RESET,
            AuthEventType.PASSWORD_RESET_REQUESTED,
            send_password_reset_email,
            now,
        )
    return ACCEPTED


async def _account_of_link(
    db: AsyncSession, raw: str, purpose: Purpose, now: datetime
) -> Optional[User]:
    """The account a still-usable link belongs to, without spending the link: a password the
    policy refuses must leave it usable for the next try."""
    try:
        token = await peek_token(db, raw, purpose, now=now)
    except InvalidOneTimeToken:
        return None
    return await _load_user(db, token.user_id)


@router.post("/password/reset", status_code=status.HTTP_204_NO_CONTENT, dependencies=_public)
@limiter.limit(TOKEN_LIMIT, key_func=get_remote_address)
async def reset_password(
    request: Request,
    response: Response,
    body: PasswordResetRequest,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """No session is opened: the next sign-in still asks for the second factor (spec §7.4). The
    account lock and its count are cleared with the password, in the same transaction."""
    now = now_utc()
    user = await _account_of_link(db, body.token, Purpose.PASSWORD_RESET, now)
    if user is None:
        return _invalid_token()
    try:
        validate_new_password(body.new_password, user.email)
    except PasswordPolicyError as exc:
        return _policy_refusal(exc)
    new_hash = await run_in_threadpool(hash_password, body.new_password)
    if not await _spend(db, body.token, Purpose.PASSWORD_RESET, now):
        await db.rollback()  # another request spent the link first
        return _invalid_token()

    user.password_hash = new_hash
    user.password_changed_at = now
    if user.email_verified_at is None:
        user.email_verified_at = now  # the link reached the inbox, as a verification link would
    await _clear_failures(db, user.id)  # an UPDATE: it also wipes a failure counted meanwhile
    revoked = await revoke_all_sessions(db, user.id, RESET_REASON)
    tokens_spent = await invalidate_user_tokens(db, user.id, now=now)
    await record_event(
        db,
        AuthEventType.PASSWORD_RESET_COMPLETED,
        user_id=user.id,
        request=request,
        sessions_revoked=revoked,
        one_time_revoked=tokens_spent,
    )
    user_id, email = user.id, user.email
    await db.commit()
    _notify(
        background,
        request,
        user_id,
        AuthEventType.PASSWORD_RESET_COMPLETED,
        lambda: send_password_changed_email(email),
    )
    return _no_content()


@router.post("/invite/accept", dependencies=_public)
@limiter.limit(TOKEN_LIMIT, key_func=get_remote_address)
async def accept_invitation(
    request: Request,
    response: Response,
    body: InviteAcceptRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The password chosen, the address proven, a pending account activated; then, as after a
    password sign-in, the step token of the second factor (its enrolment, normally). Only for a
    PENDING account without a password: any other is refused (`invalid_token`, link spent)."""
    now = now_utc()
    user = await _account_of_link(db, body.token, Purpose.INVITE, now)
    if user is None:
        return _invalid_token()
    if user.password_hash is not None or user.status != UserStatus.PENDING:
        # A stale or forwarded invitation must not become a silent password reset: refused, and
        # spent so that it cannot be tried again.
        await _spend(db, body.token, Purpose.INVITE, now)
        await db.commit()
        logger.warning("Invitation refused: the account is not pending or already has a password")
        return _invalid_token()
    try:
        validate_new_password(body.password, user.email)
    except PasswordPolicyError as exc:
        return _policy_refusal(exc)
    new_hash = await run_in_threadpool(hash_password, body.password)
    if not await _spend(db, body.token, Purpose.INVITE, now):
        await db.rollback()
        return _invalid_token()

    user.password_hash = new_hash
    user.password_changed_at = now
    if user.email_verified_at is None:
        user.email_verified_at = now
    activated = user.status == UserStatus.PENDING
    if activated:
        user.status = UserStatus.ACTIVE
    if user.mfa_enabled_at is not None:
        purpose, next_step = Purpose.MFA_CHALLENGE, "mfa"
    else:
        purpose, next_step = Purpose.MFA_ENROLL, "mfa_setup"
    raw = await issue_token(db, user.id, purpose, now=now)
    await record_event(
        db,
        AuthEventType.INVITE_ACCEPTED,
        user_id=user.id,
        request=request,
        activated=activated,
        next=next_step,
    )
    await db.commit()
    cookies.set_step_cookie(response, raw)
    return {"next": next_step}


# ── Session routes: password ─────────────────────────────────────────────────


async def _locked_out(db: AsyncSession, request: Request, user: User, step: str) -> JSONResponse:
    await record_event(
        db, AuthEventType.LOGIN_FAILED, user_id=user.id, request=request, reason="locked", step=step
    )
    await db.commit()
    return _too_many_attempts()


async def _wrong_password(
    db: AsyncSession, request: Request, user: User, step: str, now: datetime
) -> JSONResponse:
    await _count_failure(db, user.id, request, now)
    await record_event(
        db,
        AuthEventType.LOGIN_FAILED,
        user_id=user.id,
        request=request,
        reason="invalid_credentials",
        step=step,
    )
    await db.commit()
    return _bad_request("invalid_password")


async def _wrong_code(
    db: AsyncSession, request: Request, user: User, step: str, now: datetime
) -> JSONResponse:
    await _count_failure(db, user.id, request, now)
    await record_event(
        db, AuthEventType.MFA_FAILED, user_id=user.id, request=request, method="otp", step=step
    )
    await db.commit()
    return _bad_request("invalid_code")


@router.post("/password/change", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: Request,
    body: PasswordChangeRequest,
    background: BackgroundTasks,
    caller: LocalCaller = Depends(require_local_session),
    db: AsyncSession = Depends(get_db),
) -> Response:
    now = now_utc()
    user = caller.user
    if _is_locked(user, now):
        return await _locked_out(db, request, user, "password_change")
    check = await run_in_threadpool(verify_password, body.current_password, user.password_hash)
    if not check.ok:
        return await _wrong_password(db, request, user, "password_change", now)
    try:
        validate_new_password(body.new_password, user.email)
    except PasswordPolicyError as exc:
        return _policy_refusal(exc)

    user.password_hash = await run_in_threadpool(hash_password, body.new_password)
    user.password_changed_at = now
    revoked = await revoke_all_sessions(
        db, user.id, CHANGE_REASON, keep_session_id=caller.session_id
    )
    tokens_spent = await invalidate_user_tokens(db, user.id, now=now)
    await record_event(
        db,
        AuthEventType.PASSWORD_CHANGED,
        user_id=user.id,
        request=request,
        sessions_revoked=revoked,
        one_time_revoked=tokens_spent,
    )
    user_id, email = user.id, user.email
    await db.commit()
    _notify(
        background,
        request,
        user_id,
        AuthEventType.PASSWORD_CHANGED,
        lambda: send_password_changed_email(email),
    )
    return _no_content()


# ── Session routes: sessions ─────────────────────────────────────────────────


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(
    caller: LocalCaller = Depends(require_local_session), db: AsyncSession = Depends(get_db)
) -> Any:
    """The caller's live sessions (not revoked, not expired), most recently used first."""
    now = now_utc()
    rows = (
        await db.execute(
            select(AuthSession)
            .where(
                AuthSession.user_id == caller.user.id,
                AuthSession.revoked_at.is_(None),
                AuthSession.idle_expires_at > now,
                AuthSession.expires_at > now,
            )
            .order_by(AuthSession.last_used_at.desc(), AuthSession.created_at.desc())
        )
    ).scalars()
    return [
        SessionOut(
            id=session.id,
            created_at=aware_utc(session.created_at),
            last_used_at=aware_utc(session.last_used_at),
            ip=session.ip,
            user_agent=session.user_agent,
            current=session.id == caller.session_id,
        )
        for session in rows
    ]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_session(
    session_id: UUID,
    request: Request,
    caller: LocalCaller = Depends(require_local_session),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """End one of the caller's sessions. Another account's session is answered exactly as one that
    does not exist (404): `revoke_session` looks up by id alone, the ownership is checked here."""
    owned = (
        await db.execute(
            select(AuthSession.id, AuthSession.revoked_at).where(
                AuthSession.id == session_id, AuthSession.user_id == caller.user.id
            )
        )
    ).first()
    if owned is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    if owned.revoked_at is None:
        await revoke_session(db, session_id, USER_REVOKED_REASON)
        await record_event(
            db,
            AuthEventType.SESSION_REVOKED,
            user_id=caller.user.id,
            request=request,
            session_id=session_id,
            reason=USER_REVOKED_REASON,
        )
    await db.commit()
    response = _no_content()
    if session_id == caller.session_id:
        cookies.clear_session_cookies(response)
    return response


@router.post("/sessions/revoke-all")
async def end_other_sessions(
    request: Request,
    caller: LocalCaller = Depends(require_local_session),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """End every session of the caller but the one making the request."""
    revoked = await revoke_all_sessions(
        db, caller.user.id, USER_REVOKED_ALL_REASON, keep_session_id=caller.session_id
    )
    await record_event(
        db,
        AuthEventType.SESSIONS_REVOKED_ALL,
        user_id=caller.user.id,
        request=request,
        count=revoked,
        kept_session_id=caller.session_id,
    )
    await db.commit()
    return {"revoked": revoked}


# ── Session routes: second factor ────────────────────────────────────────────


async def _replace_recovery_codes(db: AsyncSession, user_id: UUID) -> list[str]:
    """Ten new recovery codes, in clear for the response only; every earlier one is gone."""
    codes = generate_recovery_codes()
    await db.execute(delete(AuthRecoveryCode).where(AuthRecoveryCode.user_id == user_id))
    db.add_all(AuthRecoveryCode(user_id=user_id, code_hash=hash_recovery_code(c)) for c in codes)
    return codes


@router.post("/mfa/rotate/start")
async def start_mfa_rotation(
    request: Request,
    response: Response,
    body: MfaRotateStartRequest,
    caller: LocalCaller = Depends(require_local_session),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """A new phone: the password and a code of the current secret, then a new secret, pending
    until `/mfa/rotate/confirm`. The current code is checked with the same compare-and-set as at
    sign-in (`_accept_totp`), so it cannot be replayed."""
    now = now_utc()
    user = caller.user
    if user.mfa_enabled_at is None:
        return _bad_request("mfa_not_enabled")
    if _is_locked(user, now):
        return await _locked_out(db, request, user, "mfa_rotate")
    check = await run_in_threadpool(verify_password, body.password, user.password_hash)
    if not check.ok:
        return await _wrong_password(db, request, user, "mfa_rotate", now)
    if not await _accept_totp(db, user, body.code, now):
        return await _wrong_code(db, request, user, "mfa_rotate", now)

    secret = new_totp_secret()
    user.mfa_pending_secret_enc = encrypt_secret(secret)
    uri = provisioning_uri(secret, user.email)
    await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return {"otpauth_uri": uri, "secret": secret}


@router.post("/mfa/rotate/confirm")
@limiter.limit(MFA_LIMIT, key_func=get_remote_address)
async def confirm_mfa_rotation(
    request: Request,
    response: Response,
    body: MfaConfirmRequest,
    caller: LocalCaller = Depends(require_local_session),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The pending secret becomes the active one, and the recovery codes are replaced (shown once).

    The code is checked against the pending secret with no earlier step to refuse: no code of that
    secret was ever accepted, and the step just used with the old secret (at `/start`, often the
    same 30 seconds) says nothing about it. The swap stores the step of this code as the last one
    used, and is conditional on the pending secret still being the one checked, so two concurrent
    confirmations cannot both pass.

    A wrong code ends the rotation: the pending secret is dropped (start again, with the password
    and a code of the current secret) and the failure counts toward the account lock. One guess per
    rotation started, so a session holder cannot guess the code of a rotation the owner began."""
    now = now_utc()
    user = caller.user
    pending = user.mfa_pending_secret_enc
    if user.mfa_enabled_at is None or not pending:
        return _bad_request("no_rotation_pending")
    if _is_locked(user, now):
        return await _locked_out(db, request, user, "mfa_rotate_confirm")

    secret = _decrypt(pending)
    step = verify_totp(secret, body.code, None, now=now) if secret else None
    swapped = False
    if step is not None:
        result = await db.execute(
            update(User)
            .where(
                User.id == user.id,
                User.mfa_enabled_at.is_not(None),
                User.mfa_pending_secret_enc == pending,
            )
            .values(
                mfa_totp_secret_enc=pending, mfa_pending_secret_enc=None, mfa_last_used_step=step
            )
            .execution_options(synchronize_session=False)
        )
        swapped = result.rowcount == 1
    if not swapped:
        # Only the secret this code was checked against: a newer `/start` keeps its own.
        await db.execute(
            update(User)
            .where(User.id == user.id, User.mfa_pending_secret_enc == pending)
            .values(mfa_pending_secret_enc=None)
            .execution_options(synchronize_session=False)
        )
        return await _wrong_code(db, request, user, "mfa_rotate_confirm", now)

    codes = await _replace_recovery_codes(db, user.id)
    await record_event(db, AuthEventType.MFA_ROTATED, user_id=user.id, request=request)
    await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return {"recovery_codes": codes}


@router.post("/mfa/recovery-codes/regenerate")
async def regenerate_recovery_codes(
    request: Request,
    response: Response,
    body: MfaConfirmRequest,
    caller: LocalCaller = Depends(require_local_session),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Ten new recovery codes for a code of the active secret; the earlier ones stop working."""
    now = now_utc()
    user = caller.user
    if user.mfa_enabled_at is None:
        return _bad_request("mfa_not_enabled")
    if _is_locked(user, now):
        return await _locked_out(db, request, user, "recovery_codes")
    if not await _accept_totp(db, user, body.code, now):
        return await _wrong_code(db, request, user, "recovery_codes", now)

    codes = await _replace_recovery_codes(db, user.id)
    await record_event(
        db, AuthEventType.RECOVERY_CODES_REGENERATED, user_id=user.id, request=request
    )
    await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return {"recovery_codes": codes}
