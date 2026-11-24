"""
The cookies of the local authentication (spec §7.3): what each one carries, which hosts receive it
and how long it lives.

- `gl_at`, the access token: HttpOnly, SameSite=Lax, Domain=AUTH_COOKIE_DOMAIN, Path=/, 15 min;
- `gl_rt`, the refresh token: HttpOnly, SameSite=Lax, API host only (no Domain),
  Path=/api/v1/auth, 30 days;
- `gl_session`, `1` ("a session exists"): NOT HttpOnly, SameSite=Lax, Domain=AUTH_COOKIE_DOMAIN,
  Path=/, 30 days;
- `gl_mfa`, the step token (MFA challenge or enrolment): HttpOnly, SameSite=Strict, API host only,
  Path=/api/v1/auth, 15 min.

All four are `Secure` when `AUTH_COOKIE_SECURE` is on (off only for plain-HTTP local development).
An empty `AUTH_COOKIE_DOMAIN` makes `gl_at` and `gl_session` host-only too, which is what local
development wants (one `localhost` cookie, shared between ports). The `/api/v1/auth` path follows
`API_V1_PREFIX`.

The refresh token never leaves the API host: it is sent to `/api/v1/auth/*` only, never to the web
app's servers nor to the other subdomains. `gl_session` holds nothing secret; the web app's
middleware reads it to tell "signed out" from "access token expired, refresh first".

`gl_rt` and `gl_session` live 30 days, the absolute lifetime of a session; a cookie that outlives
its session is refused at the next refresh, which then clears it.

Clearing a cookie names the same Path and Domain it was set with: a browser keeps a cookie whose
deletion names another scope. Settings are read at each call, never at import.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, Optional, Union

from starlette.responses import Response

from app.auth.principal import ACCESS_COOKIE
from app.auth.sessions import ABSOLUTE_TTL, IssuedSession, RefreshOutcome
from app.auth.tokens import ACCESS_TOKEN_TTL
from app.core.config import settings

REFRESH_COOKIE = "gl_rt"
SESSION_HINT_COOKIE = "gl_session"
STEP_COOKIE = "gl_mfa"
AUTH_COOKIE_PATH = f"{settings.API_V1_PREFIX}/auth"

ACCESS_MAX_AGE = int(ACCESS_TOKEN_TTL.total_seconds())
SESSION_MAX_AGE = int(ABSOLUTE_TTL.total_seconds())
STEP_MAX_AGE = int(timedelta(minutes=15).total_seconds())

__all__ = [
    "ACCESS_COOKIE",
    "AUTH_COOKIE_PATH",
    "REFRESH_COOKIE",
    "SESSION_HINT_COOKIE",
    "STEP_COOKIE",
    "clear_session_cookies",
    "clear_step_cookie",
    "set_access_cookie",
    "set_session_cookies",
    "set_step_cookie",
]


@dataclass(frozen=True)
class _Cookie:
    name: str
    path: str
    # Sent to every host of AUTH_COOKIE_DOMAIN; otherwise to the API host only (no Domain).
    shared: bool
    httponly: bool
    samesite: Literal["lax", "strict"]
    max_age: int


_ACCESS = _Cookie(ACCESS_COOKIE, "/", True, True, "lax", ACCESS_MAX_AGE)
_REFRESH = _Cookie(REFRESH_COOKIE, AUTH_COOKIE_PATH, False, True, "lax", SESSION_MAX_AGE)
_HINT = _Cookie(SESSION_HINT_COOKIE, "/", True, False, "lax", SESSION_MAX_AGE)
_STEP = _Cookie(STEP_COOKIE, AUTH_COOKIE_PATH, False, True, "strict", STEP_MAX_AGE)


def _domain(cookie: _Cookie) -> Optional[str]:
    return (settings.AUTH_COOKIE_DOMAIN or None) if cookie.shared else None


def _set(response: Response, cookie: _Cookie, value: str) -> None:
    response.set_cookie(
        cookie.name,
        value,
        max_age=cookie.max_age,
        path=cookie.path,
        domain=_domain(cookie),
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=cookie.httponly,
        samesite=cookie.samesite,
    )


def _clear(response: Response, cookie: _Cookie) -> None:
    response.delete_cookie(
        cookie.name,
        path=cookie.path,
        domain=_domain(cookie),
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=cookie.httponly,
        samesite=cookie.samesite,
    )


def set_session_cookies(response: Response, issued: Union[IssuedSession, RefreshOutcome]) -> None:
    """`gl_at`, `gl_rt` and `gl_session`: after a login, or a refresh that rotated the token.

    A refresh answered in the grace window has no refresh token (the client keeps the one it has):
    that is `set_access_cookie`'s case, and passing it here is a `ValueError`.
    """
    if issued.refresh_token is None:
        raise ValueError("no refresh token to set: use set_access_cookie")
    _set(response, _ACCESS, issued.access_token)
    _set(response, _REFRESH, issued.refresh_token)
    _set(response, _HINT, "1")


def set_access_cookie(response: Response, access_token: str) -> None:
    """`gl_at` alone: a refresh in the grace window, which issues no new refresh token."""
    _set(response, _ACCESS, access_token)


def clear_session_cookies(response: Response) -> None:
    """Remove `gl_at`, `gl_rt` and `gl_session` (logout, refused refresh)."""
    for cookie in (_ACCESS, _REFRESH, _HINT):
        _clear(response, cookie)


def set_step_cookie(response: Response, raw: str) -> None:
    """`gl_mfa`: the step token handed out by the login (MFA challenge or enrolment)."""
    _set(response, _STEP, raw)


def clear_step_cookie(response: Response) -> None:
    """Remove `gl_mfa`: the step is done, or its token can no longer be used."""
    _clear(response, _STEP)
