"""
Who is calling: the single resolution of the caller behind every protected route.

Both authentication dependencies (`app.api.deps.supabase_deps.get_current_user` and
`app.api.deps.auth.get_current_user`) delegate here, so that a credential is read, checked and
turned into a principal in exactly one place.

The credential is an `Authorization: Bearer` header, else the `gl_at` cookie. The cookie is read
only while local tokens are accepted (`AUTH_MODE=local`, or `AUTH_LOCAL_ENABLED` during the
migration window): the hosted provider never set it, and the default configuration keeps
answering exactly as it did before this module existed.

Resolution order:

1. no credential: 401 `not_authenticated` in local mode; in supabase mode, today's refusal of
   `HTTPBearer` (403 `Not authenticated`, or `Invalid authentication credentials` for a scheme
   other than Bearer), verbatim;
2. a credential from the cookie on an unsafe method (not GET, HEAD, OPTIONS) without the
   `X-GenoLens-Client` header: 403 `csrf_header_missing`. A page of another site can make the
   browser send the cookie, not set a custom header (that forces a CORS preflight, refused for an
   unlisted origin); a Bearer request is exempt, a page cannot set `Authorization` either;
3. a service token (`glst_`): `resolve_service_token`, unchanged;
4. a local access token (ES256 header), only while local tokens are accepted: its signature and
   claims, then its session (`get_session_user`): an absent, revoked or expired session, or a
   deleted account, is 401 `invalid_token`. So is a session whose `amr` (in the token and in the
   stored session alike) records no second factor (`otp` or `recovery`): sessions are only opened
   after one, and this keeps that true structurally, whatever a future code path does. The
   principal comes from the `users` row (role and plan as stored), never from the token;
5. in supabase mode, the hosted-provider token, as before;
6. anything else: 401 `invalid_token`.

The flag is checked before anything about a local token is looked at: with local tokens off, an
ES256-headed token goes down the hosted path and is refused there as before, without ever
reaching the signing keys (absent in a production that has not switched).

Everything runs on the request's own database session (`get_db`, cached per request); nothing
here opens another one.
"""

from typing import TYPE_CHECKING, Awaitable, Callable, Literal, NamedTuple, Optional, Sequence

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.sessions import get_session_user
from app.auth.tokens import InvalidAccessToken, decode_access_token, is_local_token
from app.core.config import settings
from app.core.service_tokens import is_service_token
from app.core.supabase_auth import SupabaseUser, verify_supabase_token

if TYPE_CHECKING:
    from app.models.models import User

ACCESS_COOKIE = "gl_at"
CSRF_HEADER = "X-GenoLens-Client"

CredentialSource = Literal["header", "cookie"]
HostedVerifier = Callable[[str], Awaitable[SupabaseUser]]

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# The `amr` methods that prove a second factor. Every session is opened after one of them
# (`app.api.endpoints.auth`); a session that records neither is refused all the same.
_SECOND_FACTORS = frozenset({"otp", "recovery"})

# Raises exactly what the dependencies raised before this module, when a request carried no
# Bearer credential: kept as FastAPI's own code rather than a copy of its messages.
_HOSTED_BEARER = HTTPBearer(auto_error=True)


class ResolvedCaller(NamedTuple):
    """The principal, and the `users` row when the resolution loaded it (service token, local
    token); None after a hosted-provider token, whose row is not read here."""

    principal: SupabaseUser
    user: Optional["User"]


def local_tokens_accepted() -> bool:
    """Are local access tokens (and the `gl_at` cookie) accepted at all?"""
    return settings.AUTH_MODE == "local" or settings.AUTH_LOCAL_ENABLED


def read_credentials(request: Request) -> tuple[Optional[str], Optional[CredentialSource]]:
    """The credential of `request` and where it came from: the `Authorization: Bearer` header
    first, then the `gl_at` cookie. The cookie only while local tokens are accepted.

    The header is parsed as `HTTPBearer` parses it (scheme case-insensitive, the rest verbatim),
    so the token handed on is the one the dependencies always received.
    """
    scheme, credentials = get_authorization_scheme_param(request.headers.get("Authorization"))
    if scheme.lower() == "bearer" and credentials:
        return credentials, "header"
    if local_tokens_accepted():
        cookie = request.cookies.get(ACCESS_COOKIE)
        if cookie:
            return cookie, "cookie"
    return None, None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def _no_credentials(request: Request) -> HTTPException:
    if settings.AUTH_MODE == "supabase":
        await _HOSTED_BEARER(request)  # raises today's 403
        # Unreachable: there was no Bearer credential. Kept so the refusal cannot fall open.
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated")
    return _unauthorized("not_authenticated")


def _lacks_csrf_header(request: Request, source: CredentialSource) -> bool:
    if source != "cookie" or request.method.upper() in _SAFE_METHODS:
        return False
    return not request.headers.get(CSRF_HEADER, "").strip()


async def verify_hosted_token(token: str) -> SupabaseUser:
    """The hosted provider's token, refused as `supabase_deps.get_current_user` always did."""
    user = await verify_supabase_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def _second_factor_proven(amr: Sequence[str]) -> bool:
    return any(method in _SECOND_FACTORS for method in amr)


async def _resolve_local(token: str, db: AsyncSession) -> ResolvedCaller:
    try:
        claims = decode_access_token(token)
    except InvalidAccessToken:
        raise _unauthorized("invalid_token") from None
    found = await get_session_user(db, claims.sid, claims.sub)
    if found is None:
        raise _unauthorized("invalid_token")
    user, session = found
    if not (_second_factor_proven(claims.amr) and _second_factor_proven(session.amr or ())):
        raise _unauthorized("invalid_token")
    principal = SupabaseUser(
        user_id=user.id,
        email=user.email,
        role=user.role,
        subscription_tier=user.subscription_plan,
        session_id=session.id,
        auth_source="local",
    )
    return ResolvedCaller(principal, user)


async def resolve_caller(
    request: Request,
    db: AsyncSession,
    verify_hosted: HostedVerifier = verify_hosted_token,
) -> ResolvedCaller:
    """`resolve_principal`, plus the `users` row when the resolution loaded it.

    `verify_hosted` checks a hosted-provider token (supabase mode only); a dependency that has
    always refused those tokens with its own messages passes its own verifier.
    """
    token, source = read_credentials(request)
    if token is None or source is None:
        raise await _no_credentials(request)
    if _lacks_csrf_header(request, source):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="csrf_header_missing")

    if is_service_token(token):
        # Imported here: `app.api.deps` imports this module.
        from app.api.deps.service_token import resolve_service_token

        resolved = await resolve_service_token(request, token, db)
        owner = resolved.owner
        principal = SupabaseUser(
            user_id=owner.id,
            email=owner.email,
            role=owner.role,
            subscription_tier=owner.subscription_plan,
            service_token_id=resolved.token_id,
            auth_source="service_token",
        )
        return ResolvedCaller(principal, owner)

    if local_tokens_accepted() and is_local_token(token):
        return await _resolve_local(token, db)

    if settings.AUTH_MODE == "supabase":
        return ResolvedCaller(await verify_hosted(token), None)

    raise _unauthorized("invalid_token")


async def resolve_principal(request: Request, db: AsyncSession) -> SupabaseUser:
    """The caller of `request`, or the HTTP refusal (see the module docstring for the order)."""
    return (await resolve_caller(request, db)).principal
