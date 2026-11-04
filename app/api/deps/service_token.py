"""
Service-token authentication: which routes a `glst_` token may call, and as whom.

Deny by default. A route opens itself to service tokens with `@service_token_scope(...)`, placed
right under its router decorator. On every other authenticated route a valid service token gets
403: both authentication dependencies (`supabase_deps.get_current_user`, `auth.get_current_user`)
resolve it here, and every user-facing router pulls in the first through `require_active_account`.

A valid token resolves to its owner's `User` row, and everything downstream — licence, plan,
modules, quotas, project access, account status — sees that owner, unchanged.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Collection, Optional, TypeVar
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.service_tokens import hash_service_token, is_well_formed
from app.models.models import User
from app.models.service_token import ServiceToken

_RULE_ATTR = "__service_token_rule__"
_STATE_ATTR = "service_principal"

Endpoint = TypeVar("Endpoint", bound=Callable)


@dataclass(frozen=True)
class ServiceTokenRule:
    """What a service token needs to call one route: a scope, and fixed query values."""

    scope: str
    query: tuple[tuple[str, str], ...] = ()

    def refusal(self, request: Request, scopes: Collection[str]) -> Optional[str]:
        if self.scope not in scopes:
            return f"This service token lacks the scope {self.scope}"
        for name, value in self.query:
            if request.query_params.get(name) != value:
                return f"A service token is accepted on this route only with {name}={value}"
        return None


def service_token_scope(scope: str, **query: str) -> Callable[[Endpoint], Endpoint]:
    """Open the decorated route to service tokens holding `scope`; with `query`, only to
    requests whose query string carries these exact values (e.g. template="editorial")."""
    rule = ServiceTokenRule(scope, tuple(sorted(query.items())))

    def mark(endpoint: Endpoint) -> Endpoint:
        setattr(endpoint, _RULE_ATTR, rule)
        return endpoint

    return mark


@dataclass(frozen=True)
class ServicePrincipal:
    token_id: UUID
    owner: User


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or revoked service token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _route_rule(request: Request) -> Optional[ServiceTokenRule]:
    route = request.scope.get("route")
    return getattr(getattr(route, "endpoint", None), _RULE_ATTR, None)


async def resolve_service_token(
    request: Request, token: str, db: AsyncSession
) -> ServicePrincipal:
    """The owner behind `token`, when the token is valid and open on this route.

    401 for a malformed, unknown or revoked token, or one whose owner is gone; 403 for a valid
    token on a route its scopes do not open. Resolved once per request. No refusal, log or
    stored value ever carries the token.
    """
    cached = getattr(request.state, _STATE_ATTR, None)
    if cached is not None:
        return cached
    if not is_well_formed(token):
        raise _unauthorized()
    row = await db.scalar(
        select(ServiceToken).where(ServiceToken.token_sha256 == hash_service_token(token))
    )
    if row is None or row.revoked_at is not None:
        raise _unauthorized()
    rule = _route_rule(request)
    if rule is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Service tokens are not accepted on this route",
        )
    refusal = rule.refusal(request, row.scopes or ())
    if refusal:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=refusal)
    owner = await db.get(User, row.owner_id)
    if owner is None:
        raise _unauthorized()
    # Committed with the request's own work: a request that fails leaves it as it was.
    row.last_used_at = datetime.now(timezone.utc)
    principal = ServicePrincipal(token_id=row.id, owner=owner)
    setattr(request.state, _STATE_ATTR, principal)
    return principal


def service_token_id(request: Request) -> Optional[UUID]:
    """The id of the service token this request was authenticated with; None for a user."""
    principal = getattr(request.state, _STATE_ATTR, None)
    return principal.token_id if principal is not None else None
