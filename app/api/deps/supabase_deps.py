"""
API dependencies for authentication and database sessions.
"""
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.principal import resolve_principal
from app.core.supabase_auth import SupabaseUser

# Re-exported, never redefined: a second get_db would give each request two
# sessions and deadlock the pool (see app.db.session.get_db).
from app.db.session import get_db  # noqa: F401


# Declares the Bearer scheme in the OpenAPI schema. It refuses nothing itself: the
# credential (header or `gl_at` cookie) is read and checked by `resolve_principal`.
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SupabaseUser:
    """
    Dependency to get the current authenticated user.

    Delegates to `app.auth.principal.resolve_principal`, which reads the credential (an
    `Authorization: Bearer` header, or the `gl_at` cookie while local tokens are accepted) and
    resolves, in order: a service token (`glst_…`, its owner, only on the routes its scopes open;
    401 if invalid or revoked, 403 elsewhere), a local access token (`AUTH_MODE=local` or
    `AUTH_LOCAL_ENABLED`; role and plan from the `users` row), a Supabase JWT (`AUTH_MODE=supabase`
    only; role from the Supabase profile). `db` is the request's own session (`get_db` is cached
    per request).

    Returns:
        SupabaseUser with user_id, email, role, subscription_tier, etc.

    Raises:
        HTTPException: 401 for an invalid credential; 403 for a missing credential in supabase
        mode (401 `not_authenticated` in local mode), or a cookie write without the CSRF header.
    """
    return await resolve_principal(request, db)


async def require_admin(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> SupabaseUser:
    """
    Dependency to require ADMIN role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        SupabaseUser if user is admin
        
    Raises:
        HTTPException: 403 Forbidden if user is not admin
    """
    from app.models.models import UserRole
    
    if current_user.role not in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


async def require_subscriber(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> SupabaseUser:
    """
    Dependency to require at least SUBSCRIBER role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        SupabaseUser if user is subscriber or above
        
    Raises:
        HTTPException: 403 Forbidden if user is only viewer
    """
    from app.models.models import UserRole
    
    allowed_roles = {UserRole.ADMIN, UserRole.SUBSCRIBER, UserRole.ANALYST}
    
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Subscriber access required"
        )
    
    return current_user
