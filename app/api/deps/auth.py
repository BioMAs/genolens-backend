"""
Authentication dependencies for FastAPI.
"""

from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.principal import read_credentials, resolve_caller
from app.core.security import verify_token, get_user_from_token, CurrentUser
from app.core.service_tokens import is_service_token
from app.core.supabase_auth import SupabaseUser
from app.db.session import get_db
from app.models.models import User, UserRole


# Declares the Bearer scheme in the OpenAPI schema. It refuses nothing itself: the
# credential (header or `gl_at` cookie) is read and checked by `app.auth.principal`.
security = HTTPBearer(auto_error=False)


async def _verify_supabase_jwt(token: str) -> SupabaseUser:
    """A Supabase JWT, refused with this dependency's own messages ("Token has expired", "Could
    not validate credentials"). Only the user id matters: the user is read from the database."""
    token_payload = verify_token(token)
    return SupabaseUser(user_id=get_user_from_token(token_payload), email=token_payload.email or "")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """
    Dependency to get the current authenticated user from JWT token.

    The credential is resolved by `app.auth.principal.resolve_caller`, as for every protected
    route: a service token (`glst_…`) stands for its owner, and only on the routes its scopes open
    (`app.api.deps.service_token`; 401 if invalid or revoked, 403 elsewhere); a local access token
    is accepted with `AUTH_MODE=local` or `AUTH_LOCAL_ENABLED`; a Supabase JWT in supabase mode
    only. The user is then read from the database.

    Usage:
        @app.get("/protected")
        async def protected_route(user: CurrentUser = Depends(get_current_user)):
            return {"user_id": user.id}

    Args:
        request: The request (its `Authorization` header, or `gl_at` cookie)
        credentials: Unused; declares the Bearer scheme in the OpenAPI schema
        db: Database session

    Returns:
        CurrentUser: Current authenticated user with full details

    Raises:
        HTTPException: If authentication fails
    """
    principal, db_user = await resolve_caller(request, db, verify_hosted=_verify_supabase_jwt)
    if db_user is None:
        # Load user from database
        result = await db.execute(select(User).where(User.id == principal.user_id))
        db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return CurrentUser(
        id=db_user.id,
        email=db_user.email,
        full_name=db_user.full_name,
        role=db_user.role,
        subscription_tier=db_user.subscription_plan,
        max_projects=100,  # Legacy field
        current_project_count=0,  # Legacy field
        features_access={},  # Legacy field
        is_active=db_user.is_active,
    )


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[CurrentUser]:
    """
    Dependency to optionally get the current user.
    Returns None if no valid credential is provided.

    Usage:
        @app.get("/public-or-private")
        async def route(user: Optional[CurrentUser] = Depends(get_current_user_optional)):
            if user:
                return {"message": f"Hello {user.email}"}
            return {"message": "Hello anonymous"}

    Args:
        request: The request (its `Authorization` header, or `gl_at` cookie)
        credentials: Unused; declares the Bearer scheme in the OpenAPI schema
        db: Database session

    Returns:
        Optional[CurrentUser]: User if authenticated, None otherwise
    """
    token, _source = read_credentials(request)
    # A service token is never a JWT, and opens no optional route: anonymous.
    if token is None or is_service_token(token):
        return None

    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        return None


def require_role(minimum_role: UserRole):
    """
    Dependency factory to require a minimum user role.

    Role hierarchy: ADMIN > SUBSCRIBER > ANALYST > VIEWER

    Usage:
        @app.get("/admin-only")
        async def admin_route(user: CurrentUser = Depends(require_role(UserRole.ADMIN))):
            return {"message": "Admin access granted"}

    Args:
        minimum_role: Minimum required role

    Returns:
        Dependency function that checks user role
    """
    role_hierarchy = {
        UserRole.VIEWER: 0,
        UserRole.ANALYST: 1,
        UserRole.SUBSCRIBER: 2,
        UserRole.ADMIN: 3,
    }

    async def role_checker(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(minimum_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Requires {minimum_role.value} role or higher.",
            )

        return current_user

    return role_checker


async def check_subscription_limits(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """
    Check if the user can create a new project based on their subscription tier.

    Usage:
        @app.post("/projects")
        async def create_project(
            user: CurrentUser = Depends(check_subscription_limits)
        ):
            # User can create a new project
            pass

    Args:
        current_user: Current authenticated user

    Returns:
        CurrentUser: The user if they can create more projects

    Raises:
        HTTPException: If user has reached their project limit
    """
    # Admin has unlimited projects
    if current_user.role == UserRole.ADMIN:
        return current_user

    if current_user.current_project_count >= current_user.max_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project limit reached. You can create up to {current_user.max_projects} projects with your {current_user.subscription_tier.value} subscription. Please upgrade to create more projects.",
        )

    return current_user
