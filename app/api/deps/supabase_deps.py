"""
API dependencies for authentication and database sessions.
"""
from typing import Annotated, AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.supabase_auth import verify_supabase_token, SupabaseUser
from app.db.session import AsyncSessionLocal


# HTTP Bearer security scheme for JWT tokens
security = HTTPBearer()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for database sessions.
    Provides an async SQLAlchemy session that is properly closed after use.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> SupabaseUser:
    """
    Dependency to get the current authenticated user from Supabase JWT.
    
    Validates the JWT token and extracts user information including role and subscription.
    
    Args:
        credentials: HTTP Bearer credentials containing JWT token
        
    Returns:
        SupabaseUser with user_id, email, role, subscription_tier, etc.
        
    Raises:
        HTTPException: 401 Unauthorized if token is invalid
    """
    token = credentials.credentials
    user = await verify_supabase_token(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


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
    
    if current_user.role != UserRole.ADMIN:
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
