"""
Authentication dependencies for FastAPI.
"""
from typing import Optional
from uuid import UUID
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_token, get_user_from_token, CurrentUser
from app.db.session import get_db
from app.models.models import User, UserRole, Project, ProjectMember


# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> CurrentUser:
    """
    Dependency to get the current authenticated user from JWT token.

    Usage:
        @app.get("/protected")
        async def protected_route(user: CurrentUser = Depends(get_current_user)):
            return {"user_id": user.id}

    Args:
        credentials: HTTP Bearer credentials from request header
        db: Database session

    Returns:
        CurrentUser: Current authenticated user with full details

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    token_payload = verify_token(token)
    user_id = get_user_from_token(token_payload)
    
    # Load user from database
    result = await db.execute(select(User).where(User.id == user_id))
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: AsyncSession = Depends(get_db)
) -> Optional[CurrentUser]:
    """
    Dependency to optionally get the current user.
    Returns None if no valid token is provided.

    Usage:
        @app.get("/public-or-private")
        async def route(user: Optional[CurrentUser] = Depends(get_current_user_optional)):
            if user:
                return {"message": f"Hello {user.email}"}
            return {"message": "Hello anonymous"}

    Args:
        credentials: Optional HTTP Bearer credentials
        db: Database session

    Returns:
        Optional[CurrentUser]: User if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        token_payload = verify_token(token)
        user_id = get_user_from_token(token_payload)
        
        result = await db.execute(select(User).where(User.id == user_id))
        db_user = result.scalar_one_or_none()
        
        if not db_user or not db_user.is_active:
            return None
        
        return CurrentUser(
            id=db_user.id,
            email=db_user.email,
            full_name=db_user.full_name,
            role=db_user.role,
            subscription_tier=db_user.subscription_tier,
            max_projects=db_user.max_projects,
            current_project_count=db_user.current_project_count,
            features_access=db_user.features_access or {},
            is_active=db_user.is_active,
        )
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
    
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user)
    ) -> CurrentUser:
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(minimum_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Requires {minimum_role.value} role or higher.",
            )
        
        return current_user
    
    return role_checker


async def check_project_access(
    project_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Project:
    """
    Check if the current user has access to a project.
    
    Access is granted if:
    - User is ADMIN (can access all projects)
    - User is the project owner
    - User is a project member
    
    Usage:
        @app.get("/projects/{project_id}")
        async def get_project(
            project: Project = Depends(check_project_access)
        ):
            return project
    
    Args:
        project_id: Project ID to check
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Project: The project if access is granted
        
    Raises:
        HTTPException: If project not found or access denied
    """
    # Load project
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    
    # Admin can access everything
    if current_user.role == UserRole.ADMIN:
        return project
    
    # Check if user is owner
    if project.owner_id == current_user.id:
        return project
    
    # Check if user is a project member
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.id
        )
    )
    member = result.scalar_one_or_none()
    
    if member:
        return project
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have access to this project",
    )


async def check_subscription_limits(
    current_user: CurrentUser = Depends(get_current_user)
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
