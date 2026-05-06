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
from app.models.models import User, UserRole, UserStatus, Project, ProjectMember


security = HTTPBearer()


def _raise_inactive(user: User) -> None:
    """Raise a structured 403 for non-active accounts."""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "account_inactive",
            "status": user.status.value,
            "message": _status_message(user.status),
        },
    )


def _status_message(s: UserStatus) -> str:
    messages = {
        UserStatus.PENDING: "Your account is pending activation. Please check your email.",
        UserStatus.SUSPENDED: "Your account has been suspended. Please contact support.",
        UserStatus.CANCELLED: "Your subscription has expired. Please renew to continue.",
    }
    return messages.get(s, "Account inactive.")


def _to_current_user(db_user: User) -> CurrentUser:
    return CurrentUser(
        id=db_user.id,
        email=db_user.email,
        full_name=db_user.full_name,
        role=db_user.role,
        subscription_tier=db_user.subscription_plan,
        max_projects=100,
        current_project_count=0,
        features_access={},
        is_active=db_user.status == UserStatus.ACTIVE,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    token = credentials.credentials
    token_payload = verify_token(token)
    user_id = get_user_from_token(token_payload)

    result = await db.execute(select(User).where(User.id == user_id))
    db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if db_user.status != UserStatus.ACTIVE:
        _raise_inactive(db_user)

    return _to_current_user(db_user)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: AsyncSession = Depends(get_db),
) -> Optional[CurrentUser]:
    if not credentials:
        return None
    try:
        token = credentials.credentials
        token_payload = verify_token(token)
        user_id = get_user_from_token(token_payload)
        result = await db.execute(select(User).where(User.id == user_id))
        db_user = result.scalar_one_or_none()
        if not db_user or db_user.status != UserStatus.ACTIVE:
            return None
        return _to_current_user(db_user)
    except HTTPException:
        return None


def require_role(minimum_role: UserRole):
    role_hierarchy = {UserRole.USER: 0, UserRole.ADMIN: 1}

    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(minimum_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {minimum_role.value} role.",
            )
        return current_user

    return role_checker


async def check_project_access(
    project_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    if current_user.role == UserRole.ADMIN:
        return project
    if project.owner_id == current_user.id:
        return project
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.id,
        )
    )
    if result.scalar_one_or_none():
        return project
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


async def check_subscription_limits(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    if current_user.role == UserRole.ADMIN:
        return current_user
    if current_user.current_project_count >= current_user.max_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Project limit reached. Upgrade to create more projects.",
        )
    return current_user
