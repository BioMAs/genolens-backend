"""
AccountService — manages account lifecycle: invitation, activation, suspension.
"""
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import SubscriptionPlan, User, UserRole, UserStatus
from app.services import email_service

logger = logging.getLogger(__name__)


class AccountService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def invite_user(
        self,
        email: str,
        full_name: Optional[str],
        plan: SubscriptionPlan,
        invited_by_admin_id: UUID,
        subscription_ends_at: Optional[datetime] = None,
    ) -> User:
        """Create a PENDING account and send an invitation email."""
        result = await self._db.execute(select(User).where(User.email == email))
        existing = result.scalar_one_or_none()
        if existing:
            raise ValueError(f"User with email {email} already exists")

        user = User(
            id=uuid4(),
            email=email,
            full_name=full_name,
            role=UserRole.USER,
            subscription_plan=plan,
            status=UserStatus.PENDING,
            subscription_ends_at=(
                subscription_ends_at.isoformat() if subscription_ends_at else None
            ),
        )
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user)

        await self._send_invitation_email(user)
        logger.info("Invited user %s (plan=%s) by admin %s", email, plan, invited_by_admin_id)
        return user

    async def activate_user(self, user_id: UUID) -> User:
        """Set a user's status to ACTIVE."""
        user = await self._get_user(user_id)
        user.status = UserStatus.ACTIVE
        await self._db.commit()
        logger.info("Activated user %s", user_id)
        return user

    async def suspend_user(self, user_id: UUID) -> User:
        """Manually block a user (admin action)."""
        user = await self._get_user(user_id)
        user.status = UserStatus.SUSPENDED
        await self._db.commit()
        logger.info("Suspended user %s", user_id)
        return user

    async def cancel_user(self, user_id: UUID) -> User:
        """Mark account as cancelled (subscription expired)."""
        user = await self._get_user(user_id)
        user.status = UserStatus.CANCELLED
        await self._db.commit()
        logger.info("Cancelled user %s", user_id)
        return user

    async def _get_user(self, user_id: UUID) -> User:
        result = await self._db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user

    async def _send_invitation_email(self, user: User) -> None:
        try:
            await email_service.send_invitation_email(
                to_email=user.email,
                full_name=user.full_name or "",
            )
        except Exception as exc:
            logger.warning("Failed to send invitation email to %s: %s", user.email, exc)
