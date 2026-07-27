"""
AccountService — account lifecycle: invitation, activation, suspension, cancellation.

Ported onto current `main` from feature/phase1-account-management, which branched
in April 2026 and predates the STARTER/TEAM/ON_PREMISE plan rename. Two defects in
the original were deliberately not carried over:

1. It created the local `User` row with a fresh `uuid4()`, unlinked to Supabase
   Auth. `User.id` is the Supabase auth user id everywhere else in this codebase
   (see `get_or_create_user` and `create_user` in the admin endpoints), so an
   invited person had no way to log in, and signing up later would mint a second
   row with a different id and collide on the unique email.
2. It sent a hand-rolled invitation email through a `send_invitation_email` helper
   that does not exist on `main`.

Both are solved by delegating to Supabase's admin invite endpoint, which creates
the auth user and sends the invitation magic link in one call.
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

logger = logging.getLogger(__name__)


class AccountService:
    """Account lifecycle operations. All methods are admin-initiated."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Supabase admin helpers ────────────────────────────────────────────────

    @staticmethod
    def _supabase_headers() -> dict:
        return {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

    @staticmethod
    async def _send_supabase_invite(email: str, full_name: Optional[str]) -> dict:
        """
        POST /auth/v1/invite — creates the auth user (if absent) and emails a
        magic link. Returns the Supabase user object.

        Raises RuntimeError with Supabase's own message on failure, so the caller
        can surface why rather than a generic error.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.SUPABASE_URL}/auth/v1/invite",
                headers=AccountService._supabase_headers(),
                params={"redirect_to": settings.APP_URL},
                json={"email": email, "data": {"full_name": full_name}},
            )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Supabase invite failed ({response.status_code}): {response.text}")
        return response.json()

    # ── Invitation ────────────────────────────────────────────────────────────

    async def invite_user(
        self,
        email: str,
        full_name: Optional[str],
        plan: SubscriptionPlan,
        invited_by_admin_id: UUID,
        subscription_ends_at: Optional[datetime] = None,
    ) -> User:
        """
        Invite someone: create their Supabase auth account, send the invitation
        magic link, and record a PENDING local profile keyed on the auth id.
        """
        result = await self._db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            raise ValueError(f"User with email {email} already exists")

        invited = await self._send_supabase_invite(email, full_name)
        auth_id = invited.get("id")
        if not auth_id:
            raise RuntimeError(f"Supabase invite returned no user id: {invited}")

        user = User(
            # The Supabase auth id, NOT a fresh uuid4 — this is the invariant that
            # lets the invited person actually log in and be recognised.
            id=UUID(auth_id),
            email=email,
            full_name=full_name,
            role=UserRole.USER,
            subscription_plan=plan,
            status=UserStatus.PENDING,
            # subscription_ends_at is a String(50) column; store ISO 8601.
            subscription_ends_at=subscription_ends_at.isoformat() if subscription_ends_at else None,
        )
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user)

        logger.info(
            "Invited user %s (plan=%s, auth_id=%s) by admin %s",
            email, plan.value, auth_id, invited_by_admin_id,
        )
        return user

    async def resend_invitation(self, user_id: UUID) -> User:
        """Re-send the invitation magic link. Only meaningful while PENDING."""
        user = await self._get_user(user_id)
        if user.status != UserStatus.PENDING:
            raise ValueError("Only pending users can receive an invitation resend")
        if not user.email:
            raise ValueError(f"User {user_id} has no email address to invite")

        await self._send_supabase_invite(user.email, user.full_name)
        logger.info("Resent invitation to %s", user.email)
        return user

    # ── Status transitions ────────────────────────────────────────────────────

    async def set_status(self, user_id: UUID, new_status: UserStatus) -> User:
        """Set a user's account status and persist it."""
        user = await self._get_user(user_id)
        user.status = new_status
        await self._db.commit()
        await self._db.refresh(user)
        logger.info("Set status of user %s to %s", user_id, new_status.value)
        return user

    async def activate_user(self, user_id: UUID) -> User:
        return await self.set_status(user_id, UserStatus.ACTIVE)

    async def suspend_user(self, user_id: UUID) -> User:
        return await self.set_status(user_id, UserStatus.SUSPENDED)

    async def cancel_user(self, user_id: UUID) -> User:
        return await self.set_status(user_id, UserStatus.CANCELLED)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _get_user(self, user_id: UUID) -> User:
        result = await self._db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user
