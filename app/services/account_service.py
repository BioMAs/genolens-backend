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

The identity now comes from the directory (`app.auth.directory.create_identity`),
which does exactly that in `AUTH_MODE=supabase`. In `AUTH_MODE=local` the `users`
row IS the identity (a fresh id is the right one there), and the invitation is a
one-time INVITE link issued by the admin, committed, then e-mailed. A link that
could not be e-mailed is reported (`InvitationEmailError`), never passed off as
sent; the account and its link stay, so a resend works.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.audit import record_event
from app.auth.directory import create_identity, identities_are_local
from app.auth.emails import send_invitation_email
from app.auth.one_time_tokens import issue_token
from app.core.config import settings
from app.models.auth import AuthEventType, OneTimeTokenPurpose
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus
from app.services.email_service import EmailNotConfiguredError, EmailSendError

logger = logging.getLogger(__name__)


class InvitationEmailError(Exception):
    """The local invitation e-mail did not go out (SMTP refused it or is not configured).

    The account and its invitation link are committed: the admin can resend.
    """


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
        is_demo: bool = False,
        ai_requests_limit: Optional[int] = None,
        analyses_quota_override: Optional[int] = None,
        role: UserRole = UserRole.USER,
    ) -> User:
        """
        Invite someone: create their identity (`create_identity`), record a PENDING
        local profile keyed on its id, and send the invitation.

        `is_demo` and the two per-account caps let a demo (prospect) account be
        created in one step, rather than invited and then patched.

        Local mode: the address is stored normalised and refused if another account
        has it ignoring case (local sign-in looks addresses up that way, and refuses
        one two accounts share). The link is e-mailed after the commit; if it cannot
        be, `InvitationEmailError` is raised with the account and link kept.
        """
        local = identities_are_local()
        if local:
            email = email.strip().lower()
            existing = await self._db.scalar(
                select(User.id).where(func.lower(User.email) == email).limit(1)
            )
            taken = existing is not None
        else:
            result = await self._db.execute(select(User).where(User.email == email))
            taken = bool(result.scalar_one_or_none())
        if taken:
            raise ValueError(f"User with email {email} already exists")

        # Supabase: the auth id, NOT a fresh uuid4 — this is the invariant that lets
        # the invited person actually log in and be recognised. Local: the row is the
        # identity, so a fresh id is the right one.
        user_id = await create_identity(email, full_name)

        user = User(
            id=user_id,
            email=email,
            full_name=full_name,
            role=role,
            subscription_plan=plan,
            status=UserStatus.PENDING,
            # subscription_ends_at is a String(50) column; store ISO 8601.
            subscription_ends_at=subscription_ends_at.isoformat() if subscription_ends_at else None,
            is_demo=is_demo,
            ai_requests_limit=ai_requests_limit,
            ai_requests_used=0,
            analyses_quota_override=analyses_quota_override,
        )
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user)

        logger.info(
            "Invited user %s (plan=%s, auth_id=%s) by admin %s",
            email,
            plan.value,
            user_id,
            invited_by_admin_id,
        )
        if local:
            await self._send_local_invitation(user, invited_by_admin_id, resend=False)
        return user

    async def resend_invitation(
        self, user_id: UUID, resent_by_admin_id: Optional[UUID] = None
    ) -> User:
        """Re-send the invitation. Only meaningful while PENDING.

        Local mode: a fresh link (the previous one stops working), issued by
        `resent_by_admin_id`, committed, then e-mailed; `InvitationEmailError` if not.
        """
        user = await self._get_user(user_id)
        if user.status != UserStatus.PENDING:
            raise ValueError("Only pending users can receive an invitation resend")
        if not user.email:
            raise ValueError(f"User {user_id} has no email address to invite")

        if identities_are_local():
            await self._send_local_invitation(user, resent_by_admin_id, resend=True)
        else:
            await self._send_supabase_invite(user.email, user.full_name)
        logger.info("Resent invitation to %s", user.email)
        return user

    async def _send_local_invitation(
        self, user: User, admin_id: Optional[UUID], resend: bool
    ) -> None:
        """Issue the INVITE link (replacing any earlier one), commit, e-mail it, audit it.

        The e-mail goes out only once the link is committed, so that a link that
        arrives always works. It is audited (`INVITE_SENT`) only once it has gone.
        """
        user_id, to = user.id, user.email
        raw = await issue_token(self._db, user_id, OneTimeTokenPurpose.INVITE, created_by=admin_id)
        await self._db.commit()
        try:
            await send_invitation_email(to, raw)
        except (EmailSendError, EmailNotConfiguredError) as exc:
            logger.error(
                "Invitation e-mail for user %s not sent (%s); the link stays valid for a resend",
                user_id,
                type(exc).__name__,
            )
            raise InvitationEmailError("The invitation e-mail could not be sent") from exc
        await record_event(
            self._db,
            AuthEventType.INVITE_SENT,
            user_id=user_id,
            invited_by=str(admin_id) if admin_id else None,
            resend=resend,
        )
        await self._db.commit()

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
