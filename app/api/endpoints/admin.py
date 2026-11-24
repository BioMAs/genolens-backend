"""
Admin endpoints for user and system management.
Requires ADMIN role.
"""
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import httpx

from app.auth.admin_actions import (
    reset_user_mfa,
    revoke_user_service_tokens,
    revoke_user_sessions,
    unlock_user,
)
from app.auth.audit import record_event
from app.auth.directory import (
    AuthIdentity,
    IdentityDeletionError,
    IdentityNotFoundError,
    delete_identity,
    find_identity_by_id,
    identities_are_local,
    list_identities,
    role_source_is_supabase,
)
from app.auth.emails import send_mfa_reset_email, send_password_reset_email
from app.auth.one_time_tokens import issue_token
from app.auth.principal import local_tokens_accepted
from app.models.auth import AuthEvent, AuthEventType, OneTimeTokenPurpose
from app.models.service_token import ServiceToken
from app.services.email_service import EmailNotConfiguredError, EmailSendError

logger = logging.getLogger(__name__)

from app.api.deps import get_db, require_admin
from app.api.deps.subscription import _reset_quota_if_new_month
from app.api.deps.license import require_active_license
from app.core.supabase_auth import SupabaseUser, invalidate_cached_role
from app.core.config import settings
from app.models.models import Project, Dataset, ProjectMember, User, UserRole as UserRoleEnum, SubscriptionPlan, AIUsageLog, UserLoginEvent, UserStatus
from app.services.account_service import AccountService, InvitationEmailError
from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
)


router = APIRouter(prefix="/admin", tags=["Admin"])

# Statuses an admin may set directly. PENDING is owned by the invitation flow:
# it is set when the invite is sent and cleared when the invitee accepts, so
# setting it by hand would strand an account with no pending magic link.
_ADMIN_SETTABLE_STATUSES = (UserStatus.ACTIVE, UserStatus.SUSPENDED, UserStatus.CANCELLED)


# Pydantic Schemas
class UserProfile(BaseModel):
    """User profile from Supabase and local DB."""
    id: UUID
    email: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str
    status: str = "active"
    subscription_plan: str
    ai_interpretations_used: int = 0
    ai_tokens_purchased: int = 0
    ai_tokens_used: int = 0
    # None = unlimited (TEAM/ON_PREMISE/ADMIN); -1 = no access (STARTER)
    ai_interpretations_remaining: Optional[int] = None
    cosmetics_module_enabled: bool = False
    report_customization_module_enabled: bool = False
    scientific_module_enabled: bool = False
    drug_discovery_module_enabled: bool = False
    # ISO 8601, or null for an account with no end date. Enforced by
    # `require_active_account`: past this instant the account goes read-only.
    # It was settable at invitation time and invisible ever after, so an admin
    # could neither see nor correct a limit they had set.
    subscription_starts_at: Optional[str] = None
    subscription_ends_at: Optional[str] = None
    # Compte démo et plafonds par compte (voir PATCH /users/{id}/demo).
    is_demo: bool = False
    ai_requests_limit: Optional[int] = None      # None = pas de plafond IA
    ai_requests_used: int = 0
    analyses_quota_override: Optional[int] = None  # None = quota du plan
    analyses_used_this_month: int = 0
    analyses_bonus: int = 0                      # bonus de la période, inclus dans le quota
    analyses_quota: Optional[int] = None         # quota effectif, None = illimité
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_sign_in_at: Optional[str] = None
    confirmed_at: Optional[str] = None
    # Local sign-in state, read from the `users` row in both modes (see
    # `_local_auth_fields`). In supabase mode `last_sign_in_at` / `confirmed_at`
    # above stay the provider's own values. Never a hash or a secret.
    mfa_enabled: bool = False
    email_verified: bool = False
    last_login_at: Optional[datetime] = None
    locked_until: Optional[datetime] = None


class CreatedUserProfile(UserProfile):
    """What `POST /users` answers.

    In local mode creating an account is an invitation, and `invitation_sent` says
    the link went out. Supabase mode never sets it, and an unset flag is left out
    of the response, so that the supabase answer stays exactly what it was.
    """
    invitation_sent: Optional[bool] = None

    @model_serializer(mode="wrap")
    def _omit_unset_invitation_flag(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if data.get("invitation_sent") is None:
            data.pop("invitation_sent", None)
        return data


def _local_auth_fields(local_user: Optional[User]) -> dict:
    """Second factor, verified address, last login and lock of a UserProfile.

    From the local row in both modes; defaults without one. Only whether the
    second factor is on: never its secret, nor the password hash.
    """
    if local_user is None:
        return {}
    return {
        "mfa_enabled": local_user.mfa_enabled_at is not None,
        "email_verified": local_user.email_verified_at is not None,
        "last_login_at": local_user.last_login_at,
        "locked_until": local_user.locked_until,
    }


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _local_profile_fields(user: User) -> dict:
    """A UserProfile from the `users` row alone (local mode: the row is the identity).

    Same conventions as the supabase path (lowercased role and status, raw module
    flags); `last_sign_in_at` and `confirmed_at` are the local login and e-mail
    verification, so the admin panel's existing columns keep their meaning.
    """
    return dict(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=None,
        role=user.role.value.lower(),
        status=user.status.value.lower(),
        subscription_plan=user.subscription_plan.value,
        ai_interpretations_used=user.ai_interpretations_used,
        ai_interpretations_remaining=user.ai_interpretations_remaining,
        ai_tokens_purchased=user.ai_tokens_purchased,
        ai_tokens_used=user.ai_tokens_used,
        cosmetics_module_enabled=user.cosmetics_module_enabled,
        report_customization_module_enabled=user.report_customization_module_enabled,
        scientific_module_enabled=user.scientific_module_enabled,
        drug_discovery_module_enabled=user.drug_discovery_module_enabled,
        subscription_starts_at=user.subscription_starts_at,
        subscription_ends_at=user.subscription_ends_at,
        **_demo_fields(user),
        created_at=_iso(user.created_at),
        updated_at=_iso(user.updated_at),
        last_sign_in_at=_iso(user.last_login_at),
        confirmed_at=_iso(user.email_verified_at),
        **_local_auth_fields(user),
    )


def _demo_fields(local_user: Optional[User]) -> dict:
    """Champs démo / plafonds d'un UserProfile, défauts compris sans ligne locale."""
    if local_user is None:
        return {}
    return {
        "is_demo": bool(local_user.is_demo),
        "ai_requests_limit": local_user.ai_requests_limit,
        "ai_requests_used": local_user.ai_requests_used or 0,
        "analyses_quota_override": local_user.analyses_quota_override,
        "analyses_used_this_month": local_user.analyses_used_this_month or 0,
        "analyses_bonus": local_user.analyses_bonus or 0,
        "analyses_quota": local_user.analyses_quota,
    }

class SubscriptionUpdate(BaseModel):
    """Schema for updating subscription.

    `subscription_ends_at` distinguishes three cases deliberately, which is why
    the endpoint reads `model_fields_set` rather than just the value:

      field absent   → leave the current end date alone (a plan change on its own)
      explicit null  → clear the end date, the account becomes open-ended
      a datetime     → set/replace the end date

    Collapsing the first two would make a mistakenly set expiry impossible to
    lift: every plan change would silently re-clear it, or nothing ever could.
    """
    plan: SubscriptionPlan
    subscription_ends_at: Optional[datetime] = None

class CosmeticsModuleUpdate(BaseModel):
    """Schema for toggling the Cosmetics add-on module for a user."""
    enabled: bool

class ReportCustomizationModuleUpdate(BaseModel):
    """Schema for toggling the report customization add-on module for a user."""
    enabled: bool

class ScientificModuleUpdate(BaseModel):
    """Schema for toggling the Scientific tools add-on module for a user."""
    enabled: bool

class DrugDiscoveryModuleUpdate(BaseModel):
    """Schema for toggling the Drug Discovery add-on module for a user."""
    enabled: bool

class DemoAccountUpdate(BaseModel):
    """Compte démo et plafonds par compte.

    Même sémantique que `SubscriptionUpdate`, lue via `model_fields_set` :

      champ absent   → inchangé
      null explicite → effacé (plus de plafond IA / retour au quota du plan)
      une valeur     → posée

    `reset_usage` remet à zéro les deux compteurs (requêtes IA et analyses) :
    c'est ce qui permet de démarrer une démo propre, ou d'en prolonger une.
    """
    is_demo: Optional[bool] = None
    ai_requests_limit: Optional[int] = Field(None, ge=0)
    analyses_quota_override: Optional[int] = Field(None, ge=0)
    reset_usage: bool = False

class GrantAmount(BaseModel):
    """Quantité accordée par un admin (requêtes IA ou analyses)."""
    amount: int = Field(..., ge=1, le=100_000)

class TokenAdd(BaseModel):
    """Schema for adding AI tokens."""
    tokens: int

class AIUsageLogResponse(BaseModel):
    """Schema for AI usage log."""
    id: UUID
    user_id: UUID
    dataset_id: Optional[UUID] = None
    action_type: str
    comparison_name: Optional[str] = None
    model_used: str
    tokens_used: int
    was_free: bool
    created_at: str
    user_email: Optional[str] = None


class UserRoleUpdate(BaseModel):
    """Schema for updating user role."""
    role: str


class UserCreate(BaseModel):
    """Schema for creating a new user.

    `password` is required in supabase mode only (see `_require_password`); in
    local mode creating an account is an invitation and a password is ignored.
    """
    email: str
    password: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "user"


class _UserCreateWithPassword(UserCreate):
    """`UserCreate` as it was while every account got the admin's password."""
    password: str


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    full_name: Optional[str] = None
    role: Optional[str] = None


class InviteUserRequest(BaseModel):
    """Schema for inviting a user. Defaults to the entry-level plan."""
    email: str
    full_name: Optional[str] = None
    plan: SubscriptionPlan = SubscriptionPlan.STARTER
    subscription_ends_at: Optional[datetime] = None
    # Compte démo créé en une seule étape (voir DemoAccountUpdate).
    is_demo: bool = False
    ai_requests_limit: Optional[int] = Field(None, ge=0)
    analyses_quota_override: Optional[int] = Field(None, ge=0)


class UserStatusUpdate(BaseModel):
    """Schema for changing a user's account status."""
    status: str  # "active" | "suspended" | "cancelled"


class SystemStats(BaseModel):
    """System statistics."""
    total_users: int
    total_projects: int
    total_datasets: int
    active_users: int
    users_by_plan: dict[str, int]
    estimated_revenue: float


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = None
    description: Optional[str] = None
    owner_id: Optional[UUID] = None


class ProjectMemberCreate(BaseModel):
    """Schema for adding a project member."""
    user_id: UUID
    access_level: str = "VIEWER"


class ProjectMemberResponse(BaseModel):
    """Project member with user information."""
    id: UUID
    project_id: UUID
    user_id: UUID
    access_level: str
    created_at: str
    updated_at: str
    user_email: Optional[str] = None
    user_full_name: Optional[str] = None


# Admin Endpoints

@router.get("/users", response_model=List[UserProfile])
async def list_users(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users with their profiles, auth info, and subscription status.
    Admin only.

    Local mode: the `users` table alone (no provider call).
    """
    if identities_are_local():
        users = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
        return [UserProfile(**_local_profile_fields(u)) for u in users]

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # Get local user stats first (including subscription info)
            query = select(User)
            result = await db.execute(query)
            local_users = {str(u.id): u for u in result.scalars().all()}
            
            # Get profiles
            profiles_response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"select": "id,full_name,avatar_url,role,created_at,updated_at"}
            )

            if profiles_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch profiles from Supabase"
                )

            profiles = profiles_response.json()

            # Get auth users for email and last_sign_in
            auth_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers
            )

            auth_users = {}
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                users_list = auth_data.get("users", []) if isinstance(auth_data, dict) else auth_data
                for user in users_list:
                    auth_users[user["id"]] = {
                        "email": user.get("email"),
                        "last_sign_in_at": user.get("last_sign_in_at"),
                        "confirmed_at": user.get("confirmed_at")
                    }

            # Merge profile, auth data, and local subscription data
            result = []
            for profile in profiles:
                user_id = profile["id"]
                auth_info = auth_users.get(user_id, {})
                local_user = local_users.get(user_id)
                
                # Default values if user not in local DB yet
                sub_plan = "STARTER"
                user_status = "active"
                ai_used = 0
                ai_purchased = 0
                ai_tokens_used = 0
                ai_remaining = 0
                cosmetics_enabled = False
                report_customization_enabled = False
                scientific_enabled = False
                drug_discovery_enabled = False
                subscription_starts_at = None
                subscription_ends_at = None
                role_value = profile.get("role", "USER")

                if local_user:
                    sub_plan = local_user.subscription_plan.value
                    user_status = local_user.status.value.lower()
                    ai_used = local_user.ai_interpretations_used
                    ai_purchased = local_user.ai_tokens_purchased
                    ai_tokens_used = local_user.ai_tokens_used
                    ai_remaining = local_user.ai_interpretations_remaining
                    # Local role is authoritative (it drives access gating); the Supabase
                    # profile role can be stale (hence "Blocked role downgrade" warnings).
                    # Lowercased to match the admin panel's role convention.
                    role_value = local_user.role.value.lower()
                    # Raw per-user flags (NOT has_* which is role-inflated: admins would
                    # always show ON, making the toggle look stuck). Admin panel edits
                    # the stored flag; effective access is still role-aware at feature endpoints.
                    cosmetics_enabled = local_user.cosmetics_module_enabled
                    report_customization_enabled = local_user.report_customization_module_enabled
                    scientific_enabled = local_user.scientific_module_enabled
                    drug_discovery_enabled = local_user.drug_discovery_module_enabled
                    subscription_starts_at = local_user.subscription_starts_at
                    subscription_ends_at = local_user.subscription_ends_at

                result.append(UserProfile(
                    id=UUID(user_id),
                    email=auth_info.get("email"),
                    full_name=profile.get("full_name"),
                    avatar_url=profile.get("avatar_url"),
                    role=role_value,
                    status=user_status,
                    subscription_plan=sub_plan,
                    ai_interpretations_used=ai_used,
                    ai_interpretations_remaining=ai_remaining,
                    ai_tokens_purchased=ai_purchased,
                    ai_tokens_used=ai_tokens_used,
                    cosmetics_module_enabled=cosmetics_enabled,
                    report_customization_module_enabled=report_customization_enabled,
                    scientific_module_enabled=scientific_enabled,
                    drug_discovery_module_enabled=drug_discovery_enabled,
                    subscription_starts_at=subscription_starts_at,
                    subscription_ends_at=subscription_ends_at,
                    **_demo_fields(local_user),
                    created_at=profile.get("created_at"),
                    updated_at=profile.get("updated_at"),
                    last_sign_in_at=auth_info.get("last_sign_in_at"),
                    confirmed_at=auth_info.get("confirmed_at"),
                    **_local_auth_fields(local_user),
                ))

            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users: {str(e)}"
        )


@router.post("/users/invite", status_code=status.HTTP_201_CREATED)
async def invite_user(
    body: InviteUserRequest,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a pending account and send the Supabase invitation magic link.
    Admin only.

    Local mode: the invitation is a one-time link e-mailed by the backend. If the
    e-mail cannot be sent the answer is 502 `invitation_email_failed`: the account
    and its link are kept, and `resend-invite` sends a new one.

    Declared before the parameterised `/users/{user_id}` routes so the literal
    "invite" segment cannot be captured as a user id.
    """
    if body.is_demo and body.subscription_ends_at is None:
        # Une démo sans échéance ne se termine jamais : refusé avant d'envoyer
        # l'invitation, qu'on ne pourrait plus retirer.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A demo account needs an access end date (subscription_ends_at).",
        )
    service = AccountService(db)
    try:
        user = await service.invite_user(
            email=body.email,
            full_name=body.full_name,
            plan=body.plan,
            invited_by_admin_id=current_user.user_id,
            subscription_ends_at=body.subscription_ends_at,
            is_demo=body.is_demo,
            ai_requests_limit=body.ai_requests_limit,
            analyses_quota_override=body.analyses_quota_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except InvitationEmailError:
        raise _invitation_email_failed()
    except RuntimeError as exc:
        # Supabase rejected the invite (already registered, bad address, email
        # sending not configured on the project). Surface its own message.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    return {"id": str(user.id), "email": user.email, "status": user.status.value}


def _invitation_email_failed() -> HTTPException:
    """Local mode: the invitation link exists but its e-mail did not leave. The
    admin must see it (and can resend); it is never reported as sent."""
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY, detail="invitation_email_failed"
    )


@router.post("/users/{user_id}/resend-invite")
async def resend_invite(
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Re-send the invitation magic link to a pending user. Admin only.

    Local mode: a new link (the previous one stops working); 502
    `invitation_email_failed` if its e-mail cannot be sent.
    """
    service = AccountService(db)
    try:
        user = await service.resend_invitation(user_id, resent_by_admin_id=current_user.user_id)
    except ValueError as exc:
        message = str(exc)
        code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in message
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=message)
    except InvitationEmailError:
        raise _invitation_email_failed()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    return {"detail": "Invitation email resent", "email": user.email}


@router.patch("/users/{user_id}/status")
async def update_user_status(
    user_id: UUID,
    body: UserStatusUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Activate, suspend or cancel a user account. Admin only."""
    try:
        target_status = UserStatus(body.status)
    except ValueError:
        allowed = ", ".join(s.value for s in _ADMIN_SETTABLE_STATUSES)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{body.status}'. Must be one of: {allowed}",
        )

    if target_status not in _ADMIN_SETTABLE_STATUSES:
        # PENDING is set by the invitation flow and cleared when the invitee
        # accepts; letting an admin set it by hand would strand the account.
        allowed = ", ".join(s.value for s in _ADMIN_SETTABLE_STATUSES)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot set status to '{target_status.value}'. Must be one of: {allowed}",
        )

    service = AccountService(db)
    try:
        user = await service.set_status(user_id, target_status)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return {"id": str(user.id), "email": user.email, "status": user.status.value}


# ── Security actions on an account ────────────────────────────────────────────
#
# They act on the local `users` row and its authentication tables, so they work in both AUTH_MODE
# values and never call the hosted provider. The logic itself is in `app.auth.admin_actions`
# (shared with the break-glass command); the routes check who may ask, commit, then notify.
#
# A service token is revoked by `sessions/revoke` with `revoke_service_tokens` and by nothing
# else: not by the second-factor reset, the unlock, nor the password link (spec 7.7).

_ADMIN_ROLES = (UserRoleEnum.ADMIN, UserRoleEnum.SCILICIUM_ADMIN)


class MfaResetResponse(BaseModel):
    sessions_revoked: int
    # False when the account could not be told (the reset stands either way).
    notification_sent: bool


class SessionsRevokeRequest(BaseModel):
    revoke_service_tokens: bool = False


class SessionsRevokeResponse(BaseModel):
    sessions_revoked: int
    service_tokens_revoked: int


class UnlockResponse(BaseModel):
    unlocked: bool = True


class PasswordResetResponse(BaseModel):
    email_sent: bool = True


class AuthEventResponse(BaseModel):
    """One line of the audit trail. Secrets never reach the trail (see `app.auth.audit`)."""

    event_type: str
    created_at: datetime
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    details: dict[str, Any]


class ServiceTokenResponse(BaseModel):
    """A service token of the account. Never the token, nor its hash."""

    id: UUID
    name: str
    scopes: List[str]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


async def _account_or_404(db: AsyncSession, user_id: UUID) -> User:
    user = await db.scalar(select(User).where(User.id == user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


async def _audit_unsent_notification(
    db: AsyncSession,
    request: Request,
    user_id: UUID,
    event_type: AuthEventType,
    error_kind: str,
) -> None:
    """Record that the e-mail that goes with `event_type` did not leave. Best effort: the action
    itself is committed already, and a failure here must not change the answer."""
    try:
        await record_event(
            db,
            event_type,
            user_id=user_id,
            request=request,
            notification="not_sent",
            error_kind=error_kind,
        )
        await db.commit()
    except Exception:
        logger.exception("Could not audit an unsent security e-mail")
        await db.rollback()


async def _notify_mfa_reset(db: AsyncSession, request: Request, user_id: UUID, to: str) -> bool:
    """Tell the account its second factor was reset, once the reset is committed. Best effort:
    the reset stands whatever happens here (answering 500 now would hide that it was done), so a
    failure is logged and audited (`notification: not_sent`) and reported as False."""
    try:
        await send_mfa_reset_email(to)
        return True
    except (EmailSendError, EmailNotConfiguredError) as exc:
        logger.warning("MFA reset notice for user %s not sent (%s)", user_id, type(exc).__name__)
        error_kind = type(exc).__name__
    except Exception as exc:
        logger.exception("MFA reset notice for user %s failed unexpectedly", user_id)
        error_kind = type(exc).__name__
    await _audit_unsent_notification(
        db, request, user_id, AuthEventType.MFA_RESET_BY_ADMIN, error_kind
    )
    return False


@router.post("/users/{user_id}/mfa/reset", response_model=MfaResetResponse)
async def reset_mfa(
    user_id: UUID,
    request: Request,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Remove the second factor of an account: the next sign-in is an enrolment. Admin only.

    Clears the secret, the enrolment in progress and the recovery codes, ends every session of
    the account, audits `MFA_RESET_BY_ADMIN` and e-mails the account. Service tokens are kept.

    Resetting the factor of an ADMIN or SCILICIUM_ADMIN (one's own included) takes a
    SCILICIUM_ADMIN: else 403 `scilicium_admin_required`, and nothing is changed. The e-mail is
    best effort (`notification_sent` says whether it left); the reset stands either way.
    """
    user = await _account_or_404(db, user_id)
    if user.role in _ADMIN_ROLES and current_user.role != UserRoleEnum.SCILICIUM_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="scilicium_admin_required"
        )
    to = user.email
    revoked = await reset_user_mfa(
        db,
        user,
        actor_id=current_user.user_id,
        event=AuthEventType.MFA_RESET_BY_ADMIN,
        request=request,
    )
    await db.commit()
    notified = await _notify_mfa_reset(db, request, user_id, to)
    return MfaResetResponse(sessions_revoked=revoked, notification_sent=notified)


@router.post("/users/{user_id}/sessions/revoke", response_model=SessionsRevokeResponse)
async def revoke_sessions(
    user_id: UUID,
    request: Request,
    body: Optional[SessionsRevokeRequest] = None,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """End every open session of an account. Admin only.

    Service tokens (the pipeline's credentials) are left alone unless the body says
    `{"revoke_service_tokens": true}`: ending a person's sessions must not stop a machine.
    """
    await _account_or_404(db, user_id)
    sessions_revoked = await revoke_user_sessions(
        db, user_id, actor_id=current_user.user_id, request=request
    )
    service_tokens_revoked = 0
    if body is not None and body.revoke_service_tokens:
        service_tokens_revoked = await revoke_user_service_tokens(
            db, user_id, actor_id=current_user.user_id, request=request
        )
    await db.commit()
    return SessionsRevokeResponse(
        sessions_revoked=sessions_revoked, service_tokens_revoked=service_tokens_revoked
    )


@router.post("/users/{user_id}/unlock", response_model=UnlockResponse)
async def unlock_account(
    user_id: UUID,
    request: Request,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Clear the brute-force lock of an account (failure count and lock date). Admin only.

    The account row only: an address with no account is throttled in the rate limiter's storage,
    which this does not touch.
    """
    user = await _account_or_404(db, user_id)
    await unlock_user(db, user, actor_id=current_user.user_id, request=request)
    await db.commit()
    return UnlockResponse()


@router.post("/users/{user_id}/password-reset", response_model=PasswordResetResponse)
async def send_password_reset_link(
    user_id: UUID,
    request: Request,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """E-mail an account the link to choose a new password. Admin only.

    The same link the account gets from "forgot password", issued by the administrator (and
    replacing any earlier one). Nothing about the account changes until it is used; its sessions
    and service tokens are kept. 409 `local_auth_disabled` while local authentication is off. As
    the administrator asked for this e-mail, one that cannot leave is an error: 502
    `reset_email_failed`, with the link kept so that a retry sends a new one.
    """
    user = await _account_or_404(db, user_id)
    if not local_tokens_accepted():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="local_auth_disabled")
    to = user.email
    raw = await issue_token(
        db, user_id, OneTimeTokenPurpose.PASSWORD_RESET, created_by=current_user.user_id
    )
    await record_event(
        db,
        AuthEventType.PASSWORD_RESET_REQUESTED,
        user_id=user_id,
        request=request,
        requested_by=current_user.user_id,
    )
    await db.commit()
    try:
        await send_password_reset_email(to, raw)
    except (EmailSendError, EmailNotConfiguredError) as exc:
        logger.error(
            "Password-reset e-mail for user %s not sent (%s); the link stays valid for a retry",
            user_id,
            type(exc).__name__,
        )
        await _audit_unsent_notification(
            db, request, user_id, AuthEventType.PASSWORD_RESET_REQUESTED, type(exc).__name__
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="reset_email_failed")
    return PasswordResetResponse()


@router.get("/users/{user_id}/auth-events", response_model=List[AuthEventResponse])
async def list_auth_events(
    user_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """The audit trail of an account (sign-ins, lock, second factor, sessions, administrator
    actions), newest first: `limit` lines, 50 by default, 200 at most. Admin only."""
    await _account_or_404(db, user_id)
    rows = await db.scalars(
        select(AuthEvent)
        .where(AuthEvent.user_id == user_id)
        .order_by(AuthEvent.created_at.desc(), AuthEvent.id.desc())
        .limit(limit)
    )
    return [
        AuthEventResponse(
            event_type=row.event_type,
            created_at=row.created_at,
            ip=row.ip,
            user_agent=row.user_agent,
            details=row.details,
        )
        for row in rows
    ]


@router.get("/users/{user_id}/service-tokens", response_model=List[ServiceTokenResponse])
async def list_service_tokens(
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """The service tokens an account owns, newest first, revoked ones included. Never the token
    itself nor its hash: the token exists only where it was created. Admin only."""
    await _account_or_404(db, user_id)
    rows = await db.scalars(
        select(ServiceToken)
        .where(ServiceToken.owner_id == user_id)
        .order_by(ServiceToken.created_at.desc(), ServiceToken.id)
    )
    return [
        ServiceTokenResponse(
            id=row.id,
            name=row.name,
            scopes=list(row.scopes or []),
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            revoked_at=row.revoked_at,
        )
        for row in rows
    ]


@router.get("/users/{user_id}", response_model=UserProfile)
async def get_user_details(
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific user profile.
    Admin only.

    Local mode: the `users` row alone (404 without one; no provider call).
    """
    if identities_are_local():
        local_row = await db.scalar(select(User).where(User.id == user_id))
        if local_row is None:
            raise HTTPException(status_code=404, detail="User not found")
        return UserProfile(**_local_profile_fields(local_row))

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"id": f"eq.{user_id}", "select": "id,full_name,avatar_url,role,created_at,updated_at"}
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    profile = data[0]
                    
                    # Get local DB data
                    query = select(User).where(User.id == user_id)
                    result = await db.execute(query)
                    local_user = result.scalar_one_or_none()
                    
                    # Get auth email
                    auth_response = await client.get(
                        f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                        headers=headers
                    )
                    
                    email = None
                    last_sign_in = None
                    confirmed_at = None
                    
                    if auth_response.status_code == 200:
                        user_data = auth_response.json()
                        email = user_data.get("email")
                        last_sign_in = user_data.get("last_sign_in_at")
                        confirmed_at = user_data.get("confirmed_at")

                    # Default subscription data
                    sub_plan = "STARTER"
                    ai_used = 0
                    ai_remaining = 0
                    ai_purchased = 0
                    ai_tokens_used = 0
                    cosmetics_enabled = False
                    report_customization_enabled = False
                    scientific_enabled = False
                    drug_discovery_enabled = False
                    # Unlike list_users, this path never read the real status and
                    # always reported "active" — so the detail modal showed every
                    # suspended account as healthy. Now that status actually gates
                    # access, that gap would misinform the admin acting on it.
                    user_status = "active"
                    subscription_starts_at = None
                    subscription_ends_at = None
                    role_value = profile.get("role", "USER")

                    if local_user:
                        sub_plan = local_user.subscription_plan.value
                        user_status = local_user.status.value.lower()
                        subscription_starts_at = local_user.subscription_starts_at
                        subscription_ends_at = local_user.subscription_ends_at
                        ai_used = local_user.ai_interpretations_used
                        ai_remaining = local_user.ai_interpretations_remaining
                        ai_purchased = local_user.ai_tokens_purchased
                        ai_tokens_used = local_user.ai_tokens_used
                        # Local role is authoritative (drives gating); Supabase role can be stale.
                        role_value = local_user.role.value.lower()
                        # Raw per-user flags (see note above) — not the role-inflated has_*.
                        cosmetics_enabled = local_user.cosmetics_module_enabled
                        report_customization_enabled = local_user.report_customization_module_enabled
                        scientific_enabled = local_user.scientific_module_enabled
                        drug_discovery_enabled = local_user.drug_discovery_module_enabled

                    return UserProfile(
                        id=UUID(profile["id"]),
                        email=email,
                        full_name=profile.get("full_name"),
                        avatar_url=profile.get("avatar_url"),
                        role=role_value,
                        status=user_status,
                        subscription_plan=sub_plan,
                        ai_interpretations_used=ai_used,
                        ai_interpretations_remaining=ai_remaining,
                        ai_tokens_purchased=ai_purchased,
                        ai_tokens_used=ai_tokens_used,
                        cosmetics_module_enabled=cosmetics_enabled,
                        report_customization_module_enabled=report_customization_enabled,
                        scientific_module_enabled=scientific_enabled,
                        drug_discovery_module_enabled=drug_discovery_enabled,
                        subscription_starts_at=subscription_starts_at,
                        subscription_ends_at=subscription_ends_at,
                        **_demo_fields(local_user),
                        created_at=profile.get("created_at"),
                        updated_at=profile.get("updated_at"),
                        last_sign_in_at=last_sign_in,
                        confirmed_at=confirmed_at,
                        **_local_auth_fields(local_user),
                    )
            
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/users/{user_id}/subscription", response_model=UserProfile)
async def update_user_subscription(
    user_id: UUID,
    sub_update: SubscriptionUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a user's subscription plan.
    Admin only.
    """
    from app.api.deps.subscription import get_or_create_user as get_or_create_user_dep
    
    # We can't reuse the dependency easily here as it depends on current user context
    # So we manually fetch/create the target user
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user and identities_are_local():
        # No lazy creation in local mode: an account exists once it is invited.
        raise HTTPException(status_code=404, detail="User not found")
    if not user:
        # If user doesn't exist in local DB (e.g. only in Supabase), create them
        # We need to fetch email/name from Supabase first
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers=headers
            )
            if user_response.status_code != 200:
                raise HTTPException(status_code=404, detail="User not found in Supabase")
            
            user_data = user_response.json()
            
            user = User(
                id=user_id,
                email=user_data.get("email"),
                full_name=user_data.get("user_metadata", {}).get("full_name"),
                role=UserRoleEnum.USER,
                subscription_plan=sub_update.plan
            )
            db.add(user)
    else:
        user.subscription_plan = sub_update.plan

    # Only touch the end date when the caller actually sent the field. An absent
    # field means "change the plan, leave the access period alone"; an explicit
    # null means "remove the limit". See SubscriptionUpdate.
    if "subscription_ends_at" in sub_update.model_fields_set:
        # The column is a String(50) holding ISO 8601 (see models.py) — same
        # convention as the invite path and the Stripe webhook.
        new_ends_at = (
            sub_update.subscription_ends_at.isoformat()
            if sub_update.subscription_ends_at
            else None
        )
        if new_ends_at != user.subscription_ends_at:
            # Re-arm the expiry warnings. `expiry_warning_sent_for` records the
            # smallest threshold already emailed; leaving it set would mean a user
            # warned once before a renewal is never warned again for the new
            # period. See app/worker/tasks/account_tasks.py.
            user.expiry_warning_sent_for = None
        user.subscription_ends_at = new_ends_at

    await db.commit()
    await db.refresh(user)
    
    # Return full profile (reuse get_user_details logic)
    return await get_user_details(user_id, current_user, db)


@router.patch("/users/{user_id}/cosmetics-module", response_model=UserProfile)
async def update_user_cosmetics_module(
    user_id: UUID,
    update: CosmeticsModuleUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Enable or disable the Cosmetics add-on module for a user.
    Admin only.
    """
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not initialized in local database. Please update subscription first.",
        )

    user.cosmetics_module_enabled = update.enabled
    await db.commit()

    return await get_user_details(user_id, current_user, db)


@router.patch("/users/{user_id}/report-customization-module", response_model=UserProfile)
async def update_user_report_customization_module(
    user_id: UUID,
    update: ReportCustomizationModuleUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Enable or disable the report customization add-on module for a user.
    Admin only.
    """
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not initialized in local database. Please update subscription first.",
        )

    user.report_customization_module_enabled = update.enabled
    await db.commit()

    return await get_user_details(user_id, current_user, db)


@router.patch("/users/{user_id}/scientific-module", response_model=UserProfile)
async def update_user_scientific_module(
    user_id: UUID,
    update: ScientificModuleUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Enable or disable the Scientific tools add-on module for a user (GSEA,
    contrast scatter, signature scoring, custom gene sets, DEG patterns).
    Admin only.
    """
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not initialized in local database. Please update subscription first.",
        )

    user.scientific_module_enabled = update.enabled
    await db.commit()

    return await get_user_details(user_id, current_user, db)


@router.patch("/users/{user_id}/drug-discovery-module", response_model=UserProfile)
async def update_user_drug_discovery_module(
    user_id: UUID,
    update: DrugDiscoveryModuleUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Enable or disable the Drug Discovery add-on module for a user (target
    ranking, indications, reports). Independent of the subscription plan.
    Admin only.
    """
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not initialized in local database. Please update subscription first.",
        )

    user.drug_discovery_module_enabled = update.enabled
    await db.commit()

    return await get_user_details(user_id, current_user, db)


@router.patch("/users/{user_id}/demo", response_model=UserProfile)
async def update_user_demo_account(
    user_id: UUID,
    update: DemoAccountUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Déclare ou retire un compte démo, et règle ses plafonds par compte.
    Admin only.

    Seuls les champs présents dans le corps sont appliqués (voir
    `DemoAccountUpdate`). L'échéance reste portée par `subscription_ends_at`
    (PATCH /users/{id}/subscription) : un compte démo passe en lecture seule à
    cette date, comme tout compte expiré.
    """
    user = await db.scalar(select(User).where(User.id == user_id))
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not initialized in local database. Please update subscription first.",
        )

    fields = update.model_fields_set
    if "is_demo" in fields and update.is_demo is not None:
        user.is_demo = update.is_demo
    if "ai_requests_limit" in fields:
        user.ai_requests_limit = update.ai_requests_limit
    if "analyses_quota_override" in fields:
        user.analyses_quota_override = update.analyses_quota_override
    if update.reset_usage:
        user.ai_requests_used = 0
        user.analyses_used_this_month = 0
        user.quota_reset_at = datetime.now(timezone.utc)

    if user.is_demo and not user.subscription_ends_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A demo account needs an access end date: set subscription_ends_at first.",
        )

    await db.commit()
    logger.info(
        "Demo settings for user %s updated by admin %s: %s",
        user_id, current_user.user_id, update.model_dump(include=fields),
    )
    return await get_user_details(user_id, current_user, db)


async def _local_user_or_404(db: AsyncSession, user_id: UUID) -> User:
    user = await db.scalar(select(User).where(User.id == user_id))
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not initialized in local database. Please update subscription first.",
        )
    return user


@router.post("/users/{user_id}/grant-ai-requests", response_model=UserProfile)
async def grant_ai_requests(
    user_id: UUID,
    grant: GrantAmount,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Accorde N requêtes IA à un compte. Admin only.

    Crédits à l'unité : le compte dispose ensuite d'exactement N requêtes de
    plus qu'il n'en avait de disponibles — d'où le `max(plafond, utilisées)`,
    qui ne fait pas compter un plafond déjà dépassé. Un compte STARTER, sans IA
    par son plan, y accède dans cette limite (`User.can_use_ai`).

    Refusé (409) pour un compte à l'IA illimitée : lui poser un plafond le
    restreindrait au lieu de lui accorder quoi que ce soit.
    """
    user = await _local_user_or_404(db, user_id)
    if user.ai_requests_limit is None and user.can_use_ai:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account already has unlimited AI requests.",
        )

    previous = user.ai_requests_limit
    user.ai_requests_limit = max(previous or 0, user.ai_requests_used or 0) + grant.amount
    await db.commit()
    logger.info(
        "Admin %s granted %d AI requests to user %s (limit %s -> %s)",
        current_user.user_id, grant.amount, user_id, previous, user.ai_requests_limit,
    )
    return await get_user_details(user_id, current_user, db)


@router.post("/users/{user_id}/grant-analyses", response_model=UserProfile)
async def grant_analyses(
    user_id: UUID,
    grant: GrantAmount,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Accorde N analyses (datasets) en bonus pour la période en cours. Admin only.

    Le bonus s'ajoute au quota et part avec la remise à zéro du 1er du mois ;
    pour un compte démo, jamais remis à zéro, il vaut pour toute la démo.

    Le compteur est d'abord basculé sur le mois en cours s'il ne l'a pas encore
    été : sans ça, un bonus accordé avant le premier appel du mois serait effacé
    par la remise à zéro paresseuse de ce même appel.

    Refusé (409) pour un compte au quota illimité.
    """
    user = await _local_user_or_404(db, user_id)
    if user.role in (UserRoleEnum.ADMIN, UserRoleEnum.SCILICIUM_ADMIN) or user.analyses_quota is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account already has an unlimited analyses quota.",
        )

    _reset_quota_if_new_month(user)
    user.analyses_bonus = (user.analyses_bonus or 0) + grant.amount
    await db.commit()
    logger.info(
        "Admin %s granted %d analyses to user %s (bonus now %d, quota %s)",
        current_user.user_id, grant.amount, user_id, user.analyses_bonus, user.analyses_quota,
    )
    return await get_user_details(user_id, current_user, db)


@router.post("/users/{user_id}/tokens")
async def add_ai_tokens(
    user_id: UUID,
    token_data: TokenAdd,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Add purchased AI tokens to a user.
    Admin only.
    """
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        # Create user if missing locally (basic plan)
        # This simplifies things, assumes admin knows what they are doing
        raise HTTPException(
            status_code=404, 
            detail="User not initialized in local database. Please update subscription first."
        )
    
    user.ai_tokens_purchased += token_data.tokens
    await db.commit()
    await db.refresh(user)
    
    return {"message": f"Added {token_data.tokens} tokens", "total_purchased": user.ai_tokens_purchased}


@router.get("/ai-usage", response_model=List[AIUsageLogResponse])
async def get_ai_usage_logs(
    limit: int = 100,
    user_id: Optional[UUID] = None,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent AI usage logs.
    Admin only.
    """
    query = select(AIUsageLog).order_by(AIUsageLog.created_at.desc()).limit(limit)
    
    if user_id:
        query = query.where(AIUsageLog.user_id == user_id)
        
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # We need to join with user emails, which are in Supabase or local User table
    # Let's try local User table first as it's faster
    user_ids = list(set(log.user_id for log in logs))
    
    user_query = select(User.id, User.email).where(User.id.in_(user_ids))
    user_result = await db.execute(user_query)
    user_map = {u.id: u.email for u in user_result.all()}
    
    return [
        AIUsageLogResponse(
            id=log.id,
            user_id=log.user_id,
            dataset_id=log.dataset_id,
            action_type=log.action_type,
            comparison_name=log.comparison_name,
            model_used=log.model_used,
            tokens_used=log.tokens_used,
            was_free=log.was_free,
            created_at=log.created_at.isoformat(),
            user_email=user_map.get(log.user_id)
        )
        for log in logs
    ]


@router.patch("/users/{user_id}/role", response_model=UserProfile)
async def update_user_role(
    user_id: UUID,
    role_update: UserRoleUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a user's role, in Supabase `profiles` AND in the local `users` row.
    Admin only.

    Two defects fixed here, both of which made this endpoint lie about what it did:

    1. It returned the raw PostgREST `profiles` row against
       `response_model=UserProfile`. That row carries no `subscription_plan`,
       which `UserProfile` requires, so response validation failed and the caller
       got a 500 — *after* the write had already landed in Supabase. The role did
       change; the admin panel just reported failure, which invites a retry or the
       conclusion that the feature is broken. It now returns a real profile, built
       the same way `update_user_subscription` does.

    2. It wrote the role to Supabase only. The local `users.role` syncs lazily in
       `get_or_create_user`, which deliberately refuses to downgrade ADMIN and
       SCILICIUM_ADMIN from Supabase claims — a sound protection against a stale
       or forged claim, but it also meant a *deliberate* demotion made here could
       never reach the local row. The demoted user lost the admin panel (whose
       guard reads Supabase) while keeping every local-role privilege: unlimited
       analysis quota, AI access, and a free pass through `require_active_account`.
       Writing both sides here keeps that protection intact — it guards against
       claims, not against an explicit admin action.

    `profiles` and the role cache are written only while roles come from the
    provider (`role_source_is_supabase()`). In local mode `users.role` is the
    only role there is, and writing it is the whole change (404 without a row).
    """
    # Validate role
    from app.models.models import UserRole
    try:
        UserRole(role_update.role.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
        )

    try:
        if role_source_is_supabase():
            headers = {
                "apikey": settings.SUPABASE_KEY,
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }

            async with httpx.AsyncClient() as client:
                # Update role in profiles table
                response = await client.patch(
                    f"{settings.SUPABASE_URL}/rest/v1/profiles",
                    headers=headers,
                    params={"id": f"eq.{user_id}"},
                    json={"role": role_update.role.lower()}
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to update user role: {response.text}"
                    )

                data = response.json()
                if not data:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found"
                    )

            # The role guards read a per-process cache of `profiles.role`: drop it so
            # the change applies on the user's next request, not after the TTL.
            invalidate_cached_role(user_id)

        # Mirror onto the local row, which is what every plan and quota guard
        # reads. Skipped silently when the user has no local row yet: it is
        # created on their first authenticated call, and `get_or_create_user`
        # picks the role up from the Supabase claim at that point.
        result = await db.execute(select(User).where(User.id == user_id))
        local_user = result.scalar_one_or_none()
        if local_user:
            local_user.role = UserRole(role_update.role.upper())
            await db.commit()
        elif not role_source_is_supabase():
            # Local mode: no row, no account (nothing creates it lazily any more).
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        # Build the response the same way update_user_subscription does, rather
        # than returning the PostgREST row — see the note in the docstring.
        return await get_user_details(user_id, current_user, db)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user role: {str(e)}"
        )


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get system statistics.
    Admin only.
    """
    try:
        # Count projects
        result = await db.execute(select(func.count()).select_from(Project))
        total_projects = result.scalar() or 0

        # Count datasets
        result = await db.execute(select(func.count()).select_from(Dataset))
        total_datasets = result.scalar() or 0

        # Count unique project owners (active users)
        result = await db.execute(select(Project.owner_id).distinct())
        active_users = len(result.scalars().all())

        # Get local users for detailed stats
        result = await db.execute(select(User))
        local_users = result.scalars().all()
        
        # Calculate revenue and plan distribution.
        # Both the plan list and the monthly prices come from the pricing grid
        # (app/config/pricing.json), the single source of truth — never hard-code
        # a plan or a price here again, or adding a plan would silently report
        # zero users and zero revenue for it.
        # The billing interval is not stored on User, so the estimate always
        # uses the monthly rate; a plan quoted per deal (price_monthly null,
        # e.g. ON_PREMISE) contributes 0 to the estimate.
        from app.core.pricing import get_pricing

        _grid = get_pricing()
        users_by_plan = {p.id: 0 for p in _grid.plans_ordered}
        estimated_revenue = 0.0
        PRICES = {p.id: (p.price_monthly or 0.0) for p in _grid.plans}
        PRICES["TOKEN"] = 0.10  # Per token

        for u in local_users:
            plan = u.subscription_plan.value if hasattr(u.subscription_plan, 'value') else u.subscription_plan
            users_by_plan[plan] = users_by_plan.get(plan, 0) + 1
            
            # Monthly recurring revenue
            estimated_revenue += PRICES.get(plan, 0)
            
            # One-time token revenue
            estimated_revenue += (u.ai_tokens_purchased * PRICES["TOKEN"])

        if identities_are_local():
            # The `users` table is the directory: every account has a row.
            total_users = len(local_users)
        else:
            # Fetch total users from Supabase (including those not in local DB)
            headers = {
                "apikey": settings.SUPABASE_KEY,
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.SUPABASE_URL}/rest/v1/profiles",
                    headers=headers,
                    params={"select": "id"}
                )

                total_users = 0
                if response.status_code == 200:
                    total_users = len(response.json())

        # Adjust STARTER count for users not in local DB (model default plan)
        local_users_count = sum(users_by_plan.values())
        if total_users > local_users_count:
            users_by_plan["STARTER"] += (total_users - local_users_count)

        return SystemStats(
            total_users=total_users,
            total_projects=total_projects,
            total_datasets=total_datasets,
            active_users=active_users,
            users_by_plan=users_by_plan,
            estimated_revenue=round(estimated_revenue, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching system stats: {str(e)}"
        )


async def _require_password(request: Request) -> None:
    """Supabase mode still creates the account with the admin's password.

    A body without one is refused exactly as when the schema required the field:
    the same validation errors, raised the way FastAPI raises them, so the 422
    and its body are unchanged. Only reached once the body passed `UserCreate`.
    """
    body = await request.json()
    try:
        _UserCreateWithPassword.model_validate(body)
    except ValidationError as exc:
        errors = [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        raise RequestValidationError(errors, body=body) from None


@router.post(
    "/users", response_model=CreatedUserProfile, dependencies=[Depends(require_active_license)]
)
async def create_user(
    user_data: UserCreate,
    request: Request,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user in Supabase Auth and local DB.
    Admin only.

    Local mode: an invitation (`AccountService.invite_user`) with the requested
    role. The admin no longer chooses a password: one in the request is ignored,
    and the answer carries `invitation_sent: true` (502 `invitation_email_failed`
    if the e-mail could not be sent; the account and its link are kept).
    """
    if not identities_are_local():
        await _require_password(request)

    from app.models.models import UserRole
    try:
        role_enum = UserRole(user_data.role.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
        )

    if identities_are_local():
        service = AccountService(db)
        try:
            invited = await service.invite_user(
                email=user_data.email,
                full_name=user_data.full_name,
                plan=SubscriptionPlan.STARTER,
                invited_by_admin_id=current_user.user_id,
                role=role_enum,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        except InvitationEmailError:
            raise _invitation_email_failed()
        return CreatedUserProfile(**_local_profile_fields(invited), invitation_sent=True)

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # Create user via Supabase Admin API
            response = await client.post(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers,
                json={
                    "email": user_data.email,
                    "password": user_data.password,
                    "email_confirm": True,
                    "user_metadata": {
                        "full_name": user_data.full_name
                    }
                }
            )

            if response.status_code not in [200, 201]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to create user: {response.text}"
                )

            user_response = response.json()
            user_id = UUID(user_response.get("id"))

        # Create profile in local DB
        try:
            local_user = User(
                id=user_id,
                email=user_data.email,
                full_name=user_data.full_name,
                role=role_enum,
                subscription_plan=SubscriptionPlan.STARTER
            )
            db.add(local_user)
            await db.commit()
            await db.refresh(local_user)
        except Exception as db_err:
            # Rollback Supabase user creation
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                    headers=headers
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user profile: {str(db_err)}"
            )

        return UserProfile(
            id=local_user.id,
            email=local_user.email,
            full_name=local_user.full_name,
            role=local_user.role.value,
            subscription_plan=local_user.subscription_plan.value,
            ai_interpretations_used=local_user.ai_interpretations_used,
            ai_tokens_purchased=local_user.ai_tokens_purchased,
            ai_tokens_used=local_user.ai_tokens_used,
            ai_interpretations_remaining=local_user.ai_interpretations_remaining,
            created_at=local_user.created_at.isoformat(),
            updated_at=local_user.updated_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )


@router.patch("/users/{user_id}", response_model=UserProfile)
async def update_user(
    user_id: UUID,
    user_update: UserUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a user's profile.
    Admin only.

    `profiles` (and the role cache) are written only while roles come from the
    provider. In local mode the `users` row is the profile (404 without one).
    """
    # Validate role if provided
    if user_update.role:
        from app.models.models import UserRole
        try:
            UserRole(user_update.role.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
            )

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        # Build update payload
        update_data = {}
        if user_update.full_name is not None:
            update_data["full_name"] = user_update.full_name
        if user_update.role is not None:
            update_data["role"] = user_update.role.lower()

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        if not role_source_is_supabase():
            local_row = await db.scalar(select(User).where(User.id == user_id))
            if local_row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            if user_update.full_name is not None:
                local_row.full_name = user_update.full_name
            if user_update.role is not None:
                local_row.role = UserRoleEnum(user_update.role.upper())
            await db.commit()
            return await get_user_details(user_id, current_user, db)

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"id": f"eq.{user_id}"},
                json=update_data
            )

            if response.status_code == 200:
                if "role" in update_data:
                    # See update_user_role: the role guards read a cached copy.
                    invalidate_cached_role(user_id)
                # Return full updated profile
                return await get_user_details(user_id, current_user, db)
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update user: {response.text}"
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user: {str(e)}"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a user from Supabase and local database.
    Admin only.
    WARNING: This permanently deletes the user and all their data (projects, legacy data, etc).

    Local mode: local deletion only (404 for an unknown account). The account's
    sessions, links and recovery codes go with its row (ON DELETE CASCADE); the
    Sequencing tables are not touched and the provider is not called.
    """
    # Prevent self-deletion
    if str(user_id) == str(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    try:
        from sqlalchemy import text, delete

        if identities_are_local():
            if await db.scalar(select(User.id).where(User.id == user_id)) is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        else:
            # 1. Delete legacy 'sequencing_projects' (if exists)
            # This table causes FK violation if not cleaned up
            try:
                # Use nested transaction to prevent breaking the main transaction
                # if table doesn't exist
                async with db.begin_nested():
                    await db.execute(text("DELETE FROM sequencing_projects WHERE user_id = :uid"), {"uid": user_id})
            except Exception as e:
                # Table might not exist or other error, log and continue
                logger.warning(f"Warning cleaning legacy projects: {e}")

        # 2. Delete modern 'projects' owned by user
        # This will cascade to datasets, samples, members
        await db.execute(delete(Project).where(Project.owner_id == user_id))
        
        # 3. Delete local 'users' record
        await db.execute(delete(User).where(User.id == user_id))
        
        await db.commit()

        # 4. Delete the identity: from Supabase Auth, with the provider's legacy
        # rows that block it (sequencing_projects, status_history); nothing in
        # local mode. See `delete_identity`.
        try:
            await delete_identity(user_id)
        except IdentityNotFoundError:
            # If not found in Auth but we cleaned up local DB, consider it success or just gone
            return {"message": "User deleted (was not found in Supabase Auth)"}
        except IdentityDeletionError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete user from Supabase: {exc}"
            )
        return {"message": "User deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a project.
    Admin only.
    WARNING: This permanently deletes the project and all associated datasets.
    """
    try:
        # Find project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Delete project (cascade will handle datasets)
        await db.delete(project)
        await db.commit()

        return {"message": "Project deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting project: {str(e)}"
        )


def _identity_field(
    identities: dict[UUID, AuthIdentity], user_id: UUID, field: str
) -> Optional[str]:
    """`email` or `full_name` of `user_id` in a directory listing, None if absent."""
    identity = identities.get(user_id)
    if identity is None:
        return None
    return identity.email if field == "email" else identity.full_name


@router.get("/projects")
async def list_all_projects(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    List all projects in the system with owner information.
    Admin only.
    """
    try:
        result = await db.execute(select(Project).offset(skip).limit(limit))
        projects = result.scalars().all()

        # Owner emails and names from the identity directory (the provider's user
        # list in supabase mode, the `users` table in local mode).
        owners = await list_identities(db)

        return [
            {
                "id": str(p.id),
                "name": p.name,
                "description": p.description,
                "owner_id": str(p.owner_id),
                "owner_email": _identity_field(owners, p.owner_id, "email"),
                "owner_full_name": _identity_field(owners, p.owner_id, "full_name"),
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None
            }
            for p in projects
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching projects: {str(e)}"
        )


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: UUID,
    project_update: ProjectUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a project's name or description.
    Admin only.
    """
    try:
        # Find project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Update fields
        if project_update.name is not None:
            project.name = project_update.name
        if project_update.description is not None:
            project.description = project_update.description
        if project_update.owner_id is not None:
            # Verify user exists locally
            user_result = await db.execute(select(User).where(User.id == project_update.owner_id))
            user = user_result.scalar_one_or_none()
            
            # Supabase mode only: in local mode an account exists once it is
            # invited, and nothing creates its row lazily (unknown owner → 404).
            if not user and not identities_are_local():
                # Try to fetch from Supabase to auto-create local user
                try:
                    headers = {
                        "apikey": settings.SUPABASE_KEY,
                        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
                        "Content-Type": "application/json",
                    }
                    async with httpx.AsyncClient() as client:
                        sb_res = await client.get(
                            f"{settings.SUPABASE_URL}/auth/v1/admin/users/{project_update.owner_id}",
                            headers=headers
                        )
                        if sb_res.status_code == 200:
                            sb_user = sb_res.json()
                            # Create local user
                            user = User(
                                id=project_update.owner_id,
                                email=sb_user.get("email"),
                                full_name=sb_user.get("user_metadata", {}).get("full_name"),
                                role=UserRoleEnum.USER, # Default to USER
                                subscription_plan=SubscriptionPlan.STARTER
                            )
                            db.add(user)
                            # We don't commit here, we'll commit with the project update
                except Exception as e:
                    logger.warning(f"Error fetching user from Supabase: {e}")

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {project_update.owner_id} not found"
                )
            project.owner_id = project_update.owner_id

        db.add(project)
        await db.commit()
        await db.refresh(project)

        return {
            "id": str(project.id),
            "name": project.name,
            "description": project.description,
            "owner_id": str(project.owner_id),
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating project: {str(e)}"
        )


@router.get("/projects/{project_id}/members", response_model=List[ProjectMemberResponse])
async def list_project_members(
    project_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    List all members of a project with their user information.
    Admin only.
    """
    try:
        # Verify project exists
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Get project members
        result = await db.execute(select(ProjectMember).where(ProjectMember.project_id == project_id))
        members = result.scalars().all()

        # User information from the identity directory (the provider's user list
        # in supabase mode, the `users` table in local mode).
        identities = await list_identities(db)

        return [
            ProjectMemberResponse(
                id=m.id,
                project_id=m.project_id,
                user_id=m.user_id,
                access_level=m.access_level.value if hasattr(m.access_level, 'value') else m.access_level,
                created_at=m.created_at.isoformat() if m.created_at else None,
                updated_at=m.updated_at.isoformat() if m.updated_at else None,
                user_email=_identity_field(identities, m.user_id, "email"),
                user_full_name=_identity_field(identities, m.user_id, "full_name")
            )
            for m in members
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching project members: {str(e)}"
        )


@router.post("/projects/{project_id}/members", response_model=ProjectMemberResponse)
async def add_project_member(
    project_id: UUID,
    member_data: ProjectMemberCreate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a user to a project.
    Admin only.
    """
    # Validate access level
    try:
        access_level = UserRoleEnum(member_data.access_level.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid access level. Must be one of: {', '.join([r.value for r in UserRoleEnum])}"
        )

    try:
        # Verify project exists
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Check if member already exists
        result = await db.execute(select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == member_data.user_id
        ))
        existing_member = result.scalar_one_or_none()

        if existing_member:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member of this project"
            )

        # Local mode: the account must exist (its `users` row is its identity).
        identity: Optional[AuthIdentity] = None
        if identities_are_local():
            identity = await find_identity_by_id(db, member_data.user_id)
            if identity is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        # Create new member
        new_member = ProjectMember(
            project_id=project_id,
            user_id=member_data.user_id,
            access_level=access_level
        )
        db.add(new_member)
        await db.commit()
        await db.refresh(new_member)

        # Get user information
        user_email: Optional[str] = None
        user_full_name: Optional[str] = None
        if identity is not None:
            # Local mode: read from the directory above.
            user_email, user_full_name = identity.email, identity.full_name
        else:
            headers = {
                "apikey": settings.SUPABASE_KEY,
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                user_response = await client.get(
                    f"{settings.SUPABASE_URL}/auth/v1/admin/users/{member_data.user_id}",
                    headers=headers
                )
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    user_email = user_data.get("email")
                    user_full_name = user_data.get("user_metadata", {}).get("full_name")

        return ProjectMemberResponse(
            id=new_member.id,
            project_id=new_member.project_id,
            user_id=new_member.user_id,
            access_level=new_member.access_level.value if hasattr(new_member.access_level, 'value') else new_member.access_level,
            created_at=new_member.created_at.isoformat() if new_member.created_at else None,
            updated_at=new_member.updated_at.isoformat() if new_member.updated_at else None,
            user_email=user_email,
            user_full_name=user_full_name
        )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding project member: {str(e)}"
        )


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_project_member(
    project_id: UUID,
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a user from a project.
    Admin only.
    """
    try:
        # Find member
        result = await db.execute(select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id
        ))
        member = result.scalar_one_or_none()

        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project member not found"
            )

        await db.delete(member)
        await db.commit()

        return {"message": "Project member removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing project member: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Demo data assignment
# ---------------------------------------------------------------------------


class DemoAssignRequest(BaseModel):
    """Optional overrides for the demo project (currently unused, reserved for future use)."""
    pass


class DemoDatasetInfo(BaseModel):
    """Minimal dataset info returned by assign-demo."""
    id: UUID
    name: str
    comparison_name: str
    deg_up_count: Optional[int] = None
    deg_down_count: Optional[int] = None
    deg_significant_count: Optional[int] = None
    total_genes: Optional[int] = None


class DemoAssignResponse(BaseModel):
    """Response after assigning demo data."""
    project_id: UUID
    project_name: str
    datasets: List[DemoDatasetInfo]
    message: str


@router.post("/users/{user_id}/assign-demo", response_model=DemoAssignResponse)
async def assign_demo_data(
    user_id: UUID,
    _body: DemoAssignRequest = DemoAssignRequest(),
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a synthetic demo project with 2 DEG comparisons for a user.

    The project is populated with realistic human gene data and enrichment
    pathways so the user can immediately explore all platform features.

    Returns HTTP 409 if a demo project already exists for that user.
    Admin only.
    """
    from app.services.demo_seed_service import (
        DEMO_PROJECT_NAME,
        DemoAlreadyExistsError,
        create_demo_data,
    )

    try:
        project = await create_demo_data(db, user_id)
    except DemoAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"User {user_id} already has a demo project "
                f"named '{DEMO_PROJECT_NAME}'. "
                "Delete it first before re-assigning."
            ),
        )
    except Exception as exc:
        logger.exception("[admin] assign-demo failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create demo data: {exc}",
        )

    # Eager-load datasets to build the response
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project.id)
    )
    datasets = result.scalars().all()

    dataset_infos = [
        DemoDatasetInfo(
            id=ds.id,
            name=ds.name,
            comparison_name=(
                ds.dataset_metadata.get("comparisons", ["unknown"])[0]
                if ds.dataset_metadata
                else "unknown"
            ),
            deg_up_count=ds.deg_up_count,
            deg_down_count=ds.deg_down_count,
            deg_significant_count=ds.deg_significant_count,
            total_genes=ds.total_genes,
        )
        for ds in datasets
    ]

    return DemoAssignResponse(
        project_id=project.id,
        project_name=project.name,
        datasets=dataset_infos,
        message=(
            f"Demo project '{project.name}' created successfully "
            f"with {len(dataset_infos)} comparisons."
        ),
    )


# ---------------------------------------------------------------------------
# Gene Ontology data management
# ---------------------------------------------------------------------------


async def _run_go_loader(db_session_factory):
    """Background task: download and load the GO OBO file into gene_sets."""
    from app.services.go_loader import GoLoaderService

    async with db_session_factory() as session:
        loader = GoLoaderService(session)
        await loader.load_obo(url="http://purl.obolibrary.org/obo/go/go-basic.obo")


@router.post("/ontology/reload")
async def reload_go_ontology(
    background_tasks: BackgroundTasks,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a full re-download and re-import of the Gene Ontology (go-basic.obo)
    into the gene_sets table.  This makes /tools/ontology searchable.

    The import runs in the background (~17 000 terms, can take a few minutes).
    Admin only.
    """
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.orm import sessionmaker

    # Obtain the underlying engine from the current session to build an
    # independent session factory for the background task.
    engine = db.get_bind()

    # sessionmaker produces a factory that background_tasks can call safely
    # (the request-scoped session will be closed before the task runs).
    factory = sessionmaker(engine, class_=_AS, expire_on_commit=False)

    background_tasks.add_task(_run_go_loader, factory)

    return {
        "message": "GO ontology import started in background. ~17 000 terms will be loaded.",
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Login Statistics — Schemas
# ---------------------------------------------------------------------------

class DailyLoginCount(BaseModel):
    date: str
    count: int


class RecentLoginEvent(BaseModel):
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    created_at: str


class LoginStatsResponse(BaseModel):
    daily_counts: list[DailyLoginCount]
    active_today: int
    active_7_days: int
    active_30_days: int
    recent_events: list[RecentLoginEvent]


# ---------------------------------------------------------------------------
# Login Statistics — Endpoint
# ---------------------------------------------------------------------------

@router.get("/login-stats", response_model=LoginStatsResponse)
async def get_login_stats(
    days: int = 30,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Return login activity statistics for admin dashboard.
    - daily_counts: number of login events per day for the last <days> days (1–90)
    - active_today / active_7_days / active_30_days: distinct user counts
    - recent_events: last 50 login events with user info
    Admin only.
    """
    from sqlalchemy import text as sa_text, distinct
    from datetime import date, timedelta, timezone

    days = max(1, min(days, 90))
    now = datetime.now(timezone.utc)

    # ---------- KPI counts (distinct users) ----------
    async def _count_distinct(since_delta_days: int) -> int:
        cutoff = now - timedelta(days=since_delta_days)
        result = await db.execute(
            select(func.count(distinct(UserLoginEvent.user_id)))
            .where(UserLoginEvent.created_at >= cutoff)
        )
        return result.scalar() or 0

    active_today = await _count_distinct(1)
    active_7_days = await _count_distinct(7)
    active_30_days = await _count_distinct(30)

    # ---------- Daily event counts (0-filled) ----------
    period_start = now - timedelta(days=days)
    result = await db.execute(
        select(
            func.date_trunc("day", UserLoginEvent.created_at).label("day"),
            func.count().label("cnt"),
        )
        .where(UserLoginEvent.created_at >= period_start)
        .group_by(func.date_trunc("day", UserLoginEvent.created_at))
        .order_by(func.date_trunc("day", UserLoginEvent.created_at))
    )
    rows = result.all()

    # Build a zero-filled date range
    daily_map: dict[str, int] = {r.day.strftime("%Y-%m-%d"): r.cnt for r in rows}
    daily_counts: list[DailyLoginCount] = []
    for i in range(days):
        d = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        daily_counts.append(DailyLoginCount(date=d, count=daily_map.get(d, 0)))

    # ---------- Recent events (50, newest first) ----------
    result = await db.execute(
        select(UserLoginEvent)
        .order_by(UserLoginEvent.created_at.desc())
        .limit(50)
    )
    events = result.scalars().all()

    # Enrich with user email / full_name from local users table
    user_ids = list({e.user_id for e in events})
    user_query = select(User.id, User.email, User.full_name).where(
        User.id.in_(user_ids)
    )
    user_result = await db.execute(user_query)
    user_map = {str(u.id): {"email": u.email, "full_name": u.full_name} for u in user_result.all()}

    recent_events = [
        RecentLoginEvent(
            user_id=str(e.user_id),
            email=user_map.get(str(e.user_id), {}).get("email"),
            full_name=user_map.get(str(e.user_id), {}).get("full_name"),
            created_at=e.created_at.isoformat(),
        )
        for e in events
    ]

    return LoginStatsResponse(
        daily_counts=daily_counts,
        active_today=active_today,
        active_7_days=active_7_days,
        active_30_days=active_30_days,
        recent_events=recent_events,
    )


# ---------------------------------------------------------------------------
# License management (on-premise key issuance)
# ---------------------------------------------------------------------------

from app.models.models import LicenseRecord
from app.core.license import issue_license_key


class LicenseIssueRequest(BaseModel):
    client_id: str
    plan: str
    expires_at: int  # Unix timestamp
    notes: Optional[str] = None


class LicenseRecordResponse(BaseModel):
    id: str
    client_id: str
    plan: str
    product: str
    expires_at: int
    license_key: str
    notes: Optional[str]
    is_revoked: bool
    is_expired: bool
    days_until_expiry: Optional[int]
    created_at: str
    created_by: Optional[str]

    class Config:
        from_attributes = True


@router.post("/licenses", response_model=LicenseRecordResponse)
async def issue_license(
    req: LicenseIssueRequest,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Issue a new on-premise license key. Admin only."""
    if not settings.LICENSE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LICENSE_SECRET_KEY is not configured on this instance.",
        )

    key = issue_license_key(
        client_id=req.client_id,
        plan=req.plan,
        expires_at=req.expires_at,
        product=settings.LICENSE_PRODUCT,
        secret=settings.LICENSE_SECRET_KEY,
    )

    record = LicenseRecord(
        client_id=req.client_id,
        plan=req.plan,
        product=settings.LICENSE_PRODUCT,
        expires_at=req.expires_at,
        license_key=key,
        notes=req.notes,
        created_by=str(current_user.user_id),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return LicenseRecordResponse(
        id=str(record.id),
        client_id=record.client_id,
        plan=record.plan,
        product=record.product,
        expires_at=record.expires_at,
        license_key=record.license_key,
        notes=record.notes,
        is_revoked=record.is_revoked,
        is_expired=record.is_expired,
        days_until_expiry=record.days_until_expiry,
        created_at=record.created_at.isoformat(),
        created_by=record.created_by,
    )


@router.get("/licenses", response_model=list[LicenseRecordResponse])
async def list_licenses(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all issued licenses. Admin only."""
    result = await db.execute(
        select(LicenseRecord).order_by(LicenseRecord.created_at.desc())
    )
    records = result.scalars().all()

    return [
        LicenseRecordResponse(
            id=str(r.id),
            client_id=r.client_id,
            plan=r.plan,
            product=r.product,
            expires_at=r.expires_at,
            license_key=r.license_key,
            notes=r.notes,
            is_revoked=r.is_revoked,
            is_expired=r.is_expired,
            days_until_expiry=r.days_until_expiry,
            created_at=r.created_at.isoformat(),
            created_by=r.created_by,
        )
        for r in records
    ]


@router.patch("/licenses/{license_id}/revoke", response_model=LicenseRecordResponse)
async def revoke_license(
    license_id: str,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Revoke an issued license. Admin only."""
    from uuid import UUID as _UUID
    result = await db.execute(
        select(LicenseRecord).where(LicenseRecord.id == _UUID(license_id))
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="License not found.")

    record.is_revoked = True
    await db.commit()
    await db.refresh(record)

    return LicenseRecordResponse(
        id=str(record.id),
        client_id=record.client_id,
        plan=record.plan,
        product=record.product,
        expires_at=record.expires_at,
        license_key=record.license_key,
        notes=record.notes,
        is_revoked=record.is_revoked,
        is_expired=record.is_expired,
        days_until_expiry=record.days_until_expiry,
        created_at=record.created_at.isoformat(),
        created_by=record.created_by,
    )
