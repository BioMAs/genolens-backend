"""
Subscription and quota dependencies for API endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.supabase_deps import get_current_user
from app.core.config import settings
from app.core.supabase_auth import SupabaseUser
from app.db.session import get_db
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

logger = logging.getLogger(__name__)

# Roles that must never be downgraded by Supabase claims
_PROTECTED_ROLES = {UserRole.ADMIN, UserRole.SCILICIUM_ADMIN}


# ── User resolution ────────────────────────────────────────────────────────────


async def get_or_create_user(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Get or create User profile from Supabase auth. Defaults to STARTER plan."""
    query = select(User).where(User.id == current_user.user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            id=current_user.user_id,
            email=current_user.email,
            full_name=current_user.user_metadata.get("full_name"),
            role=current_user.role,
            subscription_plan=SubscriptionPlan.STARTER,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    else:
        dirty = False

        if user.role != current_user.role:
            # Never downgrade a protected admin role via Supabase claims
            if user.role in _PROTECTED_ROLES:
                logger.warning(
                    "Blocked role downgrade attempt for user %s: "
                    "local=%s → supabase=%s (keeping local role)",
                    user.id,
                    user.role,
                    current_user.role,
                )
            else:
                logger.info(
                    "Updating role for user %s: %s → %s",
                    user.id,
                    user.role,
                    current_user.role,
                )
                user.role = current_user.role
                dirty = True

        # Invitation accepted. Reaching here means Supabase minted a valid token
        # for this account, which only happens once the invitee has followed the
        # magic link and set a password — so PENDING has served its purpose.
        #
        # Nothing else in the codebase ever clears PENDING: before this, an
        # invited user stayed PENDING forever, the admin panel kept offering
        # "Renvoyer l'invitation" for someone already using the product, and
        # `resend_invitation` was the only code that read the flag. That was
        # merely untidy while status was unenforced; once `require_active_account`
        # treats a non-ACTIVE account as inactive, leaving it here would lock out
        # every invited user on their first request. This transition is what makes
        # enforcing status safe.
        if user.status == UserStatus.PENDING:
            logger.info("Activating invited user %s on first authenticated request", user.id)
            user.status = UserStatus.ACTIVE
            dirty = True

        if dirty:
            db.add(user)
            await db.commit()
            await db.refresh(user)

    return user


# ── Admin guard ────────────────────────────────────────────────────────────────


async def require_admin(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require ADMIN or SCILICIUM_ADMIN role."""
    if user.role not in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


# ── AI access ──────────────────────────────────────────────────────────────────


async def require_ai_access(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require TEAM or ON_PREMISE plan for AI interpretation."""
    if not user.can_use_ai:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"AI interpretation requires TEAM or ON_PREMISE plan. "
                f"Current plan: {user.subscription_plan.value}"
            ),
        )
    return user


async def check_ai_quota(
    user: Annotated[User, Depends(get_or_create_user)], db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    AI quota check.
    - ON_PREMISE / ADMIN / SCILICIUM_ADMIN: unlimited.
    - TEAM: 1 AI interpretation per comparison (enforced upstream via DB unique
      index on AIInterpretation(dataset_id, comparison_name) — always passes here).
    - STARTER: blocked by require_ai_access before reaching here.
    """
    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        return user
    if user.subscription_plan in (SubscriptionPlan.ON_PREMISE, SubscriptionPlan.TEAM):
        return user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="AI features not available on STARTER plan"
    )


async def increment_ai_usage(
    user: User,
    db: AsyncSession,
    action_type: str = "interpretation",
    dataset_id: UUID = None,
    comparison_name: str = None,
    model_used: str = None,
    tokens_used: int = 1,
) -> None:
    """Log AI usage. Quota enforcement is via DB unique index, not counters.

    `tokens_used` should be the real token count from the LLM response
    (interpreter.last_usage["total_tokens"]); it drives per-user cost accounting.
    Falls back to 1 when the caller has no token count (keeps a per-action tally).
    """
    from app.models.models import AIUsageLog

    log = AIUsageLog(
        user_id=user.id,
        dataset_id=dataset_id,
        action_type=action_type,
        comparison_name=comparison_name,
        model_used=model_used or settings.LLM_MODEL,
        tokens_used=max(int(tokens_used or 0), 0),
        was_free=True,
    )
    db.add(log)
    await db.commit()


# ── Cosmetics module (per-user add-on) ───────────────────────────────────────────


async def require_cosmetics_access(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require the Cosmetics add-on module to be unlocked for this user.

    Unlocked explicitly per-user by an admin (User.cosmetics_module_enabled);
    admins always have access. The frontend keeps the Cosmetics tab visible for
    everyone and renders a locked teaser when access is denied — so a 403 here is
    expected for users without the module.
    """
    if not user.has_cosmetics_module:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cosmetics module is not enabled for this account.",
        )
    return user


# ── Report customization module (per-user add-on) ────────────────────────────────


async def require_report_customization_access(
    user: Annotated[User, Depends(get_or_create_user)]
) -> User:
    """Require the report customization add-on module to be unlocked for this user.

    Unlocked explicitly per-user by an admin (User.report_customization_module_enabled);
    admins always have access. Mirrors the Cosmetics module pattern — the frontend
    renders a locked teaser when access is denied, so a 403 here is expected for
    users without the module.
    """
    if not user.has_report_customization:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Report customization module is not enabled for this account.",
        )
    return user


# ── Scientific tools module (per-user add-on) ────────────────────────────────────


async def require_scientific_access(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require the Scientific tools add-on module to be unlocked for this user.

    Covers GSEA, the two-contrast log2FC scatter, per-sample signature scoring,
    custom gene sets and DEG patterns. Unlocked explicitly per-user by an admin
    (User.scientific_module_enabled); admins always have access. Mirrors the
    Cosmetics module pattern — the frontend renders locked cards when access is
    denied, so a 403 here is expected for users without the module.
    """
    if not user.has_scientific_module:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Scientific tools module is not enabled for this account.",
        )
    return user


# ── Drug Discovery module (per-user add-on) ──────────────────────────────────────


async def require_drug_discovery_access(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require the Drug Discovery add-on module to be unlocked for this user.

    Unlocked explicitly per-user by an admin (User.drug_discovery_module_enabled),
    independent of the subscription plan; admins always have access. Replaces the
    previous TEAM/ON_PREMISE plan gate, so a TEAM user without the flag now gets
    a 403 here — the frontend renders a locked card offering to request access.
    """
    if not user.has_drug_discovery_module:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Drug Discovery module is not enabled for this account.",
        )
    return user


# ── TEAM plan guard ────────────────────────────────────────────────────────────


async def require_advanced_export(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require TEAM or ON_PREMISE plan for PDF report generation.

    Backs the grid's `advanced_export` entitlement, which reads false for STARTER
    (`app/config/pricing.json`). `User.can_export_advanced` has expressed that rule
    since the plans were introduced and was never once called, so STARTER accounts
    have been generating PDF reports the grid says they do not have.

    Scope is PDF only. CSV/TSV (`GET /{dataset_id}/deg-stats/export`) is included
    in every plan and stays open, and no Excel export exists in the product —
    openpyxl appears only on the import side.

    Applied to the two report *triggers*, not to `/report/status` or
    `/report/download`: a STARTER can no longer start a report, so has none to
    fetch, while gating the download would retroactively hide reports already
    produced and paid for.
    """
    if not user.can_export_advanced:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"PDF report export requires a TEAM or ON_PREMISE plan. "
                f"Current plan: {user.subscription_plan.value}"
            ),
        )
    return user


async def require_team_plan(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Require TEAM or ON_PREMISE plan (multi-comparison features).

    Backs the grid's `multi_comparison` entitlement. Written when the plans were
    introduced and left with zero call sites until now, which is why Venn analysis
    and cross-comparison intersection enrichment were reachable on STARTER.

    Not applied to `logfc-scatter`: `contrast_scatter` is add-on-owned
    (`ADDON_OWNED_ENTITLEMENTS`, app/core/pricing.py) and already sits behind
    `require_scientific_access`. Adding a plan gate there would contradict the grid.
    """
    if not user.can_use_multi_comparison:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"This feature requires a TEAM or ON_PREMISE plan. "
                f"Current plan: {user.subscription_plan.value}"
            ),
        )
    return user


# ── Comparison quota ───────────────────────────────────────────────────────────


def _has_unlimited_analyses(user: User) -> bool:
    """True when no comparison quota applies: privileged roles and plans whose
    `analyses_quota` is None. The single place this condition is expressed —
    a new privileged role must not have to be added twice."""
    return user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or user.analyses_quota is None


def _reset_quota_if_new_month(user: User) -> bool:
    """
    Reset analyses_used_this_month if we're in a new calendar month.
    Returns True if a reset was performed.
    """
    now = datetime.now(timezone.utc)
    if user.quota_reset_at is None:
        user.analyses_used_this_month = 0
        user.quota_reset_at = now
        return True
    if now.year != user.quota_reset_at.year or now.month != user.quota_reset_at.month:
        user.analyses_used_this_month = 0
        user.quota_reset_at = now
        return True
    return False


async def check_analysis_quota(
    user: Annotated[User, Depends(get_or_create_user)], db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    Check that the user has remaining comparison quota for this month.
    Resets the counter automatically if a new month has started.
    Raises HTTP 429 if quota is exhausted.
    """
    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        return user

    reset_happened = _reset_quota_if_new_month(user)

    remaining = user.analyses_remaining
    if remaining is not None and remaining <= 0:
        # Persist any reset even when blocking, so next request doesn't re-reset
        if reset_happened:
            db.add(user)
            await db.commit()
        quota = user.analyses_quota
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Monthly comparison quota exhausted ({quota}/{quota}). "
                f"Quota resets on the 1st of next month. "
                f"Upgrade to a higher plan for more comparisons."
            ),
        )

    if reset_happened:
        db.add(user)
        await db.commit()
    return user


async def try_increment_analysis_usage(user: User, db: AsyncSession) -> bool:
    """
    Incrémente le compteur de comparaisons si le quota le permet.

    Contrat volontairement sans effet de bord : ne commit pas, ne rollback pas,
    ne lève rien. C'est ce qui rend la fonction appelable depuis un worker
    Celery, où une HTTPException abandonnerait une analyse dont le calcul
    coûteux a déjà abouti. La politique (répondre 429, ou logger et continuer)
    appartient à l'appelant, de même que le contrôle de la transaction.

    Retourne True si le compteur a été incrémenté, ou si l'utilisateur est
    illimité (rien n'est alors écrit). False si le quota a bloqué l'incrément.
    """
    if _has_unlimited_analyses(user):
        return True  # Illimité — rien à compter

    quota = user.analyses_quota
    result = await db.execute(
        update(User)
        .where(User.id == user.id)
        .where(User.analyses_used_this_month < quota)
        .values(analyses_used_this_month=User.analyses_used_this_month + 1)
        .returning(User.analyses_used_this_month)
    )
    return result.scalar() is not None


async def increment_analysis_usage(user: User, db: AsyncSession) -> None:
    """
    Incrémente le compteur pour un import DEG, côté HTTP.

    À appeler AVANT le commit du dataset, dans la MÊME transaction que son
    insert. C'est cette contrainte qui donne son sens au `rollback` ci-dessous :
    appelée après le commit — ce qu'elle faisait dans sa première version — elle
    annulait une transaction déjà vide, et l'appelant recevait un 429 pour un
    dataset pourtant durable, qui comptait ensuite contre la limite par projet
    sans avoir jamais été facturé.

    Ne commit pas : le commit appartient à l'appelant, seul à savoir ce que sa
    transaction porte d'autre. Lève 429 quand une requête concurrente a épuisé
    le quota entre le contrôle d'entrée (`check_analysis_quota`) et ici — le
    rollback emporte alors l'insert en cours avec le compteur.
    """
    if _has_unlimited_analyses(user):
        return  # Illimité — rien à faire

    quota = user.analyses_quota
    if not await try_increment_analysis_usage(user, db):
        # Quota épuisé par une requête concurrente — on annule le dataset,
        # dont l'insert est encore dans cette transaction.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Monthly comparison quota exhausted ({quota}/{quota}). "
                "Quota resets on the 1st of next month."
            ),
        )


async def warn_if_quota_threshold_crossed(user: User, db: AsyncSession) -> None:
    """
    Avertissement par email au franchissement des 80 % du quota.

    Séparé de l'incrément parce qu'il doit lire un compteur DURABLE : prévenir
    depuis la transaction reviendrait à annoncer une consommation qu'un commit
    raté n'aurait jamais rendue vraie. À appeler APRÈS le commit.

    Ne bloque jamais l'import : toute erreur d'envoi est journalisée et avalée.
    """
    if _has_unlimited_analyses(user):
        return

    await db.refresh(user)
    quota = user.analyses_quota
    used = user.analyses_used_this_month
    if quota and used == int(quota * 0.8):
        try:
            from app.services.email_service import send_quota_warning_email

            await send_quota_warning_email(
                to=user.email,
                used=used,
                quota=quota,
                plan=user.subscription_plan.value,
            )
        except Exception:
            logger.warning("Failed to send 80%% quota warning email to %s", user.email)


# ── Legacy alias (backward compat with existing callers) ─────────────────────


async def require_analysis_access(user: Annotated[User, Depends(get_or_create_user)]) -> User:
    """Backward-compat alias — previously required ADVANCED plan.
    Now all paid plans can launch analyses (subject to quota).
    Use check_analysis_quota for new code."""
    return user
