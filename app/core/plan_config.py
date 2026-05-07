"""
Central plan limits configuration for GenoLens subscription tiers.

Usage:
    from app.core.plan_config import get_max_projects, get_max_storage_bytes

Both helpers return None when the limit is unlimited (ADMIN role or ADVANCED plan
with no defined cap — use None-check rather than comparing to a large sentinel).
"""
from typing import Optional
from app.models.models import SubscriptionPlan, UserRole

# ---------------------------------------------------------------------------
# Plan limits table
# ---------------------------------------------------------------------------
# max_projects     : maximum number of projects an owner may create (None = ∞)
# max_storage_bytes: maximum cumulative uploaded-file storage in bytes (None = ∞)

_PLAN_LIMITS: dict[SubscriptionPlan, dict] = {
    SubscriptionPlan.BASIC: {
        "max_projects": 3,
        "max_storage_bytes": 500 * 1024 * 1024,  # 500 MB
    },
    SubscriptionPlan.PREMIUM: {
        "max_projects": 20,
        "max_storage_bytes": 10 * 1024 * 1024 * 1024,  # 10 GB
    },
    SubscriptionPlan.ADVANCED: {
        "max_projects": None,  # unlimited
        "max_storage_bytes": 50 * 1024 * 1024 * 1024,  # 50 GB
    },
}


def get_max_projects(plan: SubscriptionPlan, role: UserRole) -> Optional[int]:
    """
    Return the maximum number of owned projects for a user, or None for unlimited.
    ADMIN role is always unlimited regardless of plan.
    """
    if role == UserRole.ADMIN:
        return None
    return _PLAN_LIMITS.get(plan, _PLAN_LIMITS[SubscriptionPlan.BASIC])["max_projects"]


def get_max_storage_bytes(plan: SubscriptionPlan, role: UserRole) -> Optional[int]:
    """
    Return the maximum cumulative storage in bytes, or None for unlimited.
    ADMIN role is always unlimited regardless of plan.
    """
    if role == UserRole.ADMIN:
        return None
    return _PLAN_LIMITS.get(plan, _PLAN_LIMITS[SubscriptionPlan.BASIC])["max_storage_bytes"]


def get_plan_limits(plan: SubscriptionPlan, role: UserRole) -> dict:
    """
    Return a dict with max_projects and max_storage_bytes for a user.
    None values mean unlimited.
    """
    return {
        "max_projects": get_max_projects(plan, role),
        "max_storage_bytes": get_max_storage_bytes(plan, role),
    }
