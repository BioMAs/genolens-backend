"""
Account-state enforcement: suspension, cancellation, and subscription expiry.

Two controls that had columns, admin endpoints and even frontend UX, but no
enforcement at all until now:

1. `users.status` — an admin could suspend or cancel an account through
   `PATCH /admin/users/{id}/status`, and nothing anywhere read the flag back.
   A suspended user kept full API access.
2. `users.subscription_ends_at` — written by the admin invite flow and by the
   Stripe webhook, read only for display. It was never compared to the clock, so
   the "Accès jusqu'au" date in the admin invite form was decorative.

Both are enforced here, in one dependency, applied at the router level in
`app/main.py`. The router level is deliberate: thirteen endpoint modules resolve
their caller through `get_current_user` without ever touching
`get_or_create_user`, so a check placed inside the latter would silently miss
them. `tests/test_account_state_coverage.py` fails if a user-facing route escapes.

The 403 payload for a suspended account matches, field for field, a contract the
frontend already implements and had never once received: the axios interceptor
(`frontend/src/utils/api.ts`) keys on `detail.error === "account_inactive"`,
stores `detail.status` in a cookie and redirects to `/suspended`, where the page
picks its copy from that same status. Do not reshape this payload without
changing that interceptor.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status

from app.api.deps.subscription import get_or_create_user
from app.models.models import User, UserRole, UserStatus

logger = logging.getLogger(__name__)

# Roles that answer to neither control. Same exemption every other guard in
# app/api/deps/subscription.py grants.
_PRIVILEGED_ROLES = (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN)

# Statuses that deny access outright. PENDING is NOT here: it is cleared on the
# invitee's first authenticated request (see `get_or_create_user`), so an account
# still carrying it has not reached this code with a valid token yet.
_INACTIVE_STATUSES = (UserStatus.SUSPENDED, UserStatus.CANCELLED)

# Methods an expired subscription may still use. Expiry is read-only, not a
# lock-out: the user keeps reading the projects and results they paid for.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# POST endpoints that compute a view rather than create anything. The HTTP method
# alone would misclassify these as writes and hide results an expired account is
# entitled to keep reading, so the exception is spelled out rather than inferred.
# Matched as a suffix against the request path.
_READONLY_POST_SUFFIXES = (
    "/genes/search",
    "/venn-analysis",
    "/logfc-scatter",
    "/intersection-enrichment",
)


def parse_subscription_end(raw: Optional[str]) -> Optional[datetime]:
    """
    Parse `User.subscription_ends_at` into an aware UTC datetime.

    The column is a `String(50)` holding ISO 8601, written by two callers that do
    not agree on a format: the admin invite path stores
    `datetime.isoformat()` (naive whenever the caller passed a naive datetime),
    while the Stripe webhook stores `fromtimestamp(..., tz=utc).isoformat()`.
    Comparing those as strings would be wrong — "2026-01-02T00:00:00" sorts
    against "2026-01-02T00:00:00+00:00" on punctuation, not on time — so they are
    parsed. A naive value is read as UTC, which is what both writers mean.

    Returns None for anything unparsable. Refusing to act on a value we do not
    understand is the safe direction: the cost of ignoring a malformed date is a
    user who keeps access slightly too long, and the cost of guessing is locking
    a paying customer out of their own data.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        logger.warning("Unparsable subscription_ends_at %r — treated as no expiry", raw)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def subscription_has_expired(user: User, now: Optional[datetime] = None) -> bool:
    """True when this account's paid access ended before `now` (default: UTC now)."""
    ends_at = parse_subscription_end(user.subscription_ends_at)
    if ends_at is None:
        return False
    return ends_at <= (now or datetime.now(timezone.utc))


def _is_readonly_post(request: Request) -> bool:
    """True for the POST endpoints that read rather than create."""
    if request.method != "POST":
        return False
    path = request.url.path.rstrip("/")
    return any(path.endswith(suffix) for suffix in _READONLY_POST_SUFFIXES)


async def require_active_account(
    request: Request,
    user: Annotated[User, Depends(get_or_create_user)],
) -> User:
    """
    Deny suspended and cancelled accounts outright; hold expired ones to reads.

    Applied per-router in `app/main.py` rather than per-endpoint. FastAPI caches a
    dependency's result within a request, so this costs no extra query on the
    endpoints that already resolve `get_or_create_user`.
    """
    if user.role in _PRIVILEGED_ROLES:
        return user

    if user.status in _INACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "account_inactive",
                "status": user.status.value,
                "message": (
                    "This account is no longer active. " "Please contact support@scilicium.com."
                ),
            },
        )

    if request.method in _SAFE_METHODS or _is_readonly_post(request):
        return user

    if subscription_has_expired(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "subscription_expired",
                "expires_at": user.subscription_ends_at,
                "message": (
                    "Your access period has ended. You can still consult your existing "
                    "projects and results, but creating new ones requires renewing your "
                    "subscription. Please contact support@scilicium.com."
                ),
            },
        )

    return user
