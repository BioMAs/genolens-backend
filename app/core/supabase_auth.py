"""
Supabase JWT validation utilities.
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from uuid import UUID
import jwt
from jwt import PyJWTError
from pydantic import BaseModel
import httpx

from app.core.config import settings
from app.models.models import UserRole, SubscriptionPlan

logger = logging.getLogger(__name__)


# ── Role lookup: shared client + per-process cache ────────────────────────────
#
# `verify_supabase_token` runs on EVERY authenticated request and reads
# `profiles.role` from Supabase. It used to open a new client (fresh TLS
# handshake, 30 s timeout) per request, so a page firing 20-50 calls paid
# 20-50 round trips. Now one client is reused, the role is cached per user,
# and concurrent requests for the same user share one in-flight lookup.
#
# Role changes (this is what `require_admin` and `require_subscriber` read):
# - through the admin endpoints, which call `invalidate_cached_role`: effective
#   on the user's next request in this process. Prod runs a single uvicorn
#   process, so that means immediately;
# - any other way (Supabase dashboard, another process): effective within
#   ROLE_CACHE_TTL_SECONDS.
#
# Only successful lookups are cached. A failure still falls back to USER for
# that request, as before, but the next request asks Supabase again.
# Keys come from signature-verified JWTs, so the cache is bounded by the
# number of real users.
ROLE_CACHE_TTL_SECONDS = 60.0
_SUPABASE_TIMEOUT = httpx.Timeout(5.0)

_monotonic = time.monotonic
_role_cache: Dict[UUID, Tuple[float, str]] = {}  # user_id -> (expires_at, role)
_role_inflight: Dict[UUID, "asyncio.Task[Optional[str]]"] = {}
_http_client: Optional[httpx.AsyncClient] = None
_http_client_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_http_client() -> httpx.AsyncClient:
    """The shared client, created lazily on the running event loop."""
    global _http_client, _http_client_loop
    loop = asyncio.get_running_loop()
    # A client's pooled connections belong to the loop that opened them; a
    # script calling asyncio.run() twice gets a fresh client on the new loop.
    if _http_client is None or _http_client.is_closed or _http_client_loop is not loop:
        _http_client = httpx.AsyncClient(timeout=_SUPABASE_TIMEOUT)
        _http_client_loop = loop
    return _http_client


async def close_supabase_http_client() -> None:
    """Close the shared client (app shutdown)."""
    global _http_client, _http_client_loop
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None
    _http_client_loop = None


def invalidate_cached_role(user_id: UUID) -> None:
    """
    Forget the cached role of `user_id`, so their next request reads it again.

    Also detaches any lookup in flight: it may have read the role before the
    change, and must neither serve later requests nor be cached.
    """
    _role_cache.pop(user_id, None)
    _role_inflight.pop(user_id, None)


async def get_cached_user_role(user_id: UUID) -> Optional[str]:
    """`fetch_user_role_from_supabase`, cached for ROLE_CACHE_TTL_SECONDS."""
    entry = _role_cache.get(user_id)
    if entry is not None and entry[0] > _monotonic():
        return entry[1]

    loop = asyncio.get_running_loop()
    task = _role_inflight.get(user_id)
    if task is None or task.get_loop() is not loop:
        task = loop.create_task(_fetch_and_cache_role(user_id))
        _role_inflight[user_id] = task
        task.add_done_callback(lambda done: _forget_inflight(user_id, done))
    # Shielded: a request cancelled mid-burst (client gone) must not cancel the
    # lookup the rest of the burst is waiting on.
    return await asyncio.shield(task)


async def _fetch_and_cache_role(user_id: UUID) -> Optional[str]:
    role = await fetch_user_role_from_supabase(user_id)
    # Not cached if `invalidate_cached_role` detached this lookup meanwhile.
    if role and _role_inflight.get(user_id) is asyncio.current_task():
        _role_cache[user_id] = (_monotonic() + ROLE_CACHE_TTL_SECONDS, role)
    return role


def _forget_inflight(user_id: UUID, task: "asyncio.Task[Optional[str]]") -> None:
    if _role_inflight.get(user_id) is task:
        del _role_inflight[user_id]


class SupabaseUser(BaseModel):
    """Parsed Supabase user from JWT token."""
    user_id: UUID
    email: str
    role: UserRole = UserRole.USER
    subscription_tier: SubscriptionPlan = SubscriptionPlan.STARTER
    max_projects: int = 1
    features_access: dict = {}
    user_metadata: dict = {}


async def fetch_user_role_from_supabase(user_id: UUID) -> Optional[str]:
    """
    Fetch user role from Supabase profiles table. Uncached: requests go through
    `get_cached_user_role`.

    Args:
        user_id: User UUID

    Returns:
        Role string if found, None otherwise
    """
    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        response = await _get_http_client().get(
            f"{settings.SUPABASE_URL}/rest/v1/profiles",
            headers=headers,
            params={"id": f"eq.{user_id}", "select": "role"},
        )

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0].get("role")
        else:
            logger.warning(
                "Failed to fetch role for user %s. Status: %s, Body: %s",
                user_id,
                response.status_code,
                response.text,
            )

        return None
    except Exception as e:
        logger.warning("Error fetching user role from Supabase: %r", e)
        return None


async def verify_supabase_token(token: str) -> Optional[SupabaseUser]:
    """
    Verify Supabase JWT token and extract user information.
    Fetches the user role from the profiles table in Supabase.

    Args:
        token: JWT token from Authorization header

    Returns:
        SupabaseUser if token is valid, None otherwise
    """
    try:
        # Decode JWT using Supabase JWT secret
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True}
        )

        # Extract user information from JWT claims
        user_id = UUID(payload.get("sub"))
        email = payload.get("email")

        if not user_id or not email:
            return None

        # Role from Supabase profiles table, cached per process (see top of module)
        role_str = await get_cached_user_role(user_id)

        # Parse role or default to USER
        # Convert to uppercase to match Python Enum values
        try:
            role = UserRole(role_str.upper()) if role_str else UserRole.USER
        except (ValueError, AttributeError):
            # If role from DB is invalid or None, default to USER
            role = UserRole.USER

        # Extract user metadata for subscription info (fallback)
        user_metadata = payload.get("user_metadata", {})
        
        # Handle subscription plan mapping
        sub_plan_str = user_metadata.get("subscription_tier", "STARTER").upper()
        if sub_plan_str in ("FREE", "BASIC"):
            sub_plan_str = "STARTER"

        try:
            subscription_plan = SubscriptionPlan(sub_plan_str)
        except ValueError:
            subscription_plan = SubscriptionPlan.STARTER

        return SupabaseUser(
            user_id=user_id,
            email=email,
            role=role,
            subscription_tier=subscription_plan,
            max_projects=user_metadata.get("max_projects", 1),
            features_access=user_metadata.get("features_access", {}),
            user_metadata=user_metadata
        )

    except PyJWTError:
        return None
    except (ValueError, KeyError):
        return None


async def lookup_user_by_email(email: str) -> Optional[dict]:
    """
    Look up a Supabase Auth user by email address using the Admin API.

    Requires SUPABASE_SERVICE_ROLE_KEY to be set.

    Returns a dict with at least ``id`` and ``email`` keys, or None if not found.
    """
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        return None

    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers,
                params={"email": email},
            )

            if response.status_code == 200:
                data = response.json()
                users = data.get("users", [])
                return users[0] if users else None

            return None
    except Exception as exc:
        print(f"Error looking up user by email {email}: {repr(exc)}")
        return None


async def lookup_user_by_id(user_id: UUID) -> Optional[dict]:
    """
    Look up a Supabase Auth user by UUID using the Admin API.

    Requires SUPABASE_SERVICE_ROLE_KEY to be set.

    Returns a dict with at least ``id`` and ``email`` keys, or None if not found.
    """
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        return None

    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()

            return None
    except Exception as exc:
        print(f"Error looking up user by id {user_id}: {repr(exc)}")
        return None
