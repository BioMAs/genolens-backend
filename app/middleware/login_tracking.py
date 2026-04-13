"""
Middleware for tracking user login events.
Records a deduplicated login event per user per 30-minute window.
Non-blocking: uses BackgroundTask so the request is never delayed.
"""
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Deduplication window: one event per user per N minutes
_DEDUP_WINDOW_MINUTES = 30


def _extract_user_id(authorization: str) -> str | None:
    """
    Decode the Bearer JWT and return the 'sub' claim (Supabase user UUID).
    Returns None on any failure — never raises.
    """
    try:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        token = authorization.removeprefix("Bearer ").strip()
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True},
        )
        return payload.get("sub")
    except Exception:
        return None


async def _record_login_event(user_id: str) -> None:
    """
    Insert a UserLoginEvent if no event exists for this user in the last
    _DEDUP_WINDOW_MINUTES minutes. Silently swallows all errors.
    """
    try:
        from sqlalchemy import select, func
        from app.models.models import UserLoginEvent

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=_DEDUP_WINDOW_MINUTES)

        async with AsyncSessionLocal() as session:
            # Check for a recent event
            result = await session.execute(
                select(func.count())
                .select_from(UserLoginEvent)
                .where(UserLoginEvent.user_id == user_id)
                .where(UserLoginEvent.created_at >= cutoff)
            )
            count = result.scalar() or 0

            if count == 0:
                event = UserLoginEvent(id=uuid4(), user_id=user_id)
                session.add(event)
                await session.commit()
    except Exception as exc:
        logger.debug("login_tracking: failed to record event for %s: %s", user_id, exc)


class LoginTrackingMiddleware(BaseHTTPMiddleware):
    """
    For every authenticated request (Authorization: Bearer …), dispatch a
    non-blocking background task that may insert a UserLoginEvent.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        authorization = request.headers.get("Authorization", "")
        user_id = _extract_user_id(authorization) if authorization else None

        response: Response = await call_next(request)

        if user_id:
            # Fire-and-forget after the response is sent
            import asyncio
            asyncio.ensure_future(_record_login_event(user_id))

        return response
