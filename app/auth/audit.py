"""
The audit trail of the local authentication (`auth_events`): who did what, from where, and when.

`record_event` adds one `AuthEvent` for the caller's transaction. It is called from the middle of
login, password-reset and session flows, so two things matter more than the event itself:

- It never writes a secret. Any key of the details that names one is dropped, at any depth: a key
  that contains `password`, `code`, `token` or `secret` (case-insensitive; `recovery_code` is one
  of those by the same rule) is removed, so `password`, `new_password`, `mfa_code`, `raw_token`,
  `client_secret` and `RecoveryCode` all go. This is deliberately a substring test: an audit line
  that loses a harmless key is a nuisance, one that keeps a secret is a breach. The price is that
  innocent names are lost too (`status_code`, `error_code`, `postcode`): name them otherwise
  (`http_status`, `error_kind`). Only keys are looked at; a secret inside a value (`reason=` a
  message that quotes a token) is the caller's to keep out.
- It never fails because of what the caller passes. The details go into a JSON column, and a UUID,
  a datetime, an Enum or a set would raise at flush time, in the middle of the caller's flow.
  Values are coerced instead: UUID, datetime, date and time become strings (ISO 8601 for the
  last three), an Enum becomes its value, a set or tuple becomes a list, a non-finite float
  (`NaN` is not valid JSON on Postgres) becomes its name as a string, and anything else becomes
  `str(value)` (`<TypeName>` if even that raises). Mapping keys become strings. Nesting is cut at
  `_MAX_DEPTH` levels, which also ends a structure that contains itself.

Details are passed as keyword arguments, so `db`, `event_type`, `user_id` and `request` cannot be
used as keys of the details.

Transactions. Nothing here commits; the caller does. The event is flushed, so a problem with the row
itself (a `user_id` that is not an account: pass None for an unknown one) shows at the call site.
`created_at` is the application's clock in UTC, not the database default, which is the start of the
transaction on Postgres and has a one-second resolution on SQLite.
"""

import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auth import AuthEvent, AuthEventType

# A key containing any of these (in any case) is dropped from the details. "recovery_code" is
# already caught by "code"; it stays listed because it is a name the module promises to remove.
_SENSITIVE_KEY_WORDS = ("password", "code", "token", "secret", "recovery_code")

# Widths of the `auth_events` columns: Postgres refuses a longer value, and both come straight from
# the request, so a hostile client would otherwise turn a failed login into a 500.
_IP_MAX_LENGTH = 64
_USER_AGENT_MAX_LENGTH = 512

_MAX_DEPTH = 8
_TRUNCATED = "[truncated]"


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(word in lowered for word in _SENSITIVE_KEY_WORDS)


def _text(value: Any) -> str:
    """`str(value)`, without letting a broken `__str__` out."""
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _json_safe(value: Any, depth: int = 0) -> Any:
    """`value` as plain JSON data, with the keys that name a secret removed from every mapping."""
    if depth > _MAX_DEPTH:
        return _TRUNCATED
    if isinstance(value, Enum):
        return _json_safe(value.value, depth)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = key if isinstance(key, str) else _text(key)
            if not _is_sensitive(name):
                cleaned[name] = _json_safe(item, depth + 1)
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth + 1) for item in value]
    return _text(value)


async def record_event(
    db: AsyncSession,
    event_type: AuthEventType,
    user_id: Optional[UUID] = None,
    request: Optional[Request] = None,
    **details: Any,
) -> None:
    """Add an audit event to the caller's transaction.

    `event_type` is stored as its value. `user_id` is the account concerned, None when there is
    none (a login for an address that has no account). From `request` come the IP
    (`request.client.host`, None when the request has no client) and the user-agent header, cut to
    the width of their columns (64 and 512). `details` keeps what it is given minus the keys that
    name a secret, coerced to JSON (see the module docstring). Flushes, does not commit.
    """
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    if request is not None:
        if request.client is not None and request.client.host:
            ip = request.client.host[:_IP_MAX_LENGTH]
        header = request.headers.get("user-agent")
        if header:
            user_agent = header[:_USER_AGENT_MAX_LENGTH]

    db.add(
        AuthEvent(
            user_id=user_id,
            event_type=event_type.value,
            ip=ip,
            user_agent=user_agent,
            details=_json_safe(details),
            created_at=datetime.now(UTC),
        )
    )
    await db.flush()
