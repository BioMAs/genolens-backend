"""
The identity directory: where the identity of an account (its id, address, name, last sign-in and
confirmation) is looked up, created and deleted, whichever authentication runs.

Every function chooses its implementation from `settings.AUTH_MODE`:

- `supabase` (until the cutover, and for a rollback): identities live in the hosted provider.
  Each function runs the code that did the job before the directory existed (the lookups of
  `app.core.supabase_auth`, the invitation of `AccountService`, the requests the admin endpoints
  made), so that nothing changes in this mode: same requests, same answers, same side effects.
- `local`: the `users` table is the directory. Nothing is asked of the provider.

The account administration (`app/api/endpoints/admin.py`, `AccountService`) and the project
routes that resolve an account go through here; phase 7 deletes the `supabase` branches together
with the mode.

What an identity carries is deliberately small: no hash, no secret, no token. The second factor,
the lock and the other local sign-in state are read from the `users` row by whoever needs them.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, cast
from uuid import UUID, uuid4

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.supabase_auth import lookup_user_by_email, lookup_user_by_id
from app.models.models import User

logger = logging.getLogger(__name__)


@dataclass
class AuthIdentity:
    """An account as the directory knows it."""

    id: UUID
    email: str
    full_name: Optional[str]
    last_sign_in_at: Optional[datetime]
    email_confirmed_at: Optional[datetime]


class IdentityNotFoundError(Exception):
    """`delete_identity`: the provider had no such identity (it is gone either way)."""


class IdentityDeletionError(Exception):
    """`delete_identity`: the provider refused to delete the identity. `str(exc)` is its answer."""


def role_source_is_supabase() -> bool:
    """Whether roles come from the provider's `profiles` (and its per-process cache) or from
    `users.role` alone. Writes to `profiles` and role-cache invalidations happen only when true."""
    return settings.AUTH_MODE == "supabase"


def identities_are_local() -> bool:
    """Whether the `users` table is the directory (`AUTH_MODE=local`)."""
    return settings.AUTH_MODE == "local"


# ── Lookups ──────────────────────────────────────────────────────────────────


async def find_identity_by_email(db: AsyncSession, email: str) -> Optional[AuthIdentity]:
    """The account of `email`, or None.

    Local: the address is compared ignoring case and surrounding spaces, like the local sign-in
    does, and an address that two accounts share (possible until the migration's duplicates are
    merged) finds neither, since neither can be told from the other.
    """
    if not identities_are_local():
        record = await lookup_user_by_email(email)
        return None if record is None else _from_provider(record)

    normalized = email.strip().lower()
    rows = (
        (await db.execute(select(User).where(func.lower(User.email) == normalized).limit(2)))
        .scalars()
        .all()
    )
    if len(rows) > 1:
        logger.warning("Directory lookup refused: several accounts share one address")
        return None
    return _from_user(rows[0]) if rows else None


async def find_identity_by_id(db: AsyncSession, user_id: UUID) -> Optional[AuthIdentity]:
    """The account `user_id`, or None."""
    if not identities_are_local():
        record = await lookup_user_by_id(user_id)
        return None if record is None else _from_provider(record)

    user = await db.scalar(select(User).where(User.id == user_id))
    return None if user is None else _from_user(user)


async def list_identities(db: AsyncSession) -> dict[UUID, AuthIdentity]:
    """Every account, by id.

    Supabase: the provider's user list, as the admin endpoints read it (an answer other than 200
    gives an empty directory, a transport error propagates).
    """
    if not identities_are_local():
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users", headers=_provider_headers()
            )
        if response.status_code != 200:
            return {}
        data = response.json()
        records = data.get("users", []) if isinstance(data, dict) else data
        identities = (_from_provider(record) for record in records)
        return {identity.id: identity for identity in identities}

    users = (await db.execute(select(User))).scalars().all()
    return {user.id: _from_user(user) for user in users}


# ── Creation and deletion ────────────────────────────────────────────────────


async def create_identity(email: str, full_name: Optional[str]) -> UUID:
    """The id of a new account, to key its `users` row on.

    Supabase: the provider's invitation (`AccountService._send_supabase_invite`), which creates the
    identity and e-mails the provider's magic link; RuntimeError when it refuses or returns no id.
    Local: a fresh `uuid4()`. The local invitation (a one-time link the admin issues) is sent by
    `AccountService` once the row is committed.
    """
    if identities_are_local():
        return uuid4()

    # Deferred: `app.services.account_service` imports this module.
    from app.services.account_service import AccountService

    invited = await AccountService._send_supabase_invite(email, full_name)
    auth_id = invited.get("id")
    if not auth_id:
        raise RuntimeError(f"Supabase invite returned no user id: {invited}")
    return UUID(auth_id)


async def delete_identity(user_id: UUID) -> None:
    """Delete the identity of an account whose local rows are already gone.

    Supabase: the provider's legacy rows that block the deletion are removed first, best effort
    (`sequencing_projects`, and `status_history` as subject or actor), then the identity itself.
    `IdentityNotFoundError` when the provider had no such identity, `IdentityDeletionError` with
    its answer when it refused; a transport error propagates.
    Local: nothing to do. The account was its `users` row, and its sessions, links and recovery
    codes went with it (ON DELETE CASCADE). The Sequencing tables are never touched.
    """
    if identities_are_local():
        return

    headers = _provider_headers()
    async with httpx.AsyncClient() as client:
        # 'sequencing_projects' exists in the provider's database and blocks the deletion.
        try:
            await client.delete(
                f"{settings.SUPABASE_URL}/rest/v1/sequencing_projects",
                headers=headers,
                params={"user_id": f"eq.{user_id}"},
            )
        except Exception as e:
            logger.warning("Warning: Failed to clean up remote sequencing_projects: %s", e)

        # 'status_history' rows referencing this user, either as the subject (user_id) or the
        # actor (changed_by). Those FKs have no ON DELETE rule and otherwise block the deletion.
        for column in ("changed_by", "user_id"):
            try:
                await client.delete(
                    f"{settings.SUPABASE_URL}/rest/v1/status_history",
                    headers=headers,
                    params={column: f"eq.{user_id}"},
                )
            except Exception as e:
                logger.warning("Warning: Failed to clean up status_history.%s: %s", column, e)

        response = await client.delete(
            f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}", headers=headers
        )

    if response.status_code == 200:
        return
    if response.status_code == 404:
        raise IdentityNotFoundError(str(user_id))
    raise IdentityDeletionError(response.text)


# ── Internals ────────────────────────────────────────────────────────────────


def _provider_headers() -> dict[str, str]:
    """The headers the admin endpoints have always sent to the provider's admin API."""
    key = settings.SUPABASE_KEY
    if key is None:  # only AUTH_MODE=local leaves it unset, and the provider is never called then
        raise RuntimeError("SUPABASE_KEY is not set: the hosted provider cannot be called")
    return {
        "apikey": key,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or key}",
        "Content-Type": "application/json",
    }


def _instant(value: Any) -> Optional[datetime]:
    """A timestamp of the provider, or None when it has none (or one that does not parse)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _from_provider(record: dict[str, Any]) -> AuthIdentity:
    """An identity from a user of the provider's admin API."""
    return AuthIdentity(
        id=UUID(str(record["id"])),
        # As the admin endpoints always read it; a phone-only identity has none.
        email=cast(str, record.get("email")),
        full_name=record.get("user_metadata", {}).get("full_name"),
        last_sign_in_at=_instant(record.get("last_sign_in_at")),
        email_confirmed_at=_instant(record.get("email_confirmed_at") or record.get("confirmed_at")),
    )


def _from_user(user: User) -> AuthIdentity:
    """An identity from a `users` row."""
    return AuthIdentity(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        last_sign_in_at=user.last_login_at,
        email_confirmed_at=user.email_verified_at,
    )
