"""
One-off copy of the Supabase credentials into the local `users` table (spec, section 8).

    python -m app.auth.migrate_from_supabase [--apply] [--report PATH]
    python -m app.auth.migrate_from_supabase --send-resets

Run it in the API container. It reads `auth.users` and `public.profiles` of the Supabase project
(`SUPABASE_DB_URL`, the connection string of the pooler in session mode), read-only, and plans what
each local account would receive; without `--apply` that is all it does (a dry run). `--apply`
writes it in a single transaction. The plan is a pure function of the two lists
(`plan_migration`) and the rules are those of section 8:

- scope: the accounts of `users` only (the Supabase project is shared with another application).
  Only their rows are fetched (`WHERE id = ANY(<local ids>)`); the other accounts are counted
  (`out_of_scope`), never read, their hashes included;
- `password_hash` <- `encrypted_password` when it is a bcrypt hash (`$2...`), else nothing, and the
  account is reported (`no_password`): it gets a reset link (`--send-resets`);
- `email_verified_at` <- `email_confirmed_at`, `last_login_at` <- `last_sign_in_at`;
- e-mail: Supabase's wins, lowercased and stripped as every stored address (section 7.1), unless
  another local account has it already, ignoring case (`email_conflict`);
- role: a local ADMIN / SCILICIUM_ADMIN stays; otherwise `profiles.role` in capitals if it is a
  `UserRole`, else the local role stays (`role_invalid`);
- banned or deleted at Supabase: reported (`banned`, `deleted`), nothing applied;
- never overwrite a password set in the new system: `password_changed_at` can only have been
  written by the new system (reset, change, invitation), so a local `password_changed_at` that is
  set keeps the local password when the account was never migrated (`auth_migrated_at` NULL) or
  when it is later than `auth_migrated_at`. A Supabase hash is applied only to an account that
  has no local password change, or whose change predates the migration.

A tool for BEFORE the cutover. Once `AUTH_MODE=local` the local data is authoritative: Supabase's
roles, e-mails and last sign-ins are frozen at their cutover values (the admin routes stop
writing `profiles` then), so another `--apply` would put them back over what has changed here,
a demoted admin included. `--apply` and `apply_plan` therefore refuse in `AUTH_MODE=local`
(`MigrationError`, exit code 2); the dry run and `--report` read only and stay available.

Conventions that make a replay harmless:

- A field of a `UserUpdate` that is None is never written: it means "nothing to copy", not "blank
  it". A local password, verification date or last sign-in that Supabase cannot match is kept.
- `auth_migrated_at` is the day the account FIRST moved, so it is set once and never moved
  forward. If every run moved it, the run after a password change would see that password as
  older than the migration and overwrite it. For the same reason an account whose password was
  set in the new system BEFORE it was ever migrated is not stamped at all: a stamp would date
  that password before the migration and the next run would overwrite it.
- The password write carries its own guard in the UPDATE (not only in the plan), so a password
  chosen between the planning and the writing is kept too.

Secrets. A hash is read, compared and written, and never printed, logged or put in the report:
the standard output holds counts, the report holds `user_id`, kind and a detail line (with
e-mail addresses, so it is created readable by its owner only and belongs outside git), and the
connection string, which contains a password, is never echoed, not even in an error. The engine
this command opens hides statement parameters, so a database error cannot print a hash either, and
an error is printed as its class, its SQLSTATE and the first line of its message only (`_describe`:
asyncpg's `DETAIL:` line can quote the failing row). The dataclasses that carry a hash leave it out
of their `repr`.

Reset links (`--send-resets`, `AUTH_MODE=local` only) go to the migrated accounts left without a
password that hold no live reset link; the link of an e-mail that did not leave is withdrawn, so a
rerun mails exactly the failures.

Exit codes: 0 done; 1 something failed while running (Supabase unreachable, a database error, a
reset e-mail that did not leave); 2 usage or configuration (a missing `SUPABASE_DB_URL`, resets
outside `AUTH_MODE=local`, a report that cannot be written).
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Literal, NoReturn, Optional, Sequence, TextIO, get_args
from uuid import UUID

import asyncpg
from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.audit import record_event
from app.auth.clock import aware_utc, now_utc
from app.auth.emails import send_password_reset_email
from app.auth.one_time_tokens import InvalidOneTimeToken, consume_token, issue_token
from app.core.config import settings
from app.models.auth import AuthEventType, AuthOneTimeToken, OneTimeTokenPurpose
from app.models.models import User, UserRole, UserStatus
from app.services.email_service import EmailNotConfiguredError, EmailSendError

logger = logging.getLogger(__name__)

AnomalyKind = Literal[
    "no_password",
    "email_conflict",
    "missing_remote",
    "banned",
    "deleted",
    "role_invalid",
    "duplicate_email",
]
ANOMALY_KINDS: tuple[str, ...] = get_args(AnomalyKind)

# Reads the Supabase accounts of the given local ids: `fetch_remote_users`, or a test's stand-in.
Fetch = Callable[[str, list[UUID]], Awaitable["RemoteAccounts"]]

# A local account with one of these roles keeps it whatever the Supabase profile says.
_PROTECTED_ROLES = (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN)


class MigrationError(Exception):
    """A refusal the command explains on standard error (usage or configuration: exit code 2)."""


def _refuse_after_cutover() -> None:
    """The migration writes Supabase's values over local accounts: only until the cutover."""
    if settings.AUTH_MODE == "local":
        raise MigrationError(
            "applying the Supabase migration (--apply) is refused in AUTH_MODE=local: it is a "
            "pre-cutover tool, and since the cutover the local data (roles, e-mails, passwords, "
            "last sign-ins) is the authority, so a run would put the values Supabase froze at "
            "the cutover back over it. The dry run and --report still work"
        )


# ---- data --------------------------------------------------------------------------------------


@dataclass
class RemoteUser:
    """One row of `auth.users`, with the role of its `public.profiles` row (None if it has none)."""

    id: UUID
    email: str
    encrypted_password: Optional[str] = field(default=None, repr=False)  # a hash: never printed
    email_confirmed_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    banned_until: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    profile_role: Optional[str] = None


@dataclass
class RemoteAccounts:
    """What the read of the hosted database returns: the accounts of the local ids it was given
    (`users`), and how many accounts the project holds in all (`total`), the others counted only."""

    users: list[RemoteUser]
    total: int


@dataclass
class LocalUser:
    """What the plan needs to know of a row of `users`."""

    id: UUID
    email: str
    role: UserRole
    password_changed_at: Optional[datetime] = None
    auth_migrated_at: Optional[datetime] = None


@dataclass
class UserUpdate:
    """What to copy onto one local account. A None field is not written (see the module docstring).

    `keep_password` is True when the account set its password in the new system (never migrated,
    or after it was migrated): `password_hash` is then None whatever Supabase holds.
    """

    user_id: UUID
    password_hash: Optional[str] = field(default=None, repr=False)  # a hash: never printed
    email: Optional[str] = None
    email_verified_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    role: Optional[UserRole] = None
    keep_password: bool = False


@dataclass
class Anomaly:
    """Something the people running the migration must look at. `detail` names the e-mail
    addresses involved and never holds a hash."""

    kind: AnomalyKind
    user_id: UUID
    detail: str


@dataclass
class MigrationPlan:
    updates: list[UserUpdate] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    out_of_scope: int = 0  # Supabase accounts that have no local account


# ---- the plan (pure) ---------------------------------------------------------------------------


def _email_key(email: str) -> str:
    return email.strip().lower()


def _password_changed_since_migration(local: LocalUser) -> bool:
    """True when the local password was set in the new system and must not be overwritten.

    `password_changed_at` is only ever written by the new system, so when it is set the password
    is kept if the account was never migrated (`auth_migrated_at` NULL: nothing predates it) or
    if the change is later than the migration. Not kept: no local change at all, or a change
    that predates the migration (the Supabase hash then replaces what the migration superseded).
    """
    changed, migrated = local.password_changed_at, local.auth_migrated_at
    if changed is None:
        return False
    if migrated is None:
        return True
    return aware_utc(changed) > aware_utc(migrated)


def _role_for(local: LocalUser, remote: RemoteUser) -> tuple[Optional[UserRole], Optional[str]]:
    """`(role to apply, anomaly detail)`: at most one of them is set."""
    if local.role in _PROTECTED_ROLES:
        return None, None
    wanted = (remote.profile_role or "").strip()
    if not wanted:
        return None, None
    try:
        return UserRole(wanted.upper()), None
    except ValueError:
        return None, (
            f"{local.email}: the Supabase profile role {wanted!r} is not a GenoLens role; "
            f"kept {local.role.value}"
        )


def plan_migration(
    local: list[LocalUser],
    remote: list[RemoteUser],
    now: Optional[datetime] = None,
    remote_total: Optional[int] = None,
) -> MigrationPlan:
    """What each local account would receive from Supabase, and what needs a human. No I/O.

    `now` is the instant a ban is judged at (`banned_until` later than it is a ban); it defaults
    to the current time, and a test or a replay passes its own so that the plan is deterministic.
    `remote_total` is how many accounts the Supabase project holds in all, when `remote` lists only
    those of the local ids (what `fetch_remote_users` reads): `out_of_scope` is then that total
    minus the local accounts found. Without it, `remote` is taken as the whole project.
    """
    now = now_utc(now)
    remote_by_id = {r.id: r for r in remote}
    local_ids = {u.id for u in local}
    found = sum(1 for rid in remote_by_id if rid in local_ids)
    total = len(remote_by_id) if remote_total is None else remote_total
    plan = MigrationPlan(out_of_scope=max(total - found, 0))

    holders: dict[str, list[LocalUser]] = {}
    for user in local:
        holders.setdefault(_email_key(user.email), []).append(user)
    for group in holders.values():
        if len(group) > 1:
            for user in group:
                others = ", ".join(str(o.id) for o in group if o is not user)
                plan.anomalies.append(
                    Anomaly(
                        "duplicate_email",
                        user.id,
                        f"{user.email} is also the address, ignoring case, of account {others}",
                    )
                )

    # Addresses an earlier update of this plan has taken, so that two accounts cannot both move
    # to the same new one (Supabase keeps them unique, but not necessarily ignoring case).
    claimed: dict[str, UUID] = {}

    for user in local:
        theirs = remote_by_id.get(user.id)
        if theirs is None:
            plan.anomalies.append(
                Anomaly("missing_remote", user.id, f"{user.email} has no account at Supabase")
            )
            continue

        blocked = False
        if theirs.deleted_at is not None:
            plan.anomalies.append(
                Anomaly("deleted", user.id, f"{user.email} is deleted at Supabase")
            )
            blocked = True
        if theirs.banned_until is not None and aware_utc(theirs.banned_until) > now:
            plan.anomalies.append(
                Anomaly(
                    "banned",
                    user.id,
                    f"{user.email} is banned at Supabase until "
                    f"{aware_utc(theirs.banned_until).isoformat()}",
                )
            )
            blocked = True
        if blocked:
            continue

        update_ = UserUpdate(
            user_id=user.id,
            email_verified_at=theirs.email_confirmed_at,
            last_login_at=theirs.last_sign_in_at,
        )

        stored = theirs.encrypted_password
        if not stored or not stored.startswith("$2"):
            plan.anomalies.append(
                Anomaly(
                    "no_password",
                    user.id,
                    f"{user.email} has no bcrypt password at Supabase: it needs a reset link",
                )
            )
        elif _password_changed_since_migration(user):
            update_.keep_password = True
        else:
            update_.password_hash = stored

        wanted_email = (theirs.email or "").strip()
        if wanted_email:
            key = _email_key(wanted_email)
            # Another local account holds it (ignoring case), or an earlier update claims it.
            taken_by = next((o.id for o in holders.get(key, []) if o.id != user.id), None)
            if taken_by is None:
                taken_by = claimed.get(key)
            if taken_by is not None:
                plan.anomalies.append(
                    Anomaly(
                        "email_conflict",
                        user.id,
                        f"the Supabase address {wanted_email} of {user.email} is already "
                        f"used by account {taken_by}; kept {user.email}",
                    )
                )
            else:
                update_.email = key  # stored lowercased and stripped (spec section 7.1)
                claimed[key] = user.id

        role, problem = _role_for(user, theirs)
        update_.role = role
        if problem is not None:
            plan.anomalies.append(Anomaly("role_invalid", user.id, problem))

        plan.updates.append(update_)
    return plan


# ---- reading -----------------------------------------------------------------------------------

# `public.profiles.role` is the role the application reads today; the join is a left one so that
# an account without a profile is still listed.
_REMOTE_QUERY = """
SELECT u.id, u.email, u.encrypted_password, u.email_confirmed_at, u.last_sign_in_at,
       u.banned_until, u.deleted_at, p.role
FROM auth.users u
LEFT JOIN public.profiles p ON p.id = u.id
WHERE u.id = ANY($1::uuid[])
"""
# The other application's accounts are counted, never read (nor their hashes).
_REMOTE_COUNT = "SELECT count(*) FROM auth.users"

_DRIVER_SUFFIX = re.compile(r"^(postgres(?:ql)?)\+\w+://")


def _asyncpg_dsn(dsn: str) -> str:
    """`postgresql+asyncpg://...` (the SQLAlchemy form) as the plain `postgresql://...` asyncpg
    takes; a plain one is returned as it is."""
    return _DRIVER_SUFFIX.sub(r"\1://", dsn)


async def fetch_remote_users(dsn: str, local_ids: Sequence[UUID]) -> RemoteAccounts:
    """The Supabase accounts of `local_ids` (the scope: the project is shared with another
    application, whose accounts and hashes are not fetched), and the number of accounts of the
    whole project, read in one read-only transaction.

    `dsn` is the connection string of the pooler, in either form (`postgresql://` or
    `postgresql+asyncpg://`). It carries a password: it is passed to the driver and nowhere else.
    The accounts are ordered by id so that a report does not depend on the server's row order.
    """
    # No statement cache: a pooler in transaction mode would lose the prepared statements.
    conn = await asyncpg.connect(_asyncpg_dsn(dsn), statement_cache_size=0)
    try:
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            rows = await conn.fetch(_REMOTE_QUERY, list(local_ids))
            total = await conn.fetchval(_REMOTE_COUNT)
    finally:
        await conn.close()

    def _aware(value: Optional[datetime]) -> Optional[datetime]:
        return aware_utc(value) if value is not None else None

    users = [
        RemoteUser(
            id=UUID(str(row["id"])),
            email=row["email"] or "",
            encrypted_password=row["encrypted_password"],
            email_confirmed_at=_aware(row["email_confirmed_at"]),
            last_sign_in_at=_aware(row["last_sign_in_at"]),
            banned_until=_aware(row["banned_until"]),
            deleted_at=_aware(row["deleted_at"]),
            profile_role=row["role"],
        )
        for row in rows
    ]
    return RemoteAccounts(users=sorted(users, key=lambda u: str(u.id)), total=int(total or 0))


async def load_local_users(db: AsyncSession) -> list[LocalUser]:
    rows = await db.execute(
        select(
            User.id, User.email, User.role, User.password_changed_at, User.auth_migrated_at
        ).order_by(User.email, User.id)
    )
    return [LocalUser(*row) for row in rows.all()]


# ---- writing -----------------------------------------------------------------------------------


async def apply_plan(db: AsyncSession, plan: MigrationPlan, now: datetime) -> int:
    """Write the plan in one transaction (committed here, rolled back whole on any error).

    Returns how many accounts were updated: one that no longer exists is not counted. Replaying
    the same plan leaves the same rows (see the module docstring for what is never written).
    Raises `MigrationError`, writing nothing, in `AUTH_MODE=local` (see the module docstring).
    """
    _refuse_after_cutover()
    now = now_utc(now)
    updated = 0
    try:
        for item in plan.updates:
            values: dict = {
                # The day the account first moved: kept once set, and never given to an account
                # that has a password set in the new system (see the module docstring).
                "auth_migrated_at": case(
                    (User.password_changed_at.is_not(None), User.auth_migrated_at),
                    else_=func.coalesce(User.auth_migrated_at, now),
                ),
            }
            if item.email is not None:
                values["email"] = item.email
            if item.email_verified_at is not None:
                values["email_verified_at"] = item.email_verified_at
            if item.last_login_at is not None:
                values["last_login_at"] = item.last_login_at
            if item.role is not None:
                values["role"] = item.role
            if item.password_hash is not None and not item.keep_password:
                # Judged by the database at the time of the write, not only by the plan: a
                # password chosen since is kept. (A SET expression reads the row's old values.)
                values["password_hash"] = case(
                    (
                        or_(
                            User.password_changed_at.is_(None),
                            and_(
                                User.auth_migrated_at.is_not(None),
                                User.password_changed_at <= User.auth_migrated_at,
                            ),
                        ),
                        item.password_hash,
                    ),
                    else_=User.password_hash,
                )
            result = await db.execute(update(User).where(User.id == item.user_id).values(**values))
            updated += result.rowcount or 0
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    return updated


async def _withdraw(db: AsyncSession, raw_token: str) -> None:
    """Spend the link of an e-mail that did not leave, so that it does not count as live."""
    try:
        await consume_token(db, raw_token, OneTimeTokenPurpose.PASSWORD_RESET)
    except InvalidOneTimeToken:
        pass  # already spent or replaced meanwhile: nothing live is left either way
    await db.commit()


async def send_reset_links(db: AsyncSession, failed: Optional[list[UUID]] = None) -> int:
    """E-mail a password-reset link to every migrated account (`auth_migrated_at` set) that is
    ACTIVE or PENDING, has no password, and holds no live reset link (unused, unexpired).

    Returns how many e-mails were sent. One that cannot be sent does not stop the others: the
    account's id is appended to `failed` (when given), its link is withdrawn (spent), and the run
    goes on. So a rerun mails exactly the accounts that did not get their e-mail: those that did
    hold a live link and are skipped (for the hour the link lives; after that an account still
    without a password is mailed again). An account that asked for a link itself is skipped the
    same way. SMTP not being configured at all stops the run (the link just issued is withdrawn
    too), since every other e-mail would fail the same way. Only `AUTH_MODE=local` can send them:
    the link leads to the local reset page, which cannot sign anyone in while the hosted provider
    is the authority.

    Each link is committed before it is sent, so a crash never leaves an e-mail whose link does
    not exist; a crash between the two leaves a live link nobody received, and the account is
    mailed by a run after it expires. A link is never logged.
    """
    if settings.AUTH_MODE != "local":
        raise MigrationError(
            "reset links are sent in AUTH_MODE=local only: until then the hosted provider "
            "authenticates and the link would lead nowhere"
        )
    now = now_utc()
    live_link = (
        select(AuthOneTimeToken.id)
        .where(
            AuthOneTimeToken.user_id == User.id,
            AuthOneTimeToken.purpose == OneTimeTokenPurpose.PASSWORD_RESET,
            AuthOneTimeToken.used_at.is_(None),
            AuthOneTimeToken.expires_at > now,
        )
        .exists()
    )
    rows = await db.execute(
        select(User.id, User.email)
        .where(
            User.password_hash.is_(None),
            User.status.in_((UserStatus.ACTIVE, UserStatus.PENDING)),
            User.auth_migrated_at.is_not(None),
            ~live_link,
        )
        .order_by(User.email, User.id)
    )
    sent = 0
    for user_id, email in rows.all():
        raw_token = await issue_token(db, user_id, OneTimeTokenPurpose.PASSWORD_RESET)
        await record_event(
            db,
            AuthEventType.PASSWORD_RESET_REQUESTED,
            user_id=user_id,
            source="supabase_migration",
        )
        await db.commit()
        try:
            await send_password_reset_email(email, raw_token)
        except EmailNotConfiguredError as exc:
            await _withdraw(db, raw_token)
            raise MigrationError(
                "SMTP is not configured (SMTP_HOST, SMTP_USER, SMTP_PASSWORD, "
                "EMAIL_FROM_ADDRESS): no other e-mail was attempted"
            ) from exc
        except EmailSendError as exc:
            logger.error("Reset e-mail for user %s not sent (%s)", user_id, type(exc).__name__)
            await _withdraw(db, raw_token)
            if failed is not None:
                failed.append(user_id)
        else:
            sent += 1
    return sent


# ---- the command -------------------------------------------------------------------------------


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="python -m app.auth.migrate_from_supabase",
        description=(
            "Copy the credentials of the accounts of `users` from Supabase (read-only). "
            "Without --apply nothing is written."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="write the plan (one transaction)")
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="write the anomalies (with e-mail addresses, never a hash) as JSON; keep it off git",
    )
    parser.add_argument(
        "--send-resets",
        action="store_true",
        help="e-mail a reset link to the accounts without a password (AUTH_MODE=local; a step of "
        "its own, after --apply)",
    )
    return parser


def _summary(plan: MigrationPlan, local_count: int) -> dict:
    counts = {kind: 0 for kind in ANOMALY_KINDS}
    for anomaly in plan.anomalies:
        counts[anomaly.kind] += 1
    return {
        "local_accounts": local_count,
        "updates": len(plan.updates),
        "out_of_scope": plan.out_of_scope,
        "anomalies": counts,
    }


def _write_report(
    path: str, mode: str, now: datetime, plan: MigrationPlan, local_count: int
) -> None:
    """The report: counts and anomalies only. It names e-mail addresses, so only its owner can
    read it (also when the file already existed with wider rights)."""
    document = {
        "generated_at": now.isoformat(),
        "mode": mode,
        "counts": _summary(plan, local_count),
        "anomalies": [
            {"user_id": str(a.user_id), "kind": a.kind, "detail": a.detail} for a in plan.anomalies
        ],
    }
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                json.dump(document, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        finally:
            if fd >= 0:
                os.close(fd)
    except OSError as exc:
        raise MigrationError(f"cannot write the report to {path}: {exc.strerror or exc}") from exc


def _sqlstate(exc: BaseException) -> Optional[str]:
    """The SQLSTATE of a database error, wherever the driver or SQLAlchemy put it."""
    orig = getattr(exc, "orig", None)
    for candidate in (exc, orig, getattr(orig, "__cause__", None), exc.__cause__):
        for attribute in ("sqlstate", "pgcode"):
            code = getattr(candidate, attribute, None)
            if isinstance(code, str) and code:
                return code
    return None


def _describe(exc: BaseException, secrets: tuple[Optional[str], ...]) -> str:
    """One line about a failure: the exception's class, its SQLSTATE if it has one, and the FIRST
    line of its message, with the connection string (and its password) taken out.

    Only the first line: asyncpg appends `DETAIL:` (and `HINT:`) lines to its messages, and a
    DETAIL can quote the row that failed (an address, a hash). A SQLAlchemy error is described by
    its driver's message, never its own: the statement it carries could hold the parameters."""
    if isinstance(exc, SQLAlchemyError):
        text = str(getattr(exc, "orig", None) or "database error")
    else:
        text = str(exc)
    lines = text.strip().splitlines()
    first = lines[0].strip() if lines else ""
    for secret in secrets:
        if secret:
            first = first.replace(secret, "***")
    state = _sqlstate(exc)
    head = type(exc).__name__ + (f" [SQLSTATE {state}]" if state else "")
    return f"{head}: {first}" if first else head


def _secrets_of(dsn: Optional[str]) -> tuple[Optional[str], ...]:
    if not dsn:
        return ()
    password = re.match(r"^[^:]+://[^:@/]*:([^@]*)@", dsn)
    return (dsn, password.group(1) if password else None)


async def _migrate(
    args: argparse.Namespace,
    *,
    session_factory,
    fetch: Fetch,
    out: TextIO,
    now: datetime,
) -> None:
    if args.apply:
        _refuse_after_cutover()  # before anything is read or written, a report included
    dsn = settings.SUPABASE_DB_URL
    if not dsn:
        raise MigrationError(
            "SUPABASE_DB_URL is not set: it is the connection string of the Supabase pooler "
            "(session mode), read-only use only"
        )
    async with session_factory() as db:
        local = await load_local_users(db)
        await db.rollback()  # no transaction is held open while the hosted database is read
        remote = await fetch(dsn, [user.id for user in local])
        plan = plan_migration(local, remote.users, now, remote_total=remote.total)
        mode = "apply" if args.apply else "dry-run"
        if args.report:
            _write_report(args.report, mode, now, plan, len(local))
        applied = await apply_plan(db, plan, now) if args.apply else None

    counts = _summary(plan, len(local))
    print(f"Supabase account migration ({'APPLY' if args.apply else 'dry run'})", file=out)
    print(f"local accounts: {counts['local_accounts']}", file=out)
    print(f"updates: {counts['updates']}", file=out)
    print(f"out of scope: {counts['out_of_scope']}", file=out)
    print("anomalies:", file=out)
    for kind in ANOMALY_KINDS:
        print(f"  {kind}: {counts['anomalies'][kind]}", file=out)
    if applied is None:
        print("dry run: nothing was written (--apply writes it)", file=out)
    else:
        print(f"applied: {applied} accounts updated", file=out)


async def _resets(*, session_factory, out: TextIO, err: TextIO) -> int:
    failed: list[UUID] = []
    async with session_factory() as db:
        sent = await send_reset_links(db, failed=failed)
    print(f"reset links: {sent} sent, {len(failed)} failed", file=out)
    if failed:
        print(
            "error: the e-mail could not be sent for account(s) "
            + ", ".join(str(user_id) for user_id in failed)
            + "; run --send-resets again once SMTP answers: it mails only the accounts whose "
            "e-mail did not leave (an account that got its link is skipped while the link lives)",
            file=err,
        )
        return 1
    return 0


async def run(
    argv: list[str],
    *,
    session_factory,
    fetch: Optional[Fetch] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
    now: Optional[datetime] = None,
) -> int:
    """The command, on the database `session_factory` opens. Returns the exit code.

    `fetch` reads the Supabase accounts of the given local ids (default: `fetch_remote_users`),
    `now` fixes the clock.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    fetch = fetch_remote_users if fetch is None else fetch
    now = now_utc(now)
    try:
        args = _parser().parse_args(argv)
        if args.send_resets and (args.apply or args.report):
            raise _UsageError(
                "--send-resets is a step of its own: run it after --apply, without --apply or "
                "--report"
            )
    except _UsageError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except SystemExit as exc:  # --help
        return exc.code if isinstance(exc.code, int) else 2

    dsn = settings.SUPABASE_DB_URL
    try:
        if args.send_resets:
            return await _resets(session_factory=session_factory, out=out, err=err)
        await _migrate(args, session_factory=session_factory, fetch=fetch, out=out, now=now)
        return 0
    except MigrationError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except Exception as exc:
        print(f"error: {_describe(exc, _secrets_of(dsn))}", file=err)
        return 1


async def _main(argv: list[str]) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.db.session import engine_options

    # An engine of its own, with the app's options but the SQL echo off whatever DEBUG says and
    # the statement parameters hidden whatever SQL_ECHO_PARAMETERS says: the echo goes to the
    # standard output and a failed UPDATE prints its parameters, and here those are hashes.
    engine = create_async_engine(
        settings.DATABASE_URL, echo=False, **{**engine_options(), "hide_parameters": True}
    )
    sessions = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    try:
        return await run(argv, session_factory=sessions)
    finally:
        await engine.dispose()


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
