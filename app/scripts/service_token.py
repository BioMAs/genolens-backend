"""
Service tokens: create, revoke and list them (see docs/SERVICE_TOKENS.md).

    python -m app.scripts.service_token create --owner <email|id> --name <name> [--scope S ...]
    python -m app.scripts.service_token revoke <token id>
    python -m app.scripts.service_token list

`create` prints the token once, alone on standard output (the rest goes to standard error), and
stores only its sha256: a lost token cannot be shown again, it is revoked and replaced.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from typing import Iterable, Optional, TextIO
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.service_tokens import ALL_SCOPES, hash_service_token, new_service_token
from app.models.models import User, UserRole, UserStatus
from app.models.service_token import ServiceToken


class CommandError(Exception):
    """A refusal the command explains on standard error (exit code 1)."""


async def _owner(db: AsyncSession, owner: str) -> User:
    try:
        user = await db.get(User, UUID(owner))
    except ValueError:
        user = await db.scalar(select(User).where(User.email == owner))
    if user is None:
        raise CommandError(f"no GenoLens user {owner!r}: the owner must have an account")
    if user.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
        # Least privilege: a token acts as its owner, and an admin's rights go well past the three scopes.
        raise CommandError("an admin account cannot own a service token")
    if user.status != UserStatus.ACTIVE:
        # A token of an inactive account would be refused on every request (403).
        raise CommandError(
            f"the account {owner!r} is {user.status.value}, not active: an administrator must "
            "activate it first (docs/SERVICE_TOKENS.md)"
        )
    return user


async def create_service_token(
    db: AsyncSession, *, owner: str, name: str, scopes: Iterable[str] = ALL_SCOPES
) -> tuple[ServiceToken, str]:
    """A new token for `owner` (an email or a user id). Returns the stored row and the token,
    which exists nowhere else: the row holds its sha256 only."""
    scopes = list(dict.fromkeys(scopes))
    unknown = [s for s in scopes if s not in ALL_SCOPES]
    if unknown or not scopes:
        raise CommandError(f"scopes must be among {', '.join(ALL_SCOPES)} (got {scopes})")
    if not name.strip():
        raise CommandError("a token needs a name (what it is for)")
    user = await _owner(db, owner)
    token = new_service_token()
    row = ServiceToken(
        name=name.strip(), owner_id=user.id, token_sha256=hash_service_token(token), scopes=scopes
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row, token


async def revoke_service_token(db: AsyncSession, token_id: UUID) -> tuple[ServiceToken, bool]:
    """Revokes the token (for good). Returns the row and whether it was active until now."""
    row = await db.get(ServiceToken, token_id)
    if row is None:
        raise CommandError(f"no service token {token_id}")
    if row.revoked_at is not None:
        return row, False
    row.revoked_at = datetime.now(timezone.utc)
    await db.commit()
    return row, True


async def list_service_tokens(db: AsyncSession) -> list[tuple[ServiceToken, str]]:
    rows = await db.execute(
        select(ServiceToken, User.email)
        .join(User, User.id == ServiceToken.owner_id)
        .order_by(ServiceToken.created_at, ServiceToken.name)
    )
    return [(token, email) for token, email in rows.all()]


def _when(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "-"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.scripts.service_token",
        description="Scoped service tokens of the pipeline (docs/SERVICE_TOKENS.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a token and print it, once")
    create.add_argument("--owner", required=True, help="email or id of the owning user")
    create.add_argument("--name", required=True, help="what the token is for")
    create.add_argument(
        "--scope",
        action="append",
        choices=ALL_SCOPES,
        help="a scope to grant (repeatable; default: all of them)",
    )
    revoke = commands.add_parser("revoke", help="revoke a token (by its id)")
    revoke.add_argument("token_id", type=UUID)
    commands.add_parser("list", help="list the tokens (never the tokens themselves)")
    return parser


async def run(
    argv: list[str],
    *,
    session_factory,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    args = _parser().parse_args(argv)
    async with session_factory() as db:
        try:
            if args.command == "create":
                row, token = await create_service_token(
                    db, owner=args.owner, name=args.name, scopes=args.scope or ALL_SCOPES
                )
                print(
                    f"Created service token {row.id} ({row.name!r}, scopes "
                    f"{', '.join(row.scopes)}). The token follows, shown this once only:",
                    file=err,
                )
                print(token, file=out)
            elif args.command == "revoke":
                row, revoked = await revoke_service_token(db, args.token_id)
                state = "revoked" if revoked else "was already revoked"
                print(f"Service token {row.id} ({row.name!r}) {state}.", file=err)
            else:
                for row, email in await list_service_tokens(db):
                    print(
                        f"{row.id}  {row.name}  owner={email}  scopes={','.join(row.scopes)}  "
                        f"created={_when(row.created_at)}  last_used={_when(row.last_used_at)}  "
                        f"revoked={_when(row.revoked_at)}",
                        file=out,
                    )
        except CommandError as exc:
            print(f"error: {exc}", file=err)
            return 1
    return 0


async def _main(argv: list[str]) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.db.session import engine_options

    # An engine of its own, with the app's options but the SQL echo off whatever DEBUG says (SQLAlchemy's echo
    # writes to standard output, where the token must be alone) and the statement parameters hidden whatever
    # SQL_ECHO_PARAMETERS says: that setting is the app's, for development, and a database error of this command
    # would otherwise print its parameters, the sha256 of a token among them, to the terminal.
    engine = create_async_engine(
        settings.DATABASE_URL, echo=False, **{**engine_options(), "hide_parameters": True}
    )
    sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    try:
        return await run(argv, session_factory=sessions)
    finally:
        await engine.dispose()


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
