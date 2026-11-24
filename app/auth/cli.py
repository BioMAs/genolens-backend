"""
The operator's command of the local authentication: key generation and the break-glass actions.

    python -m app.auth.cli generate-keys
    python -m app.auth.cli reset-mfa --email <address>
    python -m app.auth.cli unlock --email <address>

Run it in the API container (`docker compose exec api python -m app.auth.cli ...`).

`generate-keys` prints the two settings the server needs, on standard output, and nothing else:

    AUTH_JWT_PRIVATE_KEYS=<base64 of a PKCS#8 PEM, P-256>
    AUTH_ENCRYPTION_KEYS=<Fernet key>

Two lines, in that order, ready to be pasted into the secret store. Nothing is written to disk,
logged or sent anywhere, and no database is opened: the keys exist in the terminal that ran the
command and nowhere else. To rotate, put the new key FIRST in the setting and keep the old one
after it (a comma-separated list): the first signs and encrypts, all of them verify and decrypt.

`reset-mfa` is the way back in when no administrator can sign in: it does what
`POST /admin/users/{id}/mfa/reset` does (`app.auth.admin_actions.reset_user_mfa`, the same code,
not a copy of it) and records `BREAK_GLASS` instead of `MFA_RESET_BY_ADMIN`, with no actor, since
no administrator is behind it. The reset is committed, then the account is told by e-mail; that
e-mail is best effort, as on the endpoint: if it cannot leave, a warning goes to standard error
and the command still succeeds, because the reset stands. `unlock` clears the brute-force lock of
the account (`app.auth.admin_actions.unlock_user`: the wrong-password count and the lock date, not
the second-factor count, which only a completed sign-in resets) and sends nothing.

Both look the account up on the local `users` table, ignoring case and surrounding spaces, and
work whatever `AUTH_MODE` says: break-glass has to work during the transition too. An address no
account has, or that several accounts share (neither can be told from the other), is refused
with exit code 2 and nothing is changed.

Exit codes: 0 done; 2 usage, or the address is unknown or shared. Any other failure (a database
that does not answer, say) ends in the traceback and Python's exit code 1: nothing was committed.

Secrets. Besides `generate-keys` (whose output is the secret), the command prints one line of
confirmation, with the address it was given and counts: no secret, hash, token or key is ever
written by it, in a message or in an error.
"""

import argparse
import asyncio
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Optional, TextIO

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.admin_actions import reset_user_mfa, unlock_user
from app.auth.emails import send_mfa_reset_email
from app.auth.keys import generate_private_key_b64
from app.auth.mfa import generate_encryption_key
from app.models.auth import AuthEventType
from app.models.models import User
from app.services.email_service import EmailNotConfiguredError, EmailSendError

SessionFactory = Callable[[], AsyncSession]


class CommandError(Exception):
    """A refusal the command explains on standard error (exit code 2)."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.auth.cli",
        description="Operator's command of the local authentication.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "generate-keys",
        help="print AUTH_JWT_PRIVATE_KEYS and AUTH_ENCRYPTION_KEYS (new keys; nothing is stored)",
    )
    reset = commands.add_parser(
        "reset-mfa",
        help="break-glass: remove the account's second factor and end its sessions",
    )
    reset.add_argument("--email", required=True, help="address of the account")
    unlock = commands.add_parser("unlock", help="clear the brute-force lock of the account")
    unlock.add_argument("--email", required=True, help="address of the account")
    return parser


def _generate_keys(out: TextIO) -> None:
    print(f"AUTH_JWT_PRIVATE_KEYS={generate_private_key_b64()}", file=out)
    print(f"AUTH_ENCRYPTION_KEYS={generate_encryption_key()}", file=out)


async def _account(db: AsyncSession, email: str) -> User:
    """The one account of `email` (ignoring case and surrounding spaces), or a refusal."""
    normalized = email.strip().lower()
    rows = (
        (await db.execute(select(User).where(func.lower(User.email) == normalized).limit(2)))
        .scalars()
        .all()
    )
    if not rows:
        raise CommandError(f"no account has the address {email.strip()!r}")
    if len(rows) > 1:
        raise CommandError(
            f"several accounts share the address {email.strip()!r}: none was changed "
            "(they differ by case only; merge or rename them first)"
        )
    return rows[0]


async def _reset_mfa(db: AsyncSession, email: str, *, out: TextIO, err: TextIO) -> None:
    user = await _account(db, email)
    to = user.email
    revoked = await reset_user_mfa(
        db, user, actor_id=None, event=AuthEventType.BREAK_GLASS, request=None
    )
    await db.commit()
    print(
        f"MFA reset for {to}: {revoked} session(s) ended, break-glass recorded in the audit trail.",
        file=out,
    )
    # The reset is done and committed: nothing that happens to the e-mail may undo it, or
    # make the command look as if it had failed.
    try:
        await send_mfa_reset_email(to)
    except (EmailSendError, EmailNotConfiguredError) as exc:
        reason = type(exc).__name__
    except Exception as exc:  # whatever it is, the reset stands
        reason = type(exc).__name__
    else:
        return
    print(
        f"warning: the e-mail to the account was not sent ({reason}); the reset itself is done. "
        "Tell the account holder yourself.",
        file=err,
    )


async def _unlock(db: AsyncSession, email: str, *, out: TextIO) -> None:
    user = await _account(db, email)
    to = user.email
    await unlock_user(db, user, actor_id=None, request=None)
    await db.commit()
    print(f"Account {to} unlocked.", file=out)


async def run(
    argv: list[str],
    *,
    session_factory: Optional[SessionFactory] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
) -> int:
    """The command, on the database `session_factory` opens (default: the application's,
    `AsyncSessionLocal`). Returns the exit code."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    try:
        # argparse writes its usage and its errors to the process streams: bring them here.
        with redirect_stdout(out), redirect_stderr(err):
            args = _parser().parse_args(argv)
    except SystemExit as exc:  # --help (0) or a usage error (2)
        return exc.code if isinstance(exc.code, int) else 2

    if args.command == "generate-keys":
        _generate_keys(out)
        return 0

    dispose = None
    if session_factory is None:
        from app.db.session import AsyncSessionLocal, engine

        session_factory = AsyncSessionLocal
        dispose = engine.dispose
    try:
        async with session_factory() as db:
            if args.command == "reset-mfa":
                await _reset_mfa(db, args.email, out=out, err=err)
            else:
                await _unlock(db, args.email, out=out)
    except CommandError as exc:
        print(f"error: {exc}", file=err)
        return 2
    finally:
        # The pool belongs to the event loop this command is about to close.
        if dispose is not None:
            await dispose()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(run(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
