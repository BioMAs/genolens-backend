"""
Password hashing and policy for the local authentication.

Pure and database-free: nothing here reads a user, a session or the network.

- New passwords are hashed with argon2id, with the default parameters of `argon2.PasswordHasher`.
- Hashes inherited from Supabase (Go bcrypt: `$2a$`, cost 10) still verify, and are handed back as
  an argon2id hash (`PasswordCheck.upgraded_hash`) for the caller to store.
- A password is used as it is received: its UTF-8 bytes, never Unicode-normalised. That is what
  Supabase did, so a migrated account keeps working with the password its owner typed.
- The "too common" list is SecLists `Passwords/Common-Credentials/10k-most-common.txt`
  (https://github.com/danielmiessler/SecLists, MIT licence), embedded in `data/` so that no
  external call is made.
"""

import logging
import re
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128

# bcrypt only reads the first 72 bytes of a password.
_BCRYPT_MAX_BYTES = 72

# A well-formed bcrypt hash: `$2a$`/`$2b$`/`$2y$`, two-digit cost, 22 characters of salt and 31 of
# digest. The shape is checked before bcrypt sees the string because the bcrypt extension panics,
# with an exception that `except Exception` does not catch, on a truncated hash.
_BCRYPT_HASH = re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}")

_COMMON_PASSWORDS_FILE = Path(__file__).parent / "data" / "common_passwords.txt"

_hasher = PasswordHasher()

# Verified against when the account has no password hash (unknown e-mail, invited account that has
# not set a password yet), so that an answer costs the same time whether the account exists or not.
# It is built from a random password nobody holds; the outcome of that check is never used.
DUMMY_HASH: str = _hasher.hash(secrets.token_urlsafe(24))


def hash_password(password: str) -> str:
    """argon2id hash of `password` (salted, default parameters of the library)."""
    return _hasher.hash(password)


@dataclass(frozen=True)
class PasswordCheck:
    """Outcome of `verify_password`.

    `upgraded_hash` is set only when the password is right AND the stored hash should be replaced
    (an inherited bcrypt hash, or an argon2 hash whose parameters are out of date): the caller then
    stores it in place of the old one.
    """

    ok: bool
    upgraded_hash: Optional[str] = None


_REJECTED = PasswordCheck(False, None)


def _spend_dummy_check(password: str) -> None:
    """Burn the time of one verification against `DUMMY_HASH`; the result is discarded."""
    try:
        _hasher.verify(DUMMY_HASH, password)
    except Exception:
        pass


def _verify(password: str, stored_hash: Optional[str]) -> PasswordCheck:
    if not stored_hash:
        _spend_dummy_check(password)
        return _REJECTED

    if stored_hash.startswith("$argon2"):
        _hasher.verify(stored_hash, password)  # raises on a mismatch or a malformed hash
        needs_rehash = _hasher.check_needs_rehash(stored_hash)
        return PasswordCheck(True, hash_password(password) if needs_rehash else None)

    if _BCRYPT_HASH.fullmatch(stored_hash):
        raw = password.encode("utf-8")
        # bcrypt would silently ignore everything past 72 bytes, so "<72 bytes>anything" would
        # match the hash of "<72 bytes>". Such a password is refused, but only after the check has
        # been paid for, so that it does not answer faster than the others.
        matches = bcrypt.checkpw(raw[:_BCRYPT_MAX_BYTES], stored_hash.encode("ascii"))
        if matches and len(raw) <= _BCRYPT_MAX_BYTES:
            return PasswordCheck(True, hash_password(password))
        return _REJECTED

    logger.warning("Unrecognised password hash format; refusing the login")
    _spend_dummy_check(password)
    return _REJECTED


def verify_password(password: str, stored_hash: Optional[str]) -> PasswordCheck:
    """Check `password` against `stored_hash` (bcrypt or argon2). Never raises.

    - a valid bcrypt hash (`$2a$`, `$2b$`, `$2y$`) that matches: `ok`, with an argon2id
      `upgraded_hash`;
    - an argon2 hash that matches but asks for a rehash: `ok`, with an `upgraded_hash`;
    - no stored hash: verified against `DUMMY_HASH`, never `ok`;
    - a wrong password, a malformed hash or any error while verifying: not `ok`.
    """
    try:
        return _verify(password, stored_hash)
    except VerifyMismatchError:
        return _REJECTED
    except Exception as exc:  # a verification failure must read as "wrong password", not a 500
        logger.warning("Password verification failed unexpectedly: %s", type(exc).__name__)
        return _REJECTED


class PasswordPolicyError(ValueError):
    """A new password the policy refuses. `code` is one of `too_short`, `too_long`, `too_common`
    and `same_as_email`."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@lru_cache(maxsize=1)
def _common_passwords() -> "frozenset[str]":
    """The embedded list of common passwords, lower-cased, read once on first use.

    The file ships with the package; if it is missing this raises, so the check fails loudly
    rather than silently letting every password through.
    """
    text = _COMMON_PASSWORDS_FILE.read_text(encoding="utf-8")
    return frozenset(line.strip().lower() for line in text.splitlines() if line.strip())


def validate_new_password(password: str, email: str) -> None:
    """Raise `PasswordPolicyError` if `password` is not acceptable for a new password.

    NIST 800-63B: 12 to 128 characters, not the account's own e-mail, not one of the ~10 000
    most common passwords; no composition rules. The checks run in that order, so a password that
    breaks several rules gets the code of the first one. Existing (migrated) passwords are never
    re-validated.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError("too_short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError("too_long")

    candidate = password.strip().lower()
    if candidate == email.strip().lower():
        raise PasswordPolicyError("same_as_email")
    if candidate in _common_passwords():
        raise PasswordPolicyError("too_common")
