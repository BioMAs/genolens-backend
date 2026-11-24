"""
Password hashing and policy for the local authentication.

Pure and database-free: nothing here reads a user, a session or the network.

- New passwords are hashed with argon2id, with the default parameters of `argon2.PasswordHasher`.
- Hashes inherited from Supabase (Go bcrypt: `$2a$`, cost 10) still verify, and are handed back as
  an argon2id hash (`PasswordCheck.upgraded_hash`) for the caller to store.
- A password is used as it is received: its UTF-8 bytes, never Unicode-normalised. That is what
  Supabase did, so a migrated account keeps working with the password its owner typed.
- The "too common" list is SecLists `Passwords/Common-Credentials/10k-most-common.txt`
  (https://github.com/danielmiessler/SecLists, MIT licence, Copyright (c) 2018 Daniel Miessler),
  embedded in `data/` so that no external call is made. The licence travels with the copy:
  `data/LICENSE-SecLists.txt`, next to `data/common_passwords.txt`.
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

# The two dummy hashes below let a refusal be padded up to the same work whatever caused it.
# The invariant, kept by `verify_password`: EVERY refusal costs exactly one argon2 verification and
# one bcrypt cost-10 check, each either real (the stored hash's own scheme) or against a dummy
# hash. A refusal therefore takes the same time for an unknown account (no hash), an inherited
# bcrypt hash, an argon2 hash, a malformed or unrecognised hash and a password over 72 bytes, so
# the time of the answer does not tell which of them the e-mail belongs to. A successful login is
# not padded: only the account's owner gets there.
# Each is built from a random password nobody holds; the outcome of a dummy check is never used.
DUMMY_HASH: str = _hasher.hash(secrets.token_urlsafe(24))

# Same cost (10) as the Go bcrypt hashes inherited from Supabase.
DUMMY_BCRYPT_HASH: str = bcrypt.hashpw(
    secrets.token_urlsafe(24).encode("ascii"), bcrypt.gensalt(rounds=10, prefix=b"2a")
).decode("ascii")


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


@dataclass
class _Spent:
    """Which of the two verifications a call has really performed (a hash computed, matching or
    not). A call that is refused up front, such as a malformed hash, did no work and is not
    marked."""

    argon2: bool = False
    bcrypt: bool = False


def _top_up(password: str, spent: _Spent) -> None:
    """Perform the verification(s) a refusal has not paid for yet, against the dummy hashes.

    This is the one place that keeps the invariant stated at `DUMMY_HASH`: whichever path refused
    the password, it ends here, and afterwards exactly one argon2 verification and one bcrypt
    check have been made. The results are discarded.
    """
    if not spent.argon2:
        try:
            _hasher.verify(DUMMY_HASH, password)
        except Exception:
            pass
    if not spent.bcrypt:
        try:
            bcrypt.checkpw(password.encode("utf-8")[:_BCRYPT_MAX_BYTES], DUMMY_BCRYPT_HASH.encode())
        except Exception:
            pass


def _verify(password: str, stored_hash: Optional[str], spent: _Spent) -> PasswordCheck:
    if not stored_hash:
        return _REJECTED

    if stored_hash.startswith("$argon2"):
        try:
            _hasher.verify(stored_hash, password)  # raises on a mismatch or a malformed hash
        except VerifyMismatchError:
            spent.argon2 = True  # the hash was computed, it simply differs
            raise
        spent.argon2 = True
        needs_rehash = _hasher.check_needs_rehash(stored_hash)
        return PasswordCheck(True, hash_password(password) if needs_rehash else None)

    if _BCRYPT_HASH.fullmatch(stored_hash):
        raw = password.encode("utf-8")
        # bcrypt would silently ignore everything past 72 bytes, so "<72 bytes>anything" would
        # match the hash of "<72 bytes>". Such a password is refused, but only after the check has
        # been paid for, so that it does not answer faster than the others.
        matches = bcrypt.checkpw(raw[:_BCRYPT_MAX_BYTES], stored_hash.encode("ascii"))
        spent.bcrypt = True
        if matches and len(raw) <= _BCRYPT_MAX_BYTES:
            return PasswordCheck(True, hash_password(password))
        return _REJECTED

    logger.warning("Unrecognised password hash format; refusing the login")
    return _REJECTED


def verify_password(password: str, stored_hash: Optional[str]) -> PasswordCheck:
    """Check `password` against `stored_hash` (bcrypt or argon2). Never raises.

    - a valid bcrypt hash (`$2a$`, `$2b$`, `$2y$`) that matches: `ok`, with an argon2id
      `upgraded_hash`;
    - an argon2 hash that matches but asks for a rehash: `ok`, with an `upgraded_hash`;
    - no stored hash: not `ok`;
    - a wrong password, a malformed or unrecognised hash, a bcrypt password over 72 bytes or any
      error while verifying: not `ok`.

    Every refusal costs exactly one argon2 verification and one bcrypt cost-10 check, real or
    against a dummy hash (see `DUMMY_HASH`); a success is not padded.
    """
    spent = _Spent()
    try:
        check = _verify(password, stored_hash, spent)
    except VerifyMismatchError:
        check = _REJECTED
    except Exception as exc:  # a verification failure must read as "wrong password", not a 500
        logger.warning("Password verification failed unexpectedly: %s", type(exc).__name__)
        check = _REJECTED
    if not check.ok:
        _top_up(password, spent)
    return check


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


def _strip_trailing_non_letters(text: str) -> str:
    """`text` without its trailing characters that are not letters (digits, punctuation, spaces)."""
    end = len(text)
    while end and not text[end - 1].isalpha():
        end -= 1
    return text[:end]


def validate_new_password(password: str, email: str) -> None:
    """Raise `PasswordPolicyError` if `password` is not acceptable for a new password.

    NIST 800-63B: 12 to 128 characters, not the account's own e-mail, not a common password; no
    composition rules. The checks run in that order, so a password that breaks several rules gets
    the code of the first one. Existing (migrated) passwords are never re-validated.

    "Common" means that the lower-cased, stripped password is in the ~10 000 most common
    passwords, or becomes one once its trailing non-letter characters are removed (`password1234`
    and `Password1234!!` are `password`). Only 10 of the listed passwords are 12 characters or
    longer, so the full-string check alone would let nearly every common password through the
    minimum length. A password that is nothing but non-letters has an empty stem, which is never
    looked up: only its full string is checked.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError("too_short")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError("too_long")

    candidate = password.strip().lower()
    if candidate == email.strip().lower():
        raise PasswordPolicyError("same_as_email")

    common = _common_passwords()
    stem = _strip_trailing_non_letters(candidate)
    if candidate in common or (stem and stem in common):
        raise PasswordPolicyError("too_common")
