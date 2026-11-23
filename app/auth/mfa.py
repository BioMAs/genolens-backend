"""
Second factor of the local authentication: TOTP, the encryption of its secret at rest, and the
recovery codes.

Pure and database-free: nothing here reads a user, a session or the network. The caller stores
what these functions return (the encrypted secret, the last accepted step, the recovery-code
hashes) and passes it back.

TOTP (RFC 6238), as every authenticator app expects it: HMAC-SHA1, 6 digits, 30-second steps.

- The secret is 32 base32 characters, i.e. 160 bits, drawn with `secrets`.
- `verify_totp` accepts the code of the current step and of the one before and after it, which
  absorbs a clock a few seconds off and a code typed as the step turns over. It returns the step
  it accepted: the caller stores it as `last_used_step` and hands it back next time, and a code
  whose step is not strictly after that one is refused. That is what makes a code single-use, even
  inside its own validity window.

Encryption at rest: `AUTH_ENCRYPTION_KEYS` holds Fernet keys (32 bytes, url-safe base64),
separated by commas. The first one encrypts; every one of them decrypts, which is what makes a
rotation painless: put the new key first, run until every stored secret has been re-encrypted (or
simply until the users have logged in again), then drop the old one. The policy is the one of the
signing keys (`app.auth.keys`): outside production an empty setting yields one ephemeral key,
generated once per process (a secret it encrypted cannot be read after a restart, which is
harmless for a development database and wrong for anything else, hence the warning); in
production an empty setting is an error, because a key that changed at each restart would
silently lock every user with a second factor out of their account.

Recovery codes: `XXXXX-XXXXX`, ten symbols of the Crockford base-32 alphabet (no `I`, `L`, `O` or
`U`, so nothing can be misread when copied from paper), 50 bits of `secrets` randomness. Only
their SHA-256 is stored; a typed code is normalised first, so case, spaces and hyphens do not
matter.
"""

import hashlib
import hmac
import logging
import re
import secrets
import threading
from datetime import UTC, datetime
from typing import Optional

import pyotp
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.core.config import settings

logger = logging.getLogger(__name__)

ISSUER = "GenoLens"

_SETTING = "AUTH_ENCRYPTION_KEYS"

TOTP_DIGITS = 6
TOTP_PERIOD = 30  # seconds per step
_TOTP_WINDOW = (-1, 0, 1)  # steps tried around the current one
_TOTP_CODE = re.compile(r"[0-9]{6}")  # ASCII digits only: `\d` and `str.isdigit` take more

_BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_SECRET_LENGTH = 32  # 32 characters x 5 bits = 160 bits

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RECOVERY_GROUP_LENGTH = 5
_RECOVERY_GROUPS = 2


class SecretDecryptionError(Exception):
    """No configured key opens this token: it was written under a key since dropped from
    `AUTH_ENCRYPTION_KEYS`, or it is corrupted. Never carries the token or a key."""


# --------------------------------------------------------------------------------------------
# TOTP
# --------------------------------------------------------------------------------------------


def new_totp_secret() -> str:
    """A new TOTP seed: 32 base32 characters (160 bits)."""
    return "".join(secrets.choice(_BASE32) for _ in range(_SECRET_LENGTH))


def _totp(secret: str) -> "pyotp.TOTP":
    return pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_PERIOD)


def provisioning_uri(secret: str, email: str) -> str:
    """The `otpauth://` URI an authenticator app reads from a QR code. Issuer: GenoLens."""
    return _totp(secret).provisioning_uri(name=email, issuer_name=ISSUER)


def verify_totp(
    secret: str, code: str, last_used_step: Optional[int], now: Optional[datetime] = None
) -> Optional[int]:
    """
    Check a TOTP code. Returns the step it was generated for, or `None` if it is refused.

    Refused: anything that is not six digits once whitespace is removed (never an exception: it
    is user input), a code that matches none of the steps `t-1`, `t`, `t+1` (`t` being the step
    of `now`), and a code whose step is not strictly after `last_used_step` (replay).

    All three steps are always computed and compared, with `hmac.compare_digest`, whatever
    matches: how long the answer takes says nothing about which step, if any, was right.

    Two steps of the window can, about one time in a million, produce the same code. The highest
    of the matching steps still unused is returned: the caller stores it, and a later replay of
    that code then finds every step it can match at or below `last_used_step`.

    `now` defaults to the current time; a naive `now` is read as UTC. A `secret` that is not
    valid base32 raises: that is corrupted data, not a wrong code.
    """
    if not isinstance(code, str):
        return None
    code = "".join(code.split())  # a code is displayed "123 456", and pasted that way
    if not _TOTP_CODE.fullmatch(code):
        return None

    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    current = int(now.timestamp()) // TOTP_PERIOD

    totp = _totp(secret)
    typed = code.encode("ascii")
    accepted: Optional[int] = None
    for step in (current + offset for offset in _TOTP_WINDOW):  # ascending: last match is highest
        matches = hmac.compare_digest(totp.generate_otp(step).encode("ascii"), typed)
        if matches and (last_used_step is None or step > last_used_step):
            accepted = step
    return accepted


# --------------------------------------------------------------------------------------------
# Encryption of the secret at rest
# --------------------------------------------------------------------------------------------

_cipher: Optional[MultiFernet] = None
_lock = threading.Lock()


def _parse_key(entry: str, position: int) -> Fernet:
    """One configured key. The error names the position, never the value: it ends up in logs."""
    try:
        return Fernet(entry)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{_SETTING} entry {position} is not a Fernet key (32 bytes, url-safe base64)"
        ) from exc


def _load() -> MultiFernet:
    entries = [entry.strip() for entry in settings.AUTH_ENCRYPTION_KEYS.split(",")]
    entries = [entry for entry in entries if entry]

    if not entries:
        if settings.is_production:
            raise RuntimeError(f"{_SETTING} is required in production")
        logger.warning(
            "%s is empty: using an ephemeral encryption key. Every second-factor secret this "
            "process stores becomes unreadable when it restarts; set the setting for anything "
            "but local development.",
            _SETTING,
        )
        return MultiFernet([Fernet(Fernet.generate_key())])

    fernets: list[Fernet] = []
    listed: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        fernet = _parse_key(entry, position)  # every entry is checked, even a repeated one
        if entry not in listed:  # the same key twice is listed once
            listed.add(entry)
            fernets.append(fernet)
    return MultiFernet(fernets)


def load_cipher() -> MultiFernet:
    """The configured keys, first one encrypting. Loaded once; `reset_encryption_cache` forgets."""
    global _cipher
    if _cipher is None:
        with _lock:
            if _cipher is None:  # another thread may have loaded them while this one waited
                _cipher = _load()
    return _cipher


def reset_encryption_cache() -> None:
    """Forget the loaded keys, so the next call reads the setting again (tests)."""
    global _cipher
    with _lock:
        _cipher = None


def generate_encryption_key() -> str:
    """A new Fernet key, in the format `AUTH_ENCRYPTION_KEYS` expects."""
    return Fernet.generate_key().decode("ascii")


def encrypt_secret(secret: str) -> str:
    """Encrypt a secret (a TOTP seed) with the first configured key. The result is ASCII text."""
    return load_cipher().encrypt(secret.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    """
    Decrypt what `encrypt_secret` returned, under any configured key.

    Raises `SecretDecryptionError` when no key opens the token, or when it is not a token at all;
    never returns anything but the exact secret that was encrypted. A missing configuration is a
    `RuntimeError`, not a decryption error: it is not the token's fault.
    """
    cipher = load_cipher()
    try:
        return cipher.decrypt(token).decode("utf-8")
    except (InvalidToken, ValueError, TypeError) as exc:
        # InvalidToken: wrong key, tampered or truncated. ValueError: text that is not even
        # base64 (non-ASCII) or whose plaintext is not UTF-8. TypeError: not a string.
        raise SecretDecryptionError(
            "the secret could not be decrypted: no key of AUTH_ENCRYPTION_KEYS opens it"
        ) from exc


# --------------------------------------------------------------------------------------------
# Recovery codes
# --------------------------------------------------------------------------------------------


def _new_recovery_code() -> str:
    return "-".join(
        "".join(secrets.choice(_CROCKFORD) for _ in range(_RECOVERY_GROUP_LENGTH))
        for _ in range(_RECOVERY_GROUPS)
    )


def generate_recovery_codes(count: int = 10) -> list[str]:
    """`count` distinct recovery codes, `XXXXX-XXXXX`, from the Crockford alphabet."""
    if count < 0:
        raise ValueError("count must not be negative")
    codes: list[str] = []
    seen: set[str] = set()
    while len(codes) < count:  # a repeat is about one in 10**15; "distinct" is still guaranteed
        code = _new_recovery_code()
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def normalize_recovery_code(code: str) -> str:
    """Upper case, with spaces and hyphens removed: how a typed code is compared."""
    return re.sub(r"[\s-]", "", code).upper()


def hash_recovery_code(code: str) -> str:
    """SHA-256 (hex) of the normalised code: the only form in which it is stored."""
    return hashlib.sha256(normalize_recovery_code(code).encode("utf-8")).hexdigest()
