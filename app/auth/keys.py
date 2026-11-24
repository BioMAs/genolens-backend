"""
ES256 signing keys of the local authentication.

`AUTH_JWT_PRIVATE_KEYS` holds PKCS#8 PEM private keys on the P-256 curve, each base64-encoded,
separated by commas. The first one signs the access tokens; every one of them verifies, which is
what makes a rotation painless: put the new key first, keep the old one until the tokens it signed
have expired (`ACCESS_TOKEN_TTL`), then drop it.

A key is identified by its `kid`: the first 16 hexadecimal characters of the SHA-256 of its public
key, DER-encoded (SubjectPublicKeyInfo). It is derived, so it never has to be configured and two
deployments holding the same key agree on it.

Outside production an empty setting yields one ephemeral key, generated once per process: handy for
development and tests, but every token it signed dies with the process. In production an empty
setting is an error: a key that changed at each restart would silently sign every user out.
"""

import base64
import binascii
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from app.core.config import settings

logger = logging.getLogger(__name__)

_SETTING = "AUTH_JWT_PRIVATE_KEYS"
_COORDINATE_BYTES = 32  # P-256: x and y are 256-bit integers, published at fixed width


@dataclass(frozen=True)
class SigningKey:
    kid: str
    private_key: ec.EllipticCurvePrivateKey
    public_jwk: dict


_cache: Optional[list[SigningKey]] = None
_lock = threading.Lock()


def _b64url(value: int) -> str:
    raw = value.to_bytes(_COORDINATE_BYTES, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _signing_key(private_key: ec.EllipticCurvePrivateKey) -> SigningKey:
    public_key = private_key.public_key()
    der = public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    kid = hashlib.sha256(der).hexdigest()[:16]
    numbers = public_key.public_numbers()
    public_jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url(numbers.x),
        "y": _b64url(numbers.y),
        "kid": kid,
        "alg": "ES256",
        "use": "sig",
    }
    return SigningKey(kid=kid, private_key=private_key, public_jwk=public_jwk)


def _parse_key(entry: str, position: int) -> ec.EllipticCurvePrivateKey:
    """One configured key. The error names the position, never the value: it ends up in logs."""
    try:
        pem = base64.b64decode(entry, validate=True)
        key = serialization.load_pem_private_key(pem, password=None)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{_SETTING} entry {position} is not a base64-encoded PKCS#8 PEM private key"
        ) from exc
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise RuntimeError(
            f"{_SETTING} entry {position} is not an EC P-256 key (ES256 needs P-256)"
        )
    return key


def _load() -> list[SigningKey]:
    entries = [entry.strip() for entry in settings.AUTH_JWT_PRIVATE_KEYS.split(",")]
    entries = [entry for entry in entries if entry]

    if not entries:
        if settings.is_production:
            raise RuntimeError(f"{_SETTING} is required in production")
        logger.warning(
            "%s is empty: using an ephemeral signing key. Every access token issued by this "
            "process stops verifying when it restarts; set the setting for anything but local "
            "development.",
            _SETTING,
        )
        return [_signing_key(ec.generate_private_key(ec.SECP256R1()))]

    keys: list[SigningKey] = []
    for position, entry in enumerate(entries, start=1):
        key = _signing_key(_parse_key(entry, position))
        if all(key.kid != known.kid for known in keys):  # the same key twice is listed once
            keys.append(key)
    return keys


def load_signing_keys() -> list[SigningKey]:
    """The configured keys, first one signing. Loaded once; `reset_key_cache` forgets them."""
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:  # another thread may have loaded them while this one waited
                _cache = _load()
    return _cache


def reset_key_cache() -> None:
    """Forget the loaded keys, so the next call reads the setting again (tests)."""
    global _cache
    with _lock:
        _cache = None


def jwks() -> dict:
    """The public keys, as a JWK Set. Nothing of the private part is ever in it."""
    return {"keys": [dict(key.public_jwk) for key in load_signing_keys()]}


def generate_private_key_b64() -> str:
    """A new P-256 key, in the format `AUTH_JWT_PRIVATE_KEYS` expects."""
    pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return base64.b64encode(pem).decode("ascii")
