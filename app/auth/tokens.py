"""
Access tokens of the local authentication, and the opaque tokens it stores.

An access token is a short-lived (15 minutes) ES256 JWT, signed with the first key of
`AUTH_JWT_PRIVATE_KEYS` and verified with the key its `kid` header names. Claims: `iss`
(`AUTH_ISSUER`), `aud` (`genolens`), `sub` (user id), `sid` (session id), `amr` (how the user
proved who they are: `pwd`, `otp`, ...), `iat`, `exp`.

Verification accepts ES256 and nothing else: a token that asks for `none`, for an HMAC (the shape
of a Supabase token, and the classic algorithm-confusion trick) or for another curve is refused
before any key is looked at. A token whose `kid` is not configured is refused too.

Opaque tokens (refresh tokens, one-time links) are random strings of which only the SHA-256 is
stored, so a leaked table gives nothing that can be replayed.
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import UUID

import jwt

from app.auth.clock import now_utc
from app.auth.keys import load_signing_keys
from app.core.config import settings

ACCESS_TOKEN_TTL = timedelta(minutes=15)
AUDIENCE = "genolens"

_ALGORITHM = "ES256"
_REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub", "sid", "amr"]


@dataclass(frozen=True)
class AccessClaims:
    sub: UUID
    sid: UUID
    amr: list[str]
    exp: datetime


class InvalidAccessToken(Exception):
    """The token is not a valid, current access token of this service. Says nothing more on
    purpose: the caller answers 401 whatever the reason."""


def issue_access_token(
    user_id: UUID,
    session_id: UUID,
    amr: list[str],
    now: Optional[datetime] = None,
) -> str:
    """Sign an access token for `user_id`, bound to `session_id`, valid for `ACCESS_TOKEN_TTL`."""
    now = now_utc(now)
    key = load_signing_keys()[0]
    payload = {
        "iss": settings.AUTH_ISSUER,
        "aud": AUDIENCE,
        "sub": str(user_id),
        "sid": str(session_id),
        "amr": list(amr),
        "iat": int(now.timestamp()),
        "exp": int((now + ACCESS_TOKEN_TTL).timestamp()),
    }
    return jwt.encode(payload, key.private_key, algorithm=_ALGORITHM, headers={"kid": key.kid})


def decode_access_token(token: str) -> AccessClaims:
    """Verify `token` and return its claims; raise `InvalidAccessToken` for any defect of the token
    (signature, algorithm, key id, expiry, audience, issuer, missing or malformed claim).

    Not caught on purpose: the `RuntimeError` of a deployment without signing keys (production,
    `AUTH_JWT_PRIVATE_KEYS` empty). That is a configuration fault, not a bad token.
    """
    try:
        header = jwt.get_unverified_header(token)
    except (jwt.PyJWTError, TypeError, ValueError, UnicodeError) as exc:
        raise InvalidAccessToken("unreadable token") from exc
    if header.get("alg") != _ALGORITHM:
        raise InvalidAccessToken("unexpected algorithm")

    kid = header.get("kid")
    key = next((k for k in load_signing_keys() if k.kid == kid), None) if kid else None
    if key is None:
        raise InvalidAccessToken("unknown key id")

    try:
        payload = jwt.decode(
            token,
            key.private_key.public_key(),
            algorithms=[_ALGORITHM],
            audience=AUDIENCE,
            issuer=settings.AUTH_ISSUER,
            options={"require": _REQUIRED_CLAIMS},
        )
        amr = payload["amr"]
        if not isinstance(amr, list) or not all(isinstance(item, str) for item in amr):
            raise InvalidAccessToken("malformed amr claim")
        return AccessClaims(
            sub=UUID(payload["sub"]),
            sid=UUID(payload["sid"]),
            amr=amr,
            exp=datetime.fromtimestamp(payload["exp"], tz=UTC),
        )
    except InvalidAccessToken:
        raise
    except (jwt.PyJWTError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InvalidAccessToken("invalid token") from exc


def is_local_token(token: str) -> bool:
    """Does this token look like one of ours (asymmetric ES256) rather than the hosted provider's
    (HS256)? Reads the header only, without verifying anything: it routes, it never authenticates.
    """
    try:
        return jwt.get_unverified_header(token).get("alg") == _ALGORITHM
    except (jwt.PyJWTError, TypeError, ValueError, UnicodeError):
        return False


def new_opaque_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
