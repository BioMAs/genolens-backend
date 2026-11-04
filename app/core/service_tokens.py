"""
Service tokens: the credential of a machine client (the analysis pipeline) instead of a user JWT.

A token is `glst_` followed by 43 base64url characters (`secrets.token_urlsafe(32)`). The prefix
tells it from a JWT before anything is parsed: a service token never reaches a JWT decoder. Only
its sha256 is stored; the token itself is shown once, when it is created
(`python -m app.scripts.service_token create`).

A token acts as its owner — rights, licence, plan, modules, quotas — on the routes its scopes
open, and nowhere else (see `app.api.deps.service_token`).
"""

import hashlib
import re
import secrets
from typing import Optional

TOKEN_PREFIX = "glst_"
_TOKEN_SHAPE = re.compile(r"glst_[A-Za-z0-9_-]{43}")

SCOPE_PROJECTS_CREATE = "projects:create"
SCOPE_DATASETS_UPLOAD = "datasets:upload"
SCOPE_REPORTS_EDITORIAL = "reports:editorial"
ALL_SCOPES = (SCOPE_PROJECTS_CREATE, SCOPE_DATASETS_UPLOAD, SCOPE_REPORTS_EDITORIAL)


def is_service_token(token: Optional[str]) -> bool:
    """True for a bearer value that claims to be a service token (whatever its validity)."""
    return bool(token) and token.startswith(TOKEN_PREFIX)


def is_well_formed(token: str) -> bool:
    return _TOKEN_SHAPE.fullmatch(token) is not None


def new_service_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_service_token(token: str) -> str:
    """The only form in which a token is stored or compared."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def bearer_value(authorization: Optional[str]) -> Optional[str]:
    """The token of an `Authorization: Bearer …` header, else None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization.removeprefix("Bearer ").strip()
