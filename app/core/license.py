"""
Application license validation.
Validates HMAC-signed license keys issued by the Scilicium license service.
"""
import hashlib
import hmac
import json
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from typing import Optional

_CACHE_TTL = 300  # seconds
_cache: Optional[tuple] = None  # (LicenseStatus, cached_at)


@dataclass
class LicenseStatus:
    valid: bool
    expires_at: Optional[int] = None
    client_id: Optional[str] = None
    plan: Optional[str] = None
    error: Optional[str] = None


def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(s: str) -> bytes:
    s += "=" * (4 - len(s) % 4)
    return urlsafe_b64decode(s)


def issue_license_key(client_id: str, plan: str, expires_at: int, product: str, secret: str) -> str:
    """Generate a signed license key for the given parameters."""
    import json as _json
    payload = {"client_id": client_id, "plan": plan, "expires_at": expires_at, "product": product}
    payload_b64 = _b64encode(_json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64encode(hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest())
    return f"{payload_b64}.{sig}"


def validate_license_key(key: str, product: str, secret: str) -> LicenseStatus:
    try:
        payload_b64, sig = key.rsplit(".", 1)
    except ValueError:
        return LicenseStatus(valid=False, error="malformed key")

    expected_sig = _b64encode(
        hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(sig, expected_sig):
        return LicenseStatus(valid=False, error="invalid signature")

    try:
        payload = json.loads(_b64decode(payload_b64))
        client_id: str = payload["client_id"]
        plan: str = payload["plan"]
        expires_at: int = int(payload["expires_at"])
        key_product: str = payload["product"]
    except Exception:
        return LicenseStatus(valid=False, error="malformed key")

    if key_product != product:
        return LicenseStatus(valid=False, error="product mismatch")

    if time.time() > expires_at:
        return LicenseStatus(
            valid=False,
            expires_at=expires_at,
            client_id=client_id,
            plan=plan,
            error="expired",
        )

    return LicenseStatus(valid=True, expires_at=expires_at, client_id=client_id, plan=plan)


def get_license_status() -> LicenseStatus:
    """Return cached license status (refreshed every 5 minutes)."""
    global _cache
    now = time.time()
    if _cache is not None and now - _cache[1] < _CACHE_TTL:
        return _cache[0]

    from app.core.config import settings

    if not settings.GENOLENS_LICENSE_KEY or not settings.LICENSE_SECRET_KEY:
        status = LicenseStatus(valid=False, error="no license configured")
    else:
        status = validate_license_key(
            settings.GENOLENS_LICENSE_KEY,
            settings.LICENSE_PRODUCT,
            settings.LICENSE_SECRET_KEY,
        )

    _cache = (status, now)
    return status
