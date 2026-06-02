"""
Symmetric encryption utilities for secrets at rest (SSH keys, deployment env vars).
Uses Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` package.

The encryption key is read from settings.DEPLOYMENT_ENCRYPTION_KEY.
Generate a new key once with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, status


def _get_fernet() -> Fernet:
    from app.core.config import settings
    key = settings.DEPLOYMENT_ENCRYPTION_KEY
    if not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DEPLOYMENT_ENCRYPTION_KEY is not configured on this server",
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    """Encrypt a string and return a URL-safe base64 token."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet token back to plaintext."""
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except (InvalidToken, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decrypt secret: {exc}",
        )
