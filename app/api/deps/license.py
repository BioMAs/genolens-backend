"""
License enforcement dependency for protected endpoints.
"""
from fastapi import HTTPException, status

from app.core.license import get_license_status


async def require_active_license() -> None:
    """Raise 403 if the application license is expired or not configured."""
    lic = get_license_status()
    if not lic.valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "license_expired",
                "expires_at": lic.expires_at,
                "message": (
                    "Your GenoLens application license has expired or is not configured. "
                    "Please contact support@scilicium.com to renew your license."
                ),
            },
        )
