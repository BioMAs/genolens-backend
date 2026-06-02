"""
License status endpoint — public, no authentication required.
"""
from fastapi import APIRouter

from app.core.license import get_license_status

router = APIRouter(prefix="/license", tags=["license"])


@router.get("/status")
def license_status():
    """Return the current application license status."""
    lic = get_license_status()
    return {
        "valid": lic.valid,
        "expires_at": lic.expires_at,
        "client_id": lic.client_id,
        "plan": lic.plan,
        "error": lic.error,
    }
