"""
Report customization settings endpoints (per-user branding).

GET    /users/me/report-settings        — fetch persistent branding settings
PUT    /users/me/report-settings        — update branding settings
POST   /users/me/report-settings/logo   — upload a custom logo
GET    /users/me/report-settings/logo   — stream the stored logo (for preview)

All routes are gated behind the report customization module
(User.has_report_customization). The frontend renders a locked teaser when a
403 is returned for users without the module.
"""
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.db import get_db
from app.api.deps.subscription import require_report_customization_access
from app.models.models import User, UserReportSettings
from app.schemas.report import ReportSettingsResponse, ReportSettingsUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users/me/report-settings", tags=["report-settings"])

_ALLOWED_LOGO_EXT = {".png", ".jpg", ".jpeg", ".pdf"}
_LOGO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
}
_MAX_LOGO_BYTES = 5 * 1024 * 1024  # 5 MB


async def _get_or_create_settings(db: AsyncSession, user_id) -> UserReportSettings:
    settings = await db.get(UserReportSettings, user_id)
    if not settings:
        settings = UserReportSettings(user_id=user_id)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


@router.get("", response_model=ReportSettingsResponse)
async def get_report_settings(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_report_customization_access)],
):
    return await _get_or_create_settings(db, user.id)


@router.put("", response_model=ReportSettingsResponse)
async def update_report_settings(
    payload: ReportSettingsUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_report_customization_access)],
):
    settings = await _get_or_create_settings(db, user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)
    await db.commit()
    await db.refresh(settings)
    return settings


@router.post("/logo", response_model=ReportSettingsResponse)
async def upload_report_logo(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_report_customization_access)],
    file: Annotated[UploadFile, File(...)],
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_LOGO_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported logo format. Allowed: {', '.join(sorted(_ALLOWED_LOGO_EXT))}",
        )
    data = await file.read()
    if len(data) > _MAX_LOGO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Logo file too large (max 5 MB).",
        )

    from app.services.storage import storage_service
    logo_path = f"report-branding/{user.id}/logo{ext}"
    await storage_service.upload_file(logo_path, data, file.content_type or "application/octet-stream")

    settings = await _get_or_create_settings(db, user.id)
    settings.logo_path = logo_path
    await db.commit()
    await db.refresh(settings)
    return settings


@router.get("/logo")
async def get_report_logo(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_report_customization_access)],
):
    """Stream the stored logo so the frontend can preview it."""
    settings = await db.get(UserReportSettings, user.id)
    if not settings or not settings.logo_path:
        raise HTTPException(status_code=404, detail="No logo uploaded")

    ext = os.path.splitext(settings.logo_path)[1].lower()
    from app.services.storage import storage_service
    data = await storage_service.download_file(settings.logo_path)
    return Response(content=data, media_type=_LOGO_MIME.get(ext, "application/octet-stream"))
