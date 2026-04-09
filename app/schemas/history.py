"""
Pydantic schemas for Project Activity History.
"""
from typing import Optional, List, Any, Dict
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.models import ActivityEventType


class ActivityLogResponse(BaseModel):
    """Single activity log entry returned by the API."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    user_id: UUID
    event_type: ActivityEventType
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    extra_metadata: Dict[str, Any] = {}
    created_at: datetime


class ActivityLogListResponse(BaseModel):
    """Paginated list of activity log entries."""
    items: List[ActivityLogResponse]
    total: int
    limit: int
    offset: int
