"""
Pydantic schemas for the agentic chat mode.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AgentSessionCreate(BaseModel):
    """Create a chat session bound to an explicitly selected context."""
    project_id: UUID
    dataset_id: UUID
    comparison_name: Optional[str] = Field(
        None, description="Selected DEG comparison (optional but recommended)."
    )
    title: Optional[str] = None


class AgentMessageOut(BaseModel):
    id: UUID
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    figures: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    sequence: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentSessionOut(BaseModel):
    id: UUID
    project_id: UUID
    dataset_id: UUID
    comparison_name: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentSessionDetail(AgentSessionOut):
    messages: List[AgentMessageOut] = Field(default_factory=list)


class ChatMessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
