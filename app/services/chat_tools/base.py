"""
Base types for chat-mode tools.

A tool declares:
  - name / description   : surfaced to the LLM in the Ollama tool schema
  - params_model         : a Pydantic model validating the model-supplied args
  - figure_type          : the frontend plot component this tool feeds (or None)
  - execute(ctx, params) : runs the underlying analysis and returns a ToolResult

Context (db session, dataset, comparison, user) is injected once via ToolContext,
so tools only ever receive *analysis* parameters from the model — never IDs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ToolContext:
    """Server-side execution context, injected into every tool call."""
    db: AsyncSession
    dataset: Any                      # app.models.models.Dataset
    dataset_id: UUID
    project_id: UUID
    current_user: Any                 # app.core.supabase_auth.SupabaseUser
    comparison_name: Optional[str] = None


@dataclass
class ToolResult:
    """
    Outcome of a tool call.

    - summary_for_model : compact dict fed back to the LLM (counts / top-N / stats).
      Must stay small — never the full plot payload.
    - figure_type       : which frontend component renders this (or None for text).
    - figure_payload    : full JSON plot data streamed straight to the frontend.
    - params            : the resolved analysis params (echoed to the frontend).
    """
    summary_for_model: Dict[str, Any]
    figure_type: Optional[str] = None
    figure_payload: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base for all chat-mode tools."""

    name: str = ""
    description: str = ""
    params_model: Type[BaseModel] = BaseModel
    figure_type: Optional[str] = None

    def schema(self) -> Dict[str, Any]:
        """Return the Ollama /api/chat tool schema for this tool."""
        params_schema = self.params_model.model_json_schema()
        # Ollama/llama expects a plain JSON-schema object for parameters.
        params_schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params_schema,
            },
        }

    @abstractmethod
    async def execute(self, ctx: ToolContext, params: BaseModel) -> ToolResult:
        """Run the underlying analysis. `params` is already validated."""
        raise NotImplementedError
