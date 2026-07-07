"""
Tool registry: holds the instantiated tool set, exposes OpenAI-compatible tool schemas, and
dispatches validated tool calls.

Validation happens here (Pydantic per tool). On invalid args we DO NOT raise — we
return a structured error so the orchestrator can feed it back to the model and let
it self-correct within its iteration budget.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.services.chat_tools.base import BaseTool, ToolContext, ToolResult
from app.services.chat_tools.tools import build_default_tools

logger = logging.getLogger(__name__)


class ToolDispatchError(Exception):
    """Raised for an unknown tool name (recoverable — reported back to the model)."""


class ToolRegistry:
    def __init__(self, ctx: ToolContext, tools: Optional[List[BaseTool]] = None):
        self.ctx = ctx
        self._tools: Dict[str, BaseTool] = {t.name: t for t in (tools or build_default_tools())}

    def schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-compatible tool schemas for every registered tool."""
        return [t.schema() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def dispatch(self, name: str, raw_args: Dict[str, Any]) -> ToolResult:
        """
        Validate `raw_args` against the tool's schema and execute it.

        Raises ToolDispatchError for an unknown tool or invalid arguments — the
        orchestrator catches this and reports it to the model for self-correction.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolDispatchError(
                f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}"
            )

        # Models frequently pass explicit nulls for unspecified params — drop them so
        # the schema defaults apply instead of failing validation.
        clean_args = {k: v for k, v in (raw_args or {}).items() if v is not None}
        try:
            params = tool.params_model.model_validate(clean_args)
        except ValidationError as exc:
            # Compact, model-readable error message
            issues = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arg'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ToolDispatchError(f"Invalid arguments for '{name}': {issues}") from exc

        return await tool.execute(self.ctx, params)
