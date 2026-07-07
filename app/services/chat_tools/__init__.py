"""
Tool catalog for the agentic chat mode.

Each tool maps a natural-language capability the LLM can request onto an existing
GenoLens analysis. Tools never receive dataset / comparison IDs from the model —
those are injected server-side via ToolContext from the validated chat session.
"""
from app.services.chat_tools.base import BaseTool, ToolContext, ToolResult
from app.services.chat_tools.registry import ToolRegistry

__all__ = ["BaseTool", "ToolContext", "ToolResult", "ToolRegistry"]
