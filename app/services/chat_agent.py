"""
Agentic chat orchestrator for GenoLens "chat mode".

Runs a bounded tool-calling loop against Ollama (llama3.1:8b): the model decides
which analysis tool to call, we execute it in-process, feed a compact summary back,
and finally stream a plain-language narrative. Figures produced by tools are emitted
as structured events so the frontend renders the existing plot components inline.

The orchestrator yields AgentEvent dicts; the SSE endpoint serialises them.
Event types: status | tool_call | figure | token | error | done
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.services.ai_interpreter import LocalAIInterpreter
from app.services.chat_tools.base import ToolContext
from app.services.chat_tools.registry import ToolDispatchError, ToolRegistry

logger = logging.getLogger(__name__)

AgentEvent = Dict[str, Any]


def _event(type_: str, **data: Any) -> AgentEvent:
    return {"type": type_, **data}


class ChatAgent:
    MAX_TOOL_ITERATIONS = 4
    MAX_CONTEXT_MESSAGES = 12  # sliding window of prior turns fed to the model

    def __init__(
        self,
        ctx: ToolContext,
        interpreter: Optional[LocalAIInterpreter] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        self.ctx = ctx
        self.interpreter = interpreter or LocalAIInterpreter()
        self.registry = registry or ToolRegistry(ctx)

    # ── prompt construction ──────────────────────────────────────────────────

    def _system_prompt(self, context_summary: Optional[Dict[str, Any]]) -> str:
        cs = context_summary or {}
        ds_name = cs.get("dataset_name") or "the selected dataset"
        comp = cs.get("comparison_name") or self.ctx.comparison_name or "the selected comparison"
        up = cs.get("deg_up")
        down = cs.get("deg_down")
        comps = cs.get("available_comparisons") or []
        stats_line = ""
        if up is not None and down is not None:
            stats_line = f"It has {up} up-regulated and {down} down-regulated genes.\n"
        comps_line = ""
        if comps:
            comps_line = f"Available comparisons in this dataset: {', '.join(map(str, comps[:20]))}.\n"

        return (
            "You are GenoLens Assistant, an expert RNA-seq / transcriptomics analyst. "
            "You help the user explore ONE selected differential-expression comparison.\n\n"
            f"Current context: dataset '{ds_name}', comparison '{comp}'.\n"
            f"{stats_line}{comps_line}\n"
            "You have tools to fetch data and generate figures for THIS comparison. "
            "When the user asks to see, draw, plot or visualise something, call the matching "
            "tool. When they ask a factual question about genes, counts or pathways, call the "
            "relevant tool to get real numbers first. Do not ask for dataset or comparison IDs "
            "— they are already fixed by the current context.\n"
            "After tools return, answer concisely in clear language for a biologist. Use ONLY "
            "data returned by the tools; never invent gene names, counts, thresholds or pathways."
        )

    def _history_messages(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Reconstruct a windowed Ollama message list from stored turns (content only)."""
        msgs: List[Dict[str, str]] = []
        for turn in history[-self.MAX_CONTEXT_MESSAGES:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        return msgs

    # ── tool-call parsing ────────────────────────────────────────────────────

    def _extract_tool_calls(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalise tool calls to [{"name": str, "arguments": dict}].

        Handles the native Ollama shape and a fallback where an 8B model emits a
        tool call as JSON inside `content` instead of the `tool_calls` field.
        """
        calls: List[Dict[str, Any]] = []
        raw_calls = msg.get("tool_calls") or []
        for tc in raw_calls:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                calls.append({"name": name, "arguments": args or {}})

        if calls:
            return calls

        # Fallback: scan content for a bare JSON object naming a known tool.
        content = msg.get("content") or ""
        if content and "{" in content:
            for match in re.finditer(r"\{[^{}]*\}", content):
                try:
                    obj = json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
                name = obj.get("name") or obj.get("tool")
                if name and self.registry.has(name):
                    args = obj.get("arguments") or obj.get("parameters") or {}
                    calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        return calls

    # ── main loop ────────────────────────────────────────────────────────────

    async def run(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        history = history or []
        tool_count = 0
        figures: List[Dict[str, Any]] = []

        # Fetch cheap context once (drives the system prompt, avoids hallucinated stats).
        context_summary = await self._safe_context_summary()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(context_summary)}
        ]
        messages.extend(self._history_messages(history))
        messages.append({"role": "user", "content": user_message})

        try:
            for _ in range(self.MAX_TOOL_ITERATIONS):
                yield _event("status", phase="thinking")
                msg = await self.interpreter.chat_with_tools(
                    messages, tools=self.registry.schemas()
                )
                tool_calls = self._extract_tool_calls(msg)

                if not tool_calls:
                    # Model is ready to answer — stop the tool loop.
                    break

                # Record the assistant's tool-call turn for the model's context.
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": [
                        {"function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in tool_calls
                    ],
                })

                for call in tool_calls:
                    name, args = call["name"], call["arguments"]
                    yield _event("status", phase="calling_tool")
                    yield _event("tool_call", tool=name, args=args)
                    try:
                        result = await self.registry.dispatch(name, args)
                    except ToolDispatchError as exc:
                        yield _event("error", message=str(exc), recoverable=True)
                        messages.append({"role": "tool", "content": f"ERROR: {exc}"})
                        continue
                    except Exception as exc:  # noqa: BLE001 — tool runtime failure
                        logger.warning("Tool '%s' failed: %s", name, exc)
                        detail = getattr(exc, "detail", None) or str(exc)
                        yield _event(
                            "error", message=f"Tool '{name}' failed: {detail}", recoverable=True
                        )
                        messages.append({"role": "tool", "content": f"ERROR: {detail}"})
                        continue

                    tool_count += 1
                    if result.figure_type and result.figure_payload is not None:
                        # For Plotly figures the payload IS the {data, layout} spec that
                        # the frontend renders directly; carry it as `spec`.
                        figure = {
                            "call_id": f"fig{len(figures) + 1}",
                            "figure_type": result.figure_type,
                            "params": result.params,
                            "spec": result.figure_payload,
                        }
                        figures.append(figure)
                        yield _event("figure", **figure)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result.summary_for_model, default=str),
                    })
            else:
                # Iteration budget exhausted — force a final narrative below.
                logger.info("ChatAgent: max tool iterations reached")

            # Final narrative turn — no tools bound, forces prose.
            yield _event("status", phase="generating")
            messages.append({
                "role": "user",
                "content": (
                    "Now write your answer to my request in clear language, using only the "
                    "tool results above. Do not call any more tools."
                ),
            })
            produced = False
            async for token in self.interpreter.chat_stream(messages):
                produced = True
                yield _event("token", text=token)

            if not produced:
                yield _event("token", text="(no response generated)")

            yield _event("done", tool_count=tool_count, figure_count=len(figures))

        except Exception as exc:  # graceful degradation
            logger.exception("ChatAgent run failed: %s", exc)
            yield _event(
                "error",
                message="The assistant could not complete the request.",
                recoverable=False,
            )
            # Best-effort plain fallback answer.
            try:
                fallback = await self.interpreter.generate_simple_answer(
                    f"Answer this transcriptomics question briefly: {user_message}"
                )
                if fallback:
                    yield _event("token", text=fallback)
            except Exception:
                pass
            yield _event("done", tool_count=tool_count, figure_count=len(figures))

    async def _safe_context_summary(self) -> Optional[Dict[str, Any]]:
        """Best-effort dataset summary for the system prompt (never fatal)."""
        try:
            if self.registry.has("get_dataset_summary"):
                result = await self.registry.dispatch("get_dataset_summary", {})
                return result.summary_for_model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build context summary: %s", exc)
        return None
