"""
Agentic chat-mode endpoints.

- Session CRUD (bound to an explicitly selected project / dataset / comparison).
- SSE agentic endpoint that streams narrative tokens + figure events, driven by the
  ChatAgent tool-calling loop.

The SSE generator uses its OWN database session (not the request-scoped one, which is
closed once the response starts streaming). Auth / quota checks run on the request
session via dependencies before streaming begins.
"""
import json
import logging
from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.license import require_active_license
from app.api.deps.subscription import check_ai_quota, increment_ai_usage, require_ai_access
from app.api.endpoints.datasets import _check_project_read_access
from app.core.supabase_auth import SupabaseUser
from app.db.session import AsyncSessionLocal
from app.models.models import AgentMessage, AgentSession, Dataset, User
from app.schemas.chat import (
    AgentMessageOut,
    AgentSessionCreate,
    AgentSessionDetail,
    AgentSessionOut,
    ChatMessageIn,
)
from app.services.ai_interpreter import LocalAIInterpreter
from app.services.chat_agent import ChatAgent
from app.services.chat_tools.base import ToolContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: dict) -> str:
    """Serialise an agent event as a named SSE frame."""
    etype = event.get("type", "message")
    return f"event: {etype}\ndata: {json.dumps(event, default=str)}\n\n"


async def _load_owned_session(
    session_id: UUID, current_user: SupabaseUser, db: AsyncSession
) -> AgentSession:
    session = await db.get(AgentSession, session_id)
    if not session or session.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    return session


@router.post("/sessions", response_model=AgentSessionOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_active_license)])
async def create_chat_session(
    payload: AgentSessionCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
) -> AgentSessionOut:
    """Create a chat session bound to a selected project / dataset / comparison."""
    await _check_project_read_access(payload.project_id, current_user.user_id, db)

    dataset = await db.get(Dataset, payload.dataset_id)
    if not dataset or dataset.project_id != payload.project_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    session = AgentSession(
        user_id=current_user.user_id,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        comparison_name=payload.comparison_name,
        title=payload.title,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return AgentSessionOut.model_validate(session)


@router.get("/sessions", response_model=List[AgentSessionOut])
async def list_chat_sessions(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> List[AgentSessionOut]:
    """List the current user's chat sessions, most recent first."""
    rows = (await db.execute(
        select(AgentSession)
        .where(AgentSession.user_id == current_user.user_id)
        .order_by(AgentSession.updated_at.desc())
    )).scalars().all()
    return [AgentSessionOut.model_validate(r) for r in rows]


@router.get("/sessions/{session_id}", response_model=AgentSessionDetail)
async def get_chat_session(
    session_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> AgentSessionDetail:
    """Full session with its message history (for reload — figures re-render from payload)."""
    session = await _load_owned_session(session_id, current_user, db)
    messages = (await db.execute(
        select(AgentMessage)
        .where(AgentMessage.session_id == session_id)
        .order_by(AgentMessage.sequence.asc())
    )).scalars().all()
    detail = AgentSessionDetail.model_validate(session)
    detail.messages = [AgentMessageOut.model_validate(m) for m in messages]
    return detail


@router.post("/sessions/{session_id}/message", dependencies=[Depends(require_active_license)])
async def send_chat_message(
    session_id: UUID,
    payload: ChatMessageIn,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
    quota_check: Annotated[User, Depends(check_ai_quota)],
) -> StreamingResponse:
    """
    Send a message to the agent and stream the response (SSE).

    Emits named events: status, tool_call, figure, token, error, done.
    """
    session = await _load_owned_session(session_id, current_user, db)

    # Load prior turns (content only) for conversational memory.
    prior = (await db.execute(
        select(AgentMessage)
        .where(AgentMessage.session_id == session_id)
        .order_by(AgentMessage.sequence.asc())
    )).scalars().all()
    history = [{"role": m.role, "content": m.content or ""} for m in prior]
    next_seq = (await db.scalar(
        select(func.max(AgentMessage.sequence)).where(AgentMessage.session_id == session_id)
    ) or 0)

    # Persist the user's message immediately; set the session title on first turn.
    user_msg = AgentMessage(
        session_id=session_id, role="user", content=payload.message, sequence=next_seq + 1
    )
    db.add(user_msg)
    if not session.title:
        session.title = payload.message[:120]
        db.add(session)
    await db.commit()

    user_id = current_user.user_id
    dataset_id = session.dataset_id
    project_id = session.project_id
    comparison_name = session.comparison_name
    assistant_seq = next_seq + 2

    async def event_stream():
        interpreter = LocalAIInterpreter()
        narrative_parts: List[str] = []
        figures: List[dict] = []
        tool_calls: List[dict] = []
        async with AsyncSessionLocal() as sdb:
            try:
                dataset = await sdb.get(Dataset, dataset_id)
                ctx = ToolContext(
                    db=sdb,
                    dataset=dataset,
                    dataset_id=dataset_id,
                    project_id=project_id,
                    current_user=current_user,
                    comparison_name=comparison_name,
                )
                agent = ChatAgent(ctx, interpreter=interpreter)

                async for event in agent.run(payload.message, history=history):
                    etype = event.get("type")
                    if etype == "token":
                        narrative_parts.append(event.get("text", ""))
                    elif etype == "figure":
                        figures.append({k: event[k] for k in event if k != "type"})
                    elif etype == "tool_call":
                        tool_calls.append({"tool": event.get("tool"), "args": event.get("args")})
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat stream failed: %s", exc)
                yield _sse({"type": "error", "message": "Streaming failed.", "recoverable": False})
                yield _sse({"type": "done", "tool_count": 0, "figure_count": 0})
            finally:
                # Persist the assistant turn + log usage, best-effort even on disconnect.
                try:
                    assistant_msg = AgentMessage(
                        session_id=session_id,
                        role="assistant",
                        content="".join(narrative_parts).strip() or None,
                        tool_calls=tool_calls or None,
                        figures=figures or None,
                        model=interpreter.model,
                        sequence=assistant_seq,
                    )
                    sdb.add(assistant_msg)
                    await sdb.commit()
                    db_user = await sdb.get(User, user_id)
                    if db_user is not None:
                        await increment_ai_usage(
                            user=db_user, db=sdb, action_type="chat_agent",
                            dataset_id=dataset_id, comparison_name=comparison_name,
                            model_used=interpreter.model,
                            tokens_used=interpreter.last_usage["total_tokens"],
                        )
                except Exception as persist_exc:  # noqa: BLE001
                    logger.warning("Failed to persist assistant turn: %s", persist_exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
