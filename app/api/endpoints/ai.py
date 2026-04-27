"""
AI Chart Assistant endpoints.
Provides interpret, ask, and conversation-history endpoints for any chart type.
All inference runs via local Ollama — no data leaves the server.
"""
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.api.deps import get_current_user, get_db
from app.api.deps.subscription import require_ai_access, check_ai_quota
from app.core.supabase_auth import SupabaseUser
from app.models.models import Dataset, AIInterpretation, AIConversation, User

from app.services.ai_interpreter import LocalAIInterpreter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["ai"])


# ── Pydantic request schemas ─────────────────────────────────────────────────

class InterpretRequest(BaseModel):
    chart_type: str   # "volcano" | "pca" | "umap" | "heatmap" | "enrichment"
    context: dict
    context_key: str  # e.g. comparison name or "pca-default"
    force_regenerate: bool = False


class AskRequest(BaseModel):
    chart_type: str
    context_key: str
    question: str
    context: dict


# ── Shared auth helper ────────────────────────────────────────────────────────

async def _get_dataset_or_404(dataset_id: UUID, current_user: SupabaseUser, db: AsyncSession) -> Dataset:
    result = await db.execute(
        select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.project.owner_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    return dataset


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/{dataset_id}/ai/interpret")
async def interpret_chart(
    dataset_id: UUID,
    body: InterpretRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
    quota_check: Annotated[User, Depends(check_ai_quota)],
) -> dict:
    """
    Generate or return cached plain-English interpretation for a chart.
    Caches result in ai_interpretations so subsequent loads are instant.
    """
    await _get_dataset_or_404(dataset_id, current_user, db)

    # Check cache
    existing = await db.scalar(
        select(AIInterpretation)
        .where(AIInterpretation.dataset_id == dataset_id)
        .where(AIInterpretation.chart_type == body.chart_type)
        .where(AIInterpretation.comparison_name == body.context_key)
    )
    if existing and not body.force_regenerate:
        return {
            "interpretation": existing.interpretation,
            "cached": True,
            "model": existing.model,
            "chart_type": body.chart_type,
        }
    if existing and body.force_regenerate:
        await db.delete(existing)
        await db.flush()

    # Generate
    interpreter = LocalAIInterpreter()
    availability = await interpreter.check_availability()
    if not availability.get("available"):
        raise HTTPException(
            status_code=503,
            detail="AI service (Ollama) is not available. Ensure the container is running."
        )

    try:
        interpretation = await interpreter.interpret_chart(body.chart_type, body.context)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI chart interpretation error: {e}")
        raise HTTPException(status_code=503, detail=f"AI generation failed: {str(e)}")

    # Persist
    record = AIInterpretation(
        dataset_id=dataset_id,
        chart_type=body.chart_type,
        comparison_name=body.context_key,
        interpretation=interpretation,
        model=interpreter.model,
    )
    db.add(record)
    await db.commit()

    return {
        "interpretation": interpretation,
        "cached": False,
        "model": interpreter.model,
        "chart_type": body.chart_type,
    }


@router.post("/{dataset_id}/ai/ask")
async def ask_chart_question(
    dataset_id: UUID,
    body: AskRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
    quota_check: Annotated[User, Depends(check_ai_quota)],
) -> dict:
    """
    Answer a follow-up question about a chart. Persists to AIConversation.
    """
    await _get_dataset_or_404(dataset_id, current_user, db)

    interpreter = LocalAIInterpreter()
    try:
        builders = {
            "volcano": interpreter._build_volcano_prompt,
            "pca": interpreter._build_pca_prompt,
            "umap": interpreter._build_umap_prompt,
            "heatmap": interpreter._build_heatmap_prompt,
            "enrichment": interpreter._build_enrichment_prompt,
        }
        builder = builders.get(body.chart_type)
        if not builder:
            raise HTTPException(status_code=400, detail=f"Unknown chart_type: {body.chart_type!r}")
        chart_summary = builder(body.context)
        prompt = (
            f"{interpreter._CHART_SYSTEM}\n\n"
            f"Chart context:\n{chart_summary}\n\n"
            f"User question: {body.question}\n\n"
            f"Answer in plain language, 2-4 sentences."
        )
        answer = await interpreter._call_ollama_raw(prompt, max_tokens=300)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI ask error: {e}")
        raise HTTPException(status_code=503, detail=f"AI Q&A failed: {str(e)}")

    conv = AIConversation(
        dataset_id=dataset_id,
        chart_type=body.chart_type,
        comparison_name=body.context_key,
        question=body.question,
        answer=answer,
        model=interpreter.model,
        user_id=current_user.user_id,
    )
    db.add(conv)
    await db.commit()

    return {"answer": answer, "chart_type": body.chart_type}


@router.get("/{dataset_id}/ai/conversations")
async def get_chart_conversations(
    dataset_id: UUID,
    chart_type: str = Query(...),
    context_key: str = Query(...),
    db: Annotated[AsyncSession, Depends(get_db)] = None,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)] = None,
) -> dict:
    """
    Load conversation history for a specific chart (ordered oldest-first).
    """
    await _get_dataset_or_404(dataset_id, current_user, db)

    rows = (await db.execute(
        select(AIConversation)
        .where(AIConversation.dataset_id == dataset_id)
        .where(AIConversation.chart_type == chart_type)
        .where(AIConversation.comparison_name == context_key)
        .order_by(AIConversation.created_at.asc())
    )).scalars().all()

    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row.question, "timestamp": row.created_at.isoformat()})
        messages.append({"role": "assistant", "content": row.answer, "timestamp": row.created_at.isoformat()})

    return {"messages": messages}
