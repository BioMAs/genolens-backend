"""Helpers shared by the comparison report renderers (LaTeX and OmicSkin editorial) and by the
AI-text endpoints: the label every AI text carries, the fingerprint that ties a cached text to the
data it was written from, and the record of the texts a report printed."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional


def merge_cover_info(
    branding: Optional[dict[str, Any]], cover_info: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Cover / sign-off fields: the user's saved defaults, then the job's non-empty values."""
    merged = dict((branding or {}).get("cover_info") or {})
    if cover_info:
        merged.update({k: v for k, v in cover_info.items() if v})
    return merged


# ── AI texts ────────────────────────────────────────────────────────────────


class AIText(NamedTuple):
    """An AI text as a renderer receives it.

    `text` is None when there is nothing to print; `stale` says it is because a stored text was
    written from other data than today's (it is then neither printed nor passed off as current).
    """

    text: Optional[str] = None
    model: Optional[str] = None
    generated_at: Optional[datetime] = None
    stale: bool = False


def _utc(moment: datetime) -> datetime:
    """The moment in UTC; a naive one is taken to be UTC already."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def ai_label(model: str, generated_at: Optional[datetime]) -> str:
    """The label of every AI text, in the panel and in both reports.

    "AI-generated text (model <model>, generated <YYYY-MM-DD>)", with no date part when the moment
    of generation is unknown. The date is the one of the generation, in UTC: the model server has
    no dated snapshot to name.
    """
    named = model or "unknown"
    if generated_at is None:
        return f"AI-generated text (model {named})"
    return f"AI-generated text (model {named}, generated {_utc(generated_at):%Y-%m-%d})"


def ai_text_record(kind: str, model: Optional[str], generated_at: Optional[datetime]) -> dict:
    """One entry of `report_jobs.ai_texts`: its kind ("nova" or "comparison"), model and date."""
    return {
        "kind": kind,
        "model": model or "unknown",
        "generated_at": _utc(generated_at).isoformat() if generated_at is not None else None,
    }


def _canonical(value: Any) -> Any:
    """The payload with the one list whose order is not fixed by the payload sorted: the terms of
    a sub-group tie on (p, label) and then follow the order the database returned them in."""
    if isinstance(value, dict):
        out = {k: _canonical(v) for k, v in value.items()}
        terms = out.get("terms")
        if isinstance(terms, list) and all(isinstance(t, dict) for t in terms):
            out["terms"] = sorted(terms, key=lambda t: (str(t.get("id")), str(t.get("obs"))))
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def nova_inputs_sha256(nova: dict[str, Any], model: str) -> str:
    """The fingerprint of what an AI reading of the nova is written from.

    The sha256 of the canonical JSON of the nova's `claims` and `unlinked` sub-groups, its `alpha`,
    the referential version, the enrichment datasets the terms come from, and the model. It leaves
    out the clock (`provenance.generated_at`) and everything else that is not an input of the text,
    so two novas that read the same give the same fingerprint, and the text of one is not served
    for the other.
    """
    provenance = nova.get("provenance") or {}
    datasets = provenance.get("enrichment_dataset_ids")
    payload = {
        "claims": nova.get("claims"),
        "unlinked": nova.get("unlinked"),
        "alpha": nova.get("alpha"),
        "referential_version": provenance.get("referential_version"),
        "enrichment_dataset_ids": sorted(str(d) for d in datasets) if datasets else datasets,
        "model": model,
    }
    blob = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def comparison_text_is_current(ai: Any, deg_up: int, deg_down: int, pathways_count: int) -> bool:
    """Whether a stored comparison interpretation was written from today's counts.

    It records the numbers of up- and down-regulated genes and of pathways it was written from
    (`AIInterpretation.deg_up`, `deg_down`, `pathways_count`); other counts mean other data.
    """
    return (ai.deg_up, ai.deg_down, ai.pathways_count) == (deg_up, deg_down, pathways_count)
