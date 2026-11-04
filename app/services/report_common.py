"""Helpers shared by the comparison report renderers (LaTeX and OmicSkin editorial)."""

from __future__ import annotations

from typing import Any, Optional


def merge_cover_info(
    branding: Optional[dict[str, Any]], cover_info: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Cover / sign-off fields: the user's saved defaults, then the job's non-empty values."""
    merged = dict((branding or {}).get("cover_info") or {})
    if cover_info:
        merged.update({k: v for k, v in cover_info.items() if v})
    return merged
