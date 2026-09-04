"""
Which dataset holds the enrichment for a comparison.

A DEG dataset does not hold its own enrichment. On a self-service analysis the R pipeline emits a
separate ENRICHMENT dataset carrying the annoDB results (GO + KEGG/Reactome/Hallmark/…), while the
legacy Python path writes `EnrichmentPathway` rows under the DEG dataset's own id. So "the
enrichment for this comparison" is a lookup, not a given.

The UI does that lookup (`useComparisonContext.ts`, then `enrichmentDataset?.id ?? dataset.id`).
Two server-side AI surfaces did not: `interpret_comparison` and the chat's pathways tool both read
`EnrichmentPathway` under the id they were called with — the DEG dataset on a comparison page.
Since the worker deliberately skips legacy enrichment for self-service datasets, those rows do not
exist there, so the panels showed annoDB pathways while the AI and the chat saw **none at all**.

This module is the one resolver they share, deliberately mirroring the client's rules so the two
cannot drift apart again.
"""

import logging
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Dataset, DatasetStatus, DatasetType

logger = logging.getLogger(__name__)


def _metadata(dataset: Dataset) -> dict[str, Any]:
    return dataset.dataset_metadata or {}


def _names_match(dataset: Dataset, comparison_name: str) -> bool:
    """Enrichment named after the comparison, either in metadata or as the dataset name."""
    meta = _metadata(dataset)
    return meta.get("comparison_name") == comparison_name or dataset.name == comparison_name


def _lists_comparison(dataset: Dataset, comparison_name: str) -> bool:
    """Multi-comparison enrichment file that declares this comparison among its columns."""
    listed = _metadata(dataset).get("enrichment_comparisons")
    return isinstance(listed, (list, tuple)) and comparison_name in listed


def _prefer_ready(candidates: Sequence[Dataset]) -> Optional[Dataset]:
    """A READY dataset first, so a failed or half-written duplicate is never picked."""
    if not candidates:
        return None
    for dataset in candidates:
        if dataset.status == DatasetStatus.READY:
            return dataset
    return candidates[0]


def match_enrichment_dataset(
    candidates: Sequence[Dataset],
    comparison_name: str,
) -> Optional[Dataset]:
    """
    Pick the enrichment dataset for a comparison out of an already-scoped list.

    Pure, so callers that have their own scoping — the report builders work from an analysis's
    result datasets, not from a project query — share the matching rules without giving it up.

    A direct name match wins over a multi-comparison enrichment file that merely lists the
    comparison, matching the client's `byName || byComparisons` precedence.
    """
    by_name = [d for d in candidates if _names_match(d, comparison_name)]
    by_comparisons = [d for d in candidates if _lists_comparison(d, comparison_name)]
    return _prefer_ready(by_name or by_comparisons)


def deg_comparison_names(deg_dataset: Dataset) -> list[str]:
    """
    The comparison names a DEG dataset declares, most specific first.

    A per-comparison dataset names one in `comparison_name`; a multi-comparison file lists them
    under `comparisons`, as a dict keyed by name or as a plain list. Lets a caller with no
    comparison in hand — the project report iterates datasets, not comparisons — still match on
    comparison rather than on dataset names.
    """
    meta = _metadata(deg_dataset)
    names: list[str] = []

    named = meta.get("comparison_name")
    if isinstance(named, str) and named:
        names.append(named)

    comparisons = meta.get("comparisons")
    if isinstance(comparisons, dict):
        names.extend(str(k) for k in comparisons)
    elif isinstance(comparisons, (list, tuple)):
        names.extend(str(c) for c in comparisons)

    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def match_enrichment_for_deg(
    candidates: Sequence[Dataset],
    deg_dataset: Dataset,
) -> Optional[Dataset]:
    """
    The enrichment dataset paired with a DEG dataset, for callers with no comparison in hand.

    Tries every comparison the DEG dataset declares, then falls back to pairing by dataset name.

    That fallback is the project report's original and *only* rule
    (``deg.name in enr.name or enr.name in deg.name``). It is kept last because it is fragile —
    a substring match pairs "X" with "XY" just as happily — and because it matched **0 of 34**
    READY DEG datasets on the development database, where the comparison rules matched the one
    pair that existed. Last resort, not first choice.
    """
    for comparison_name in deg_comparison_names(deg_dataset):
        found = match_enrichment_dataset(candidates, comparison_name)
        if found is not None:
            return found

    name = deg_dataset.name or ""
    paired = [
        d for d in candidates
        if d.name and (name in d.name or d.name in name)
    ]
    return _prefer_ready(paired)


async def find_enrichment_dataset(
    db: AsyncSession,
    deg_dataset: Dataset,
    comparison_name: str,
) -> Optional[Dataset]:
    """
    The ENRICHMENT dataset for this comparison, or ``None`` if the project has none.

    Scoped to the DEG dataset's project and, when it belongs to an analysis, to that analysis —
    otherwise an enrichment file from a *different* analysis that happens to share a comparison
    name bleeds in. The client hit exactly that bug and fixed it the same way
    (`useComparisonContext.ts`: "mélange entre analyses").
    """
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == deg_dataset.project_id,
            Dataset.type == DatasetType.ENRICHMENT,
        )
    )
    candidates = list(result.scalars().all())
    if not candidates:
        return None

    analysis_id = _metadata(deg_dataset).get("analysis_id")
    if analysis_id:
        same_analysis = [
            d for d in candidates if _metadata(d).get("analysis_id") == analysis_id
        ]
        if same_analysis:
            candidates = same_analysis

    return match_enrichment_dataset(candidates, comparison_name)


async def resolve_pathway_dataset_id(
    db: AsyncSession,
    deg_dataset: Dataset,
    comparison_name: str,
) -> UUID:
    """
    The id to read ``EnrichmentPathway`` rows under, for this comparison.

    The ENRICHMENT dataset when one exists, else the DEG dataset itself — where the legacy Python
    enrichment writes its rows, and the only enrichment a plain DEG upload ever has. Same
    `?? dataset.id` fallback the enrichment panel applies, so the AI reads what the user sees.
    """
    enrichment = await find_enrichment_dataset(db, deg_dataset, comparison_name)
    if enrichment is None:
        return deg_dataset.id

    logger.debug(
        "Pathways for %s/%s resolved to ENRICHMENT dataset %s",
        deg_dataset.id, comparison_name, enrichment.id,
    )
    return enrichment.id
