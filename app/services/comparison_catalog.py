"""
Comparison catalog: turns raw Dataset rows into the ComparisonSummary list the
API exposes.

This is the single source of truth for how a comparison is derived from a
dataset — which datasets count, where the DEG numbers come from, and how
enrichment is attributed. Several other places in the backend count DEGs their
own way; do not add a variant here, fix the caller instead.
"""
from typing import Iterable

from app.schemas.project import ComparisonSummary


def build_comparisons_from_datasets(datasets: Iterable) -> list[ComparisonSummary]:
    """
    Build the list of ComparisonSummary from a list of Dataset objects.

    Accepts anything with the Dataset attributes (ORM instances or SQLAlchemy
    Rows selecting the same columns), so callers can select narrow column sets.
    """
    # Keyed by (project_id, dataset_id, comparison name): a comparison name is
    # only unique within one dataset. Keying by name alone made two datasets —
    # and, once results span several projects, two projects — collapse onto a
    # single entry.
    comparisons_dict: dict[tuple, ComparisonSummary] = {}

    for d in datasets:
        metadata = d.dataset_metadata or {}

        # Skip datasets that never finished processing. A FAILED/PROCESSING DEG
        # dataset carries no counts (deg_up_count NULL, no `comparisons` metadata);
        # because entries are keyed by comparison name and the loop is ordered
        # newest-first, a stale failed dataset from an earlier run is processed
        # last and would overwrite a good comparison's counts with zeros.
        if d.status != "READY":
            continue

        # Single file per comparison (old way)
        if d.type == "DEG":
            # Prefer explicit comparison_name; fall back to first item in list; then dataset name
            comp_name = metadata.get('comparison_name') if metadata else None
            if not comp_name and isinstance(metadata.get('comparisons'), list) and metadata['comparisons']:
                comp_name = metadata['comparisons'][0]
            if not comp_name:
                comp_name = d.name
            # Use pre-calculated DB counts as primary source, metadata as fallback
            deg_up = d.deg_up_count or (metadata.get('deg_up', 0) if metadata else 0)
            deg_down = d.deg_down_count or (metadata.get('deg_down', 0) if metadata else 0)
            deg_total = d.deg_significant_count or (metadata.get('deg_total', deg_up + deg_down) if metadata else deg_up + deg_down)
            comparisons_dict[(d.project_id, d.id, comp_name)] = ComparisonSummary(
                name=comp_name,
                deg_up=deg_up,
                deg_down=deg_down,
                deg_total=deg_total,
                has_enrichment=False,
                dataset_id=d.id,
                dataset_type='SINGLE'
            )

        # Global DEG file (new way)
        if metadata and 'comparisons' in metadata and isinstance(metadata['comparisons'], dict):
            for comp_name, comp_info in metadata['comparisons'].items():
                deg_up = comp_info.get('deg_up', 0)
                deg_down = comp_info.get('deg_down', 0)
                deg_total = comp_info.get('deg_total', deg_up + deg_down)
                comparisons_dict[(d.project_id, d.id, comp_name)] = ComparisonSummary(
                    name=comp_name,
                    deg_up=deg_up,
                    deg_down=deg_down,
                    deg_total=deg_total,
                    has_enrichment=False,
                    dataset_id=d.id,
                    dataset_type='GLOBAL'
                )

    # Mark comparisons with enrichment (check both ENRICHMENT datasets and DEG datasets).
    # An ENRICHMENT dataset names the comparisons it covers but is a distinct row, so
    # the flag is attributed per project, not per dataset.
    for d in datasets:
        metadata = d.dataset_metadata or {}
        if metadata:
            enrichment_comparisons = metadata.get('enrichment_comparisons', [])
            for comp_name in enrichment_comparisons:
                for key, summary in comparisons_dict.items():
                    if key[0] == d.project_id and key[2] == comp_name:
                        summary.has_enrichment = True

    return list(comparisons_dict.values())
