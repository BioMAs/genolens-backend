"""
Reusable pre-ranked GSEA computation.

Shared by the synchronous endpoint (POST /datasets/{id}/gsea) and the async
Celery task (app.worker.tasks.gsea_task). Fetches the comparison's DEG rows,
ranks them, loads the requested gene-set database for the dataset's organism,
runs GSEA, and returns a JSON-serialisable payload with a *lean* results list
(the heavy per-set running-enrichment arrays are dropped — the enrichment-plot
endpoint recomputes them on demand for a single gene set).
"""
import logging
from uuid import UUID

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Dataset, GeneSetDatabase
from app.services.gsea_processor import GSEAProcessor, prepare_ranked_gene_list, GeneSetsLoader
from app.services.gene_set_loader import GeneSetLoader

logger = logging.getLogger(__name__)

# Common-name → scientific-name mapping used by the gene_sets table (organism column).
_GSEA_ORGANISM_MAP = {
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "rat": "Rattus norvegicus",
    "zebrafish": "Danio rerio",
    "medaka": "Oryzias latipes",
}


def resolve_gsea_organism(dataset: Dataset) -> str:
    """Resolve the organism scientific name used to look up gene sets.

    Reads ``dataset_metadata['species']`` (a common name like ``"human"`` or an
    already-scientific name) and maps it to the scientific name stored in the
    ``gene_sets`` table. Defaults to ``"Homo sapiens"``.
    """
    meta = (dataset.dataset_metadata or {}) if dataset else {}
    species = str(meta.get("species") or "").strip()
    if not species:
        return "Homo sapiens"
    return _GSEA_ORGANISM_MAP.get(species.lower(), species)


class GSEANoDegError(Exception):
    """Raised when a comparison has no DEG rows to rank."""


class GSEANoGeneSetsError(Exception):
    """Raised when no gene sets are available for the requested database/scope."""


async def compute_gsea(
    db: AsyncSession,
    dataset: Dataset,
    comparison_name: str,
    gene_set_database: str = "GO_BP",
    ranking_metric: str = "signed_pvalue",
    min_size: int = 15,
    max_size: int = 500,
    n_permutations: int = 1000,
    fdr_threshold: float = 0.25,
) -> dict:
    """Run pre-ranked GSEA for a comparison and return a lean JSON payload.

    Raises:
        GSEANoDegError: if the comparison has no DEG rows.
        ValueError: if ``gene_set_database`` is not a valid database.
    """
    dataset_id = dataset.id

    # Fetch all DEG genes for this comparison
    query = text(
        """
        SELECT gene_id, log_fc, padj, gene_name
        FROM deg_genes
        WHERE dataset_id = :dataset_id
        AND comparison_name = :comparison_name
        ORDER BY padj ASC
        """
    )
    result = await db.execute(
        query, {"dataset_id": str(dataset_id), "comparison_name": comparison_name}
    )
    rows = result.fetchall()
    if not rows:
        raise GSEANoDegError(f"No DEG data found for comparison: {comparison_name}")

    deg_data = pd.DataFrame(rows, columns=["gene_id", "log_fc", "padj", "gene_name"])
    ranked_genes = prepare_ranked_gene_list(deg_data, ranking_metric=ranking_metric)

    database_enum = GeneSetDatabase(gene_set_database)  # raises ValueError if invalid

    if database_enum == GeneSetDatabase.CUSTOM:
        # Custom sets are project-scoped, not organism-scoped.
        from sqlalchemy import select as _select
        from app.models.models import GeneSet

        stmt = _select(GeneSet).where(
            GeneSet.database == GeneSetDatabase.CUSTOM,
            GeneSet.project_id == dataset.project_id,
            GeneSet.size >= min_size,
            GeneSet.size <= max_size,
        )
        gene_sets_db = (await db.execute(stmt)).scalars().all()
        gene_sets = {gs.name: gs.genes for gs in gene_sets_db}
        if not gene_sets:
            raise GSEANoGeneSetsError(
                "No custom gene sets in this project match the size filters "
                f"(min_size={min_size}, max_size={max_size})."
            )
        logger.info("Loaded %d custom gene sets for project %s", len(gene_sets), dataset.project_id)
    else:
        # Built-in databases are organism-scoped.
        gene_set_loader = GeneSetLoader(db)
        organism = resolve_gsea_organism(dataset)
        gene_sets_db = await gene_set_loader.get_gene_sets(
            database=database_enum, organism=organism, min_size=min_size, max_size=max_size
        )
        if not gene_sets_db:
            logger.warning(
                "No gene sets found for %s/%s, using placeholders", gene_set_database, organism
            )
            gene_sets = GeneSetsLoader.get_default_gene_sets()
        else:
            gene_sets = {gs.name: gs.genes for gs in gene_sets_db}
            logger.info("Loaded %d gene sets from database", len(gene_sets))

    gsea_processor = GSEAProcessor(min_size=min_size, max_size=max_size, power=1.0)
    logger.info(
        "Running GSEA with %d genes and %d gene sets", len(ranked_genes), len(gene_sets)
    )
    results = gsea_processor.run_gsea(
        ranked_genes=ranked_genes,
        gene_sets=gene_sets,
        metric_column="metric",
        n_permutations=n_permutations,
    )

    significant_results = [r for r in results if r.fdr_q_value <= fdr_threshold]

    # Lean per-set payload — drop heavy running-enrichment arrays (100+ MB otherwise).
    results_dict = []
    for r in significant_results:
        d = r.to_dict()
        d.pop("running_enrichment_scores", None)
        d.pop("gene_positions", None)
        results_dict.append(d)

    n_enriched_positive = sum(1 for r in significant_results if r.normalized_enrichment_score > 0)
    n_enriched_negative = sum(1 for r in significant_results if r.normalized_enrichment_score < 0)

    return {
        "dataset_id": str(dataset_id),
        "comparison_name": comparison_name,
        "parameters": {
            "gene_set_database": gene_set_database,
            "ranking_metric": ranking_metric,
            "min_size": min_size,
            "max_size": max_size,
            "n_permutations": n_permutations,
            "fdr_threshold": fdr_threshold,
        },
        "summary": {
            "total_genes": len(ranked_genes),
            "total_gene_sets_tested": len(gene_sets),
            "significant_gene_sets": len(significant_results),
            "enriched_in_phenotype_pos": n_enriched_positive,
            "enriched_in_phenotype_neg": n_enriched_negative,
        },
        "results": results_dict,
    }


async def persist_gsea_result(db: AsyncSession, dataset_id: UUID, payload: dict) -> None:
    """Persist a GSEA payload to the shared computation cache (best-effort)."""
    try:
        from app.services.persistent_cache_service import persistent_cache_service

        await persistent_cache_service.set_cached(
            db=db,
            dataset_id=dataset_id,
            computation_type="gsea",
            params={"comparison_name": payload.get("comparison_name"), **payload.get("parameters", {})},
            result_data=payload,
        )
    except Exception as cache_err:  # persistence is best-effort, never fail the run
        logger.warning("Failed to persist GSEA result: %s", cache_err)
