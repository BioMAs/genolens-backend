"""
Celery tasks for background data processing.
"""
import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID
from celery import Task
from sqlalchemy import select, update, delete, and_, insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.worker.celery_app import celery_app
from app.models.models import Dataset, DatasetStatus, DatasetType, DegGene, EnrichmentPathway, GeneSet, GeneSetDatabase
from app.services.storage import storage_service
from app.services.data_processor import data_processor, DEG_LOGFC_THRESHOLD, DEG_PADJ_THRESHOLD
from app.services.go_service import GOService
from app.core.config import settings

# GeneSetDatabase categories that run_ora_enrichment supports
# (only databases actually loaded in the gene_sets table are worth trying)
_ORA_DATABASES = [
    GeneSetDatabase.KEGG,
    GeneSetDatabase.REACTOME,
    GeneSetDatabase.HALLMARK,
    GeneSetDatabase.C5_ONTOLOGY,
    GeneSetDatabase.C7_IMMUNOLOGIC,
]

# Map GeneSetDatabase → EnrichmentPathway.category string
_DB_TO_CAT: dict[GeneSetDatabase, str] = {
    GeneSetDatabase.KEGG: "KEGG",
    GeneSetDatabase.REACTOME: "REACTOME",
    GeneSetDatabase.HALLMARK: "HALLMARK",
    GeneSetDatabase.C5_ONTOLOGY: "C5_ONTOLOGY",
    GeneSetDatabase.C7_IMMUNOLOGIC: "C7_IMMUNOLOGIC",
}


def _make_worker_session() -> async_sessionmaker:
    """
    Create a fresh AsyncSessionLocal with NullPool for the current event loop.
    NullPool disables connection pooling so each task gets a clean connection
    unbound from any previous event loop — required for Celery workers.
    """
    worker_engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        future=True,
    )
    return async_sessionmaker(
        bind=worker_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

_NS_TO_CAT = {
    "biological_process": "GO:BP",
    "molecular_function": "GO:MF",
    "cellular_component": "GO:CC",
}


async def _auto_run_enrichment(db: "AsyncSession", dataset_id: str) -> None:
    """
    Automatically run enrichment (GO + ORA on KEGG/Reactome/Hallmark/…) for all
    comparisons in a DEG dataset, persisting results to EnrichmentPathway.
    Called after DEG genes are stored.
    """
    go_service = GOService()

    from sqlalchemy import distinct as _distinct
    comp_result = await db.execute(
        select(_distinct(DegGene.comparison_name)).where(DegGene.dataset_id == dataset_id)
    )
    comparisons = [row[0] for row in comp_result.fetchall()]

    if not comparisons:
        logger.info(f"[WORKER] No comparisons for auto enrichment on dataset {dataset_id}")
        return

    bg_result = await db.execute(select(DegGene).where(DegGene.dataset_id == dataset_id))
    all_deg_rows = bg_result.scalars().all()
    logger.info(f"[WORKER] Auto enrichment: {len(comparisons)} comparisons")

    # Pre-check which ORA databases have gene sets loaded
    available_ora_dbs: list[GeneSetDatabase] = []
    for ora_db in _ORA_DATABASES:
        count_result = await db.execute(
            select(GeneSet.id).where(GeneSet.database == ora_db).limit(1)
        )
        if count_result.scalar_one_or_none() is not None:
            available_ora_dbs.append(ora_db)
    logger.info(f"[WORKER] Available ORA databases: {[d.value for d in available_ora_dbs]}")

    for comp_name in comparisons:
        try:
            all_rows = [d for d in all_deg_rows if d.comparison_name == comp_name]
            sig_rows = [
                d for d in all_rows
                if d.padj is not None
                and d.padj <= DEG_PADJ_THRESHOLD
                and abs(d.log_fc or 0) >= DEG_LOGFC_THRESHOLD
            ]

            if not sig_rows:
                logger.info(f"[WORKER] No significant DEGs for '{comp_name}', skipping")
                continue

            for regulation in ("ALL", "UP", "DOWN"):
                if regulation == "UP":
                    subset = [d for d in sig_rows if (d.log_fc or 0) > 0]
                elif regulation == "DOWN":
                    subset = [d for d in sig_rows if (d.log_fc or 0) < 0]
                else:
                    subset = sig_rows

                if not subset:
                    continue

                study_genes = [d.gene_name or d.gene_id for d in subset if (d.gene_name or d.gene_id)]
                if not study_genes:
                    continue

                # ── GO enrichment (BP, MF, CC) ──────────────────────────────
                logger.info(f"[WORKER] GO enrichment '{comp_name}' ({regulation}): {len(study_genes)} genes")
                go_results = await go_service.go_enrichment_analysis(
                    db=db,
                    gene_list=study_genes,
                    background=None,
                    pvalue_threshold=DEG_PADJ_THRESHOLD,
                )

                if go_results:
                    await db.execute(
                        delete(EnrichmentPathway).where(
                            EnrichmentPathway.dataset_id == dataset_id,
                            EnrichmentPathway.comparison_name == comp_name,
                            EnrichmentPathway.regulation == regulation,
                            EnrichmentPathway.category.in_(["GO:BP", "GO:MF", "GO:CC"]),
                        )
                    )
                    await db.commit()

                    go_records = []
                    for r in go_results:
                        cat = _NS_TO_CAT.get(r.get("namespace", ""))
                        if not cat:
                            continue
                        go_records.append({
                            "dataset_id": dataset_id,
                            "comparison_name": comp_name,
                            "pathway_id": r["go_id"],
                            "pathway_name": r.get("term_name", r["go_id"]),
                            "category": cat,
                            "description": r.get("definition"),
                            "gene_count": r["study_count"],
                            "pvalue": r["pvalue"],
                            "padj": r.get("fdr", r["pvalue"]),
                            "gene_ratio": f"{r['study_count']}/{len(study_genes)}",
                            "bg_ratio": f"{r['background_count']}/all_annotated",
                            "genes": r.get("study_genes", []),
                            "regulation": regulation,
                        })

                    chunk_size = 500
                    for i in range(0, len(go_records), chunk_size):
                        await db.execute(insert(EnrichmentPathway), go_records[i:i + chunk_size])
                        await db.commit()
                    logger.info(f"[WORKER] Stored {len(go_records)} GO terms for '{comp_name}' ({regulation})")

                # ── ORA enrichment (KEGG, Reactome, Hallmark, …) ────────────
                for ora_db in available_ora_dbs:
                    cat = _DB_TO_CAT.get(ora_db, ora_db.value)
                    try:
                        ora_results = await go_service.run_ora_enrichment(
                            db=db,
                            gene_list=study_genes,
                            database=ora_db,
                            background=None,
                            padj_threshold=DEG_PADJ_THRESHOLD,
                        )
                    except Exception as ora_err:
                        logger.warning(f"[WORKER] ORA {ora_db.value} failed for '{comp_name}': {ora_err}")
                        continue

                    if not ora_results:
                        continue

                    await db.execute(
                        delete(EnrichmentPathway).where(
                            EnrichmentPathway.dataset_id == dataset_id,
                            EnrichmentPathway.comparison_name == comp_name,
                            EnrichmentPathway.regulation == regulation,
                            EnrichmentPathway.category == cat,
                        )
                    )
                    await db.commit()

                    ora_records = [
                        {
                            "dataset_id": dataset_id,
                            "comparison_name": comp_name,
                            "pathway_id": r["pathway_id"],
                            "pathway_name": r["pathway_name"],
                            "category": cat,
                            "description": r.get("pathway_name"),
                            "gene_count": r["study_count"],
                            "pvalue": r["pvalue"],
                            "padj": r["fdr"],
                            "gene_ratio": r.get("gene_ratio"),
                            "bg_ratio": r.get("bg_ratio"),
                            "genes": r.get("study_genes", []),
                            "regulation": regulation,
                        }
                        for r in ora_results
                    ]

                    chunk_size = 500
                    for i in range(0, len(ora_records), chunk_size):
                        await db.execute(insert(EnrichmentPathway), ora_records[i:i + chunk_size])
                        await db.commit()
                    logger.info(f"[WORKER] Stored {len(ora_records)} {cat} pathways for '{comp_name}' ({regulation})")

        except Exception as e:
            await db.rollback()
            logger.warning(f"[WORKER] Auto enrichment failed for '{comp_name}': {e}")
            # Non-fatal: continue with other comparisons


# Keep backward-compatible alias
_auto_run_go_enrichment = _auto_run_enrichment

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session support."""
    pass


def run_async(coro):
    """Helper to run async functions in Celery tasks."""
    return asyncio.run(coro)


@celery_app.task(bind=True, base=DatabaseTask, name="app.worker.tasks.process_dataset_upload")
def process_dataset_upload(self, dataset_id: str, raw_file_path: str, is_reprocess: bool = False) -> dict:
    """
    Process an uploaded dataset file:
    1. Download raw file from Supabase Storage
    2. Convert to Parquet format
    3. Upload Parquet file to Storage
    4. Update database with new status and metadata

    Args:
        dataset_id: UUID of the dataset
        raw_file_path: Path to raw file in Supabase Storage
        is_reprocess: If True, reuse existing parquet file instead of re-uploading

    Returns:
        dict: Processing result with status and message
    """
    async def _process():
        async with _make_worker_session()() as db:
            try:
                # Update status to PROCESSING
                stmt = update(Dataset).where(Dataset.id == dataset_id).values(status=DatasetStatus.PROCESSING)
                await db.execute(stmt)
                await db.commit()

                # Download raw file
                self.update_state(state="PROGRESS", meta={"step": "downloading"})
                raw_data = await storage_service.download_file(raw_file_path)

                # Get file extension
                file_extension = Path(raw_file_path).suffix

                # Extract metadata
                self.update_state(state="PROGRESS", meta={"step": "analyzing"})
                metadata = await data_processor.get_file_metadata(raw_data, file_extension)

                # Convert to Parquet
                self.update_state(state="PROGRESS", meta={"step": "converting"})
                parquet_data = await data_processor.convert_to_parquet(raw_data, file_extension)

                # Upload Parquet file (skip if reprocessing)
                parquet_path = raw_file_path.replace("/raw/", "/processed/").replace(
                    file_extension, ".parquet"
                )

                if not is_reprocess:
                    self.update_state(state="PROGRESS", meta={"step": "uploading"})
                    await storage_service.upload_file(
                        parquet_path,
                        parquet_data,
                        content_type="application/octet-stream"
                    )

                # Calculate PCA for MATRIX datasets
                result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
                dataset = result.scalar_one()

                pca_results = {}
                if dataset.type == "MATRIX":
                    self.update_state(state="PROGRESS", meta={"step": "calculating_pca"})
                    try:
                        # Calculate PCA with 2 and 3 components
                        pca_2d = await data_processor.calculate_pca(parquet_data, n_components=2)
                        pca_3d = await data_processor.calculate_pca(parquet_data, n_components=3)

                        # Upload PCA results to storage
                        if pca_2d:
                            path = f"projects/{dataset.project_id}/datasets/{dataset_id}/plots/pca_2d.json"
                            await storage_service.upload_file(path, json.dumps(pca_2d).encode('utf-8'), content_type="application/json")
                            pca_results["pca_2d_path"] = path
                        
                        if pca_3d:
                            path = f"projects/{dataset.project_id}/datasets/{dataset_id}/plots/pca_3d.json"
                            await storage_service.upload_file(path, json.dumps(pca_3d).encode('utf-8'), content_type="application/json")
                            pca_results["pca_3d_path"] = path

                    except Exception as e:
                        logger.warning(f"PCA calculation failed: {e}")
                        # Continue without PCA results
                        pass

            # Pre-calculate volcano plots and DEG statistics for DEG datasets
                plot_results = {}
                deg_stats = {}
                if metadata.get("comparisons"):
                    self.update_state(state="PROGRESS", meta={"step": "calculating_volcano_plots"})
                    try:
                        volcano_plots = await data_processor.calculate_volcano_plots(parquet_data, metadata["comparisons"])
                        
                        # Upload volcano plots to storage
                        volcano_path = f"projects/{dataset.project_id}/datasets/{dataset_id}/plots/volcano_plots.json"
                        await storage_service.upload_file(volcano_path, json.dumps(volcano_plots).encode('utf-8'), content_type="application/json")
                        
                        plot_results["volcano_plots_path"] = volcano_path
                    except Exception as e:
                        plot_results["volcano_error"] = str(e)

                    # Calculate DEG statistics
                    self.update_state(state="PROGRESS", meta={"step": "calculating_deg_statistics"})
                    try:
                        deg_statistics = await data_processor.calculate_deg_statistics(parquet_data, metadata["comparisons"])
                        deg_stats = deg_statistics
                    except Exception as e:
                        deg_stats["deg_stats_error"] = str(e)

                    # Extract and store DEG genes in database for fast querying
                    self.update_state(state="PROGRESS", meta={"step": "storing_deg_genes"})
                    try:
                        logger.info(f"[WORKER] Extracting DEG genes for DB...")
                        deg_genes_data = await data_processor.extract_deg_genes_for_db(parquet_data, metadata["comparisons"])
                        logger.info(f"[WORKER] Extracted DEG genes for {len(deg_genes_data)} comparisons")

                        # Delete existing DEG genes for this dataset (in case of reprocessing)
                        try:
                            logger.info(f"[WORKER] Deleting existing DEG genes for dataset {dataset_id}")
                            await db.execute(delete(DegGene).where(DegGene.dataset_id == dataset_id))
                            await db.commit()
                        except Exception as e:
                            logger.warning(f"[WORKER] Error deleting existing DEG genes: {e}")
                            pass  # Table might not exist or no records to delete

                        # Insert new DEG genes
                        for comp_name, genes_list in deg_genes_data.items():
                            logger.info(f"[WORKER] Processing comparison '{comp_name}' with {len(genes_list)} genes")
                            if genes_list:
                                # Prepare bulk insert data
                                deg_records = [
                                    {
                                        "dataset_id": dataset_id,
                                        "comparison_name": comp_name,
                                        "gene_id": gene['gene_id'],
                                        "log_fc": gene['log_fc'],
                                        "padj": gene['padj'],
                                        "regulation": gene.get('regulation'),
                                        "pvalue": gene.get('pvalue'),
                                        "gene_name": gene.get('gene_name')
                                    }
                                    for gene in genes_list
                                ]

                                # Bulk insert via SQLAlchemy in chunks
                                chunk_size = 1000
                                for i in range(0, len(deg_records), chunk_size):
                                    chunk = deg_records[i:i + chunk_size]
                                    await db.execute(insert(DegGene), chunk)
                                    await db.commit()
                                    await asyncio.sleep(0.1)
                            
                                logger.info(f"[WORKER] Finished storing {len(genes_list)} DEG genes for comparison '{comp_name}'")
                            else:
                                logger.info(f"[WORKER] No genes to store for comparison '{comp_name}'")

                    except Exception as e:
                        await db.rollback()
                        logger.error(f"[WORKER] Error storing DEG genes: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Don't fail the whole task if DEG storage fails

                    # Automatically run GO enrichment on the stored DEG genes
                    self.update_state(state="PROGRESS", meta={"step": "computing_go_enrichment"})
                    try:
                        await _auto_run_go_enrichment(db, dataset_id)
                    except Exception as e:
                        logger.warning(f"[WORKER] Auto GO enrichment step failed: {e}")
                        # Non-fatal: dataset still usable without pre-computed enrichment

                # Extract and store enrichment pathways in database for ENRICHMENT datasets
                if metadata.get("enrichment_comparisons"):
                    self.update_state(state="PROGRESS", meta={"step": "storing_enrichment_pathways"})
                    try:
                        enrichment_pathways_data = await data_processor.extract_enrichment_pathways_for_db(
                            parquet_data,
                            {comp: {} for comp in metadata["enrichment_comparisons"]}
                        )

                        # Delete existing enrichment pathways for this dataset
                        try:
                            await db.execute(delete(EnrichmentPathway).where(EnrichmentPathway.dataset_id == dataset_id))
                            await db.commit()
                        except Exception:
                            pass  # Table might not exist or no records to delete

                        # Insert new enrichment pathways
                        for comp_name, pathways_list in enrichment_pathways_data.items():
                            if pathways_list:
                                # Prepare bulk insert data
                                pathway_records = [
                                    {
                                        "dataset_id": dataset_id,
                                        "comparison_name": comp_name,
                                        "pathway_id": pathway['pathway_id'],
                                        "pathway_name": pathway['pathway_name'],
                                        "gene_count": pathway.get('gene_count'),
                                        "pvalue": pathway.get('pvalue'),
                                        "padj": pathway['padj'],
                                        "gene_ratio": pathway.get('gene_ratio'),
                                        "bg_ratio": pathway.get('bg_ratio'),
                                        "genes": pathway.get('genes'),
                                        "category": pathway.get('category'),
                                        "description": pathway.get('description'),
                                        "regulation": pathway.get('regulation', 'ALL')
                                    }
                                    for pathway in pathways_list
                                ]

                                # Bulk insert via SQLAlchemy in chunks
                                chunk_size = 1000
                                for i in range(0, len(pathway_records), chunk_size):
                                    chunk = pathway_records[i:i + chunk_size]
                                    await db.execute(insert(EnrichmentPathway), chunk)
                                    await db.commit()
                                    await asyncio.sleep(0.1)
                            
                                logger.info(f"[WORKER] Stored {len(pathways_list)} enrichment pathways for comparison '{comp_name}'")

                    except Exception as e:
                        await db.rollback()
                        logger.error(f"[WORKER] Error storing enrichment pathways: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Don't fail the whole task if enrichment storage fails

                # Pre-calculate dotplots for ENRICHMENT datasets
                if metadata.get("enrichment_comparisons"):
                    self.update_state(state="PROGRESS", meta={"step": "calculating_dotplots"})
                    try:
                        dotplots = await data_processor.calculate_enrichment_dotplots(parquet_data, metadata["enrichment_comparisons"])
                        
                        # Upload dotplots to storage
                        dotplots_path = f"projects/{dataset.project_id}/datasets/{dataset_id}/plots/enrichment_dotplots.json"
                        await storage_service.upload_file(dotplots_path, json.dumps(dotplots).encode('utf-8'), content_type="application/json")
                        
                        plot_results["enrichment_dotplots_path"] = dotplots_path
                    except Exception as e:
                        plot_results["dotplot_error"] = str(e)

                # Pre-calculate heatmaps for DEG datasets (requires MATRIX dataset)
                if metadata.get("comparisons"):
                    self.update_state(state="PROGRESS", meta={"step": "calculating_heatmaps"})
                    try:
                        # Find MATRIX dataset for this project
                        result = await db.execute(
                            select(Dataset).where(
                                Dataset.project_id == dataset.project_id,
                                Dataset.type == "MATRIX",
                                Dataset.status == DatasetStatus.READY
                            ).limit(1)
                        )
                        matrix_dataset = result.scalar_one_or_none()

                        if matrix_dataset and matrix_dataset.parquet_file_path:
                            matrix_path = matrix_dataset.parquet_file_path
                            logger.info(f"[WORKER] Found matrix dataset {matrix_dataset.id} with path: {matrix_path}")
                        
                            # Fix path if it includes bucket name
                            if matrix_path.startswith(f"{settings.SUPABASE_STORAGE_BUCKET}/"):
                                matrix_path = matrix_path.replace(f"{settings.SUPABASE_STORAGE_BUCKET}/", "", 1)
                                logger.info(f"[WORKER] Corrected matrix path to: {matrix_path}")
                            
                            try:
                                matrix_data = await storage_service.download_file(matrix_path)
                                heatmaps = await data_processor.calculate_deg_heatmaps(parquet_data, matrix_data, metadata["comparisons"])
                                
                                # Upload heatmaps to storage
                                heatmaps_path = f"projects/{dataset.project_id}/datasets/{dataset_id}/plots/heatmaps.json"
                                await storage_service.upload_file(heatmaps_path, json.dumps(heatmaps).encode('utf-8'), content_type="application/json")

                                plot_results["heatmaps_path"] = heatmaps_path
                            except Exception as download_error:
                                logger.error(f"[WORKER] Failed to download matrix file: {str(download_error)}")
                                plot_results["heatmap_error"] = f"Matrix download failed: {str(download_error)}"
                    except Exception as e:
                        logger.error(f"[WORKER] Heatmap calculation failed: {str(e)}")
                        plot_results["heatmap_error"] = str(e)

                # Validate enrichment comparisons against project comparisons
                validation_warnings = []
                if "enrichment_comparisons" in metadata:
                    # Get all datasets for this project to extract comparison names
                    result = await db.execute(select(Dataset).where(Dataset.project_id == dataset.project_id))
                    project_datasets = result.scalars().all()

                    # Extract comparison names from all datasets
                    project_comparisons = set()
                    for ds in project_datasets:
                        # From DEG datasets with comparison_name
                        ds_metadata = ds.dataset_metadata or {}
                        if ds_metadata.get('comparison_name'):
                            project_comparisons.add(ds_metadata['comparison_name'])
                        # From global DEG files with comparisons metadata
                        if ds_metadata.get('comparisons'):
                            project_comparisons.update(ds_metadata['comparisons'].keys())

                    # Check for enrichment comparisons not in project
                    # Remove (up)/(down) suffixes for validation since enrichment files add these
                    unlisted_comparisons = []
                    for comp in metadata["enrichment_comparisons"]:
                        # Strip regulation suffix for comparison matching
                        base_comp = comp
                        if comp.endswith(' (up)') or comp.endswith(' (down)'):
                            base_comp = comp.rsplit(' (', 1)[0]
                        
                        # Check if base comparison exists in project
                        if base_comp not in project_comparisons and comp not in project_comparisons:
                            unlisted_comparisons.append(comp)

                    if unlisted_comparisons:
                        warning_msg = f"Warning: The following comparisons from enrichment file are not listed in the project: {', '.join(unlisted_comparisons)}"
                        validation_warnings.append(warning_msg)
                        logger.warning(warning_msg)  # Log to worker console

                # Merge: start with existing dataset metadata (preserves analysis_id, source,
                # comparison_name, etc. set at creation time), then layer computed metadata on top.
                final_metadata = {**(dataset.dataset_metadata or {}), **metadata, **pca_results, **plot_results}

                # Add DEG statistics to metadata
                # For datasets with comparisons, store statistics per comparison
                if deg_stats and "comparisons" in metadata:
                    # Store in comparisons metadata structure
                    if "comparisons" not in final_metadata:
                        final_metadata["comparisons"] = {}

                    for comp_name, stats in deg_stats.items():
                        if comp_name in final_metadata["comparisons"]:
                            final_metadata["comparisons"][comp_name].update(stats)
                        else:
                            final_metadata["comparisons"][comp_name] = {**metadata["comparisons"][comp_name], **stats}

                if validation_warnings:
                    final_metadata["validation_warnings"] = validation_warnings

                # Update database
                self.update_state(state="PROGRESS", meta={"step": "finalizing"})
            
                stmt = update(Dataset).where(Dataset.id == dataset_id).values(
                    status=DatasetStatus.READY,
                    parquet_file_path=parquet_path,
                    dataset_metadata=final_metadata,
                    error_message=None
                )
                await db.execute(stmt)
                await db.commit()

                return {
                    "status": "success",
                    "dataset_id": dataset_id,
                    "parquet_path": parquet_path,
                    "dataset_metadata": final_metadata
                }

            except Exception as e:
                # Rollback the failed transaction
                await db.rollback()

                # Log full error details
                import traceback
                error_details = traceback.format_exc()
                error_message = str(e) if str(e) else error_details[:500]  # Use traceback if error message is empty

                logger.error(f"[WORKER] Dataset processing failed for {dataset_id}")
                logger.error(f"[WORKER] Error: {error_message}")
                logger.error(f"[WORKER] Full traceback:\n{error_details}")

                # Update status to FAILED
                stmt = update(Dataset).where(Dataset.id == dataset_id).values(
                    status=DatasetStatus.FAILED,
                    error_message=error_message
                )
            await db.execute(stmt)
            await db.commit()

            return {
                "status": "failed",
                "dataset_id": dataset_id,
                "error": error_message
            }

    return run_async(_process())


@celery_app.task(name="app.worker.tasks.health_check")
def health_check() -> dict:
    """
    Simple health check task for monitoring.

    Returns:
        dict: Health status
    """
    return {"status": "healthy", "message": "Celery worker is running"}


@celery_app.task(bind=True, base=DatabaseTask, name="app.worker.tasks.run_self_service_analysis", queue="r_analysis")
def run_self_service_analysis(self, analysis_id: str) -> dict:
    """
    Run a self-service DESeq2/multi-method analysis pipeline (R-based).
    After R completes, registers output files as Dataset records so comparisons appear
    in the project comparisons tab.
    """
    import os
    import shutil
    import subprocess
    import traceback
    from datetime import datetime, timezone
    from uuid import uuid4 as _uuid4
    from app.models.models import SelfServiceAnalysis, SelfServiceAnalysisStatus

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def _run():
        outdir = f"/tmp/analyses/{analysis_id}"
        async with _make_worker_session()() as db:
            try:
                analysis = await db.scalar(
                    select(SelfServiceAnalysis).where(
                        SelfServiceAnalysis.id == UUID(analysis_id)
                    )
                )
                if not analysis:
                    logger.error("[ANALYSIS] Analysis %s not found", analysis_id)
                    return {"status": "error", "error": "not found"}

                def _log(step: str, message: str = ""):
                    entry = {"step": step, "message": message, "timestamp": _now_iso()}
                    analysis.progress_log = (analysis.progress_log or []) + [entry]
                    logger.info("[ANALYSIS] %s: %s", step, message)

                # Mark as running
                analysis.status = SelfServiceAnalysisStatus.RUNNING
                analysis.current_step = "initializing"
                _log("initializing", "Starting analysis pipeline")
                db.add(analysis)
                await db.commit()
                await db.refresh(analysis)

                # Resolve raw TSV file paths on local filesystem
                # (R script reads TSV directly; parquet_file_path is the processed binary)
                async def _get_local_raw_path(dataset_id):
                    if not dataset_id:
                        return None
                    d = await db.scalar(select(Dataset).where(Dataset.id == dataset_id))
                    if not d or not d.raw_file_path:
                        return None
                    return str(Path(settings.LOCAL_STORAGE_PATH) / d.raw_file_path)

                matrix_path = await _get_local_raw_path(analysis.matrix_dataset_id)
                samples_path = await _get_local_raw_path(analysis.samples_dataset_id)
                comparisons_path = await _get_local_raw_path(analysis.comparisons_dataset_id)

                if not matrix_path or not samples_path or not comparisons_path:
                    raise ValueError(
                        "One or more input datasets have no raw file. "
                        "Ensure all datasets are in READY status."
                    )

                # Prepare output directory
                os.makedirs(outdir, exist_ok=True)

                params = analysis.params or {}
                r_script = Path("/app/r_scripts/run_multimethod_pipeline.R")
                if not r_script.exists():
                    r_script = Path("/app/r_scripts/run_deseq_pipeline.R")
                if not r_script.exists():
                    raise FileNotFoundError("R pipeline script not found on this server.")

                _log("running_pipeline", f"Executing {r_script.name}")
                analysis.current_step = "running_pipeline"
                db.add(analysis)
                await db.commit()

                cmd = [
                    "Rscript", str(r_script),
                    "--counts", matrix_path,
                    "--samples", samples_path,
                    "--comparisons", comparisons_path,
                    "--outdir", outdir,
                    "--design", str(params.get("design", "auto")),
                    "--fdr", str(params.get("fdr", 0.05)),
                    "--min-log2fc", str(params.get("min_log2fc", 1.0)),
                    "--min-reads", str(params.get("min_reads", 10)),
                    "--min-genes", str(params.get("min_genes", 200)),
                    "--min-count", str(params.get("min_count", 5)),
                    "--min-reps", str(params.get("min_reps", 2)),
                    "--threads", str(params.get("threads", 4)),
                    "--species", str(params.get("species", "human")),
                ]
                enrichment_dbs = params.get("enrichment_databases")
                if enrichment_dbs:
                    dbs_str = ",".join(enrichment_dbs) if isinstance(enrichment_dbs, list) else enrichment_dbs
                    cmd.extend(["--enrichment-databases", dbs_str])

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"R pipeline exited with code {proc.returncode}.\n"
                        f"stderr: {proc.stderr[-2000:]}"
                    )

                # ── Register output files as Dataset records ─────────────────
                _log("registering_results", "Registering output datasets")
                analysis.current_step = "registering_results"
                db.add(analysis)
                await db.commit()

                manifest = {}
                manifest_path = Path(outdir) / "pipeline_manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)

                result_dataset_ids: list[str] = []
                intermediate_dataset_ids: dict[str, str] = {}

                # VST intermediate dataset (MATRIX type — used for PCA display)
                vst_local = Path(outdir) / "vst_counts.tsv"
                if manifest.get("has_vst") and vst_local.exists():
                    vst_storage = f"projects/{analysis.project_id}/analyses/{analysis_id}/vst_counts.tsv"
                    await storage_service.upload_file(vst_storage, vst_local.read_bytes())
                    vst_ds = Dataset(
                        id=_uuid4(),
                        project_id=analysis.project_id,
                        name=f"{analysis.name} — VST counts",
                        type=DatasetType.MATRIX,
                        status=DatasetStatus.PENDING,
                        raw_file_path=vst_storage,
                        dataset_metadata={"source": "vst", "analysis_id": str(analysis_id)},
                        column_mapping={},
                    )
                    db.add(vst_ds)
                    await db.flush()
                    intermediate_dataset_ids["vst"] = str(vst_ds.id)
                    process_dataset_upload.delay(str(vst_ds.id), vst_storage)
                    logger.info("[ANALYSIS] Registered VST dataset %s", vst_ds.id)

                # DEG result datasets (one per comparison)
                comparisons_dir = Path(outdir) / "comparisons"
                if comparisons_dir.exists():
                    for comp_dir in sorted(comparisons_dir.iterdir()):
                        if not comp_dir.is_dir():
                            continue
                        deg_csv = comp_dir / "genolens_deg.csv"
                        if not deg_csv.exists():
                            continue
                        comp_id = comp_dir.name
                        deg_storage = (
                            f"projects/{analysis.project_id}/analyses/{analysis_id}"
                            f"/comparisons/{comp_id}/genolens_deg.csv"
                        )
                        await storage_service.upload_file(deg_storage, deg_csv.read_bytes())
                        deg_ds = Dataset(
                            id=_uuid4(),
                            project_id=analysis.project_id,
                            name=comp_id,
                            type=DatasetType.DEG,
                            status=DatasetStatus.PENDING,
                            raw_file_path=deg_storage,
                            dataset_metadata={"analysis_id": str(analysis_id), "comparison_name": comp_id},
                            column_mapping={},
                        )
                        db.add(deg_ds)
                        await db.flush()
                        result_dataset_ids.append(str(deg_ds.id))
                        process_dataset_upload.delay(str(deg_ds.id), deg_storage)
                        logger.info("[ANALYSIS] Registered DEG dataset %s for comparison %s", deg_ds.id, comp_id)

                analysis.result_dataset_ids = result_dataset_ids
                analysis.intermediate_dataset_ids = intermediate_dataset_ids
                analysis.status = SelfServiceAnalysisStatus.DONE
                analysis.current_step = "done"
                _log("done", f"Pipeline completed. Registered {len(result_dataset_ids)} DEG dataset(s).")
                db.add(analysis)
                await db.commit()

                shutil.rmtree(outdir, ignore_errors=True)
                return {"status": "done", "analysis_id": analysis_id, "result_dataset_ids": result_dataset_ids}

            except Exception as exc:
                error_details = traceback.format_exc()
                error_msg = str(exc) if str(exc) else error_details[:500]
                logger.error("[ANALYSIS] Failed for %s: %s", analysis_id, error_msg)
                logger.error("[ANALYSIS] Traceback:\n%s", error_details)
                shutil.rmtree(outdir, ignore_errors=True)
                await db.execute(
                    update(SelfServiceAnalysis)
                    .where(SelfServiceAnalysis.id == UUID(analysis_id))
                    .values(
                        status=SelfServiceAnalysisStatus.FAILED,
                        error_message=error_msg,
                        current_step="failed",
                    )
                )
                await db.commit()
                return {"status": "failed", "analysis_id": analysis_id, "error": error_msg}

    return run_async(_run())
