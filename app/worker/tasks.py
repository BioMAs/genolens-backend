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
from sqlalchemy.ext.asyncio import AsyncSession

from app.worker.celery_app import celery_app
from app.db.session import AsyncSessionLocal
from app.models.models import Dataset, DatasetStatus, DegGene, EnrichmentPathway, Project
from app.services.storage import storage_service
from app.services.data_processor import data_processor
from app.services.email_service import send_dataset_ready_email, send_dataset_failed_email
from app.core.supabase_auth import lookup_user_by_id
from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session support."""
    pass


def run_async(coro):
    """Helper to run async functions in Celery tasks."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


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
        async with AsyncSessionLocal() as db:
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

                # Merge PCA results, plot results, DEG stats, metadata and warnings
                final_metadata = {**metadata, **pca_results, **plot_results}

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

                # Send job-completion email (fire-and-forget — never blocks the task)
                try:
                    proj_result = await db.execute(
                        select(Project).where(Project.id == dataset.project_id)
                    )
                    project = proj_result.scalar_one_or_none()
                    if project:
                        owner_info = await lookup_user_by_id(project.owner_id)
                        if owner_info and owner_info.get("email"):
                            await send_dataset_ready_email(
                                to_email=owner_info["email"],
                                dataset_name=dataset.name,
                                dataset_type=dataset.type.value,
                                project_id=str(dataset.project_id),
                                project_name=project.name,
                            )
                except Exception as email_exc:
                    logger.warning(
                        "[WORKER] Could not send dataset-ready email for %s: %s",
                        dataset_id, email_exc,
                    )

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

            # Send failure email (fire-and-forget — never blocks the task)
            try:
                failed_ds_result = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id)
                )
                failed_ds = failed_ds_result.scalar_one_or_none()
                if failed_ds:
                    proj_result = await db.execute(
                        select(Project).where(Project.id == failed_ds.project_id)
                    )
                    project = proj_result.scalar_one_or_none()
                    if project:
                        owner_info = await lookup_user_by_id(project.owner_id)
                        if owner_info and owner_info.get("email"):
                            await send_dataset_failed_email(
                                to_email=owner_info["email"],
                                dataset_name=failed_ds.name,
                                error_message=error_message,
                                project_id=str(failed_ds.project_id),
                                project_name=project.name,
                            )
            except Exception as email_exc:
                logger.warning(
                    "[WORKER] Could not send dataset-failed email for %s: %s",
                    dataset_id, email_exc,
                )

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


# ---------------------------------------------------------------------------
# Self-service DESeq2 analysis task
# Runs in the dedicated r-worker container (queue: r_analysis).
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="app.worker.tasks.run_self_service_analysis",
    soft_time_limit=3300,
    time_limit=3600,
)
def run_self_service_analysis(self, analysis_id: str) -> dict:
    """
    Execute a self-service DESeq2 analysis pipeline.

    Updates SelfServiceAnalysis status in the database at each step and
    stores result_dataset_ids on successful completion.

    Args:
        analysis_id: UUID string of the SelfServiceAnalysis record.

    Returns:
        dict: {status: "success"|"failed", result_dataset_ids: [...], error: ...}
    """
    from app.models.models import SelfServiceAnalysis, SelfServiceAnalysisStatus
    from app.services.analysis_service import analysis_service

    async def _run():
        analysis_uuid = UUID(analysis_id)
        async with AsyncSessionLocal() as db:
            # Mark as RUNNING
            await db.execute(
                update(SelfServiceAnalysis)
                .where(SelfServiceAnalysis.id == analysis_uuid)
                .values(
                    status=SelfServiceAnalysisStatus.RUNNING,
                    celery_task_id=self.request.id,
                    current_step="starting",
                )
            )
            await db.commit()

            try:
                result_ids = await analysis_service.run(analysis_uuid, db)

                await db.execute(
                    update(SelfServiceAnalysis)
                    .where(SelfServiceAnalysis.id == analysis_uuid)
                    .values(
                        status=SelfServiceAnalysisStatus.DONE,
                        current_step="done",
                        result_dataset_ids=[str(r) for r in result_ids],
                    )
                )
                await db.commit()
                logger.info(f"[r-worker] Analysis {analysis_id} completed: {result_ids}")
                return {
                    "status": "success",
                    "analysis_id": analysis_id,
                    "result_dataset_ids": [str(r) for r in result_ids],
                }

            except Exception as exc:
                import traceback
                error_msg = str(exc) or traceback.format_exc()[:1000]
                logger.error(f"[r-worker] Analysis {analysis_id} FAILED: {error_msg}")
                logger.error(traceback.format_exc())

                try:
                    await db.execute(
                        update(SelfServiceAnalysis)
                        .where(SelfServiceAnalysis.id == analysis_uuid)
                        .values(
                            status=SelfServiceAnalysisStatus.FAILED,
                            current_step="failed",
                            error_message=error_msg,
                        )
                    )
                    await db.commit()
                except Exception:
                    pass  # best-effort status update

                return {
                    "status": "failed",
                    "analysis_id": analysis_id,
                    "error": error_msg,
                }

    return run_async(_run())
