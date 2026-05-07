"""
Self-service DE analysis service.

Downloads source datasets, runs the DESeq2 R pipeline via subprocess,
uploads the resulting DEG CSV files as new Dataset records.
"""
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    DegGene,
    EnrichmentPathway,
    GOAnnotation,
    Project,
    SelfServiceAnalysis,
    SelfServiceAnalysisStatus,
)
from app.services.go_service import go_service
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

# Path to the R pipeline script (in the r-worker container or local dev)
R_SCRIPTS_PATH = os.environ.get("R_SCRIPTS_PATH", "/app/r_scripts")
R_PIPELINE_SCRIPT = os.path.join(R_SCRIPTS_PATH, "run_multimethod_pipeline.R")

# Path to anno.db directory (mounted in r-worker container)
ANNO_DB_PATH = os.environ.get("ANNO_DB_PATH", "/app/anno_db")

# Timeout for the R subprocess (seconds)
R_TIMEOUT = int(os.environ.get("R_PIPELINE_TIMEOUT", 3600))


class AnalysisService:
    """Orchestrates the self-service DESeq2 analysis pipeline."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, analysis_id: UUID, db: AsyncSession) -> list[UUID]:
        """
        Execute the full analysis pipeline for a SelfServiceAnalysis job.

        1. Load analysis record from DB
        2. Download source datasets (matrix, samples, comparisons)
        3. Run R pipeline script
        4. Upload DEG CSV results as new Dataset records
        5. Upload VST + normalized intermediate datasets
        6. Return list of created Dataset UUIDs

        Raises on any fatal error (caller should update status to FAILED).
        """
        analysis = await self._load_analysis(analysis_id, db)

        # Species is now stored per-analysis in params (fallback to "human" for older analyses)
        species = (analysis.params or {}).get("species", "human") or "human"

        workdir = Path(tempfile.mkdtemp(prefix=f"analysis_{analysis_id}_"))
        try:
            await self._set_step(analysis_id, "downloading", db)

            await self._download_datasets(
                workdir,
                analysis.matrix_dataset_id,
                analysis.samples_dataset_id,
                analysis.comparisons_dataset_id,
                db,
            )

            await self._set_step(analysis_id, "running_deseq2", db)

            progress_file = str(workdir / "progress.txt")
            self._run_r_pipeline(workdir, analysis.params, progress_file, species)

            await self._set_step(analysis_id, "uploading_results", db)

            result_pairs = await self._upload_results(
                workdir,
                analysis.project_id,
                analysis.name,
                analysis_id,
                db,
            )
            result_ids = [uid for uid, _ in result_pairs]

            await self._set_step(analysis_id, "computing_go_enrichment", db)
            await self._compute_go_enrichment(result_pairs, species, db)

            intermediate_ids = await self._upload_intermediate_datasets(
                workdir,
                analysis.project_id,
                analysis.name,
                analysis_id,
                db,
            )

            # Persist intermediate dataset IDs on the analysis record
            if intermediate_ids:
                await db.execute(
                    update(SelfServiceAnalysis)
                    .where(SelfServiceAnalysis.id == analysis_id)
                    .values(intermediate_dataset_ids=intermediate_ids)
                )
                await db.commit()

            return result_ids

        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_analysis(self, analysis_id: UUID, db: AsyncSession) -> SelfServiceAnalysis:
        result = await db.execute(
            select(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()
        if not analysis:
            raise ValueError(f"SelfServiceAnalysis {analysis_id} not found")
        return analysis

    async def _set_step(self, analysis_id: UUID, step: str, db: AsyncSession) -> None:
        """Update current_step and append to progress_log."""
        now = datetime.now(timezone.utc).isoformat()
        result = await db.execute(
            select(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id)
        )
        analysis = result.scalar_one()
        log = list(analysis.progress_log or [])
        log.append({"step": step, "timestamp": now})
        await db.execute(
            update(SelfServiceAnalysis)
            .where(SelfServiceAnalysis.id == analysis_id)
            .values(current_step=step, progress_log=log)
        )
        await db.commit()
    async def _set_step(self, analysis_id: UUID, step: str, db: AsyncSession) -> None:
        """Update current_step and append to progress_log."""
        now = datetime.now(timezone.utc).isoformat()
        result = await db.execute(
            select(SelfServiceAnalysis).where(SelfServiceAnalysis.id == analysis_id)
        )
        analysis = result.scalar_one()
        log = list(analysis.progress_log or [])
        log.append({"step": step, "timestamp": now})
        await db.execute(
            update(SelfServiceAnalysis)
            .where(SelfServiceAnalysis.id == analysis_id)
            .values(current_step=step, progress_log=log)
        )
        await db.commit()
        logger.info(f"[Analysis {analysis_id}] step={step}")

    @staticmethod
    def _go_params_hash(
        namespace: Optional[str],
        regulation: str,
        padj_threshold: float,
        log_fc_threshold: float,
        min_term_size: int,
        max_term_size: int,
        pvalue_threshold: float,
        fdr_method: str = "fdr_bh",
        propagate_annotations: bool = True,
        organism: str = "Homo sapiens",
    ) -> str:
        """Reproduce the same hash logic as _compute_go_params_hash in datasets.py."""
        key = json.dumps([
            namespace, regulation, padj_threshold, log_fc_threshold,
            min_term_size, max_term_size, pvalue_threshold,
            fdr_method, propagate_annotations, organism,
        ])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def _compute_go_enrichment(
        self,
        result_pairs: list[tuple[UUID, str]],
        species: str,
        db: AsyncSession,
    ) -> None:
        """
        Automatically compute GO enrichment for each DEG dataset produced by the pipeline.

        For every (dataset_id, comparison_name) pair and every regulation direction
        (ALL, UP, DOWN), runs GO enrichment and saves results to EnrichmentPathway.

        This is non-fatal: any error is logged as a warning and execution continues.
        """
        _species_map = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "rat": "Rattus norvegicus",
        }
        organism = _species_map.get(species.lower(), "Homo sapiens")

        _ns_map = {
            "biological_process": "GO:BP",
            "molecular_function": "GO:MF",
            "cellular_component": "GO:CC",
        }

        # Default enrichment parameters
        PADJ_THRESHOLD = 0.05
        LOGFC_THRESHOLD = 0.5
        MIN_TERM_SIZE = 5
        MAX_TERM_SIZE = 500
        PVALUE_THRESHOLD = 0.05

        try:
            # Fetch background genes once (all GO-annotated genes for this organism)
            bg_result = await db.execute(
                select(GOAnnotation.gene_symbol).distinct().where(
                    GOAnnotation.organism == organism
                )
            )
            background_genes = [row[0] for row in bg_result.all()]

            if not background_genes:
                logger.warning(
                    f"[GO pipeline] No GO annotations found for '{organism}'. "
                    "Skipping automatic GO enrichment. Load GO annotations to enable this feature."
                )
                return

            logger.info(
                f"[GO pipeline] Background: {len(background_genes)} annotated genes for {organism}"
            )

            for dataset_id, comparison_name in result_pairs:
                for regulation in ("ALL", "UP", "DOWN"):
                    try:
                        # Build DEG query
                        stmt = (
                            select(DegGene.gene_id, DegGene.gene_name, DegGene.log_fc, DegGene.padj)
                            .where(
                                DegGene.dataset_id == dataset_id,
                                DegGene.comparison_name == comparison_name,
                                DegGene.padj <= PADJ_THRESHOLD,
                            )
                        )
                        if regulation == "UP":
                            stmt = stmt.where(DegGene.log_fc >= LOGFC_THRESHOLD)
                        elif regulation == "DOWN":
                            stmt = stmt.where(DegGene.log_fc <= -LOGFC_THRESHOLD)
                        else:
                            # ALL: both directions, absolute filter
                            from sqlalchemy import or_
                            stmt = stmt.where(
                                or_(
                                    DegGene.log_fc >= LOGFC_THRESHOLD,
                                    DegGene.log_fc <= -LOGFC_THRESHOLD,
                                )
                            )

                        deg_result = await db.execute(stmt)
                        degs = deg_result.all()

                        if not degs:
                            logger.info(
                                f"[GO pipeline] {comparison_name}/{regulation}: 0 DEGs — skipping"
                            )
                            continue

                        # Prefer gene symbol (gene_name), fall back to gene_id
                        study_genes = [
                            (row.gene_name.strip() if row.gene_name and row.gene_name.strip() else row.gene_id)
                            for row in degs
                        ]

                        logger.info(
                            f"[GO pipeline] {comparison_name}/{regulation}: "
                            f"{len(study_genes)} DEGs → running enrichment"
                        )

                        # Run GO enrichment (namespace=None → all namespaces, one call)
                        enrichment_results = await go_service.go_enrichment_analysis(
                            db=db,
                            gene_list=study_genes,
                            background=background_genes,
                            namespace=None,
                            organism=organism,
                            min_gene_count=MIN_TERM_SIZE,
                            max_gene_count=MAX_TERM_SIZE,
                            pvalue_threshold=PVALUE_THRESHOLD,
                        )

                        if not enrichment_results:
                            logger.info(
                                f"[GO pipeline] {comparison_name}/{regulation}: no enriched terms"
                            )
                            continue

                        params_hash = self._go_params_hash(
                            namespace=None,
                            regulation=regulation,
                            padj_threshold=PADJ_THRESHOLD,
                            log_fc_threshold=LOGFC_THRESHOLD,
                            min_term_size=MIN_TERM_SIZE,
                            max_term_size=MAX_TERM_SIZE,
                            pvalue_threshold=PVALUE_THRESHOLD,
                            organism=organism,
                        )

                        # Delete old results for this (dataset, comparison, regulation, GO:ALL)
                        await db.execute(
                            delete(EnrichmentPathway).where(
                                EnrichmentPathway.dataset_id == dataset_id,
                                EnrichmentPathway.comparison_name == comparison_name,
                                EnrichmentPathway.regulation == regulation,
                                EnrichmentPathway.category == "GO:ALL",
                            )
                        )

                        study_size = len(study_genes)
                        bg_size = len(background_genes)

                        new_rows = [
                            EnrichmentPathway(
                                dataset_id=dataset_id,
                                comparison_name=comparison_name,
                                pathway_id=r["go_id"],
                                pathway_name=r["go_name"],
                                category=_ns_map.get(r.get("namespace", ""), "GO:ALL"),
                                description=r.get("definition"),
                                gene_count=r.get("study_count", 0),
                                pvalue=r["pvalue"],
                                padj=r["fdr"],
                                gene_ratio=f"{r.get('study_count', 0)}/{study_size}",
                                bg_ratio=f"{r.get('background_count', 0)}/{bg_size}",
                                genes=r.get("study_genes", []),
                                regulation=regulation,
                                enrichment_ratio=r.get("enrichment_ratio"),
                                level=r.get("level"),
                                parameters_hash=params_hash,
                            )
                            for r in enrichment_results
                        ]
                        db.add_all(new_rows)
                        await db.commit()

                        logger.info(
                            f"[GO pipeline] {comparison_name}/{regulation}: "
                            f"{len(new_rows)} terms saved"
                        )

                    except Exception as e:
                        logger.warning(
                            f"[GO pipeline] {comparison_name}/{regulation} failed (non-fatal): {e}"
                        )
                        await db.rollback()

        except Exception as e:
            logger.warning(f"[GO pipeline] Enrichment step failed (non-fatal): {e}")

    async def _download_datasets(
        self,
        workdir: Path,
        matrix_id: Optional[UUID],
        samples_id: Optional[UUID],
        comparisons_id: Optional[UUID],
        db: AsyncSession,
    ) -> None:
        """Download the three source datasets and write them as TSV files."""
        mapping = {
            "counts.tsv": matrix_id,
            "samples.tsv": samples_id,
            "comparisons.tsv": comparisons_id,
        }
        for filename, dataset_id in mapping.items():
            if dataset_id is None:
                raise ValueError(f"Missing source dataset for {filename}")
            await self._dataset_to_tsv(dataset_id, workdir / filename, db)

    async def _dataset_to_tsv(
        self, dataset_id: UUID, dest: Path, db: AsyncSession
    ) -> None:
        """Download a dataset's Parquet file and write it as TSV."""
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        if dataset.status != DatasetStatus.READY:
            raise ValueError(
                f"Dataset {dataset_id} is not READY (status={dataset.status})"
            )
        if not dataset.parquet_file_path:
            raise ValueError(f"Dataset {dataset_id} has no parquet_file_path")

        parquet_bytes = await storage_service.download_file(dataset.parquet_file_path)
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
        df.to_csv(dest, sep="\t", index=False)
        logger.info(f"  Downloaded {dataset_id} → {dest.name} ({len(df)} rows)")

    def _run_r_pipeline(self, workdir: Path, params: dict, progress_file: str, species: str = "human") -> None:
        """Execute the R pipeline script via subprocess."""
        if not os.path.exists(R_PIPELINE_SCRIPT):
            raise FileNotFoundError(
                f"R pipeline script not found: {R_PIPELINE_SCRIPT}. "
                "Is the r-worker container being used?"
            )

        # Build enrichment databases argument
        enrichment_dbs = params.get("enrichment_databases")
        if enrichment_dbs and isinstance(enrichment_dbs, list) and len(enrichment_dbs) > 0:
            enrichment_databases_arg = ",".join(str(db) for db in enrichment_dbs)
        else:
            enrichment_databases_arg = "all"

        cmd = [
            "Rscript", R_PIPELINE_SCRIPT,
            "--counts",       str(workdir / "counts.tsv"),
            "--samples",      str(workdir / "samples.tsv"),
            "--comparisons",  str(workdir / "comparisons.tsv"),
            "--outdir",       str(workdir / "output"),
            "--design",       str(params.get("design", "auto")),
            "--fdr",          str(params.get("fdr", 0.05)),
            "--min-log2fc",   str(params.get("min_log2fc", 1.0)),
            "--min-reads",    str(params.get("min_reads", 100000)),
            "--min-genes",    str(params.get("min_genes", 500)),
            "--min-count",    str(params.get("min_count", 10)),
            "--min-reps",     str(params.get("min_reps", 2)),
            "--threads",      str(params.get("threads", 1)),
            "--species",      species,
            "--anno-db-dir",  ANNO_DB_PATH,
            "--enrichment-databases", enrichment_databases_arg,
            "--progress-file", progress_file,
        ]

        logger.info(f"  Running R pipeline: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=R_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"R pipeline timed out after {R_TIMEOUT}s")

        if proc.stdout:
            logger.info(f"  [R stdout]\n{proc.stdout[-3000:]}")
        if proc.stderr:
            logger.warning(f"  [R stderr]\n{proc.stderr[-3000:]}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"R pipeline failed (exit {proc.returncode}).\n"
                f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}"
            )

    async def _upload_results(
        self,
        workdir: Path,
        project_id: UUID,
        analysis_name: str,
        analysis_id: UUID,
        db: AsyncSession,
    ) -> list[tuple[UUID, str]]:
        """
        Upload each genolens_deg.csv as a DEG Dataset and return pairs of (dataset_uuid, comparison_name).
        Also uploads method_pvalues.tsv as METADATA dataset and stores
        the enrichment_legacy_path in the DEG dataset metadata.
        """
        output_dir = workdir / "output" / "comparisons"
        if not output_dir.exists():
            raise RuntimeError(
                f"Output directory not found: {output_dir}. "
                "The R pipeline may have failed silently."
            )

        created_ids: list[tuple[UUID, str]] = []

        for comp_dir in sorted(output_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            deg_file = comp_dir / "genolens_deg.csv"
            if not deg_file.exists():
                logger.warning(f"  No genolens_deg.csv for {comp_dir.name}, skipping")
                continue

            comparison_id = comp_dir.name
            dataset_name = f"{analysis_name} — {comparison_id}"

            # Locate optional enrichment file
            enrich_file = comp_dir / "enrichment_legacy" / "functional_enrichment_onlyenriched.txt"
            enrich_storage_path: Optional[str] = None
            if enrich_file.exists():
                try:
                    enrich_bytes = enrich_file.read_bytes()
                    enrich_storage_path = (
                        f"projects/{project_id}/analyses/{analysis_id}"
                        f"/comparisons/{comparison_id}/enrichment_legacy.tsv"
                    )
                    await storage_service.upload_file(
                        enrich_storage_path,
                        enrich_bytes,
                        content_type="text/tab-separated-values",
                    )
                    logger.info(f"  Uploaded enrichment → {enrich_storage_path}")
                except Exception as e:
                    logger.warning(f"  Could not upload enrichment for {comparison_id}: {e}")
                    enrich_storage_path = None

            dataset_uuid = await self._create_deg_dataset(
                project_id=project_id,
                name=dataset_name,
                comparison_name=comparison_id,
                deg_file=deg_file,
                analysis_id=analysis_id,
                enrichment_legacy_path=enrich_storage_path,
                db=db,
            )
            created_ids.append((dataset_uuid, comparison_id))
            logger.info(f"  Uploaded DEG dataset: {dataset_name} → {dataset_uuid}")

            # Upload method_pvalues.tsv as METADATA dataset
            method_pvalues_file = comp_dir / "method_pvalues.tsv"
            if method_pvalues_file.exists():
                try:
                    await self._upload_method_pvalues(
                        project_id=project_id,
                        analysis_id=analysis_id,
                        comparison_id=comparison_id,
                        analysis_name=analysis_name,
                        method_pvalues_file=method_pvalues_file,
                        db=db,
                    )
                except Exception as e:
                    logger.warning(f"  Could not upload method_pvalues for {comparison_id}: {e}")

        if not created_ids:
            raise RuntimeError(
                "R pipeline finished but no genolens_deg.csv files were produced. "
                "Check that condition names in the comparisons file match sample conditions."
            )

        return created_ids

    async def _upload_method_pvalues(
        self,
        project_id: UUID,
        analysis_id: UUID,
        comparison_id: str,
        analysis_name: str,
        method_pvalues_file: Path,
        db: AsyncSession,
    ) -> UUID:
        """Upload method_pvalues.tsv as a METADATA Dataset."""
        from app.services.data_processor import data_processor

        tsv_bytes = method_pvalues_file.read_bytes()
        parquet_bytes = await data_processor.convert_to_parquet(tsv_bytes, ".tsv")

        dataset_id = uuid4()
        raw_path = (
            f"projects/{project_id}/datasets/{dataset_id}/raw/method_pvalues.tsv"
        )
        parquet_path = (
            f"projects/{project_id}/datasets/{dataset_id}/processed/method_pvalues.parquet"
        )
        await storage_service.upload_file(
            raw_path, tsv_bytes, content_type="text/tab-separated-values"
        )
        await storage_service.upload_file(
            parquet_path, parquet_bytes, content_type="application/octet-stream"
        )

        dataset = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=f"{analysis_name} — {comparison_id} — Method P-values",
            description=f"Per-method p-values (DESeq2/edgeR/limma/Stouffer) for {comparison_id}",
            type=DatasetType.METADATA,
            status=DatasetStatus.READY,
            raw_file_path=raw_path,
            parquet_file_path=parquet_path,
            dataset_metadata={
                "subtype": "multimethod_pvalues",
                "comparison": comparison_id,
                "generated_by": "self_service_analysis",
                "analysis_id": str(analysis_id),
            },
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        logger.info(f"  Uploaded method_pvalues dataset → {dataset_id}")
        return dataset_id

    async def _create_deg_dataset(
        self,
        project_id: UUID,
        name: str,
        comparison_name: str,
        deg_file: Path,
        analysis_id: UUID,
        db: AsyncSession,
        enrichment_legacy_path: Optional[str] = None,
    ) -> UUID:
        """Create a DEG Dataset record and upload the CSV + Parquet files."""
        from app.services.data_processor import data_processor

        csv_bytes = deg_file.read_bytes()
        parquet_bytes = await data_processor.convert_to_parquet(csv_bytes, ".csv")

        dataset_id = uuid4()
        raw_path = f"projects/{project_id}/datasets/{dataset_id}/raw/genolens_deg.csv"
        parquet_path = f"projects/{project_id}/datasets/{dataset_id}/processed/genolens_deg.parquet"

        await storage_service.upload_file(raw_path, csv_bytes, content_type="text/csv")
        await storage_service.upload_file(
            parquet_path, parquet_bytes, content_type="application/octet-stream"
        )

        # Read parquet to get metadata
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
        padj_col = next((c for c in df.columns if c.startswith("padj:")), None)
        sig_count = int((df[padj_col] < 0.05).sum()) if padj_col else None

        # Calculate up/down counts from the data directly
        logfc_col = next((c for c in df.columns if c.startswith("logFC:")), None)
        up_count: Optional[int] = None
        down_count: Optional[int] = None
        if padj_col and logfc_col:
            sig_mask = df[padj_col] < 0.05
            up_count = int((sig_mask & (df[logfc_col] > 0)).sum())
            down_count = int((sig_mask & (df[logfc_col] < 0)).sum())

        metadata = {
            "comparisons": [comparison_name],
            "total_genes": len(df),
            "significant_genes": sig_count,
            "generated_by": "self_service_analysis",
            "analysis_id": str(analysis_id),
        }
        if enrichment_legacy_path:
            metadata["enrichment_legacy_path"] = enrichment_legacy_path

        dataset = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=name,
            description=f"DESeq2 results generated by self-service analysis (comparison: {comparison_name})",
            type=DatasetType.DEG,
            status=DatasetStatus.READY,
            raw_file_path=raw_path,
            parquet_file_path=parquet_path,
            dataset_metadata=metadata,
            total_genes=len(df),
            deg_significant_count=sig_count,
            deg_up_count=up_count,
            deg_down_count=down_count,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        # ----------------------------------------------------------------
        # Populate deg_genes table for fast SQL querying (used by DEGTable)
        # ----------------------------------------------------------------
        try:
            comparisons_map = data_processor._detect_comparisons(df.columns.tolist())
            if comparisons_map:
                deg_genes_data = await data_processor.extract_deg_genes_for_db(
                    parquet_bytes, comparisons_map
                )
                # Delete any stale records (idempotent re-runs)
                await db.execute(
                    delete(DegGene).where(DegGene.dataset_id == str(dataset_id))
                )
                await db.commit()

                chunk_size = 1000
                for comp, genes_list in deg_genes_data.items():
                    if not genes_list:
                        continue
                    records = [
                        {
                            "dataset_id": str(dataset_id),
                            "comparison_name": comp,
                            "gene_id": g["gene_id"],
                            "log_fc": g.get("log_fc"),
                            "padj": g.get("padj"),
                            "pvalue": g.get("pvalue"),
                            "regulation": g.get("regulation"),
                            "gene_name": g.get("gene_name"),
                        }
                        for g in genes_list
                    ]
                    for i in range(0, len(records), chunk_size):
                        await db.execute(insert(DegGene), records[i : i + chunk_size])
                        await db.commit()
                    logger.info(
                        f"  Stored {len(records)} DEG genes for comparison '{comp}' "
                        f"in dataset {dataset_id}"
                    )
            else:
                logger.warning(
                    f"  No comparison columns detected in {deg_file.name}, "
                    "deg_genes table not populated"
                )
        except Exception as exc:
            await db.rollback()
            logger.error(
                f"  Failed to populate deg_genes for dataset {dataset_id}: {exc}",
                exc_info=True,
            )
            # Non-fatal: dataset record is already saved

        return dataset_id

    # ------------------------------------------------------------------
    # Upload intermediate datasets (VST + normalized counts)
    # ------------------------------------------------------------------

    async def _upload_intermediate_datasets(
        self,
        workdir: Path,
        project_id: UUID,
        analysis_name: str,
        analysis_id: UUID,
        db: AsyncSession,
    ) -> dict:
        """
        Upload VST and normalized count matrices as MATRIX Datasets.
        Returns a dict with keys 'vst' and 'normalized' mapping to dataset UUIDs.
        Non-fatal: logs warnings but does not raise.
        """
        from app.services.data_processor import data_processor

        result: dict = {}
        output_dir = workdir / "output"

        # Load PCA data if available
        pca_data: Optional[dict] = None
        pca_file = output_dir / "pca_data.json"
        if pca_file.exists():
            try:
                import json
                pca_data = json.loads(pca_file.read_text())
            except Exception as e:
                logger.warning(f"  Could not read pca_data.json: {e}")

        # Load QC report if available
        qc_data: Optional[dict] = None
        qc_file = output_dir / "qc_report.json"
        if qc_file.exists():
            try:
                import json
                qc_data = json.loads(qc_file.read_text())
            except Exception as e:
                logger.warning(f"  Could not read qc_report.json: {e}")

        for source_key, filename, label in [
            ("vst",        "vst_counts.tsv",        "VST"),
            ("normalized", "normalized_counts.tsv",  "Normalized"),
        ]:
            tsv_file = output_dir / filename
            if not tsv_file.exists():
                logger.warning(f"  Intermediate file not found, skipping: {filename}")
                continue

            try:
                tsv_bytes = tsv_file.read_bytes()
                parquet_bytes = await data_processor.convert_to_parquet(tsv_bytes, ".tsv")

                dataset_id = uuid4()
                raw_path = f"projects/{project_id}/datasets/{dataset_id}/raw/{filename}"
                parquet_path = (
                    f"projects/{project_id}/datasets/{dataset_id}/processed/"
                    f"{filename.replace('.tsv', '.parquet')}"
                )

                await storage_service.upload_file(raw_path, tsv_bytes, content_type="text/tab-separated-values")
                await storage_service.upload_file(
                    parquet_path, parquet_bytes, content_type="application/octet-stream"
                )

                df = pd.read_parquet(io.BytesIO(parquet_bytes))
                metadata: dict = {
                    "source": source_key,
                    "generated_by": "self_service_analysis",
                    "analysis_id": str(analysis_id),
                }
                if source_key == "vst" and pca_data is not None:
                    metadata["pca_data"] = pca_data
                if qc_data is not None:
                    metadata["qc_report"] = qc_data

                dataset = Dataset(
                    id=dataset_id,
                    project_id=project_id,
                    name=f"{analysis_name} — {label} counts",
                    description=f"{label}-transformed count matrix from self-service analysis",
                    type=DatasetType.MATRIX,
                    status=DatasetStatus.READY,
                    raw_file_path=raw_path,
                    parquet_file_path=parquet_path,
                    dataset_metadata=metadata,
                    total_genes=len(df),
                )
                db.add(dataset)
                await db.commit()
                await db.refresh(dataset)
                result[source_key] = str(dataset_id)
                logger.info(f"  Uploaded {label} counts dataset → {dataset_id}")

            except Exception as e:
                logger.error(
                    f"  Failed to upload {label} counts dataset: {e}", exc_info=True
                )

        # ------------------------------------------------------------------
        # UMAP dataset (new — from run_multimethod_pipeline.R)
        # ------------------------------------------------------------------
        umap_file = output_dir / "umap_data.json"
        if umap_file.exists():
            try:
                import json as _json

                umap_raw = umap_file.read_text()
                umap_payload = _json.loads(umap_raw)

                # Only store if non-empty (pipeline writes "{}" on failure)
                if isinstance(umap_payload, list) and len(umap_payload) > 0:
                    dataset_id = uuid4()
                    storage_path = (
                        f"projects/{project_id}/datasets/{dataset_id}/raw/umap_data.json"
                    )
                    await storage_service.upload_file(
                        storage_path,
                        umap_raw.encode(),
                        content_type="application/json",
                    )
                    umap_metadata: dict = {
                        "source": "umap",
                        "generated_by": "self_service_analysis",
                        "analysis_id": str(analysis_id),
                        "umap_data": umap_payload,
                    }
                    if qc_data is not None:
                        umap_metadata["qc_report"] = qc_data

                    dataset = Dataset(
                        id=dataset_id,
                        project_id=project_id,
                        name=f"{analysis_name} — UMAP",
                        description="UMAP 2D projection from multi-method analysis",
                        type=DatasetType.METADATA,
                        status=DatasetStatus.READY,
                        raw_file_path=storage_path,
                        parquet_file_path=None,
                        dataset_metadata=umap_metadata,
                    )
                    db.add(dataset)
                    await db.commit()
                    await db.refresh(dataset)
                    result["umap"] = str(dataset_id)
                    logger.info(f"  Uploaded UMAP dataset → {dataset_id}")
            except Exception as e:
                logger.error(f"  Failed to upload UMAP dataset: {e}", exc_info=True)

        return result


analysis_service = AnalysisService()
