"""
Celery task: ad-hoc annoDB functional enrichment of a Venn/UpSet intersection.

Runs on the `r_analysis` queue (the r-worker), where /app/anno_db is mounted and
the R toolchain lives. Invokes functional_enrichment.R in --gene-list mode on the
intersection's gene symbols and stores the parsed enrichment rows on the job.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from uuid import UUID

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse_enrichment_csv(csv_path: Path) -> list[dict]:
    """Parse the genolens_enrichment.csv produced by functional_enrichment.R."""
    import math

    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    def _num(v):
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    rows: list[dict] = []
    for _, r in df.iterrows():
        genes_raw = r.get("genes")
        genes = (
            [g for g in str(genes_raw).replace(",", "/").split("/") if g and g != "nan"]
            if genes_raw is not None and str(genes_raw) != "nan"
            else []
        )
        rows.append(
            {
                "pathway_id": str(r.get("term", "")),
                "pathway_name": str(r.get("Description", "") or r.get("term", "")),
                "category": str(r.get("category", "")),
                "pvalue": _num(r.get("pvalue")),
                "padj": _num(r.get("p.adjust")),
                "gene_count": int(r.get("Count")) if _num(r.get("Count")) is not None else len(genes),
                "gene_ratio": str(r.get("GeneRatio", "")),
                "bg_ratio": str(r.get("BgRatio", "")),
                "genes": genes,
            }
        )

    # Most-significant first
    rows.sort(key=lambda x: (x["padj"] if x["padj"] is not None else 1.0))
    return rows


@celery_app.task(
    bind=True,
    name="app.worker.tasks.intersection_enrichment_task.run_intersection_enrichment",
    queue="r_analysis",
    max_retries=1,
    default_retry_delay=30,
)
def run_intersection_enrichment(self, job_id: str) -> dict:
    return _run_async(_async_run(self, job_id))


async def _async_run(task, job_id: str) -> dict:
    from app.db.session import AsyncSessionLocal
    from app.models.intersection_enrichment_job import (
        IntersectionEnrichmentJob,
        IntersectionEnrichmentStatus,
    )

    async with AsyncSessionLocal() as db:
        job = await db.get(IntersectionEnrichmentJob, UUID(job_id))
        if not job:
            raise ValueError(f"IntersectionEnrichmentJob {job_id} not found")

        job.status = IntersectionEnrichmentStatus.RUNNING
        await db.commit()

        try:
            params = job.params or {}
            genes = [str(g).strip() for g in params.get("genes", []) if str(g).strip()]
            species = params.get("species") or "human"
            label = params.get("label") or "intersection"
            if not genes:
                raise ValueError("No genes provided for enrichment")

            enrich_script = Path(
                os.environ.get("R_SCRIPTS_PATH", "/app/r_scripts")
            ) / "functional_enrichment.R"
            anno_db_dir = os.environ.get("ANNO_DB_PATH", "/app/anno_db")
            if not enrich_script.exists():
                raise FileNotFoundError(f"functional_enrichment.R not found at {enrich_script}")

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                gene_file = tmp_path / "genes.txt"
                gene_file.write_text("\n".join(genes))
                out_csv = tmp_path / "genolens_enrichment.csv"

                cmd = [
                    "Rscript", str(enrich_script),
                    "--gene-list", str(gene_file),
                    "--anno-db-dir", anno_db_dir,
                    "--species", str(species),
                    "--comparison", label,
                    "--output", str(out_csv),
                    "--padj-cutoff", str(params.get("padj_cutoff", 0.05)),
                ]
                logger.info("[IntersectionEnrichment] %s — %d genes, species=%s", job_id, len(genes), species)
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if proc.returncode != 0:
                    raise RuntimeError(f"functional_enrichment.R failed: {proc.stderr[-1500:]}")

                rows = _parse_enrichment_csv(out_csv) if out_csv.exists() else []

            job.result = rows
            job.status = IntersectionEnrichmentStatus.DONE
            await db.commit()
            logger.info("[IntersectionEnrichment] %s — DONE, %d terms", job_id, len(rows))
            return {"status": "done", "terms": len(rows)}

        except Exception as exc:
            logger.exception("Intersection enrichment failed for job %s", job_id)
            job.status = IntersectionEnrichmentStatus.FAILED
            job.error_message = str(exc)[:2000]
            await db.commit()
            return {"status": "failed", "error": str(exc)}
