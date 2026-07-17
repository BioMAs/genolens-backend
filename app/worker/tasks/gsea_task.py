"""
Celery task: asynchronous pre-ranked GSEA for a comparison.

Pure-Python compute (no R) → runs on the default queue. Loads the GSEAJob,
runs the shared compute (app.services.gsea_runner.compute_gsea), stores the
payload on the job and in the shared computation cache (so the gsea-results
GET can restore it), and marks the job DONE/FAILED.
"""
import asyncio
import logging
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


@celery_app.task(
    bind=True,
    name="app.worker.tasks.gsea_task.run_gsea_job",
    queue="default",
    max_retries=1,
    default_retry_delay=30,
)
def run_gsea_job(self, job_id: str) -> dict:
    return _run_async(_async_run(self, job_id))


async def _async_run(task, job_id: str) -> dict:
    from sqlalchemy import select

    from app.db.session import AsyncSessionLocal
    from app.models.gsea_job import GSEAJob, GSEAJobStatus
    from app.models.models import Dataset
    from app.services.gsea_runner import (
        compute_gsea, persist_gsea_result, GSEANoDegError, GSEANoGeneSetsError,
    )

    async with AsyncSessionLocal() as db:
        job = await db.get(GSEAJob, UUID(job_id))
        if not job:
            raise ValueError(f"GSEAJob {job_id} not found")

        job.status = GSEAJobStatus.RUNNING
        await db.commit()

        try:
            params = job.params or {}
            ds_result = await db.execute(select(Dataset).where(Dataset.id == job.dataset_id))
            dataset = ds_result.scalar_one_or_none()
            if not dataset:
                raise ValueError(f"Dataset {job.dataset_id} not found")

            comparison_name = params.get("comparison_name")
            logger.info("[GSEA] %s — comparison=%s, db=%s", job_id, comparison_name, params.get("gene_set_database"))

            payload = await compute_gsea(
                db=db,
                dataset=dataset,
                comparison_name=comparison_name,
                gene_set_database=params.get("gene_set_database", "GO_BP"),
                ranking_metric=params.get("ranking_metric", "signed_pvalue"),
                min_size=int(params.get("min_size", 15)),
                max_size=int(params.get("max_size", 500)),
                n_permutations=int(params.get("n_permutations", 1000)),
                fdr_threshold=float(params.get("fdr_threshold", 0.25)),
            )

            await persist_gsea_result(db, job.dataset_id, payload)

            job.result = payload
            job.status = GSEAJobStatus.DONE
            await db.commit()
            logger.info(
                "[GSEA] %s — DONE, %d significant sets",
                job_id, payload["summary"]["significant_gene_sets"],
            )
            return {"status": "done", "significant": payload["summary"]["significant_gene_sets"]}

        except (GSEANoDegError, GSEANoGeneSetsError) as exc:
            logger.warning("GSEA job %s: %s", job_id, exc)
            job.status = GSEAJobStatus.FAILED
            job.error_message = str(exc)[:2000]
            await db.commit()
            return {"status": "failed", "error": str(exc)}
        except Exception as exc:
            logger.exception("GSEA failed for job %s", job_id)
            job.status = GSEAJobStatus.FAILED
            job.error_message = str(exc)[:2000]
            await db.commit()
            return {"status": "failed", "error": str(exc)}
