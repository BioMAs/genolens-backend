"""
Celery task: generate the branded SciLicium LaTeX PDF report for one analysis.

Runs on the `r_analysis` queue (the r-worker), where the LaTeX toolchain
(pdflatex/texlive + ghostscript) is installed.
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
    name="app.worker.tasks.report_task.generate_analysis_report",
    queue="r_analysis",
    max_retries=1,
    default_retry_delay=30,
)
def generate_analysis_report(self, job_id: str, analysis_id: str) -> dict:
    return _run_async(_async_generate(self, job_id, analysis_id))


async def _async_generate(task, job_id: str, analysis_id: str) -> dict:
    from app.db.session import AsyncSessionLocal
    from app.models.report_job import ReportJob, ReportJobStatus

    async with AsyncSessionLocal() as db:
        job = await db.get(ReportJob, UUID(job_id))
        if not job:
            raise ValueError(f"ReportJob {job_id} not found")

        job.status = ReportJobStatus.RUNNING
        await db.commit()

        try:
            from app.services.report_latex_service import report_latex_service
            from app.services.storage import storage_service

            pdf_bytes = await report_latex_service.render_pdf(db, UUID(analysis_id))

            pdf_path = f"reports/{analysis_id}/{job_id}.pdf"
            await storage_service.upload_file(pdf_path, pdf_bytes, "application/pdf")

            job.status = ReportJobStatus.DONE
            job.pdf_path = pdf_path
            await db.commit()
            return {"status": "done", "pdf_path": pdf_path}

        except Exception as exc:
            logger.exception("Report generation failed for job %s", job_id)
            job.status = ReportJobStatus.FAILED
            job.error_message = str(exc)[:2000]
            await db.commit()
            return {"status": "failed", "error": str(exc)}
