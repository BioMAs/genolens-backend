from app.models.report_job import ReportJob, ReportJobStatus


def test_report_job_status_values():
    assert ReportJobStatus.PENDING == "PENDING"
    assert ReportJobStatus.RUNNING == "RUNNING"
    assert ReportJobStatus.DONE == "DONE"
    assert ReportJobStatus.FAILED == "FAILED"


def test_report_job_can_be_instantiated():
    job = ReportJob(status=ReportJobStatus.PENDING)
    assert job.status == ReportJobStatus.PENDING
