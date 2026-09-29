"""Celery task modules.

This package re-exports the original tasks from the adjacent ``tasks.py``
(which Python can no longer see directly, since this ``tasks/`` directory
shadows it). We load it explicitly via ``importlib`` so existing callers
such as ``app.worker.__init__`` and ``app.api.endpoints.datasets`` continue
to work without changes.
"""

import importlib.util
import os as _os
import sys as _sys

# ── Re-export legacy tasks from the shadowed tasks.py ────────────────────────
_legacy_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "tasks.py",
)
_spec = importlib.util.spec_from_file_location("app.worker._tasks_legacy", _legacy_path)
_legacy_mod = importlib.util.module_from_spec(_spec)
_sys.modules.setdefault("app.worker._tasks_legacy", _legacy_mod)
_spec.loader.exec_module(_legacy_mod)

process_dataset_upload = _legacy_mod.process_dataset_upload  # noqa: F401
import_geo_dataset = _legacy_mod.import_geo_dataset  # noqa: F401
health_check = _legacy_mod.health_check  # noqa: F401
run_self_service_analysis = _legacy_mod.run_self_service_analysis  # noqa: F401
_count_pipeline_analysis = _legacy_mod._count_pipeline_analysis  # noqa: F401
_analysis_cancelled = _legacy_mod._analysis_cancelled  # noqa: F401
# Appele par POST /datasets/{id}/rerun-enrichment. Son absence de cette liste
# faisait lever ImportError a la route, donc 500 pour tout appelant : le
# package masque `tasks.py`, et un symbole non repris ici n'existe plus pour
# personne. `tests/test_rerun_enrichment_route.py` epingle le re-export.
_auto_run_enrichment = _legacy_mod._auto_run_enrichment  # noqa: F401

# ── Periodic tasks ────────────────────────────────────────────────────────────
# These imports are what actually REGISTER the tasks with Celery. Listing a
# module in `celery_app.conf.include` schedules it but does not import it here,
# so a task missing from this file is one beat enqueues and the worker answers
# with "Received unregistered task". `tests/test_periodic_tasks_registered.py`
# walks the beat schedule and fails when that happens.
from app.worker.tasks.quota_tasks import reset_monthly_analysis_quotas  # noqa: F401
from app.worker.tasks.account_tasks import check_account_expirations  # noqa: F401

__all__ = [
    "process_dataset_upload",
    "import_geo_dataset",
    "health_check",
    "run_self_service_analysis",
    "reset_monthly_analysis_quotas",
    "check_account_expirations",
]
