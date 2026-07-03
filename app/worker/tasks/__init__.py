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

# ── New periodic quota task ───────────────────────────────────────────────────
from app.worker.tasks.quota_tasks import reset_monthly_comparison_quotas  # noqa: F401

__all__ = [
    "process_dataset_upload",
    "import_geo_dataset",
    "health_check",
    "run_self_service_analysis",
    "reset_monthly_comparison_quotas",
]
