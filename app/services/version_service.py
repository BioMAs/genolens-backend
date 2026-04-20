"""
Version fingerprinting for scientific algorithm provenance.

Returns a stable snapshot of key package versions used in analysis pipelines.
The result is cached for the process lifetime — versions don't change at runtime.
"""
import logging
import importlib.metadata
from functools import lru_cache

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = [
    "scipy",
    "scikit-learn",
    "goatools",
    "pandas",
    "numpy",
    "fastcluster",
    "umap-learn",
]


@lru_cache(maxsize=1)
def get_algorithm_versions() -> dict[str, str]:
    """
    Return a dict of {package_name: version} for all tracked scientific packages.
    Uses the distribution name (pip name) as the key.
    Returns "unknown" for any package that cannot be found.
    """
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            logger.debug("Package %r not found in metadata — recording as 'unknown'", pkg)
            versions[pkg] = "unknown"
    return versions


def get_app_version() -> str:
    """Return the application version from settings."""
    from app.core.config import settings
    return settings.APP_VERSION
