"""
Unit tests for version_service.

Covers:
- get_algorithm_versions() returns a dict
- All expected package keys are present
- Each value is a non-empty string
- Calling it twice returns the same object (lru_cache works)
- A missing package returns "unknown"
"""
import importlib.metadata
from unittest.mock import patch

import pytest


_EXPECTED_PACKAGES = [
    "scipy",
    "scikit-learn",
    "goatools",
    "pandas",
    "numpy",
    "fastcluster",
    "umap-learn",
]


class TestGetAlgorithmVersions:

    def setup_method(self):
        """Clear the lru_cache before each test so tests are independent."""
        from app.services.version_service import get_algorithm_versions
        get_algorithm_versions.cache_clear()

    def test_returns_dict(self):
        """get_algorithm_versions() should return a dict."""
        from app.services.version_service import get_algorithm_versions
        result = get_algorithm_versions()
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self):
        """All tracked packages should appear as keys in the result."""
        from app.services.version_service import get_algorithm_versions
        result = get_algorithm_versions()
        for pkg in _EXPECTED_PACKAGES:
            assert pkg in result, f"Expected key {pkg!r} not found in versions dict"

    def test_values_are_non_empty_strings(self):
        """Every value in the result dict should be a non-empty string."""
        from app.services.version_service import get_algorithm_versions
        result = get_algorithm_versions()
        for pkg, version in result.items():
            assert isinstance(version, str), f"Version for {pkg!r} is not a string"
            assert len(version) > 0, f"Version for {pkg!r} is an empty string"

    def test_lru_cache_returns_same_object(self):
        """Calling get_algorithm_versions() twice should return the identical object."""
        from app.services.version_service import get_algorithm_versions
        first = get_algorithm_versions()
        second = get_algorithm_versions()
        assert first is second

    def test_missing_package_returns_unknown(self):
        """A package not found in metadata should be recorded as 'unknown'."""
        from app.services.version_service import get_algorithm_versions

        def raise_for_missing(pkg):
            if pkg == "goatools":
                raise importlib.metadata.PackageNotFoundError(pkg)
            return "1.0.0"

        with patch("importlib.metadata.version", side_effect=raise_for_missing):
            result = get_algorithm_versions()

        assert result["goatools"] == "unknown"

    def test_all_missing_packages_return_unknown(self):
        """When all packages are missing, every value should be 'unknown'."""
        from app.services.version_service import get_algorithm_versions

        def always_raise(pkg):
            raise importlib.metadata.PackageNotFoundError(pkg)

        with patch("importlib.metadata.version", side_effect=always_raise):
            result = get_algorithm_versions()

        for pkg in _EXPECTED_PACKAGES:
            assert result[pkg] == "unknown", f"Expected 'unknown' for {pkg!r}"


class TestGetAppVersion:

    def test_returns_string(self):
        """get_app_version() should return a non-empty string."""
        from app.services.version_service import get_app_version
        version = get_app_version()
        assert isinstance(version, str)
        assert len(version) > 0
