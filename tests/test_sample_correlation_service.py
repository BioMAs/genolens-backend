"""
Unit tests for SampleCorrelationService.

Covers:
- get_cached_correlations: returns None when empty, reconstructs matrices when data present
- store_correlations: calls db.add for each pairwise correlation
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

from tests.conftest import TEST_DATASET_ID


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalars_all(items):
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


def make_sample_correlation(
    sample_a: str,
    sample_b: str,
    correlation: float = 0.95,
    distance: float = 0.05,
    dataset_id=TEST_DATASET_ID,
    method: str = "hierarchical",
    metric: str = "euclidean",
    top_n_genes: int = 500,
):
    """Build a minimal SampleCorrelation mock."""
    from app.models.models import SampleCorrelation
    sc = MagicMock(spec=SampleCorrelation)
    sc.dataset_id = dataset_id
    sc.sample_a = sample_a
    sc.sample_b = sample_b
    sc.correlation = correlation
    sc.distance = distance
    sc.method = method
    sc.metric = metric
    sc.top_n_genes = top_n_genes
    return sc


# ─────────────────────────────────────────────────────────────────────────────
# get_cached_correlations
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCachedCorrelations:

    @pytest.mark.asyncio
    async def test_returns_none_when_no_entries(self, mock_db):
        """Should return None when no correlations are cached."""
        from app.services.sample_correlation_service import SampleCorrelationService

        mock_db.execute.return_value = _scalars_all([])

        svc = SampleCorrelationService()
        result = await svc.get_cached_correlations(
            mock_db, TEST_DATASET_ID, method="hierarchical", metric="euclidean", top_n_genes=500
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_reconstructs_correlation_matrix(self, mock_db):
        """Should reconstruct correlation/distance matrices from stored pairs."""
        from app.services.sample_correlation_service import SampleCorrelationService

        sc = make_sample_correlation("sampleA", "sampleB", correlation=0.90, distance=0.10)
        mock_db.execute.return_value = _scalars_all([sc])

        svc = SampleCorrelationService()
        result = await svc.get_cached_correlations(
            mock_db, TEST_DATASET_ID, method="hierarchical", metric="euclidean", top_n_genes=500
        )

        assert result is not None
        assert "samples" in result
        assert "sampleA" in result["samples"]
        assert "sampleB" in result["samples"]

    @pytest.mark.asyncio
    async def test_correlation_matrix_is_symmetric(self, mock_db):
        """Correlation matrix should be symmetric (M[i,j] == M[j,i])."""
        from app.services.sample_correlation_service import SampleCorrelationService

        sc = make_sample_correlation("alpha", "beta", correlation=0.80)
        mock_db.execute.return_value = _scalars_all([sc])

        svc = SampleCorrelationService()
        result = await svc.get_cached_correlations(
            mock_db, TEST_DATASET_ID, method="hierarchical", metric="euclidean", top_n_genes=500
        )

        matrix = np.array(result["correlation_matrix"])
        # The matrix should be symmetric (M[i,j] == M[j,i])
        assert matrix.shape[0] == matrix.shape[1]
        np.testing.assert_array_almost_equal(matrix, matrix.T)


# ─────────────────────────────────────────────────────────────────────────────
# store_correlations
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreCorrelations:

    @pytest.mark.asyncio
    async def test_stores_pairwise_correlations(self, mock_db):
        """Should call db.add for each pairwise correlation."""
        from app.services.sample_correlation_service import SampleCorrelationService
        import pandas as pd

        svc = SampleCorrelationService()

        samples = ["s1", "s2", "s3"]
        # Symmetric correlation matrix (3x3)
        corr_matrix = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0],
        ])
        dist_matrix = 1 - corr_matrix

        await svc.cache_correlations(
            db=mock_db,
            dataset_id=TEST_DATASET_ID,
            samples=samples,
            correlation_matrix=corr_matrix,
            distance_matrix=dist_matrix,
            method="hierarchical",
            metric="euclidean",
            top_n_genes=500,
        )

        # add_all called once with 3 pairs (n*(n-1)/2 = 3)
        mock_db.add_all.assert_called_once()
        pairs = mock_db.add_all.call_args[0][0]
        assert len(pairs) == 3
        # commit called at least once (invalidate_cache + cache_correlations each commit)
        mock_db.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_stores_nothing_for_single_sample(self, mock_db):
        """A single-sample dataset has no pairs to store."""
        from app.services.sample_correlation_service import SampleCorrelationService
        import numpy as np

        svc = SampleCorrelationService()
        corr = np.array([[1.0]])
        dist = np.array([[0.0]])

        await svc.cache_correlations(
            db=mock_db,
            dataset_id=TEST_DATASET_ID,
            samples=["only_sample"],
            correlation_matrix=corr,
            distance_matrix=dist,
            method="hierarchical",
            metric="euclidean",
            top_n_genes=500,
        )

        mock_db.add.assert_not_called()
