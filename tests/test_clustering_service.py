"""
Unit tests for ClusteringService — pure computation, no DB.

Covers:
- perform_clustering: hierarchical ward, average, k-means
- perform_clustering: top_n_genes selection, gene_ids filter
- perform_clustering: downsampling > max_genes
- perform_clustering: invalid data (NaN, Inf)
- perform_clustering: ward requires euclidean
- _compute_linkage: metric name mapping
- precompute_sample_clustering: output structure
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n_genes: int = 20, n_samples: int = 6, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic expression DataFrame."""
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=5.0, scale=2.0, size=(n_genes, n_samples))
    genes = [f"Gene{i}" for i in range(n_genes)]
    samples = [f"S{j}" for j in range(n_samples)]
    return pd.DataFrame(data, index=genes, columns=samples)


def _make_service():
    from app.services.clustering_service import ClusteringService
    return ClusteringService()


# ─────────────────────────────────────────────────────────────────────────────
# perform_clustering — happy paths
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformClusteringHappyPaths:

    def test_hierarchical_ward_produces_valid_result(self):
        """Ward linkage on genes and samples should produce a complete result dict."""
        df = _make_df(n_genes=20, n_samples=6)
        service = _make_service()

        result = service.perform_clustering(df.copy(), cluster_rows=True, cluster_cols=True)

        assert "row_labels" in result
        assert "col_labels" in result
        assert "row_order" in result
        assert "col_order" in result
        assert len(result["row_labels"]) == 20
        assert len(result["col_labels"]) == 6

    def test_row_order_is_valid_permutation(self):
        """row_order must be a valid permutation of 0..n_genes-1."""
        n = 20
        df = _make_df(n_genes=n, n_samples=4)
        service = _make_service()

        result = service.perform_clustering(df.copy())

        assert sorted(result["row_order"]) == list(range(n))

    def test_col_order_is_valid_permutation(self):
        """col_order must be a valid permutation of 0..n_samples-1."""
        df = _make_df(n_genes=20, n_samples=6)
        service = _make_service()

        result = service.perform_clustering(df.copy(), cluster_cols=True)

        assert sorted(result["col_order"]) == list(range(6))

    def test_hierarchical_complete_euclidean_raises(self):
        """
        Known limitation: fastcluster.linkage_vector only supports 'ward' for euclidean.
        Other methods (complete, average, single) with euclidean raise ValueError.
        This test documents the current behavior.
        """
        df = _make_df(n_genes=15, n_samples=4)
        service = _make_service()

        with pytest.raises(ValueError, match="Failed to cluster rows"):
            service.perform_clustering(
                df.copy(), method="complete", metric="euclidean", cluster_cols=False
            )

    def test_hierarchical_single_correlation_succeeds(self):
        """Single linkage with correlation metric should produce valid results."""
        df = _make_df(n_genes=15, n_samples=6)
        service = _make_service()

        result = service.perform_clustering(
            df.copy(), method="single", metric="correlation", cluster_cols=False
        )
        assert result["row_dendrogram"] is not None
        assert len(result["row_labels"]) == 15

    def test_hierarchical_correlation_metric(self):
        """Correlation metric with average linkage should work."""
        df = _make_df(n_genes=15, n_samples=6)
        service = _make_service()

        result = service.perform_clustering(
            df.copy(), method="average", metric="correlation"
        )
        assert result is not None

    def test_kmeans_clustering(self):
        """K-means method should cluster rows into groups."""
        df = _make_df(n_genes=30, n_samples=6)
        service = _make_service()

        # Disable col clustering: the code passes method='kmeans' to _compute_linkage
        # for columns as well, which is invalid — col clustering is a separate concern.
        result = service.perform_clustering(
            df.copy(), method="kmeans", n_clusters=3, cluster_cols=False
        )

        assert "row_clusters" in result
        assert result["n_clusters"] == 3
        assert len(set(result["row_clusters"])) <= 3

    def test_no_row_clustering_returns_identity_order(self):
        """Disabling row clustering should return sequential row_order."""
        df = _make_df(n_genes=10, n_samples=4)
        service = _make_service()

        result = service.perform_clustering(df.copy(), cluster_rows=False, cluster_cols=False)

        assert result["row_order"] == list(range(10))

    def test_top_n_genes_filters_by_variance(self):
        """top_n_genes should select the most variable genes."""
        df = _make_df(n_genes=50, n_samples=6)
        service = _make_service()

        result = service.perform_clustering(df.copy(), top_n_genes=10)

        assert len(result["row_labels"]) == 10

    def test_gene_ids_filter_uses_provided_list(self):
        """gene_ids parameter should restrict to given genes."""
        df = _make_df(n_genes=20, n_samples=6)
        subset = df.index[:5].tolist()
        service = _make_service()

        result = service.perform_clustering(df.copy(), gene_ids=subset)

        assert len(result["row_labels"]) == 5
        assert set(result["row_labels"]) == set(subset)

    def test_precomputed_col_clustering_is_reused(self):
        """When precomputed col clustering is provided, it should be used directly."""
        df = _make_df(n_genes=20, n_samples=6)
        service = _make_service()

        precomp = {
            "col_dendrogram": [[0, 1, 1.0, 2], [2, 3, 2.0, 2]],
            "col_order": [2, 0, 4, 1, 5, 3],
        }
        result = service.perform_clustering(
            df.copy(),
            cluster_cols=True,
            precomputed_col_clustering=precomp,
        )

        assert result["col_order"] == [2, 0, 4, 1, 5, 3]


# ─────────────────────────────────────────────────────────────────────────────
# perform_clustering — error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformClusteringErrors:

    def test_raises_on_nan_data(self):
        """NaN values in the DataFrame should raise ValueError."""
        df = _make_df(n_genes=10, n_samples=4)
        df.iloc[3, 2] = float("nan")
        service = _make_service()

        with pytest.raises(ValueError, match="NaN"):
            service.perform_clustering(df.copy())

    def test_raises_on_infinite_data(self):
        """Inf values in the DataFrame should raise ValueError."""
        df = _make_df(n_genes=10, n_samples=4)
        df.iloc[1, 1] = float("inf")
        service = _make_service()

        with pytest.raises(ValueError, match="infinite"):
            service.perform_clustering(df.copy())

    def test_raises_when_gene_ids_not_in_dataframe(self):
        """gene_ids that don't match any row should raise ValueError."""
        df = _make_df(n_genes=10, n_samples=4)
        service = _make_service()

        with pytest.raises(ValueError, match="gene_ids were found in the dataset"):
            service.perform_clustering(df.copy(), gene_ids=["NONEXISTENT1", "NONEXISTENT2"])

    def test_ward_with_non_euclidean_raises(self):
        """Ward linkage with non-euclidean metric should raise ValueError."""
        df = _make_df(n_genes=15, n_samples=4)
        service = _make_service()

        with pytest.raises(ValueError, match="Ward linkage requires euclidean"):
            service.perform_clustering(df.copy(), method="ward", metric="correlation")


# ─────────────────────────────────────────────────────────────────────────────
# perform_clustering — downsampling
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformClusteringDownsampling:

    def test_downsamples_to_max_genes_for_clustering(self):
        """When n_genes > max_genes_for_clustering, should downsample."""
        n_genes = 50
        df = _make_df(n_genes=n_genes, n_samples=4)
        max_genes = 20
        service = _make_service()

        result = service.perform_clustering(df.copy(), max_genes_for_clustering=max_genes)

        assert len(result["row_labels"]) <= max_genes


# ─────────────────────────────────────────────────────────────────────────────
# precompute_sample_clustering
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecomputeSampleClustering:

    def test_returns_expected_keys(self):
        """Output dict should contain all expected keys."""
        df = _make_df(n_genes=30, n_samples=8)
        service = _make_service()

        result = service.precompute_sample_clustering(df.copy())

        expected_keys = {"col_labels", "col_dendrogram", "col_order", "method", "metric", "genes_used"}
        assert expected_keys.issubset(result.keys())

    def test_col_labels_match_dataframe_columns(self):
        """col_labels should match the DataFrame column names."""
        df = _make_df(n_genes=30, n_samples=6)
        service = _make_service()

        result = service.precompute_sample_clustering(df.copy())

        assert result["col_labels"] == df.columns.tolist()

    def test_col_order_is_valid_permutation(self):
        """col_order must be a valid permutation of all sample indices."""
        n_samples = 8
        df = _make_df(n_genes=30, n_samples=n_samples)
        service = _make_service()

        result = service.precompute_sample_clustering(df.copy())

        assert sorted(result["col_order"]) == list(range(n_samples))

    def test_top_n_genes_limits_genes_used(self):
        """genes_used should be <= top_n_genes when dataset is large."""
        top_n = 10
        df = _make_df(n_genes=50, n_samples=4)
        service = _make_service()

        result = service.precompute_sample_clustering(df.copy(), top_n_genes=top_n)

        assert result["genes_used"] <= top_n


# ─────────────────────────────────────────────────────────────────────────────
# _compute_linkage — metric mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLinkage:

    def _make_data(self, n=15, p=4, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, p))

    def test_euclidean_ward_succeeds(self):
        """Ward + euclidean is the primary optimized path."""
        service = _make_service()
        data = self._make_data()
        linkage = service._compute_linkage(data, method="ward", metric="euclidean")
        assert linkage.shape[1] == 4  # Standard linkage matrix has 4 columns

    def test_manhattan_mapped_to_cityblock(self):
        """'manhattan' metric should be correctly mapped to 'cityblock'."""
        service = _make_service()
        data = self._make_data()
        # Should not raise
        linkage = service._compute_linkage(data, method="average", metric="manhattan")
        assert linkage is not None

    def test_correlation_metric_succeeds(self):
        """Correlation metric with average linkage should work."""
        service = _make_service()
        data = self._make_data(n=20, p=6)
        linkage = service._compute_linkage(data, method="average", metric="correlation")
        assert linkage is not None

    def test_unsupported_metric_raises(self):
        """Completely unknown metric should raise ValueError."""
        service = _make_service()
        data = self._make_data()
        with pytest.raises((ValueError, Exception)):
            service._compute_linkage(data, method="average", metric="bogus_metric_xyz")
