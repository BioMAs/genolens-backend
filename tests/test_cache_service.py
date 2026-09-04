"""
Unit tests for CacheService (in-memory TTLCache / LRUCache).

Covers:
- _generate_cache_key (determinism, kwarg ordering)
- Clustering cache: set/get (hit/miss), key stability regardless of gene list order
- Volcano cache: set/get with different thresholds produce distinct keys
- Stats cache: set/get with/without comparison_name
- DataFrame cache: set/get, LRU eviction
- Cache management: clear_dataset_cache, clear_all, get_cache_stats
"""
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# _generate_cache_key
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateCacheKey:

    def test_same_args_produce_same_key(self):
        """Same input should always produce the same hash."""
        from app.services.cache_service import CacheService
        key1 = CacheService._generate_cache_key("ds1", ["A", "B"], "ward")
        key2 = CacheService._generate_cache_key("ds1", ["A", "B"], "ward")
        assert key1 == key2

    def test_different_args_produce_different_keys(self):
        """Different inputs should yield different hashes."""
        from app.services.cache_service import CacheService
        key1 = CacheService._generate_cache_key("ds1", "cmp1")
        key2 = CacheService._generate_cache_key("ds2", "cmp1")
        assert key1 != key2

    def test_kwargs_ordering_does_not_affect_key(self):
        """Keyword arguments should be order-independent."""
        from app.services.cache_service import CacheService
        key1 = CacheService._generate_cache_key("ds1", alpha=0.05, beta=2)
        key2 = CacheService._generate_cache_key("ds1", beta=2, alpha=0.05)
        assert key1 == key2

    def test_returns_32_char_hex_string(self):
        """Result should be a 32-character MD5 hex digest."""
        from app.services.cache_service import CacheService
        key = CacheService._generate_cache_key("test")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


# ─────────────────────────────────────────────────────────────────────────────
# Clustering cache
# ─────────────────────────────────────────────────────────────────────────────

class TestClusteringCache:

    def _make_service(self, **kwargs):
        from app.services.cache_service import CacheService
        return CacheService(clustering_ttl_seconds=3600, **kwargs)

    def test_cache_miss_returns_none(self):
        """get_clustering_result should return None on a miss."""
        svc = self._make_service()
        result = svc.get_clustering_result("ds1", ["A", "B"], method="ward")
        assert result is None

    def test_set_then_get_returns_result(self):
        """After setting, get should return the same result."""
        svc = self._make_service()
        data = {"dendro": [1, 2, 3]}
        svc.set_clustering_result("ds1", ["A", "B"], data, method="ward")
        fetched = svc.get_clustering_result("ds1", ["A", "B"], method="ward")
        assert fetched == data

    def test_gene_list_order_does_not_matter(self):
        """Gene list is sorted internally, so order should not affect the key."""
        svc = self._make_service()
        data = {"result": True}
        svc.set_clustering_result("ds1", ["B", "A"], data, method="ward")
        fetched = svc.get_clustering_result("ds1", ["A", "B"], method="ward")
        assert fetched == data

    def test_different_methods_use_different_keys(self):
        """Clustering results for different methods should be cached separately."""
        svc = self._make_service()
        ward_data = {"method": "ward"}
        avg_data = {"method": "average"}
        svc.set_clustering_result("ds1", ["A", "B"], ward_data, method="ward")
        svc.set_clustering_result("ds1", ["A", "B"], avg_data, method="average")

        assert svc.get_clustering_result("ds1", ["A", "B"], method="ward") == ward_data
        assert svc.get_clustering_result("ds1", ["A", "B"], method="average") == avg_data

    def test_different_datasets_do_not_collide(self):
        """Results for different datasets should not collide."""
        svc = self._make_service()
        svc.set_clustering_result("ds1", ["A"], {"x": 1}, method="ward")
        svc.set_clustering_result("ds2", ["A"], {"x": 2}, method="ward")

        assert svc.get_clustering_result("ds1", ["A"], method="ward") == {"x": 1}
        assert svc.get_clustering_result("ds2", ["A"], method="ward") == {"x": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Volcano cache
# ─────────────────────────────────────────────────────────────────────────────

class TestVolcanoCache:

    def _make_service(self):
        from app.services.cache_service import CacheService
        return CacheService()

    def test_set_then_get_returns_data(self):
        """Cached volcano data should be retrievable."""
        svc = self._make_service()
        payload = {"points": [1, 2, 3]}
        svc.set_volcano_data("ds1", "KO_vs_WT", payload)
        result = svc.get_volcano_data("ds1", "KO_vs_WT")
        assert result == payload

    def test_different_thresholds_have_different_keys(self):
        """Volcano data with different thresholds must not collide."""
        svc = self._make_service()
        data_strict = {"threshold": "strict"}
        data_lenient = {"threshold": "lenient"}

        svc.set_volcano_data("ds1", "KO_vs_WT", data_strict,
                             padj_threshold=0.01, logfc_threshold=1.0)
        svc.set_volcano_data("ds1", "KO_vs_WT", data_lenient,
                             padj_threshold=0.05, logfc_threshold=0.58)

        assert svc.get_volcano_data("ds1", "KO_vs_WT",
                                    padj_threshold=0.01, logfc_threshold=1.0) == data_strict
        assert svc.get_volcano_data("ds1", "KO_vs_WT",
                                    padj_threshold=0.05, logfc_threshold=0.58) == data_lenient

    def test_miss_returns_none(self):
        """get_volcano_data should return None if nothing is cached."""
        svc = self._make_service()
        assert svc.get_volcano_data("ds_unknown", "cmp") is None


# ─────────────────────────────────────────────────────────────────────────────
# Stats cache
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsCache:

    def _make_service(self):
        from app.services.cache_service import CacheService
        return CacheService()

    def test_set_and_get_without_comparison(self):
        """Stats without comparison_name should be cached correctly."""
        svc = self._make_service()
        stats = {"mean": 5.2, "std": 1.1}
        svc.set_stats("ds1", stats)
        assert svc.get_stats("ds1") == stats

    def test_set_and_get_with_comparison(self):
        """Stats with comparison_name should be distinct from global stats."""
        svc = self._make_service()
        global_stats = {"type": "global"}
        cmp_stats = {"type": "comparison"}
        svc.set_stats("ds1", global_stats)
        svc.set_stats("ds1", cmp_stats, comparison_name="KO_vs_WT")

        assert svc.get_stats("ds1") == global_stats
        assert svc.get_stats("ds1", comparison_name="KO_vs_WT") == cmp_stats

    def test_miss_returns_none(self):
        svc = self._make_service()
        assert svc.get_stats("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame cache
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFrameCache:

    def _make_service(self):
        from app.services.cache_service import CacheService
        return CacheService(dataframe_cache_size=3)

    def test_set_and_get(self):
        """A DataFrame stored under a dataset ID should be retrievable."""
        import pandas as pd
        svc = self._make_service()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        svc.set_dataframe("ds1", df)
        fetched = svc.get_dataframe("ds1")
        assert fetched is df

    def test_miss_returns_none(self):
        svc = self._make_service()
        assert svc.get_dataframe("missing_ds") is None

    def test_lru_eviction_on_overflow(self):
        """When maxsize is exceeded, least recently used entry is evicted."""
        import pandas as pd
        svc = self._make_service()  # maxsize=3
        for i in range(4):
            svc.set_dataframe(f"ds{i}", pd.DataFrame({"v": [i]}))

        # One of the 4 should have been evicted (LRU)
        cached = sum(1 for i in range(4) if svc.get_dataframe(f"ds{i}") is not None)
        assert cached == 3


# ─────────────────────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheManagement:

    def _make_service(self):
        from app.services.cache_service import CacheService
        return CacheService()

    def test_clear_dataset_cache_removes_dataframe_entry(self):
        """clear_dataset_cache should remove the DataFrame entry."""
        import pandas as pd
        svc = self._make_service()
        svc.set_dataframe("ds1", pd.DataFrame())
        svc.clear_dataset_cache("ds1")
        assert svc.get_dataframe("ds1") is None

    def test_clear_dataset_cache_evicts_keyed_caches(self):
        """
        The volcano, clustering and stats entries must go too.

        These keys are MD5 hashes, so clear_dataset_cache had no way to find them and left them
        behind: a reprocessed dataset kept serving its stale volcano cloud until the TTL expired.
        """
        svc = self._make_service()
        svc.set_clustering_result("ds1", ["A"], {"x": 1})
        svc.set_volcano_data("ds1", "cmp", {"y": 2})
        svc.set_stats("ds1", {"z": 3})

        svc.clear_dataset_cache("ds1")

        assert svc.get_clustering_result("ds1", ["A"]) is None
        assert svc.get_volcano_data("ds1", "cmp") is None
        assert svc.get_stats("ds1") is None

    def test_clear_dataset_cache_spares_other_datasets(self):
        """Eviction is scoped to one dataset — a prefix scan, not a flush."""
        svc = self._make_service()
        svc.set_volcano_data("ds1", "cmp", {"y": 1})
        svc.set_volcano_data("ds2", "cmp", {"y": 2})
        svc.set_stats("ds2", {"z": 3})

        svc.clear_dataset_cache("ds1")

        assert svc.get_volcano_data("ds1", "cmp") is None
        assert svc.get_volcano_data("ds2", "cmp") == {"y": 2}
        assert svc.get_stats("ds2") == {"z": 3}

    def test_clear_dataset_cache_keeps_other_thresholds_of_same_dataset_out(self):
        """All threshold variants of a dataset's volcano go, not just the default one."""
        svc = self._make_service()
        svc.set_volcano_data("ds1", "cmp", {"y": 1}, padj_threshold=0.05, logfc_threshold=0.58)
        svc.set_volcano_data("ds1", "cmp", {"y": 2}, padj_threshold=0.01, logfc_threshold=1.0)

        svc.clear_dataset_cache("ds1")

        assert svc.get_volcano_data("ds1", "cmp", padj_threshold=0.05, logfc_threshold=0.58) is None
        assert svc.get_volcano_data("ds1", "cmp", padj_threshold=0.01, logfc_threshold=1.0) is None

    def test_clear_all_empties_all_caches(self):
        """clear_all should remove all entries from all caches."""
        import pandas as pd
        svc = self._make_service()
        svc.set_clustering_result("ds1", ["A"], {"x": 1})
        svc.set_volcano_data("ds1", "cmp", {"y": 2})
        svc.set_stats("ds1", {"z": 3})
        svc.set_dataframe("ds1", pd.DataFrame())

        svc.clear_all()

        assert svc.get_clustering_result("ds1", ["A"]) is None
        assert svc.get_volcano_data("ds1", "cmp") is None
        assert svc.get_stats("ds1") is None
        assert svc.get_dataframe("ds1") is None

    def test_get_cache_stats_structure(self):
        """get_cache_stats should return dict with expected top-level keys."""
        svc = self._make_service()
        stats = svc.get_cache_stats()

        assert "clustering" in stats
        assert "volcano" in stats
        assert "stats" in stats
        assert "dataframe" in stats

        # Each sub-dict should have size information
        for section in ("clustering", "volcano", "stats"):
            assert "size" in stats[section]
            assert "maxsize" in stats[section]

    def test_get_cache_stats_reflects_content(self):
        """After inserting an entry, size should increase."""
        svc = self._make_service()
        svc.set_clustering_result("ds1", ["G1"], {"r": 1})

        stats = svc.get_cache_stats()
        assert stats["clustering"]["size"] >= 1
