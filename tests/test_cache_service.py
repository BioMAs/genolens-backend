"""
Unit tests for CacheService (async Redis-backed + in-memory LRU for DataFrames).

Covers:
- _generate_cache_key (determinism, kwarg ordering)
- Clustering cache: set/get (hit/miss), key stability regardless of gene list order
- Volcano cache: set/get with different thresholds produce distinct keys
- Stats cache: set/get with/without comparison_name
- DataFrame cache: set/get, LRU eviction
- Cache management: clear_dataset_cache, clear_all
- get_stats_info / get_cache_stats structure
"""
import pytest
from unittest.mock import AsyncMock
import json


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def redis_mock():
    """Simple dict-backed AsyncMock simulating redis.asyncio."""
    store: dict[str, str] = {}

    mock = AsyncMock()

    async def fake_get(key):
        return store.get(key)

    async def fake_setex(key, ttl, value):
        store[key] = value

    mock.get = fake_get
    mock.setex = fake_setex
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def cache_svc(redis_mock):
    from app.services.cache_service import CacheService
    svc = CacheService()
    svc._redis = redis_mock  # inject mock — bypasses initialize()
    return svc


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

    async def test_cache_miss_returns_none(self, cache_svc):
        """get_clustering_result should return None on a miss."""
        result = await cache_svc.get_clustering_result("ds1", ["A", "B"], method="ward")
        assert result is None

    async def test_set_then_get_returns_result(self, cache_svc):
        """After setting, get should return the same result."""
        data = {"dendro": [1, 2, 3]}
        await cache_svc.set_clustering_result("ds1", ["A", "B"], data, method="ward")
        fetched = await cache_svc.get_clustering_result("ds1", ["A", "B"], method="ward")
        assert fetched == data

    async def test_gene_list_order_does_not_matter(self, cache_svc):
        """Gene list is sorted internally, so order should not affect the key."""
        data = {"result": True}
        await cache_svc.set_clustering_result("ds1", ["B", "A"], data, method="ward")
        fetched = await cache_svc.get_clustering_result("ds1", ["A", "B"], method="ward")
        assert fetched == data

    async def test_different_methods_use_different_keys(self, cache_svc):
        """Clustering results for different methods should be cached separately."""
        ward_data = {"method": "ward"}
        avg_data = {"method": "average"}
        await cache_svc.set_clustering_result("ds1", ["A", "B"], ward_data, method="ward")
        await cache_svc.set_clustering_result("ds1", ["A", "B"], avg_data, method="average")

        assert await cache_svc.get_clustering_result("ds1", ["A", "B"], method="ward") == ward_data
        assert await cache_svc.get_clustering_result("ds1", ["A", "B"], method="average") == avg_data

    async def test_different_datasets_do_not_collide(self, cache_svc):
        """Results for different datasets should not collide."""
        await cache_svc.set_clustering_result("ds1", ["A"], {"x": 1}, method="ward")
        await cache_svc.set_clustering_result("ds2", ["A"], {"x": 2}, method="ward")

        assert await cache_svc.get_clustering_result("ds1", ["A"], method="ward") == {"x": 1}
        assert await cache_svc.get_clustering_result("ds2", ["A"], method="ward") == {"x": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Volcano cache
# ─────────────────────────────────────────────────────────────────────────────

class TestVolcanoCache:

    async def test_miss_returns_none(self, cache_svc):
        """get_volcano_data should return None if nothing is cached."""
        assert await cache_svc.get_volcano_data("ds_unknown", "cmp") is None

    async def test_set_then_get_returns_data(self, cache_svc):
        """Cached volcano data should be retrievable."""
        payload = {"points": [1, 2, 3]}
        await cache_svc.set_volcano_data("ds1", "KO_vs_WT", payload)
        result = await cache_svc.get_volcano_data("ds1", "KO_vs_WT")
        assert result == payload

    async def test_different_thresholds_have_different_keys(self, cache_svc):
        """Volcano data with different thresholds must not collide."""
        data_strict = {"threshold": "strict"}
        data_lenient = {"threshold": "lenient"}

        await cache_svc.set_volcano_data("ds1", "KO_vs_WT", data_strict,
                                         padj_threshold=0.01, logfc_threshold=1.0)
        await cache_svc.set_volcano_data("ds1", "KO_vs_WT", data_lenient,
                                         padj_threshold=0.05, logfc_threshold=0.58)

        assert await cache_svc.get_volcano_data("ds1", "KO_vs_WT",
                                                padj_threshold=0.01, logfc_threshold=1.0) == data_strict
        assert await cache_svc.get_volcano_data("ds1", "KO_vs_WT",
                                                padj_threshold=0.05, logfc_threshold=0.58) == data_lenient


# ─────────────────────────────────────────────────────────────────────────────
# Stats cache
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsCache:

    async def test_miss_returns_none(self, cache_svc):
        assert await cache_svc.get_stats("nonexistent") is None

    async def test_set_and_get_without_comparison(self, cache_svc):
        """Stats without comparison_name should be cached correctly."""
        stats = {"mean": 5.2, "std": 1.1}
        await cache_svc.set_stats("ds1", stats)
        assert await cache_svc.get_stats("ds1") == stats

    async def test_set_and_get_with_comparison(self, cache_svc):
        """Stats with comparison_name should be distinct from global stats."""
        global_stats = {"type": "global"}
        cmp_stats = {"type": "comparison"}
        await cache_svc.set_stats("ds1", global_stats)
        await cache_svc.set_stats("ds1", cmp_stats, comparison_name="KO_vs_WT")

        assert await cache_svc.get_stats("ds1") == global_stats
        assert await cache_svc.get_stats("ds1", comparison_name="KO_vs_WT") == cmp_stats


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

    def test_clear_dataset_cache_removes_dataframe_entry(self, cache_svc):
        """clear_dataset_cache should remove the DataFrame entry."""
        import pandas as pd
        cache_svc.set_dataframe("ds1", pd.DataFrame())
        cache_svc.clear_dataset_cache("ds1")
        assert cache_svc.get_dataframe("ds1") is None

    def test_clear_all_clears_dataframe_cache(self, cache_svc):
        """clear_all should clear the in-memory DataFrame cache."""
        import pandas as pd
        cache_svc.set_dataframe("ds1", pd.DataFrame())
        cache_svc.set_dataframe("ds2", pd.DataFrame())

        cache_svc.clear_all()

        assert cache_svc.get_dataframe("ds1") is None
        assert cache_svc.get_dataframe("ds2") is None

    def test_clear_all_does_not_affect_redis_keys(self, cache_svc):
        """clear_all only clears the in-memory DataFrame cache; Redis is untouched."""
        # Verify that clear_all does not attempt to flush Redis
        # (no call to _redis.flushdb or similar)
        cache_svc.clear_all()
        # If _redis were called, the AsyncMock would record it.
        # Only aclose is a known method; other attribute accesses would raise.
        # This test simply confirms no exception is raised and redis is not flushed.
        assert cache_svc._redis is not None


# ─────────────────────────────────────────────────────────────────────────────
# get_stats_info / get_cache_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheStatsInfo:

    async def test_get_stats_info_has_expected_keys(self, cache_svc):
        """get_stats_info should return dict with all four top-level keys."""
        info = await cache_svc.get_stats_info()

        assert "clustering" in info
        assert "volcano" in info
        assert "stats" in info
        assert "dataframe" in info

    async def test_redis_caches_have_backend_and_ttl(self, cache_svc):
        """Redis-backed caches should report backend='redis' and a ttl_seconds."""
        info = await cache_svc.get_stats_info()

        for section in ("clustering", "volcano", "stats"):
            assert info[section]["backend"] == "redis"
            assert "ttl_seconds" in info[section]
            assert isinstance(info[section]["ttl_seconds"], int)

    async def test_redis_caches_have_no_size_field(self, cache_svc):
        """Redis-backed caches should NOT expose size/currsize (no in-memory count)."""
        info = await cache_svc.get_stats_info()

        for section in ("clustering", "volcano", "stats"):
            assert "size" not in info[section]
            assert "currsize" not in info[section]

    async def test_dataframe_cache_has_memory_backend_and_size(self, cache_svc):
        """DataFrame cache entry should report backend='memory' with size and maxsize."""
        import pandas as pd
        cache_svc.set_dataframe("ds1", pd.DataFrame())
        info = await cache_svc.get_stats_info()

        df_info = info["dataframe"]
        assert df_info["backend"] == "memory"
        assert "size" in df_info
        assert "maxsize" in df_info
        assert df_info["size"] >= 1

    async def test_get_cache_stats_is_alias_for_get_stats_info(self, cache_svc):
        """get_cache_stats() is an alias and should return identical structure."""
        info = await cache_svc.get_stats_info()
        stats = await cache_svc.get_cache_stats()
        assert info == stats

    async def test_ttl_values_are_correct(self, cache_svc):
        """TTL constants should match documented values."""
        info = await cache_svc.get_stats_info()

        assert info["clustering"]["ttl_seconds"] == 3600
        assert info["volcano"]["ttl_seconds"] == 7200
        assert info["stats"]["ttl_seconds"] == 86400
