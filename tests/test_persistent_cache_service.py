"""
Unit tests for PersistentCacheService (DB-backed cache).

Covers:
- _generate_params_hash (determinism, order-independence)
- get_cached (miss / hit / expired entry)
- set_cached (insert new / update existing)
- invalidate_dataset (deletes all entries for dataset)
- cleanup_expired (deletes only expired entries)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from tests.conftest import TEST_DATASET_ID, make_cached_computation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_one_or_none(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _rowcount_result(count: int):
    r = MagicMock()
    r.rowcount = count
    return r


# ─────────────────────────────────────────────────────────────────────────────
# _generate_params_hash
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateParamsHash:

    def test_same_params_same_hash(self):
        """Same parameters must produce the same hash."""
        from app.services.persistent_cache_service import PersistentCacheService
        svc = PersistentCacheService()
        params = {"method": "ward", "n_genes": 500}
        h1 = svc._generate_params_hash(params)
        h2 = svc._generate_params_hash(params)
        assert h1 == h2

    def test_different_params_different_hash(self):
        """Different parameters should yield different hashes."""
        from app.services.persistent_cache_service import PersistentCacheService
        svc = PersistentCacheService()
        h1 = svc._generate_params_hash({"method": "ward"})
        h2 = svc._generate_params_hash({"method": "average"})
        assert h1 != h2

    def test_key_order_does_not_matter(self):
        """Dict key order must not affect the hash (uses sort_keys=True)."""
        from app.services.persistent_cache_service import PersistentCacheService
        svc = PersistentCacheService()
        h1 = svc._generate_params_hash({"a": 1, "b": 2})
        h2 = svc._generate_params_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_returns_32_char_hex(self):
        """Hash should be a 32-character MD5 hex string."""
        from app.services.persistent_cache_service import PersistentCacheService
        svc = PersistentCacheService()
        h = svc._generate_params_hash({"x": 42})
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# get_cached
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCached:

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mock_db):
        """Should return None when no cached entry exists."""
        from app.services.persistent_cache_service import PersistentCacheService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        svc = PersistentCacheService()
        result = await svc.get_cached(
            mock_db, TEST_DATASET_ID, "clustering", {"method": "ward"}
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_result_data(self, mock_db):
        """Should return result_data from a valid (non-expired) cache entry."""
        from app.services.persistent_cache_service import PersistentCacheService

        cached = make_cached_computation(is_expired=False, hit_count=0)
        mock_db.execute.return_value = _scalar_one_or_none(cached)

        svc = PersistentCacheService()
        result = await svc.get_cached(
            mock_db, TEST_DATASET_ID, "clustering", {"method": "ward"}
        )

        assert result == cached.result_data
        assert cached.hit_count == 1  # incremented
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none_and_deletes(self, mock_db):
        """An expired cache entry should be deleted and None returned."""
        from app.services.persistent_cache_service import PersistentCacheService

        cached = make_cached_computation(is_expired=True)
        mock_db.execute.return_value = _scalar_one_or_none(cached)
        mock_db.delete = AsyncMock()

        svc = PersistentCacheService()
        result = await svc.get_cached(
            mock_db, TEST_DATASET_ID, "clustering", {"method": "ward"}
        )

        assert result is None
        mock_db.delete.assert_awaited_once_with(cached)
        mock_db.commit.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# set_cached
# ─────────────────────────────────────────────────────────────────────────────

class TestSetCached:

    @pytest.mark.asyncio
    async def test_creates_new_entry_when_not_exists(self, mock_db):
        """Should add a new CachedComputation when none exists."""
        from app.services.persistent_cache_service import PersistentCacheService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        svc = PersistentCacheService()
        await svc.set_cached(
            mock_db,
            dataset_id=TEST_DATASET_ID,
            computation_type="volcano",
            params={"comparison": "KO_vs_WT"},
            result_data={"points": [1, 2, 3]},
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updates_existing_entry(self, mock_db):
        """Should update result_data when an entry already exists."""
        from app.services.persistent_cache_service import PersistentCacheService

        existing = make_cached_computation()
        existing.result_data = {"old": True}
        mock_db.execute.return_value = _scalar_one_or_none(existing)

        svc = PersistentCacheService()
        await svc.set_cached(
            mock_db,
            dataset_id=TEST_DATASET_ID,
            computation_type="clustering",
            params={"method": "ward"},
            result_data={"new": True},
        )

        assert existing.result_data == {"new": True}
        mock_db.add.assert_not_called()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_cached_with_ttl(self, mock_db):
        """expires_at should be set when ttl_seconds is provided."""
        from app.services.persistent_cache_service import PersistentCacheService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        svc = PersistentCacheService()
        await svc.set_cached(
            mock_db,
            dataset_id=TEST_DATASET_ID,
            computation_type="stats",
            params={},
            result_data={"stat": 1},
            ttl_seconds=3600,
        )

        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.expires_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# invalidate_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidateDataset:

    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, mock_db):
        """Should return number of deleted cache entries."""
        from app.services.persistent_cache_service import PersistentCacheService

        mock_db.execute.return_value = _rowcount_result(3)

        svc = PersistentCacheService()
        count = await svc.invalidate_dataset(mock_db, TEST_DATASET_ID)

        assert count == 3
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_to_delete(self, mock_db):
        """Should return 0 when no entries exist for dataset."""
        from app.services.persistent_cache_service import PersistentCacheService

        mock_db.execute.return_value = _rowcount_result(0)

        svc = PersistentCacheService()
        count = await svc.invalidate_dataset(mock_db, uuid4())

        assert count == 0
