"""
Cache service for optimizing data retrieval and computations.

Provides:
- Redis-backed async TTL caching for clustering, volcano, and stats results
  (shared across API instances for horizontal scaling)
- In-memory LRU caching for DataFrames (not serializable to Redis efficiently)
"""
import logging
import hashlib
import json
from typing import Any, Optional
from urllib.parse import urlparse
from cachetools import LRUCache

import redis.asyncio

logger = logging.getLogger(__name__)

# TTL constants (seconds)
_TTL_CLUSTERING = 3600    # 1 hour
_TTL_VOLCANO = 7200       # 2 hours
_TTL_STATS = 86400        # 24 hours


class CacheService:
    """
    Service for caching expensive computations and data retrievals.

    TTL caches (clustering, volcano, stats) are backed by Redis and are
    therefore shared across all API worker processes.

    The DataFrame cache remains in-memory (LRU) because pandas DataFrames
    cannot be serialized to Redis efficiently.
    """

    def __init__(self, dataframe_cache_size: int = 5):
        """
        Initialize the cache service.

        Args:
            dataframe_cache_size: Max number of DataFrames to keep in memory.
        """
        # DataFrame cache — in-memory LRU, no TTL
        self.dataframe_cache: LRUCache = LRUCache(maxsize=dataframe_cache_size)

        # Redis client — set by initialize()
        self._redis: redis.asyncio.Redis | None = None

        logger.info(
            "CacheService created (Redis not yet connected). "
            "Call initialize(redis_url) before use."
        )

    # ===== Lifecycle =====

    async def initialize(self, redis_url: str) -> None:
        """
        Create the Redis connection and verify connectivity.

        Args:
            redis_url: Redis connection URL, e.g. ``redis://redis:6379/0``.
        """
        parsed = urlparse(redis_url)
        safe_url = parsed._replace(netloc=f"{parsed.hostname}:{parsed.port}").geturl()

        client = redis.asyncio.from_url(redis_url, decode_responses=True)
        try:
            await client.ping()
            self._redis = client
            logger.info("CacheService connected to Redis at %s", safe_url)
        except redis.asyncio.RedisError as exc:
            logger.error(
                "CacheService could not reach Redis at %s: %s — cache disabled", safe_url, exc
            )
            await client.aclose()

    async def close(self) -> None:
        """Close the Redis connection if it is open."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("CacheService Redis connection closed")

    # ===== Internal helpers =====

    @staticmethod
    def _generate_cache_key(*args, **kwargs) -> str:
        """
        Generate a stable MD5 cache key from the given arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            str: MD5 hex digest of the serialized arguments.
        """
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    @staticmethod
    def _redis_key(cache_type: str, hash_: str) -> str:
        """Build the full Redis key: ``genolens:{cache_type}:{hash}``."""
        return f"genolens:{cache_type}:{hash_}"

    async def _redis_get(self, key: str) -> Optional[dict]:
        """Fetch and JSON-decode a value from Redis; returns None on miss, error, or if Redis is unavailable."""
        if self._redis is None:
            logger.warning("Redis not initialized — cache get skipped for key: %s", key)
            return None
        try:
            raw = await self._redis.get(key)
        except redis.asyncio.RedisError as exc:
            logger.warning("Redis get failed for key %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Cache value for key %s is not valid JSON: %s", key, exc)
            return None

    async def _redis_set(self, key: str, value: dict, ttl: int) -> None:
        """JSON-encode and store a value in Redis with the given TTL; silently skips on error or if unavailable."""
        if self._redis is None:
            logger.warning("Redis not initialized — cache set skipped for key: %s", key)
            return
        try:
            await self._redis.setex(key, ttl, json.dumps(value))
        except redis.asyncio.RedisError as exc:
            logger.warning("Redis set failed for key %s: %s", key, exc)

    # ===== Clustering Cache =====

    async def get_clustering_result(
        self,
        dataset_id: str,
        gene_list: list[str],
        method: str = "hierarchical",
        **params,
    ) -> Optional[dict]:
        """
        Get cached clustering result.

        Args:
            dataset_id: Dataset identifier.
            gene_list: List of gene IDs.
            method: Clustering method.
            **params: Additional clustering parameters.

        Returns:
            Cached result dict, or None if not found.
        """
        cache_key = self._generate_cache_key(
            dataset_id, tuple(sorted(gene_list)), method, **params
        )
        redis_key = self._redis_key("clustering", cache_key)
        result = await self._redis_get(redis_key)

        if result is not None:
            logger.debug(
                "Clustering cache HIT for %s... (%d genes)", dataset_id[:8], len(gene_list)
            )
        else:
            logger.debug(
                "Clustering cache MISS for %s... (%d genes)", dataset_id[:8], len(gene_list)
            )

        return result

    async def set_clustering_result(
        self,
        dataset_id: str,
        gene_list: list[str],
        result: dict,
        method: str = "hierarchical",
        **params,
    ) -> None:
        """
        Cache clustering result.

        Args:
            dataset_id: Dataset identifier.
            gene_list: List of gene IDs.
            result: Clustering result to cache.
            method: Clustering method.
            **params: Additional clustering parameters.
        """
        cache_key = self._generate_cache_key(
            dataset_id, tuple(sorted(gene_list)), method, **params
        )
        redis_key = self._redis_key("clustering", cache_key)
        await self._redis_set(redis_key, result, _TTL_CLUSTERING)
        logger.debug(
            "Clustering cache SET for %s... (%d genes)", dataset_id[:8], len(gene_list)
        )

    # ===== Volcano Plot Cache =====

    async def get_volcano_data(
        self,
        dataset_id: str,
        comparison_name: str,
        max_points: int = 5000,
        padj_threshold: float = 0.05,
        logfc_threshold: float = 0.58,
    ) -> Optional[dict]:
        """
        Get cached volcano plot data.

        Args:
            dataset_id: Dataset identifier.
            comparison_name: Comparison name.
            max_points: Maximum points to include.
            padj_threshold: P-value threshold for significance.
            logfc_threshold: Log fold change threshold for significance.

        Returns:
            Cached result dict, or None if not found.
        """
        cache_key = self._generate_cache_key(
            dataset_id, comparison_name, max_points, padj_threshold, logfc_threshold
        )
        redis_key = self._redis_key("volcano", cache_key)
        result = await self._redis_get(redis_key)

        if result is not None:
            logger.debug(
                "Volcano cache HIT for %s.../%s (padj<%s, |logFC|>%s)",
                dataset_id[:8], comparison_name, padj_threshold, logfc_threshold,
            )
        else:
            logger.debug(
                "Volcano cache MISS for %s.../%s (padj<%s, |logFC|>%s)",
                dataset_id[:8], comparison_name, padj_threshold, logfc_threshold,
            )

        return result

    async def set_volcano_data(
        self,
        dataset_id: str,
        comparison_name: str,
        result: dict,
        max_points: int = 5000,
        padj_threshold: float = 0.05,
        logfc_threshold: float = 0.58,
    ) -> None:
        """
        Cache volcano plot data.

        Args:
            dataset_id: Dataset identifier.
            comparison_name: Comparison name.
            result: Volcano plot data to cache.
            max_points: Maximum points included.
            padj_threshold: P-value threshold used.
            logfc_threshold: Log fold change threshold used.
        """
        cache_key = self._generate_cache_key(
            dataset_id, comparison_name, max_points, padj_threshold, logfc_threshold
        )
        redis_key = self._redis_key("volcano", cache_key)
        await self._redis_set(redis_key, result, _TTL_VOLCANO)
        logger.debug(
            "Volcano cache SET for %s.../%s (padj<%s, |logFC|>%s)",
            dataset_id[:8], comparison_name, padj_threshold, logfc_threshold,
        )

    # ===== Statistics Cache =====

    async def get_stats(
        self,
        dataset_id: str,
        comparison_name: Optional[str] = None,
        **params,
    ) -> Optional[dict]:
        """
        Get cached statistics.

        Args:
            dataset_id: Dataset identifier.
            comparison_name: Optional comparison name.
            **params: Additional parameters.

        Returns:
            Cached result dict, or None if not found.
        """
        cache_key = self._generate_cache_key(dataset_id, comparison_name, **params)
        redis_key = self._redis_key("stats", cache_key)
        result = await self._redis_get(redis_key)

        if result is not None:
            logger.debug("Stats cache HIT for %s...", dataset_id[:8])
        else:
            logger.debug("Stats cache MISS for %s...", dataset_id[:8])

        return result

    async def set_stats(
        self,
        dataset_id: str,
        result: dict,
        comparison_name: Optional[str] = None,
        **params,
    ) -> None:
        """
        Cache statistics result.

        Args:
            dataset_id: Dataset identifier.
            result: Statistics to cache.
            comparison_name: Optional comparison name.
            **params: Additional parameters.
        """
        cache_key = self._generate_cache_key(dataset_id, comparison_name, **params)
        redis_key = self._redis_key("stats", cache_key)
        await self._redis_set(redis_key, result, _TTL_STATS)
        logger.debug("Stats cache SET for %s...", dataset_id[:8])

    # ===== DataFrame Cache =====

    def get_dataframe(self, dataset_id: str) -> Optional[Any]:
        """
        Get cached DataFrame (for hot datasets).

        Args:
            dataset_id: Dataset identifier.

        Returns:
            Cached DataFrame or None if not found.
        """
        result = self.dataframe_cache.get(dataset_id)

        if result is not None:
            logger.debug("DataFrame cache HIT for %s...", dataset_id[:8])
        else:
            logger.debug("DataFrame cache MISS for %s...", dataset_id[:8])

        return result

    def set_dataframe(self, dataset_id: str, dataframe: Any) -> None:
        """
        Cache DataFrame for a hot dataset.

        Args:
            dataset_id: Dataset identifier.
            dataframe: pandas DataFrame to cache.
        """
        self.dataframe_cache[dataset_id] = dataframe
        logger.debug("DataFrame cache SET for %s...", dataset_id[:8])

    # ===== Cache Management =====

    def clear_dataset_cache(self, dataset_id: str) -> None:
        """
        Clear in-memory DataFrame cache entry for a specific dataset.

        Redis TTL-based entries will expire naturally; no key scan is performed.

        Args:
            dataset_id: Dataset identifier.
        """
        if dataset_id in self.dataframe_cache:
            del self.dataframe_cache[dataset_id]
            logger.info("Cleared DataFrame cache for %s...", dataset_id[:8])

        logger.info(
            "Redis cache entries for dataset %s... will expire naturally (TTL-based).",
            dataset_id[:8],
        )

    def clear_all(self) -> None:
        """
        Clear the in-memory DataFrame cache.

        Redis caches are NOT cleared here (would require FLUSHDB, which is
        too destructive in a shared environment). Redis entries will expire
        via their TTLs.
        """
        self.dataframe_cache.clear()
        logger.info(
            "DataFrame cache cleared. Redis caches not cleared — entries will expire via TTL."
        )

    # ===== Stats / Monitoring =====

    async def get_stats_info(self) -> dict:
        """
        Return cache metadata for monitoring.

        Returns:
            dict: Backend type, TTL, and size information per cache.
        """
        return {
            "clustering": {"ttl_seconds": _TTL_CLUSTERING, "backend": "redis"},
            "volcano": {"ttl_seconds": _TTL_VOLCANO, "backend": "redis"},
            "stats": {"ttl_seconds": _TTL_STATS, "backend": "redis"},
            "dataframe": {
                "size": len(self.dataframe_cache),
                "maxsize": self.dataframe_cache.maxsize,
                "backend": "memory",
            },
        }

    async def get_cache_stats(self) -> dict:
        """Alias for get_stats_info() — kept for backward compatibility."""
        return await self.get_stats_info()


# Singleton instance
cache_service = CacheService()
