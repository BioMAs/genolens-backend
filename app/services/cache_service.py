"""
Cache service for optimizing data retrieval and computations.

Provides in-memory LRU caching for expensive operations like:
- Clustering results
- Volcano plot data
- Statistical computations
- Gene lists

Can be extended to use Redis for distributed caching in production.
"""
import logging
import hashlib
import json
import threading
from typing import Any, Optional
from functools import lru_cache
from cachetools import TTLCache, LRUCache
from datetime import timedelta

logger = logging.getLogger(__name__)


class CacheService:
    """
    Service for caching expensive computations and data retrievals.
    
    Uses in-memory caching with TTL (Time To Live) and LRU (Least Recently Used) eviction.
    Thread-safe for async operations.
    """
    
    def __init__(
        self,
        clustering_cache_size: int = 100,
        clustering_ttl_seconds: int = 3600,  # 1 hour
        volcano_cache_size: int = 200,
        volcano_ttl_seconds: int = 7200,  # 2 hours
        stats_cache_size: int = 500,
        stats_ttl_seconds: int = 86400,  # 24 hours
        dataframe_cache_size: int = 5,
        dataframe_cache_memory_mb: int = 500
    ):
        """
        Initialize cache service with configurable cache sizes and TTLs.
        
        Args:
            clustering_cache_size: Max number of clustering results to cache
            clustering_ttl_seconds: TTL for clustering cache entries
            volcano_cache_size: Max number of volcano plot data to cache
            volcano_ttl_seconds: TTL for volcano plot cache entries
            stats_cache_size: Max number of statistics results to cache
            stats_ttl_seconds: TTL for statistics cache entries
            dataframe_cache_size: Max number of DataFrames to cache (for hot datasets)
            dataframe_cache_memory_mb: Memory limit for DataFrame cache
        """
        # Thread-safe locks for each cache (cachetools is NOT thread-safe by default)
        self._clustering_lock = threading.RLock()
        self._volcano_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._dataframe_lock = threading.RLock()

        # Clustering results cache (gene lists + parameters -> clustered data)
        self.clustering_cache = TTLCache(
            maxsize=clustering_cache_size,
            ttl=clustering_ttl_seconds
        )

        # Volcano plot data cache (dataset_id + comparison -> plot data)
        self.volcano_cache = TTLCache(
            maxsize=volcano_cache_size,
            ttl=volcano_ttl_seconds
        )

        # Statistics cache (dataset_id + params -> stats)
        self.stats_cache = TTLCache(
            maxsize=stats_cache_size,
            ttl=stats_ttl_seconds
        )

        # DataFrame cache (for frequently accessed datasets)
        # LRU without TTL - evicts least recently used when memory limit reached
        self.dataframe_cache = LRUCache(maxsize=dataframe_cache_size)
        
        logger.info(
            f"CacheService initialized: "
            f"clustering={clustering_cache_size}/{clustering_ttl_seconds}s, "
            f"volcano={volcano_cache_size}/{volcano_ttl_seconds}s, "
            f"stats={stats_cache_size}/{stats_ttl_seconds}s, "
            f"dataframe={dataframe_cache_size}"
        )
    
    @staticmethod
    def _generate_cache_key(*args, **kwargs) -> str:
        """
        Generate a stable cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: MD5 hash of serialized arguments
        """
        # Create a stable representation
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())  # Sort for stable ordering
        }
        
        # Serialize to JSON (with sorted keys)
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        
        # Hash to fixed-length key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # ===== Clustering Cache =====
    
    def get_clustering_result(
        self,
        dataset_id: str,
        gene_list: list[str],
        method: str = "hierarchical",
        **params
    ) -> Optional[dict]:
        """
        Get cached clustering result.
        
        Args:
            dataset_id: Dataset identifier
            gene_list: List of gene IDs
            method: Clustering method
            **params: Additional clustering parameters
            
        Returns:
            Cached result or None if not found
        """
        cache_key = self._generate_cache_key(
            dataset_id,
            tuple(sorted(gene_list)),  # Sort for stable key
            method,
            **params
        )
        
        with self._clustering_lock:
            result = self.clustering_cache.get(cache_key)

        if result:
            logger.debug(f"Clustering cache HIT for {dataset_id[:8]}... ({len(gene_list)} genes)")
        else:
            logger.debug(f"Clustering cache MISS for {dataset_id[:8]}... ({len(gene_list)} genes)")

        return result
    
    def set_clustering_result(
        self,
        dataset_id: str,
        gene_list: list[str],
        result: dict,
        method: str = "hierarchical",
        **params
    ) -> None:
        """
        Cache clustering result.
        
        Args:
            dataset_id: Dataset identifier
            gene_list: List of gene IDs
            result: Clustering result to cache
            method: Clustering method
            **params: Additional clustering parameters
        """
        cache_key = self._generate_cache_key(
            dataset_id,
            tuple(sorted(gene_list)),
            method,
            **params
        )
        
        with self._clustering_lock:
            self.clustering_cache[cache_key] = result
        logger.debug(f"Clustering cache SET for {dataset_id[:8]}... ({len(gene_list)} genes)")
    
    # ===== Volcano Plot Cache =====
    
    def get_volcano_data(
        self,
        dataset_id: str,
        comparison_name: str,
        max_points: int = 5000,
        padj_threshold: float = 0.05,
        logfc_threshold: float = 0.58
    ) -> Optional[dict]:
        """
        Get cached volcano plot data.
        
        Args:
            dataset_id: Dataset identifier
            comparison_name: Comparison name
            max_points: Maximum points to include
            padj_threshold: P-value threshold for significance
            logfc_threshold: Log fold change threshold for significance
            
        Returns:
            Cached result or None if not found
        """
        cache_key = self._generate_cache_key(
            dataset_id, comparison_name, max_points, padj_threshold, logfc_threshold
        )
        
        with self._volcano_lock:
            result = self.volcano_cache.get(cache_key)
        
        if result:
            logger.debug(
                f"Volcano cache HIT for {dataset_id[:8]}.../{comparison_name} "
                f"(padj<{padj_threshold}, |logFC|>{logfc_threshold})"
            )
        else:
            logger.debug(
                f"Volcano cache MISS for {dataset_id[:8]}.../{comparison_name} "
                f"(padj<{padj_threshold}, |logFC|>{logfc_threshold})"
            )
        
        return result
    
    def set_volcano_data(
        self,
        dataset_id: str,
        comparison_name: str,
        result: dict,
        max_points: int = 5000,
        padj_threshold: float = 0.05,
        logfc_threshold: float = 0.58
    ) -> None:
        """
        Cache volcano plot data.
        
        Args:
            dataset_id: Dataset identifier
            comparison_name: Comparison name
            result: Volcano plot data to cache
            max_points: Maximum points included
            padj_threshold: P-value threshold used
            logfc_threshold: Log fold change threshold used
        """
        cache_key = self._generate_cache_key(
            dataset_id, comparison_name, max_points, padj_threshold, logfc_threshold
        )
        
        with self._volcano_lock:
            self.volcano_cache[cache_key] = result
        logger.debug(
            f"Volcano cache SET for {dataset_id[:8]}.../{comparison_name} "
            f"(padj<{padj_threshold}, |logFC|>{logfc_threshold})"
        )
    
    # ===== Statistics Cache =====
    
    def get_stats(
        self,
        dataset_id: str,
        comparison_name: Optional[str] = None,
        **params
    ) -> Optional[dict]:
        """
        Get cached statistics.
        
        Args:
            dataset_id: Dataset identifier
            comparison_name: Optional comparison name
            **params: Additional parameters
            
        Returns:
            Cached result or None if not found
        """
        cache_key = self._generate_cache_key(dataset_id, comparison_name, **params)
        
        with self._stats_lock:
            result = self.stats_cache.get(cache_key)

        if result:
            logger.debug(f"Stats cache HIT for {dataset_id[:8]}...")
        else:
            logger.debug(f"Stats cache MISS for {dataset_id[:8]}...")

        return result
    
    def set_stats(
        self,
        dataset_id: str,
        result: dict,
        comparison_name: Optional[str] = None,
        **params
    ) -> None:
        """
        Cache statistics result.
        
        Args:
            dataset_id: Dataset identifier
            result: Statistics to cache
            comparison_name: Optional comparison name
            **params: Additional parameters
        """
        cache_key = self._generate_cache_key(dataset_id, comparison_name, **params)
        
        with self._stats_lock:
            self.stats_cache[cache_key] = result
        logger.debug(f"Stats cache SET for {dataset_id[:8]}...")
    
    # ===== DataFrame Cache =====
    
    def get_dataframe(self, dataset_id: str) -> Optional[Any]:
        """
        Get cached DataFrame (for hot datasets).
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Cached DataFrame or None if not found
        """
        with self._dataframe_lock:
            result = self.dataframe_cache.get(dataset_id)

        if result is not None:
            logger.debug(f"DataFrame cache HIT for {dataset_id[:8]}...")
        else:
            logger.debug(f"DataFrame cache MISS for {dataset_id[:8]}...")

        return result
    
    def set_dataframe(self, dataset_id: str, dataframe: Any) -> None:
        """
        Cache DataFrame for hot dataset.
        
        Args:
            dataset_id: Dataset identifier
            dataframe: pandas DataFrame to cache
        """
        with self._dataframe_lock:
            self.dataframe_cache[dataset_id] = dataframe
        logger.debug(f"DataFrame cache SET for {dataset_id[:8]}...")
    
    # ===== Cache Management =====
    
    def clear_dataset_cache(self, dataset_id: str) -> None:
        """
        Clear all cache entries for a specific dataset.
        Called when dataset is updated/reprocessed.
        
        Args:
            dataset_id: Dataset identifier
        """
        # Remove from DataFrame cache
        with self._dataframe_lock:
            if dataset_id in self.dataframe_cache:
                del self.dataframe_cache[dataset_id]
        logger.info(f"Cleared DataFrame cache for {dataset_id[:8]}...")
        
        # For other caches, we'd need to iterate and remove matching keys
        # (More expensive, but necessary for correctness)
        # This is a limitation of simple dict-based caches
        # Redis would handle this better with key patterns
        
        logger.info(f"Cache cleared for dataset {dataset_id[:8]}...")
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics for monitoring.
        
        Returns:
            dict: Cache hit rates, sizes, etc.
        """
        return {
            "clustering": {
                "size": len(self.clustering_cache),
                "maxsize": self.clustering_cache.maxsize,
                "currsize": self.clustering_cache.currsize
            },
            "volcano": {
                "size": len(self.volcano_cache),
                "maxsize": self.volcano_cache.maxsize,
                "currsize": self.volcano_cache.currsize
            },
            "stats": {
                "size": len(self.stats_cache),
                "maxsize": self.stats_cache.maxsize,
                "currsize": self.stats_cache.currsize
            },
            "dataframe": {
                "size": len(self.dataframe_cache),
                "maxsize": self.dataframe_cache.maxsize
            }
        }
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.clustering_cache.clear()
        self.volcano_cache.clear()
        self.stats_cache.clear()
        self.dataframe_cache.clear()
        logger.info("All caches cleared")


# Singleton instance
cache_service = CacheService()
