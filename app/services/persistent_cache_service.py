"""
Persistent cache service using database backend.

Provides hybrid caching strategy:
1. In-memory cache (fast, temporary) - from cache_service.py
2. Database cache (persistent, survives restarts) - this module

This service works alongside the in-memory cache for optimal performance.
"""
import logging
import hashlib
import json
from typing import Any, Optional
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import CachedComputation

logger = logging.getLogger(__name__)


class PersistentCacheService:
    """
    Database-backed cache service for computation results.
    
    Complements in-memory cache with persistent storage that:
    - Survives server restarts
    - Can be shared across multiple instances
    - Tracks usage metrics (hit_count, last_accessed)
    - Supports automatic expiration
    """
    
    @staticmethod
    def _generate_params_hash(params: dict) -> str:
        """
        Generate MD5 hash from parameters dictionary.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            str: MD5 hash (32 chars)
        """
        # Sort keys for stable hash
        params_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    async def get_cached(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        computation_type: str,
        params: dict
    ) -> Optional[dict]:
        """
        Get cached computation result.
        
        Args:
            db: Database session
            dataset_id: Dataset UUID
            computation_type: Type of computation (clustering, volcano, etc.)
            params: Computation parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        params_hash = self._generate_params_hash(params)
        
        # Query for cached entry
        query = select(CachedComputation).where(
            CachedComputation.dataset_id == dataset_id,
            CachedComputation.computation_type == computation_type,
            CachedComputation.params_hash == params_hash
        )
        
        result = await db.execute(query)
        cached = result.scalar_one_or_none()
        
        if not cached:
            logger.debug(f"DB cache MISS: {computation_type} for dataset {dataset_id}")
            return None
        
        # Check expiration
        if cached.is_expired:
            logger.info(f"DB cache EXPIRED: {computation_type} for dataset {dataset_id}")
            # Delete expired entry
            await db.delete(cached)
            await db.commit()
            return None
        
        # Update hit count and last accessed
        cached.hit_count += 1
        cached.last_accessed_at = datetime.now(timezone.utc)
        await db.commit()
        
        logger.info(f"DB cache HIT: {computation_type} for dataset {dataset_id} (hits: {cached.hit_count})")
        return cached.result_data
    
    async def set_cached(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        computation_type: str,
        params: dict,
        result_data: dict,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Cache computation result in database.
        
        Args:
            db: Database session
            dataset_id: Dataset UUID
            computation_type: Type of computation
            params: Computation parameters
            result_data: Result to cache
            ttl_seconds: Time to live in seconds (None = never expires)
        """
        params_hash = self._generate_params_hash(params)
        
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        
        # Check if entry already exists (upsert)
        query = select(CachedComputation).where(
            CachedComputation.dataset_id == dataset_id,
            CachedComputation.computation_type == computation_type,
            CachedComputation.params_hash == params_hash
        )
        
        result = await db.execute(query)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing entry
            existing.result_data = result_data
            existing.expires_at = expires_at
            existing.hit_count = 0  # Reset hit count on update
            existing.last_accessed_at = None
        else:
            # Create new entry
            cached = CachedComputation(
                dataset_id=dataset_id,
                computation_type=computation_type,
                params_hash=params_hash,
                result_data=result_data,
                expires_at=expires_at,
                hit_count=0
            )
            db.add(cached)
        
        await db.commit()
        logger.info(f"DB cache SET: {computation_type} for dataset {dataset_id}")
    
    async def invalidate_dataset(
        self,
        db: AsyncSession,
        dataset_id: UUID
    ) -> int:
        """
        Invalidate all cache entries for a dataset.
        Called when dataset is reprocessed or deleted.
        
        Args:
            db: Database session
            dataset_id: Dataset UUID
            
        Returns:
            Number of entries deleted
        """
        query = delete(CachedComputation).where(
            CachedComputation.dataset_id == dataset_id
        )
        
        result = await db.execute(query)
        await db.commit()
        
        count = result.rowcount
        logger.info(f"DB cache INVALIDATED: {count} entries for dataset {dataset_id}")
        return count
    
    async def cleanup_expired(
        self,
        db: AsyncSession
    ) -> int:
        """
        Clean up expired cache entries.
        Should be called periodically (e.g., daily cron job).
        
        Args:
            db: Database session
            
        Returns:
            Number of entries deleted
        """
        now = datetime.now(timezone.utc)
        
        query = delete(CachedComputation).where(
            CachedComputation.expires_at.isnot(None),
            CachedComputation.expires_at < now
        )
        
        result = await db.execute(query)
        await db.commit()
        
        count = result.rowcount
        logger.info(f"DB cache CLEANUP: {count} expired entries deleted")
        return count
    
    async def get_cache_stats(
        self,
        db: AsyncSession,
        dataset_id: Optional[UUID] = None
    ) -> dict:
        """
        Get cache statistics for monitoring.
        
        Args:
            db: Database session
            dataset_id: Optional dataset to filter by
            
        Returns:
            Cache statistics dictionary
        """
        from sqlalchemy import func
        
        # Base query
        query = select(
            CachedComputation.computation_type,
            func.count(CachedComputation.id).label('count'),
            func.sum(CachedComputation.hit_count).label('total_hits'),
            func.avg(CachedComputation.hit_count).label('avg_hits')
        ).group_by(CachedComputation.computation_type)
        
        if dataset_id:
            query = query.where(CachedComputation.dataset_id == dataset_id)
        
        result = await db.execute(query)
        rows = result.fetchall()
        
        stats = {
            "by_type": {},
            "total_entries": 0,
            "total_hits": 0
        }
        
        for row in rows:
            stats["by_type"][row.computation_type] = {
                "count": row.count,
                "total_hits": row.total_hits or 0,
                "avg_hits": float(row.avg_hits or 0)
            }
            stats["total_entries"] += row.count
            stats["total_hits"] += row.total_hits or 0
        
        return stats


# Singleton instance
persistent_cache_service = PersistentCacheService()
