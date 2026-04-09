"""
Service for managing pre-computed sample correlations and distances.
Enables instant heatmap rendering by caching pairwise sample computations.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.models import SampleCorrelation
from app.core.monitoring import timing_decorator

logger = logging.getLogger(__name__)


class SampleCorrelationService:
    """Service for managing cached sample correlations."""

    @timing_decorator(name="get_cached_correlations")
    async def get_cached_correlations(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        method: str,
        metric: str,
        top_n_genes: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached sample correlations for a dataset.
        
        Args:
            db: Database session
            dataset_id: Dataset UUID
            method: Clustering method used
            metric: Distance metric used
            top_n_genes: Number of genes used for computation
            
        Returns:
            Dictionary with correlation matrix or None if not cached
        """
        query = select(SampleCorrelation).where(
            and_(
                SampleCorrelation.dataset_id == dataset_id,
                SampleCorrelation.method == method,
                SampleCorrelation.metric == metric,
                SampleCorrelation.top_n_genes == top_n_genes
            )
        )
        
        result = await db.execute(query)
        correlations = result.scalars().all()
        
        if not correlations:
            return None
        
        # Build correlation/distance matrix
        samples = set()
        for corr in correlations:
            samples.add(corr.sample_a)
            samples.add(corr.sample_b)
        
        samples_list = sorted(list(samples))
        n_samples = len(samples_list)
        sample_idx_map = {s: i for i, s in enumerate(samples_list)}
        
        # Initialize matrices
        corr_matrix = np.zeros((n_samples, n_samples))
        dist_matrix = np.zeros((n_samples, n_samples))
        
        # Fill matrices
        for corr in correlations:
            i = sample_idx_map[corr.sample_a]
            j = sample_idx_map[corr.sample_b]
            
            if corr.correlation is not None:
                corr_matrix[i, j] = corr.correlation
                corr_matrix[j, i] = corr.correlation
            
            if corr.distance is not None:
                dist_matrix[i, j] = corr.distance
                dist_matrix[j, i] = corr.distance
        
        # Diagonal is always 1 for correlation, 0 for distance
        np.fill_diagonal(corr_matrix, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)
        
        logger.info(f"✅ Retrieved cached correlations for {n_samples} samples")
        
        return {
            "samples": samples_list,
            "correlation_matrix": corr_matrix.tolist(),
            "distance_matrix": dist_matrix.tolist(),
            "method": method,
            "metric": metric,
            "top_n_genes": top_n_genes
        }

    @timing_decorator(name="cache_correlations")
    async def cache_correlations(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        samples: List[str],
        correlation_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        method: str,
        metric: str,
        top_n_genes: int
    ) -> int:
        """
        Cache pairwise sample correlations and distances.
        
        Args:
            db: Database session
            dataset_id: Dataset UUID
            samples: List of sample names
            correlation_matrix: NxN correlation matrix
            distance_matrix: NxN distance matrix
            method: Clustering method
            metric: Distance metric
            top_n_genes: Number of genes used
            
        Returns:
            Number of correlation pairs cached
        """
        from uuid import uuid4
        
        # First, delete existing correlations for this configuration
        await self.invalidate_cache(db, dataset_id, method, metric, top_n_genes)
        
        # Create SampleCorrelation objects for upper triangle (excluding diagonal)
        n_samples = len(samples)
        correlations = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):  # Upper triangle only
                corr_obj = SampleCorrelation(
                    id=uuid4(),
                    dataset_id=dataset_id,
                    sample_a=samples[i],
                    sample_b=samples[j],
                    correlation=float(correlation_matrix[i, j]) if correlation_matrix is not None else None,
                    distance=float(distance_matrix[i, j]) if distance_matrix is not None else None,
                    method=method,
                    metric=metric,
                    top_n_genes=top_n_genes
                )
                correlations.append(corr_obj)
        
        # Bulk insert
        db.add_all(correlations)
        await db.commit()
        
        logger.info(f"✅ Cached {len(correlations)} sample correlations for dataset {dataset_id}")
        
        return len(correlations)

    async def invalidate_cache(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        method: Optional[str] = None,
        metric: Optional[str] = None,
        top_n_genes: Optional[int] = None
    ) -> int:
        """
        Invalidate cached correlations for a dataset.
        
        Args:
            db: Database session
            dataset_id: Dataset UUID
            method: Optional method filter
            metric: Optional metric filter
            top_n_genes: Optional gene count filter
            
        Returns:
            Number of entries deleted
        """
        from sqlalchemy import delete
        
        conditions = [SampleCorrelation.dataset_id == dataset_id]
        
        if method is not None:
            conditions.append(SampleCorrelation.method == method)
        if metric is not None:
            conditions.append(SampleCorrelation.metric == metric)
        if top_n_genes is not None:
            conditions.append(SampleCorrelation.top_n_genes == top_n_genes)
        
        stmt = delete(SampleCorrelation).where(and_(*conditions))
        result = await db.execute(stmt)
        await db.commit()
        
        deleted = result.rowcount
        logger.info(f"🗑️ Invalidated {deleted} cached correlations for dataset {dataset_id}")
        
        return deleted


# Global instance
sample_correlation_service = SampleCorrelationService()
