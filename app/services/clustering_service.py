"""
Service for hierarchical clustering and K-means analysis of expression data.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import fastcluster
from app.core.monitoring import timing_decorator

logger = logging.getLogger(__name__)

class ClusteringService:
    """Service for performing hierarchical clustering on expression data."""

    @timing_decorator(name="perform_clustering")
    def perform_clustering(
        self,
        df: pd.DataFrame,
        top_n_genes: int = 1000,
        gene_ids: Optional[List[str]] = None,
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        method: str = "ward",
        metric: str = "euclidean",
        sort_by: Optional[str] = None,
        max_genes_for_clustering: int = 2000,
        precomputed_col_clustering: Optional[Dict[str, Any]] = None,
        n_clusters: Optional[int] = None,  # For K-means
        return_zscore: bool = False  # If True, z in result is normalized [-1, 1]
    ) -> Dict[str, Any]:
        """
        Perform hierarchical or K-means clustering on the provided DataFrame.
        
        Args:
            df: Expression DataFrame (rows=genes, cols=samples)
            top_n_genes: Number of most variable genes to keep (0 for all)
            gene_ids: Specific list of genes to include (e.g., DEGs)
            cluster_rows: Whether to cluster genes
            cluster_cols: Whether to cluster samples
            method: Linkage method ('ward', 'average', 'complete', 'single', 'kmeans')
            metric: Distance metric ('euclidean', 'manhattan', 'correlation', 'cosine')
            sort_by: If cluster_rows is False, sort genes by this column
            max_genes_for_clustering: Maximum genes to cluster (downsample if exceeded)
            n_clusters: Number of clusters for K-means (default: auto)
            
        Returns:
            Dictionary containing clustering result.
        """
        # 1. Filter genes
        if gene_ids and len(gene_ids) > 0:
             # Filter by specific gene IDs
             # Use intersection to avoid errors if some IDs are missing
             valid_genes = [g for g in gene_ids if g in df.index]
             if not valid_genes:
                 raise ValueError(
                     f"None of the {len(gene_ids)} provided gene_ids were found in the dataset. "
                     f"Dataset has {len(df)} genes. Check gene ID format compatibility."
                 )
             if len(valid_genes) < len(gene_ids):
                 missing_count = len(gene_ids) - len(valid_genes)
                 logger.warning(f"⚠️ {missing_count}/{len(gene_ids)} genes not found in dataset, using {len(valid_genes)} valid genes")
             df = df.loc[valid_genes]
        elif top_n_genes > 0 and len(df) > top_n_genes:
            # Calculate variance for each gene
            variances = df.var(axis=1)
            # Get indices of top N variable genes
            top_indices = variances.nlargest(top_n_genes).index
            df = df.loc[top_indices]
        
        # Intelligent downsampling: if we have too many genes for clustering, reduce to top variable genes
        if cluster_rows and len(df) > max_genes_for_clustering:
            logger.info(f"⚡ Downsampling from {len(df)} to {max_genes_for_clustering} genes for clustering performance")
            variances = df.var(axis=1)
            top_indices = variances.nlargest(max_genes_for_clustering).index
            df = df.loc[top_indices]
        
        # Prepare result structure
        result = {
            "row_labels": df.index.tolist(),
            "col_labels": df.columns.tolist(),
            "data": df.values.tolist(), # Can be large, consider sending separately or client-side format
            "row_order": list(range(len(df))),
            "col_order": list(range(len(df.columns))),
            "row_dendrogram": None,
            "col_dendrogram": None
        }

        # Pre-compute Z-scores for correct clustering of expression profiles
        # We cluster on normalized data so genes with similar *shapes* are grouped, ignoring magnitude.
        data_values = df.values.astype(float)
        
        # Handle NaN and Inf values by imputing with gene mean (per row)
        if np.any(np.isnan(data_values)) or np.any(np.isinf(data_values)):
            nan_count = np.sum(np.isnan(data_values)) + np.sum(np.isinf(data_values))
            logger.warning(f"⚠️ Found {nan_count} NaN/Inf values in expression data, imputing with gene mean")
            data_values = np.where(np.isinf(data_values), np.nan, data_values)
            row_means = np.nanmean(data_values, axis=1, keepdims=True)
            row_means = np.where(np.isnan(row_means), 0.0, row_means)
            nan_mask = np.isnan(data_values)
            data_values[nan_mask] = np.take(row_means.flatten(), np.where(nan_mask)[0])
        
        means = np.mean(data_values, axis=1, keepdims=True)
        stds = np.std(data_values, axis=1, keepdims=True)
        stds[stds == 0] = 1.0 # Avoid div/0
        normalized_values = (data_values - means) / stds
        
        # Validate normalized data
        if np.any(np.isnan(normalized_values)) or np.any(np.isinf(normalized_values)):
            logger.warning("Normalization produced NaN/Inf values, replacing with zeros")
            normalized_values = np.nan_to_num(normalized_values, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Cluster Rows (Genes) with hierarchical or K-means
        if cluster_rows and len(df) > 1:
            if method == 'kmeans':
                # K-means clustering
                try:
                    k = n_clusters if n_clusters else min(10, len(df) // 10)
                    k = max(2, min(k, len(df)))  # Ensure valid k
                    
                    logger.info(f"⚡ K-means clustering {len(df)} genes into {k} clusters")
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(normalized_values)
                    
                    # Sort genes by cluster assignment
                    sorted_indices = np.argsort(cluster_labels)
                    result["row_order"] = sorted_indices.tolist()
                    result["row_clusters"] = cluster_labels.tolist()
                    result["n_clusters"] = k
                    logger.info(f"✓ K-means completed: {k} clusters")
                except Exception as e:
                    raise ValueError(
                        f"K-means clustering failed: {str(e)}. "
                        f"Try hierarchical clustering or check data quality."
                    )
            else:
                # Hierarchical clustering
                try:
                    row_linkage = self._compute_linkage(normalized_values, method=method, metric=metric)
                    result["row_dendrogram"] = row_linkage.tolist()
                    result["row_order"] = hierarchy.leaves_list(row_linkage).tolist()
                except Exception as e:
                    raise ValueError(
                        f"Failed to cluster rows (genes) with method='{method}', metric='{metric}': {str(e)}. "
                        f"Try different method/metric combination or check data quality."
                    )
        elif not cluster_rows and sort_by and sort_by in df.columns:
            # Sort genes by specified column (e.g., logFC) instead of clustering
            logger.info(f"⚡ Sorting {len(df)} genes by '{sort_by}' instead of clustering")
            sorted_indices = df[sort_by].abs().sort_values(ascending=False).index
            row_order_map = {gene: i for i, gene in enumerate(df.index)}
            result["row_order"] = [row_order_map[gene] for gene in sorted_indices]

        # 3. Cluster Cols (Samples) - use precomputed if available
        if cluster_cols and len(df.columns) > 1:
            if precomputed_col_clustering:
                # Use pre-computed sample clustering (MUCH faster!)
                logger.info(f"⚡ Using pre-computed sample clustering ({len(df.columns)} samples)")
                result["col_dendrogram"] = precomputed_col_clustering.get("col_dendrogram")
                result["col_order"] = precomputed_col_clustering.get("col_order")
            else:
                # Compute on the fly
                try:
                    # Transpose for column clustering
                    col_linkage = self._compute_linkage(normalized_values.T, method=method, metric=metric)
                    result["col_dendrogram"] = col_linkage.tolist()
                    result["col_order"] = hierarchy.leaves_list(col_linkage).tolist()
                except Exception as e:
                    raise ValueError(
                        f"Failed to cluster columns (samples) with method='{method}', metric='{metric}': {str(e)}. "
                        f"Try different method/metric combination or check data quality."
                    )

        # Reorder data according to clustering for easy heatmap rendering if client doesn't do it
        # Actually client usually needs raw data + order. 
        # But let's verify if we should reorder here. 
        # If we return 'data' as raw, and indices, client can map.
        # But simple clients prefer pre-ordered data.
        # Let's return raw data + indices, frontend (plotly) handles mapping or we map there.
        # However, for heatmap, we usually send x, y, and z.
        
        if return_zscore:
            # Return z-scores normalized to [-1, 1] so the frontend can render directly
            z_values = normalized_values  # already z-scored
            # Scale each row to [-1, 1]
            max_abs = np.abs(z_values).max(axis=1, keepdims=True)
            max_abs[max_abs == 0] = 1.0
            result["z"] = (z_values / max_abs).tolist()
        else:
            result["z"] = df.values.tolist()  # raw values

        return result

    def _compute_linkage(self, data: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
        """
        Compute hierarchical clustering linkage matrix.
        Supports multiple distance metrics with proper mapping.
        """
        # Metric name mapping (user-friendly → scipy/fastcluster names)
        METRIC_MAPPING = {
            'manhattan': 'cityblock',  # scipy uses 'cityblock' for L1 distance
            'cityblock': 'cityblock',
            'euclidean': 'euclidean',
            'correlation': 'correlation',
            'cosine': 'cosine',
            'hamming': 'hamming',
            'jaccard': 'jaccard'
        }
        
        # Map user metric to internal metric
        internal_metric = METRIC_MAPPING.get(metric, metric)
        
        # Validate method/metric combination
        if method == 'ward' and internal_metric != 'euclidean':
            raise ValueError(
                f"Ward linkage requires euclidean metric, got '{metric}'. "
                f"Use 'average', 'complete', or 'single' with other metrics."
            )
        
        # Optimized paths for specific metrics
        if internal_metric == 'euclidean' and method == 'ward':
            # fastcluster optimized path for Ward+Euclidean
            return fastcluster.linkage_vector(data, method=method, metric='euclidean')
        
        if internal_metric == 'euclidean':
            # fastcluster optimized for euclidean (any method)
            return fastcluster.linkage_vector(data, method=method, metric='euclidean')
        
        # For all other metrics, use pdist path (more compatible)
        if internal_metric in ['correlation', 'cosine', 'cityblock', 'hamming', 'jaccard']:
            try:
                d = distance.pdist(data, metric=internal_metric)
                return fastcluster.linkage(d, method=method)
            except Exception as e:
                raise ValueError(
                    f"Linkage computation failed with metric '{metric}': {str(e)}. "
                    f"Supported metrics: euclidean, manhattan, correlation, cosine."
                )
        
        # Generic fallback for unknown metrics
        try:
            d = distance.pdist(data, metric=internal_metric)
            return fastcluster.linkage(d, method=method)
        except Exception as e:
            raise ValueError(
                f"Unsupported metric '{metric}'. Use: euclidean, manhattan, correlation, or cosine."
            )

    @timing_decorator(name="precompute_sample_clustering")
    def precompute_sample_clustering(
        self,
        df: pd.DataFrame,
        method: str = "ward",
        metric: str = "euclidean",
        top_n_genes: int = 2000
    ) -> Dict[str, Any]:
        """
        Pre-compute sample (column) clustering for a full dataset.
        This can be done once per dataset and reused for all visualizations.
        
        Args:
            df: Full expression DataFrame (rows=genes, cols=samples)
            method: Linkage method
            metric: Distance metric
            top_n_genes: Use top N most variable genes for clustering
            
        Returns:
            Dictionary with column dendrogram and optimal ordering
        """
        # For sample clustering, use most variable genes to get meaningful clusters
        if len(df) > top_n_genes:
            variances = df.var(axis=1)
            top_indices = variances.nlargest(top_n_genes).index
            df_subset = df.loc[top_indices]
        else:
            df_subset = df
        
        # Normalize for clustering
        data_values = df_subset.values.astype(float)
        means = np.mean(data_values, axis=1, keepdims=True)
        stds = np.std(data_values, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        normalized_values = (data_values - means) / stds
        
        # Cluster samples (transpose)
        col_linkage = self._compute_linkage(normalized_values.T, method=method, metric=metric)
        col_order = hierarchy.leaves_list(col_linkage).tolist()
        
        return {
            "col_labels": df.columns.tolist(),
            "col_dendrogram": col_linkage.tolist(),
            "col_order": col_order,
            "method": method,
            "metric": metric,
            "genes_used": len(df_subset)
        }

    def compute_silhouette_profile(
        self,
        df: pd.DataFrame,
        top_n_genes: int = 500,
        metric: str = "euclidean",
        k_min: int = 2,
        k_max: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute silhouette scores for k in [k_min..k_max] to help choose optimal k.

        Returns profile list and the recommended k (highest silhouette score).
        """
        # Select top variable genes
        if top_n_genes > 0 and len(df) > top_n_genes:
            variances = df.var(axis=1)
            df = df.loc[variances.nlargest(top_n_genes).index]

        n_genes = len(df)
        if n_genes < 20:
            raise ValueError(f"Not enough genes for silhouette analysis ({n_genes} < 20)")

        # Z-score normalize
        data = df.values.astype(float)
        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        normalized = (data - means) / stds
        normalized = np.nan_to_num(normalized, nan=0.0)

        k_max_effective = min(k_max, n_genes - 1)
        profile = []
        for k in range(k_min, k_max_effective + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(normalized)
            score = float(silhouette_score(normalized, labels, metric=metric))
            profile.append({
                "k": k,
                "silhouette_score": round(score, 4),
                "inertia": round(float(km.inertia_), 2),
            })
            logger.info(f"Silhouette k={k}: {score:.4f}")

        recommended = max(profile, key=lambda x: x["silhouette_score"])
        return {
            "profile": profile,
            "recommended_k": recommended["k"],
            "recommended_score": recommended["silhouette_score"],
        }
