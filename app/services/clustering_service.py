"""
Service for hierarchical clustering analysis of expression data.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.cluster import hierarchy
from scipy.spatial import distance
import fastcluster

class ClusteringService:
    """Service for performing hierarchical clustering on expression data."""

    def perform_clustering(
        self,
        df: pd.DataFrame,
        top_n_genes: int = 1000,
        gene_ids: Optional[List[str]] = None,
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        method: str = "ward",
        metric: str = "euclidean"
    ) -> Dict[str, Any]:
        """
        Perform hierarchical clustering on the provided DataFrame.
        
        Args:
            df: Expression DataFrame (rows=genes, cols=samples)
            top_n_genes: Number of most variable genes to keep (0 for all) - Ignored if gene_ids is provided
            gene_ids: Specific list of genes to include (e.g., DEGs)
            cluster_rows: Whether to cluster genes
            cluster_cols: Whether to cluster samples
            method: Linkage method ('ward', 'average', 'complete', 'single')
            metric: Distance metric ('euclidean', 'correlation', etc.)
            
        Returns:
            Dictionary containing result.
        """
        # 1. Filter genes
        if gene_ids and len(gene_ids) > 0:
             # Filter by specific gene IDs
             # Use intersection to avoid errors if some IDs are missing
             valid_genes = [g for g in gene_ids if g in df.index]
             if not valid_genes:
                 raise ValueError("None of the provided gene_ids were found in the dataset")
             df = df.loc[valid_genes]
        elif top_n_genes > 0 and len(df) > top_n_genes:
            # Calculate variance for each gene
            variances = df.var(axis=1)
            # Get indices of top N variable genes
            top_indices = variances.nlargest(top_n_genes).index
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
        means = np.mean(data_values, axis=1, keepdims=True)
        stds = np.std(data_values, axis=1, keepdims=True)
        stds[stds == 0] = 1.0 # Avoid div/0
        normalized_values = (data_values - means) / stds

        # 2. Cluster Rows (Genes)
        if cluster_rows and len(df) > 1:
            try:
                row_linkage = self._compute_linkage(normalized_values, method=method, metric=metric)
                result["row_dendrogram"] = row_linkage.tolist()
                result["row_order"] = hierarchy.leaves_list(row_linkage).tolist()
            except Exception as e:
                print(f"Error clustering rows: {e}")

        # 3. Cluster Cols (Samples)
        if cluster_cols and len(df.columns) > 1:
            try:
                # Transpose for column clustering
                col_linkage = self._compute_linkage(normalized_values.T, method=method, metric=metric)
                result["col_dendrogram"] = col_linkage.tolist()
                result["col_order"] = hierarchy.leaves_list(col_linkage).tolist()
            except Exception as e:
                print(f"Error clustering cols: {e}")

        # Reorder data according to clustering for easy heatmap rendering if client doesn't do it
        # Actually client usually needs raw data + order. 
        # But let's verify if we should reorder here. 
        # If we return 'data' as raw, and indices, client can map.
        # But simple clients prefer pre-ordered data.
        # Let's return raw data + indices, frontend (plotly) handles mapping or we map there.
        # However, for heatmap, we usually send x, y, and z.
        
        result["z"] = df.values.tolist() # raw values
        
        # Return z in row-major order ? No, just return as is, let client handle via row_order/col_order
        
        return result

    def _compute_linkage(self, data: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
        """
        Compute hierarchical clustering linkage matrix.
        """
        # Fastcluster is much faster than scipy for Euclidean/Ward
        # But specific combinations might need pdist
        
        if metric == 'euclidean' and method == 'ward':
             # fastcluster generic optimized path
             return fastcluster.linkage_vector(data, method=method, metric=metric)
        
        if metric == 'correlation':
             # fastcluster doesn't support correlation directly in linkage_vector commonly without pdist
             # transform correlation to distance
             d = distance.pdist(data, metric='correlation')
             return fastcluster.linkage(d, method=method)

        # Fallback to general generic
        try:
             # Try optimized vector linkage first
             return fastcluster.linkage_vector(data, method=method, metric=metric)
        except:
             # Fallback to pairwise distance calculation
             d = distance.pdist(data, metric=metric)
             return fastcluster.linkage(d, method=method)

