"""
DEG expression-pattern clustering (à la DEGreport::degPatterns).

Group the significant DEGs of a comparison into clusters that share an expression
*trajectory* across conditions, and summarise each cluster as a median z-score
profile over the (ordered) groups. Complements the DEG heatmap: instead of a
gene × sample matrix, it answers "which coordinated up/down/transient patterns
do my DEGs fall into across conditions?".

Method (shape-focused, like degPatterns):
1. z-score each gene across samples,
2. collapse to a gene × group median-z matrix (robust to replicate noise),
3. cluster genes on that matrix (correlation distance + average linkage),
4. cut into k clusters, drop clusters smaller than min_cluster_size,
5. per cluster: median trajectory over groups + each gene's group trajectory.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DEGPatternsError(Exception):
    """Raised for recoverable DEG-patterns input problems (mapped to 4xx)."""


def _resolve_group_order(
    sample_condition_map: Dict[str, str],
    expression_cols: List[str],
    group_order: Optional[List[str]],
) -> List[str]:
    present = []
    seen = set()
    for s in expression_cols:
        g = sample_condition_map.get(s)
        if g is not None and g not in seen:
            seen.add(g)
            present.append(g)
    if group_order:
        # Keep requested order, but only groups actually present
        ordered = [g for g in group_order if g in seen]
        # Append any present groups the caller forgot to list
        ordered += [g for g in present if g not in set(ordered)]
        return ordered
    return present


def compute_deg_patterns(
    matrix_df: pd.DataFrame,
    gene_col: str,
    expression_cols: List[str],
    genes: List[str],
    sample_condition_map: Dict[str, str],
    group_order: Optional[List[str]] = None,
    n_clusters: int = 6,
    min_cluster_size: int = 15,
    max_genes: int = 2000,
) -> dict:
    """Cluster DEGs by expression trajectory across groups. See module docstring."""
    from scipy.cluster.hierarchy import fcluster

    from app.services.clustering_service import ClusteringService

    groups = _resolve_group_order(sample_condition_map, expression_cols, group_order)
    if len(groups) < 2:
        raise DEGPatternsError(
            "Need at least 2 conditions/groups (provide a sample→condition mapping)."
        )

    # Restrict to samples that have a group, grouped by condition
    group_to_samples: Dict[str, List[str]] = {g: [] for g in groups}
    for s in expression_cols:
        g = sample_condition_map.get(s)
        if g in group_to_samples:
            group_to_samples[g].append(s)
    usable_samples = [s for g in groups for s in group_to_samples[g]]

    # Subset to requested DEGs (case-insensitive)
    wanted = {str(g).strip().upper() for g in genes if g and str(g).strip()}
    n_requested = len(wanted)
    if not wanted:
        raise DEGPatternsError("No genes provided.")
    col_upper = matrix_df[gene_col].astype(str).str.upper()
    sub = matrix_df[col_upper.isin(wanted)].copy()
    if sub.empty:
        raise DEGPatternsError(
            "None of the DEG genes were found in the matrix (check gene id namespace)."
        )

    labels = sub[gene_col].astype(str).tolist()
    expr = sub[usable_samples].apply(pd.to_numeric, errors="coerce")

    # z-score per gene across samples; drop zero-variance genes
    row_mean = expr.mean(axis=1)
    row_std = expr.std(axis=1, ddof=0)
    keep = row_std > 0
    expr, row_mean, row_std = expr[keep], row_mean[keep], row_std[keep]
    labels = [g for g, k in zip(labels, keep.tolist()) if k]
    if expr.empty:
        raise DEGPatternsError("All DEG genes have zero variance across samples.")
    z = expr.sub(row_mean, axis=0).div(row_std, axis=0)

    # Collapse to gene × group median-z matrix
    group_median = pd.DataFrame(
        {g: z[group_to_samples[g]].median(axis=1) for g in groups}, index=z.index
    )

    # Cap for clustering performance (variance across groups = most informative)
    downsampled = False
    if len(group_median) > max_genes:
        top = group_median.var(axis=1).sort_values(ascending=False).head(max_genes).index
        group_median = group_median.loc[top]
        keep_idx = set(top)
        labels = [g for g, idx in zip(labels, z.index) if idx in keep_idx]
        downsampled = True

    n_genes = len(group_median)
    gene_labels = labels

    data = group_median.values  # genes × groups
    k = max(2, min(n_clusters, n_genes - 1)) if n_genes > 2 else 1

    if k == 1:
        cluster_ids = np.ones(n_genes, dtype=int)
    else:
        cs = ClusteringService()
        linkage = cs._compute_linkage(data, method="average", metric="correlation")
        cluster_ids = fcluster(linkage, t=k, criterion="maxclust")

    # Build clusters, drop those below min_cluster_size, renumber sequentially
    raw: Dict[int, List[int]] = {}
    for i, cid in enumerate(cluster_ids):
        raw.setdefault(int(cid), []).append(i)

    clusters = []
    for cid in sorted(raw, key=lambda c: -len(raw[c])):  # largest first
        idxs = raw[cid]
        if len(idxs) < min_cluster_size:
            continue
        sub_matrix = data[idxs, :]
        median_traj = np.median(sub_matrix, axis=0)
        gene_trajectories = [
            {"gene": gene_labels[i], "values": [float(v) for v in data[i, :]]}
            for i in idxs
        ]
        clusters.append({
            "id": len(clusters) + 1,
            "n_genes": len(idxs),
            "genes": [gene_labels[i] for i in idxs],
            "median": [float(v) for v in median_traj],
            "gene_trajectories": gene_trajectories,
        })

    n_in_clusters = sum(c["n_genes"] for c in clusters)
    return {
        "groups": groups,
        "n_deg_requested": n_requested,
        "n_deg_used": n_genes,
        "n_genes_clustered": n_in_clusters,
        "n_clusters": len(clusters),
        "min_cluster_size": min_cluster_size,
        "downsampled": downsampled,
        "clusters": clusters,
    }
