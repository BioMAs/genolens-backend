"""
Per-sample signature scoring.

Given an expression matrix (genes × samples) and a gene signature (list of gene
symbols/ids), compute one score per sample summarising how strongly the signature
is expressed in that sample. Two transparent methods are offered:

- ``mean_z`` (default): per-gene z-score across samples, then average over the
  signature genes. Robust, unbiased by signature size, interpretable (0 ≈ average
  sample, >0 higher-than-average signature expression).
- ``mean_rank``: per-sample gene ranking (AUCell/mean-rank style); score is the
  mean normalised rank of the signature genes in [0, 1]. Non-parametric, scale-free.

The heavy lifting is pure pandas/numpy; the endpoint handles I/O and grouping.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("mean_z", "mean_rank")


def _match_signature_rows(
    df: pd.DataFrame, gene_col: str, gene_list: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (matched rows, matched gene labels) using case-insensitive matching."""
    wanted = {g.strip().upper() for g in gene_list if g and str(g).strip()}
    if not wanted:
        return df.iloc[0:0], []
    col_upper = df[gene_col].astype(str).str.upper()
    mask = col_upper.isin(wanted)
    matched = df[mask]
    matched_labels = matched[gene_col].astype(str).tolist()
    return matched, matched_labels


def score_signature(
    df: pd.DataFrame,
    gene_col: str,
    expression_cols: List[str],
    gene_list: List[str],
    method: str = "mean_z",
) -> Dict:
    """Compute a per-sample signature score.

    Returns a dict with ``scores`` ({sample: score}), ``n_genes_used``,
    ``n_genes_requested`` and ``matched_genes``.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown scoring method: {method}. Options: {list(SUPPORTED_METHODS)}")

    n_requested = len({g.strip().upper() for g in gene_list if g and str(g).strip()})

    if method == "mean_rank":
        # Rank every gene within each sample, then average the signature genes' ranks.
        expr_all = df[expression_cols].apply(pd.to_numeric, errors="coerce")
        n_genes_total = len(expr_all)
        if n_genes_total < 2:
            return {"scores": {}, "n_genes_used": 0, "n_genes_requested": n_requested, "matched_genes": []}
        ranks = expr_all.rank(axis=0, method="average", na_option="keep")  # 1..N per column
        matched, matched_labels = _match_signature_rows(df, gene_col, gene_list)
        if matched.empty:
            return {"scores": {}, "n_genes_used": 0, "n_genes_requested": n_requested, "matched_genes": []}
        sig_ranks = ranks.loc[matched.index]
        # Normalise ranks to [0, 1] and average over signature genes per sample
        norm = sig_ranks / float(n_genes_total)
        scores = norm.mean(axis=0, skipna=True)
        return {
            "scores": {s: (float(v) if pd.notna(v) else None) for s, v in scores.items()},
            "n_genes_used": len(matched_labels),
            "n_genes_requested": n_requested,
            "matched_genes": matched_labels,
        }

    # method == "mean_z"
    matched, matched_labels = _match_signature_rows(df, gene_col, gene_list)
    if matched.empty:
        return {"scores": {}, "n_genes_used": 0, "n_genes_requested": n_requested, "matched_genes": []}

    expr = matched[expression_cols].apply(pd.to_numeric, errors="coerce")
    # Per-gene (row) z-score across samples
    row_mean = expr.mean(axis=1)
    row_std = expr.std(axis=1, ddof=0)
    # Drop zero-variance genes (no information, would divide by zero)
    keep = row_std > 0
    expr = expr[keep]
    row_mean = row_mean[keep]
    row_std = row_std[keep]
    kept_labels = matched[keep][gene_col].astype(str).tolist()
    if expr.empty:
        return {"scores": {}, "n_genes_used": 0, "n_genes_requested": n_requested, "matched_genes": []}

    z = expr.sub(row_mean, axis=0).div(row_std, axis=0)
    scores = z.mean(axis=0, skipna=True)
    return {
        "scores": {s: (float(v) if pd.notna(v) else None) for s, v in scores.items()},
        "n_genes_used": len(kept_labels),
        "n_genes_requested": n_requested,
        "matched_genes": kept_labels,
    }


def group_and_test(
    scores: Dict[str, float],
    sample_condition_map: Dict[str, str],
) -> Dict:
    """Group per-sample scores by condition and run a between-group test.

    Mann-Whitney U for 2 groups, Kruskal-Wallis for >2. Returns groups + test.
    """
    from scipy.stats import mannwhitneyu, kruskal

    groups: Dict[str, List[float]] = {}
    for sample, score in scores.items():
        if score is None:
            continue
        cond = sample_condition_map.get(sample)
        if cond is None:
            continue
        groups.setdefault(str(cond), []).append(float(score))

    group_list = [(name, vals) for name, vals in groups.items() if len(vals) >= 1]
    test_name = None
    stat = None
    pvalue = None

    testable = [vals for _, vals in group_list if len(vals) >= 2]
    if len(group_list) == 2 and all(len(v) >= 1 for _, v in group_list) and len(testable) == 2:
        try:
            stat, pvalue = mannwhitneyu(group_list[0][1], group_list[1][1], alternative="two-sided")
            test_name = "Mann-Whitney U"
        except ValueError as e:  # e.g. all identical
            logger.warning("Mann-Whitney U failed: %s", e)
    elif len(group_list) > 2 and len(testable) >= 2:
        try:
            stat, pvalue = kruskal(*[vals for _, vals in group_list if len(vals) >= 1])
            test_name = "Kruskal-Wallis"
        except ValueError as e:
            logger.warning("Kruskal-Wallis failed: %s", e)

    return {
        "groups": {name: vals for name, vals in group_list},
        "test": test_name,
        "stat": float(stat) if stat is not None else None,
        "pvalue": float(pvalue) if pvalue is not None else None,
    }
