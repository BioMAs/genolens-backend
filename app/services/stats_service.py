"""
Multiple testing correction service for GenoLens.

Implements four standard methods entirely with NumPy (no statsmodels dependency):

 - Benjamini-Hochberg (BH)   — FDR control [Benjamini & Hochberg 1995]
 - Bonferroni                — Family-wise error rate (FWER)
 - Holm-Bonferroni           — Step-down FWER [Holm 1979]
 - Benjamini-Yekutieli (BY)  — FDR control under arbitrary dependency [BY 2001]

All functions accept a Python list (or NumPy array) of raw p-values and
return a dict with corrected p-values + summary statistics.
"""
import math
from enum import Enum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Correction method enum
# ---------------------------------------------------------------------------

class CorrectionMethod(str, Enum):
    BH = "bh"                  # Benjamini-Hochberg (FDR)
    BONFERRONI = "bonferroni"  # Bonferroni (FWER)
    HOLM = "holm"              # Holm-Bonferroni (FWER, step-down)
    BY = "by"                  # Benjamini-Yekutieli (FDR, conservative)


# ---------------------------------------------------------------------------
# Core correction algorithms
# ---------------------------------------------------------------------------

def _bh(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg step-up FDR correction.

    For each sorted p-value p_i (rank i out of n):
        q_i = p_i * n / i

    Applied step-up (cumulative minimum from the right) to preserve monotonicity.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    # BH adjusted value for rank i (1-based): p_i * n / i
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks

    # Step-up: enforce monotonicity from right to left
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(n)
    result[order] = adjusted
    return result


def _by(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Yekutieli FDR correction (valid under arbitrary dependency).

    Same as BH but multiplied by the harmonic number c(n) = Σ 1/i for i=1..n.
    """
    n = len(pvalues)
    c_n = np.sum(1.0 / np.arange(1, n + 1))

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n * c_n / ranks

    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n)
    result[order] = adjusted
    return result


def _bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Bonferroni correction: p_adj = min(p * n, 1)."""
    return np.clip(pvalues * len(pvalues), 0.0, 1.0)


def _holm(pvalues: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni step-down correction.

    For sorted p-values (ascending), rank i (0-based):
        threshold_i = alpha / (n - i)
        adjusted_i  = p_i * (n - i)

    Step-down: enforce monotonicity from left to right (cumulative max).
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    factors = n - np.arange(n)          # n, n-1, n-2, …, 1
    adjusted = sorted_p * factors

    # Step-down: cumulative max (each value >= previous)
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n)
    result[order] = adjusted
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METHODS = {
    CorrectionMethod.BH: _bh,
    CorrectionMethod.BY: _by,
    CorrectionMethod.BONFERRONI: _bonferroni,
    CorrectionMethod.HOLM: _holm,
}


def correct_pvalues(
    pvalues: list[float],
    method: CorrectionMethod,
) -> list[float]:
    """
    Apply *method* correction to *pvalues*.

    NaN values are preserved as NaN; only finite non-NaN values are corrected.

    Returns a list of adjusted p-values (same length as input).
    """
    arr = np.array(pvalues, dtype=float)
    valid_mask = np.isfinite(arr)

    result = np.full_like(arr, fill_value=np.nan)
    if valid_mask.sum() > 0:
        result[valid_mask] = _METHODS[method](arr[valid_mask])

    return result.tolist()


# ---------------------------------------------------------------------------
# High-level comparison helper
# ---------------------------------------------------------------------------

def compute_correction_comparison(
    gene_ids: list[str],
    raw_pvalues: list[Optional[float]],
    original_padj: list[Optional[float]],
    methods: list[CorrectionMethod] | None = None,
    threshold: float = 0.05,
) -> dict:
    """
    Compute multiple testing corrections and return a rich comparison dict.

    Args:
        gene_ids: Gene identifiers (same order as pvalues).
        raw_pvalues: Raw nominal p-values from the statistical test.
        original_padj: Original adjusted p-values as stored in the DB (e.g. from DESeq2).
        methods: List of methods to apply. Defaults to all 4.
        threshold: Significance threshold (default 0.05).

    Returns a dict with:
        - ``summary``: per-method counts of significant genes
        - ``gene_results``: list of per-gene dicts with corrected values
        - ``n_tested``: number of genes with a valid p-value
        - ``threshold``: threshold used
    """
    if methods is None:
        methods = list(CorrectionMethod)

    n = len(gene_ids)
    raw_arr = np.array(
        [p if p is not None else np.nan for p in raw_pvalues],
        dtype=float,
    )
    orig_arr = np.array(
        [p if p is not None else np.nan for p in original_padj],
        dtype=float,
    )

    valid_mask = np.isfinite(raw_arr)
    n_tested = int(valid_mask.sum())

    # Apply each method to valid p-values
    corrected: dict[str, np.ndarray] = {}
    for method in methods:
        adj = np.full(n, fill_value=np.nan)
        if n_tested > 0:
            adj[valid_mask] = _METHODS[method](raw_arr[valid_mask])
        corrected[method.value] = adj

    # Build per-gene result rows
    gene_results = []
    for i in range(n):
        row: dict = {
            "gene_id": gene_ids[i],
            "pvalue": None if np.isnan(raw_arr[i]) else float(raw_arr[i]),
            "original_padj": None if np.isnan(orig_arr[i]) else float(orig_arr[i]),
        }
        for method in methods:
            val = corrected[method.value][i]
            row[f"padj_{method.value}"] = None if np.isnan(val) else float(val)
        gene_results.append(row)

    # Build summary
    summary: dict = {
        "original": {
            "n_significant": int(np.sum(orig_arr[valid_mask] < threshold)),
            "label": "Original (DESeq2/tool)",
            "description": "Adjusted p-values as provided in the uploaded file.",
        }
    }
    for method in methods:
        adj = corrected[method.value]
        n_sig = int(np.sum(adj[valid_mask] < threshold))
        labels = {
            CorrectionMethod.BH: ("Benjamini-Hochberg (BH)", "FDR control — standard for genomics. Less conservative than Bonferroni."),
            CorrectionMethod.BONFERRONI: ("Bonferroni", "FWER control — most conservative. Multiplies p-values by n."),
            CorrectionMethod.HOLM: ("Holm-Bonferroni", "FWER step-down — uniformly more powerful than Bonferroni."),
            CorrectionMethod.BY: ("Benjamini-Yekutieli (BY)", "FDR under arbitrary dependency — more conservative than BH."),
        }
        label, desc = labels[method]
        summary[method.value] = {
            "n_significant": n_sig,
            "label": label,
            "description": desc,
        }

    return {
        "n_total": n,
        "n_tested": n_tested,
        "threshold": threshold,
        "summary": summary,
        "gene_results": gene_results,
        "methods_applied": [m.value for m in methods],
    }
